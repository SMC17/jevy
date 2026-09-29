"""N products × M sleeves paper desk.

Spot paths are a shared index factor plus an idiosyncratic shock. Surfaces
are refit from synthetic quotes on ``fit_stride``. Sleeves share that book
and keep separate PnL streams. Residual PnL strips index beta, the gamma
bucket, and (by default) the vega bucket.

No live orders. No OPRA tape. ``synthetic_fixture`` stays 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from jev_omm.decisions.client import DeterministicFallbackClient
from jev_omm.decisions.policy import apply_desk_policy
from jev_omm.decisions.schemas import build_desk_questions, build_mm_state
from jev_omm.desk.allocator import DEFAULT_CORR_CAP, DEFAULT_MAX_WEIGHT, allocate
from jev_omm.desk.fixtures import EXPIRIES, LOG_MONEYNESS, UNDERLIERS, UnderlierSpec
from jev_omm.desk.scoreboard import (
    DeskScoreboard,
    SleeveRow,
    cumulative_drawdown,
    per_step_sharpe,
    to_markdown,
)
from jev_omm.desk.sleeves import RISK_BUDGET, SLEEVE_IDS, SleeveContext, quote_or_target
from jev_omm.pnl.residual import pearson, spearman, strip_residual
from jev_omm.surface.book import SurfaceBook, quotes_from_svi

_CLIENT = DeterministicFallbackClient()


@dataclass
class DeskConfig:
    products: tuple[str, ...] = ("EQ_INDEX", "EQ_SINGLE", "FX_PAIR")
    sleeves: tuple[str, ...] = SLEEVE_IDS
    n_steps: int = 80
    seed: int = 11
    dt: float = 1.0 / 252.0
    max_abs_delta: float = 1.0e9
    max_abs_gamma: float = 1.0e9
    max_abs_vega: float = 1.0e9
    max_sleeve_weight: float = DEFAULT_MAX_WEIGHT
    corr_cap: float = DEFAULT_CORR_CAP
    research_fee: float = 0.001
    fit_stride: int = 20
    strip_vega: bool = True
    gates_on: bool = True
    jev_desk: bool = False
    enabled: dict[str, bool] | None = None
    synthetic_fixture: int = 1


@dataclass
class _NamePath:
    spec: UnderlierSpec
    product_return: np.ndarray
    skew: np.ndarray
    d_skew: np.ndarray
    fly: np.ndarray
    d_fly: np.ndarray
    term: np.ndarray
    d_term: np.ndarray
    tox: np.ndarray
    gex: np.ndarray
    instability: np.ndarray
    box: np.ndarray
    d_box: np.ndarray
    iv: np.ndarray
    d_sigma: np.ndarray


@dataclass
class DeskRun:
    config: DeskConfig
    book: SurfaceBook
    scoreboard: DeskScoreboard
    raw: dict[str, np.ndarray]
    residual: dict[str, np.ndarray]
    f_beta: np.ndarray
    f_gamma: dict[str, np.ndarray]
    f_vega: dict[str, np.ndarray]
    weights: dict[str, float]
    suspect_fits: int = 0
    n_fits: int = 0
    scale_path: np.ndarray = field(default_factory=lambda: np.zeros(0))
    markdown: str = ""


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _simulate_names(cfg: DeskConfig) -> tuple[np.ndarray, np.ndarray, dict[str, _NamePath], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_steps
    index_ret = rng.normal(0.0, 0.01, n)
    smile = rng.normal(0.0, 1.0, n)
    stress = rng.normal(0.0, 1.0, n)
    vol_shock = rng.normal(0.0, 1.0, n)
    rate_shock = rng.normal(0.0, 1.0, n)
    fill_rng = np.random.default_rng(cfg.seed + 17)
    fills = fill_rng.random((len(cfg.products), len(cfg.sleeves), n))
    prev_index = np.zeros(n)
    if n > 1:
        prev_index[1:] = index_ret[:-1]
    names: dict[str, _NamePath] = {}
    for spec_id in cfg.products:
        spec = UNDERLIERS[spec_id]
        product_return = np.zeros(n)
        skew = np.zeros(n)
        d_skew = np.zeros(n)
        fly = np.zeros(n)
        d_fly = np.zeros(n)
        term = np.zeros(n)
        d_term = np.zeros(n)
        tox = np.zeros(n)
        gex = np.zeros(n)
        inst = np.zeros(n)
        box = np.zeros(n)
        d_box = np.zeros(n)
        iv = np.zeros(n)
        d_sigma = np.zeros(n)
        s = f = tm = 0.0
        tx = 0.20
        gx = 0.30
        bx = 0.0
        iv_level = spec.iv0
        for t in range(n):
            idio = float(rng.normal(0.0, spec.idio_vol))
            product_return[t] = spec.beta_to_index * float(index_ret[t]) + idio
            skew[t] = s
            fly[t] = f
            term[t] = tm
            tox[t] = tx
            gex[t] = gx
            inst[t] = abs(gx)
            box[t] = bx
            iv[t] = iv_level
            shock_s = 0.85 * float(smile[t]) + 0.15 * float(rng.normal())
            shock_f = 0.80 * float(smile[t]) + 0.20 * float(rng.normal())
            s2 = 0.78 * s + 0.035 * shock_s
            f2 = 0.78 * f + 0.030 * shock_f
            tm2 = 0.88 * tm + 0.02 * float(rng.normal())
            tx2 = _clip(0.70 * tx + 0.28 * max(float(stress[t]), 0.0), 0.0, 1.0)
            gx2 = 0.82 * gx - 0.70 * float(stress[t])
            bx2 = 0.90 * bx + 0.015 * float(rate_shock[t])
            iv2 = _clip(spec.iv0 + 0.92 * (iv_level - spec.iv0) + 0.006 * float(vol_shock[t]), 0.04, 0.80)
            d_skew[t] = s2 - s
            d_fly[t] = f2 - f
            d_term[t] = tm2 - tm
            d_box[t] = bx2 - bx
            d_sigma[t] = iv2 - iv_level
            s, f, tm, tx, gx, bx, iv_level = s2, f2, tm2, tx2, gx2, bx2, iv2
        names[spec_id] = _NamePath(
            spec=spec,
            product_return=product_return,
            skew=skew,
            d_skew=d_skew,
            fly=fly,
            d_fly=d_fly,
            term=term,
            d_term=d_term,
            tox=tox,
            gex=gex,
            instability=inst,
            box=box,
            d_box=d_box,
            iv=iv,
            d_sigma=d_sigma,
        )
    return index_ret, prev_index, names, fills


def _fit_book(cfg: DeskConfig, names: dict[str, _NamePath], t: int, book: SurfaceBook) -> int:
    """Refit every name. Returns how many slices came back suspect."""
    suspect = 0
    ks = list(LOG_MONEYNESS)
    for spec_id, path in names.items():
        spec = path.spec
        if spec_id not in book.underliers:
            book.add_underlier(
                spec_id,
                asset_class=spec.asset_class,
                beta_to_index=spec.beta_to_index,
                spot=spec.spot,
                rate=spec.rate,
                div_yield=spec.div_yield,
                synthetic_fixture=cfg.synthetic_fixture,
            )
        quotes = {}
        for expiry, params in spec.slices.items():
            if expiry not in EXPIRIES and expiry not in spec.slices:
                continue
            fwd = spec.spot * np.exp((spec.rate - spec.div_yield) * expiry)
            quotes[expiry] = quotes_from_svi(
                params,
                forward=float(fwd),
                expiry_years=expiry,
                ks=ks,
                skew=float(path.skew[t]),
                fly=float(path.fly[t]),
            )
        slices = book.fit_expiries(spec_id, quotes)
        suspect += sum(1 for sl in slices if sl.surface_suspect)
    return suspect


def _risk_scale(delta: float, gamma: float, vega: float, cfg: DeskConfig) -> float:
    scale = 1.0
    if abs(delta) > cfg.max_abs_delta > 0.0:
        scale = min(scale, cfg.max_abs_delta / abs(delta))
    if abs(gamma) > cfg.max_abs_gamma > 0.0:
        scale = min(scale, cfg.max_abs_gamma / abs(gamma))
    if abs(vega) > cfg.max_abs_vega > 0.0:
        scale = min(scale, cfg.max_abs_vega / abs(vega))
    return scale


def _enabled(cfg: DeskConfig, sleeve_id: str) -> bool:
    if cfg.enabled is None:
        return True
    return bool(cfg.enabled.get(sleeve_id, True))


def run_desk(cfg: DeskConfig | None = None) -> DeskRun:
    cfg = cfg or DeskConfig()
    index_ret, prev_index, names, fills = _simulate_names(cfg)
    n = cfg.n_steps
    sleeves = list(cfg.sleeves)
    book = SurfaceBook(synthetic_fixture=cfg.synthetic_fixture)
    raw = {s: np.zeros(n) for s in sleeves}
    f_gamma = {s: np.zeros(n) for s in sleeves}
    f_vega = {s: np.zeros(n) for s in sleeves}
    fills_n = {s: 0 for s in sleeves}
    fees = {s: 0.0 for s in sleeves}
    prev = {(p, s): 0.0 for p in cfg.products for s in sleeves}
    scale_path = np.zeros(n)
    suspect = 0
    n_fits = 0
    sleeve_index = {s: i for i, s in enumerate(sleeves)}
    questions = build_desk_questions() if cfg.jev_desk else None

    for t in range(n):
        if t % max(cfg.fit_stride, 1) == 0:
            suspect += _fit_book(cfg, names, t, book)
            n_fits += 1
        jev_mult = {s: 1.0 for s in sleeves}
        jev_kill = {s: False for s in sleeves}
        if cfg.jev_desk and questions is not None:
            mean_tox = float(np.mean([names[p].tox[t] for p in cfg.products]))
            rmse = 0.0
            for u in book.underliers.values():
                for sl in u.slices.values():
                    rmse = max(rmse, sl.rmse)
            state = build_mm_state(
                time=float(t),
                spot=100.0,
                option_mid=1.0,
                iv=0.2,
                inventory=0,
                delta=0.0,
                gamma=0.0,
                vega=0.0,
                cash_pnl=0.0,
                half_spread=0.2,
                quoting_allowed=True,
            )
            state["desk"] = {
                "enabled": 1.0,
                "sleeve_toxic": 1.0 if mean_tox >= 0.55 else 0.0,
                "surface_rmse": rmse,
            }
            ans = _CLIENT.system_one(state, questions)
            adj = apply_desk_policy(ans, state)
            if adj.kill:
                jev_kill["flow_toxicity"] = True
            jev_mult["flow_toxicity"] = adj.weight_mult

        step_delta = 0.0
        step_gamma = 0.0
        step_vega = 0.0
        pending: list[tuple[str, str, object]] = []
        for pi, pid in enumerate(cfg.products):
            path = names[pid]
            for sleeve_id in sleeves:
                ctx = SleeveContext(
                    product_id=pid,
                    beta_to_index=path.spec.beta_to_index,
                    product_return=float(path.product_return[t]),
                    index_return=float(index_ret[t]),
                    d_sigma=float(path.d_sigma[t]),
                    dt=cfg.dt,
                    skew=float(path.skew[t]),
                    d_skew=float(path.d_skew[t]),
                    fly=float(path.fly[t]),
                    d_fly=float(path.d_fly[t]),
                    term=float(path.term[t]),
                    d_term=float(path.d_term[t]),
                    tox=float(path.tox[t]),
                    gex_norm=float(path.gex[t]),
                    instability=float(path.instability[t]),
                    f_signed=float(prev_index[t]),
                    box_edge=float(path.box[t]),
                    d_box=float(path.d_box[t]),
                    iv=float(path.iv[t]),
                    prev_target=prev[(pid, sleeve_id)],
                    gates_on=cfg.gates_on,
                    enabled=_enabled(cfg, sleeve_id),
                    jev_size_mult=jev_mult.get(sleeve_id, 1.0),
                    jev_kill=jev_kill.get(sleeve_id, False),
                    fill_draw=float(fills[pi, sleeve_index[sleeve_id], t]),
                    rate=path.spec.rate,
                )
                quote = quote_or_target(sleeve_id, ctx, fee_rate=cfg.research_fee)
                pending.append((pid, sleeve_id, quote))
                step_delta += quote.delta
                step_gamma += quote.gamma
                step_vega += quote.vega
        scale = _risk_scale(step_delta, step_gamma, step_vega, cfg)
        scale_path[t] = scale
        for pid, sleeve_id, quote in pending:
            raw[sleeve_id][t] += quote.raw_pnl * scale
            f_gamma[sleeve_id][t] += quote.f_gamma * scale
            f_vega[sleeve_id][t] += quote.f_vega * scale
            fills_n[sleeve_id] += int(quote.fill)
            fees[sleeve_id] += quote.fee * scale
            prev[(pid, sleeve_id)] = quote.target * scale

    residual: dict[str, np.ndarray] = {}
    fits = {}
    for sleeve_id in sleeves:
        fv = f_vega[sleeve_id] if cfg.strip_vega else None
        fit = strip_residual(raw[sleeve_id], index_ret, f_gamma[sleeve_id], fv)
        residual[sleeve_id] = fit.residual
        fits[sleeve_id] = fit

    enabled_flags = [_enabled(cfg, s) for s in sleeves]
    # Disabled sleeves are a zero series so they cannot pick up a spurious ρ.
    resid_for_alloc = [
        residual[s] if enabled_flags[i] else np.zeros(n) for i, s in enumerate(sleeves)
    ]
    w = allocate(
        resid_for_alloc,
        enabled_flags,
        max_weight=cfg.max_sleeve_weight,
        corr_cap=cfg.corr_cap,
    )
    weights = {s: float(w[i]) for i, s in enumerate(sleeves)}

    active = [s for s, flag in zip(sleeves, enabled_flags) if flag]
    pear = np.eye(len(active))
    spear = np.eye(len(active))
    flagged: list[tuple[str, str, float]] = []
    for i, a in enumerate(active):
        for j in range(i + 1, len(active)):
            b = active[j]
            rho = pearson(residual[a], residual[b])
            pear[i, j] = pear[j, i] = rho
            sp = spearman(residual[a], residual[b])
            spear[i, j] = spear[j, i] = sp
            if abs(rho) > cfg.corr_cap:
                flagged.append((a, b, rho))

    desk_raw = np.zeros(n)
    desk_res = np.zeros(n)
    for sleeve_id, weight in weights.items():
        desk_raw += weight * raw[sleeve_id]
        desk_res += weight * residual[sleeve_id]

    rows: list[SleeveRow] = []
    notes: list[str] = []
    for sleeve_id in sleeves:
        fit = fits[sleeve_id]
        series = residual[sleeve_id]
        rows.append(
            SleeveRow(
                sleeve_id=sleeve_id,
                enabled=_enabled(cfg, sleeve_id),
                weight=weights[sleeve_id],
                risk_budget=RISK_BUDGET[sleeve_id],
                raw_pnl=float(np.sum(raw[sleeve_id])),
                residual_pnl=float(np.sum(series)),
                mean_residual=float(np.mean(series)) if n else 0.0,
                sharpe_residual=per_step_sharpe(series),
                r2=float(fit.r2),
                beta=float(fit.beta),
                gamma_coef=float(fit.gamma_coef),
                vega_coef=float(fit.vega_coef),
                max_dd_raw=cumulative_drawdown(raw[sleeve_id]),
                max_dd_residual=cumulative_drawdown(series),
                n_fills=fills_n[sleeve_id],
                fees=fees[sleeve_id],
            )
        )
        if fit.r2 < 0.15 and _enabled(cfg, sleeve_id):
            notes.append(
                f"{sleeve_id} factor R² is {fit.r2:.3f}. Beta, gamma, and vega explain little of this sample's variance."
            )
        elif fit.r2 > 0.70 and _enabled(cfg, sleeve_id):
            notes.append(
                f"{sleeve_id} factor R² is {fit.r2:.3f}. Most of the variance sits in beta, gamma, or vega on this sample."
            )
    if flagged:
        pair_txt = ", ".join(f"{a}/{b} ({rho:.2f})" for a, b, rho in flagged)
        notes.append(f"Residual |ρ| > {cfg.corr_cap:.2f} (not yet orthogonal): {pair_txt}.")
    else:
        notes.append(
            f"No enabled residual pair exceeded |ρ| {cfg.corr_cap:.2f} on this seed."
        )
    notes.append(
        "Gamma and vega columns are the greek PnL buckets. Beta is the index simple return. "
        "Slopes come from a demeaned regression. The residual keeps the intercept, so a constant premium stays in the mean."
    )
    notes.append("Quotes are synthetic. synthetic_fixture=1. This is not an OPRA surface.")

    board = DeskScoreboard(
        rows=rows,
        products=list(cfg.products),
        flagged_pairs=flagged,
        sleeve_ids=active,
        pearson=pear,
        spearman=spear,
        desk_raw_pnl=float(np.sum(desk_raw)),
        desk_residual_pnl=float(np.sum(desk_res)),
        desk_sharpe_residual=per_step_sharpe(desk_res),
        desk_max_dd_residual=cumulative_drawdown(desk_res),
        weight_sum=float(sum(weights.values())),
        synthetic_fixture=cfg.synthetic_fixture,
        notes=notes,
    )
    md = to_markdown(board, title="Sleeve residual correlation")
    return DeskRun(
        config=cfg,
        book=book,
        scoreboard=board,
        raw=raw,
        residual=residual,
        f_beta=index_ret,
        f_gamma=f_gamma,
        f_vega=f_vega,
        weights=weights,
        suspect_fits=suspect,
        n_fits=n_fits,
        scale_path=scale_path,
        markdown=md,
    )
