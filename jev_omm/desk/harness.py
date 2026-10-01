"""N products × M sleeves paper desk.

Spot paths are a shared index factor plus an idiosyncratic shock. Smile
slope, curvature, and the far wing are separate factors. The smile-shock
regime plants a shared draw and then residualizes curvature and the wing
against the slope. Surfaces are refit from synthetic quotes on
``fit_stride`` unless ``fit_surfaces`` is false. Sleeves keep separate PnL
streams.

Residual PnL strips index beta, spot gamma, vega, and (by default) volga,
vanna, and a variance column orthogonal to spot gamma. A hard gate then
residualizes or merges any sleeve pair that is still too correlated. The
gate update is projected back onto the same factor columns so the published
residual stays orthogonal to beta and the greek strip.

No live orders. No OPRA tape. ``synthetic_fixture`` stays 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from jev_omm.decisions.client import DeterministicFallbackClient
from jev_omm.decisions.policy import apply_desk_policy
from jev_omm.decisions.schemas import build_desk_questions, build_mm_state
from jev_omm.desk.allocator import (
    DEFAULT_CORR_CAP,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_SHARPE_TILT,
    allocate,
    apply_min_weight_floor,
    renorm_cap,
)
from jev_omm.desk.honesty import smoothness_penalty
from jev_omm.desk.fixtures import DEFAULT_PRODUCTS, EXPIRIES, LOG_MONEYNESS, UNDERLIERS, UnderlierSpec
from jev_omm.desk.orthogonal import enforce_orthogonality, research_pca, residualize_keep_mean
from jev_omm.desk.scoreboard import (
    DeskScoreboard,
    SleeveRow,
    cumulative_drawdown,
    per_step_sharpe,
    to_markdown,
)
from jev_omm.desk.sleeves import LEGACY_SLEEVE_IDS, RISK_BUDGET, SLEEVE_IDS, SleeveContext, quote_or_target
from jev_omm.execution.executable import attribute_sleeve_fill
from jev_omm.pnl.residual import pearson, spearman, strip_factors, strip_residual
from jev_omm.surface.book import SurfaceBook, quotes_from_svi
from jev_omm.surface.rough_vol import rough_bergomi_variance

_CLIENT = DeterministicFallbackClient()

EVAL_SEEDS: tuple[int, ...] = (11, 23, 42, 7, 99)
EVAL_REGIMES: tuple[str, ...] = ("baseline", "smile_shock", "jump")
# Stress paths used to falsify the book. Not part of the correlation grid.
ADVERSARIAL_REGIMES: tuple[str, ...] = (
    "smile_shock",
    "jump",
    "toxic_flow",
    "wide_spread",
    "no_fill",
)
# Research sleeves inverse-vol was starving on the 1.1 book.
RESEARCH_SLEEVES: tuple[str, ...] = (
    "sticky_regime",
    "vanna_tilt",
    "queue_sniper",
    "rough_vol_stress",
)
# Sleeves the 1.2 baseline walk-forward killed. Off on the paper desk.
# ``legacy_honest_config`` turns the book back on for the 1.2 regression.
LEAN_OFF: tuple[str, ...] = (
    "parity_box",
    "roll_yield",
    "cot_fade_sleeve",
    "warehouse_autocall",
    "box_rate",
)
SURVIVOR_SLEEVES: tuple[str, ...] = (
    "mm_spread",
    "sticky_regime",
    "vanna_tilt",
    "queue_sniper",
    "charm_bleed",
)


@dataclass
class DeskConfig:
    products: tuple[str, ...] = DEFAULT_PRODUCTS
    sleeves: tuple[str, ...] = SLEEVE_IDS
    n_steps: int = 80
    seed: int = 11
    dt: float = 1.0 / 252.0
    max_abs_delta: float = 1.0e9
    max_abs_gamma: float = 1.0e9
    max_abs_vega: float = 1.0e9
    max_abs_vanna: float = 1.0e9
    max_abs_volga: float = 1.0e9
    max_sleeve_weight: float = DEFAULT_MAX_WEIGHT
    corr_cap: float = DEFAULT_CORR_CAP
    sharpe_tilt: float = DEFAULT_SHARPE_TILT
    gate_threshold: float = 0.40
    research_fee: float = 0.001
    fit_stride: int = 20
    fit_surfaces: bool = True
    strip_vega: bool = True
    extended_strip: bool = True
    factorize_smile: bool = True
    separate_convexity: bool = True
    harden_edges: bool = True
    book_higher_greeks: bool = True
    ortho_gate: bool = True
    regime: str = "baseline"
    gates_on: bool = True
    jev_desk: bool = False
    enabled: dict[str, bool] | None = None
    synthetic_fixture: int = 1
    # 1.2 honesty. Off is the identity: no penalty, no σ clip, no floor,
    # no capacity scale, no walk-forward kill, no roll/vanna split.
    honesty: bool = True
    honest_allocator: bool = True
    sigma_clip_quantile: float = 0.75
    min_weight_floor: float = 0.03
    split_roll_vanna: bool = True
    capacity_caps: bool = True
    # Gross |Δinventory| across products. 0 disables that leg.
    turnover_cap: float = 600.0
    # Peak of |sum of targets| across products. 0 disables that leg.
    inventory_cap: float = 30.0
    walkforward_kill: bool = True
    # 1.3 executable fills. ``lean_book`` drops the sleeves 1.2 killed.
    # ``executable_fills=False`` leaves fill PnL, join rate, and markout at 0.
    lean_book: bool = True
    executable_fills: bool = True
    fill_model: str = "lob"
    queue_edge: bool = True


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
    wing: np.ndarray
    d_wing: np.ndarray
    sticky: np.ndarray
    d_sticky: np.ndarray
    roll: np.ndarray
    d_roll: np.ndarray
    vanna: np.ndarray
    d_vanna: np.ndarray
    queue: np.ndarray
    d_queue: np.ndarray
    cot: np.ndarray
    d_cot: np.ndarray
    autocall: np.ndarray
    d_autocall: np.ndarray
    funding: np.ndarray
    d_funding: np.ndarray
    rough: np.ndarray
    d_rough: np.ndarray
    vrp: np.ndarray
    d_vrp: np.ndarray
    lagged_rv: np.ndarray
    dispersion: np.ndarray
    d_dispersion: np.ndarray
    cross_ret: np.ndarray
    d_cross: np.ndarray
    charm_clock: np.ndarray


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
    pre_gate_max_abs_rho: float = 0.0
    ortho_notes: list[str] = field(default_factory=list)
    penalties: dict[str, float] = field(default_factory=dict)
    capacity_scale: dict[str, float] = field(default_factory=dict)
    product_kills: list[str] = field(default_factory=list)
    sleeve_kills: list[str] = field(default_factory=list)


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _rebuild_from_innovations(d: np.ndarray) -> np.ndarray:
    """Level at the start of each step, given the innovation applied during it."""
    level = np.zeros_like(d)
    s = 0.0
    for t in range(d.size):
        level[t] = s
        s = s + float(d[t])
    return level


def _rough_path(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    level = np.zeros(n)
    innov = np.zeros(n)
    if n < 2:
        return level, innov
    _times, var = rough_bergomi_variance(n, 0.15, 0.40, 0.04, seed, t_end=1.0)
    log_var = np.log(np.maximum(var, 1e-8))
    log_var = log_var - log_var[0]
    innov[1:] = np.diff(log_var)
    return _rebuild_from_innovations(innov), innov


def _ar_update(level: float, shock: float, decay: float, scale: float) -> tuple[float, float]:
    nxt = decay * level + scale * shock
    return nxt, nxt - level


def _simulate_names(cfg: DeskConfig) -> tuple[np.ndarray, np.ndarray, dict[str, _NamePath], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    rng_x = np.random.default_rng(cfg.seed + 101)
    n = cfg.n_steps
    index_ret = rng.normal(0.0, 0.01, n)
    if cfg.regime == "jump":
        jump_mask = rng_x.random(n) < 0.05
        jump_sign = rng_x.choice(np.array([-1.0, 1.0]), size=n)
        index_ret = index_ret + np.where(jump_mask, jump_sign * 0.04, 0.0)
    smile = rng.normal(0.0, 1.0, n)
    stress = rng.normal(0.0, 1.0, n)
    vol_shock = rng.normal(0.0, 1.0, n)
    rate_shock = rng.normal(0.0, 1.0, n)
    z_skew = rng_x.normal(0.0, 1.0, n)
    z_fly = rng_x.normal(0.0, 1.0, n)
    z_wing = rng_x.normal(0.0, 1.0, n)
    z_sticky = rng_x.normal(0.0, 1.0, n)
    z_roll = rng_x.normal(0.0, 1.0, n)
    z_vanna = rng_x.normal(0.0, 1.0, n)
    z_queue = rng_x.normal(0.0, 1.0, n)
    z_cot = rng_x.normal(0.0, 1.0, n)
    z_auto = rng_x.normal(0.0, 1.0, n)
    z_fund = rng_x.normal(0.0, 1.0, n)
    z_vrp = rng_x.normal(0.0, 1.0, n)
    z_disp = rng_x.normal(0.0, 1.0, n)
    z_cross = rng_x.normal(0.0, 1.0, n)
    rough_level, rough_d = _rough_path(n, cfg.seed + 3)
    fill_rng = np.random.default_rng(cfg.seed + 17)
    fills = fill_rng.random((len(cfg.products), len(cfg.sleeves), n))
    prev_index = np.zeros(n)
    if n > 1:
        prev_index[1:] = index_ret[:-1]

    # Weekly COT clock and the charm/weekend clock are common, not per name.
    cot = np.zeros(n)
    d_cot = np.zeros(n)
    c = 0.0
    charm_clock = np.ones(n)
    for t in range(n):
        cot[t] = c
        # COT prints on one weekday. Charm's weekend bleed is a different day,
        # so the two clocks are not the same spike.
        if t % 5 == 4:
            c2 = 0.80 * c + 0.40 * float(z_cot[t])
        else:
            c2 = c
        if t % 5 == 1:
            charm_clock[t] = 3.0
        d_cot[t] = c2 - c
        c = c2

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
        wing = np.zeros(n)
        d_wing = np.zeros(n)
        sticky = np.zeros(n)
        d_sticky = np.zeros(n)
        roll = np.zeros(n)
        d_roll = np.zeros(n)
        vanna = np.zeros(n)
        d_vanna = np.zeros(n)
        queue = np.zeros(n)
        d_queue = np.zeros(n)
        autocall = np.zeros(n)
        d_autocall = np.zeros(n)
        funding = np.zeros(n)
        d_funding = np.zeros(n)
        vrp = np.zeros(n)
        d_vrp = np.zeros(n)
        s = f = w = tm = 0.0
        tx = 0.20
        gx = 0.30
        bx = 0.0
        st = va = qu = au = fu = vp = 0.0
        iv_level = spec.iv0
        for t in range(n):
            idio = float(rng.normal(0.0, spec.idio_vol))
            product_return[t] = spec.beta_to_index * float(index_ret[t]) + idio
            shock_s_priv = float(rng.normal())
            shock_f_priv = float(rng.normal())
            term_priv = float(rng.normal())
            skew[t] = s
            fly[t] = f
            wing[t] = w
            term[t] = tm
            tox[t] = tx
            gex[t] = gx
            inst[t] = abs(gx)
            box[t] = bx
            iv[t] = iv_level
            sticky[t] = st
            vanna[t] = va
            queue[t] = qu
            autocall[t] = au
            funding[t] = fu
            vrp[t] = vp
            roll[t] = 0.0
            if cfg.factorize_smile:
                shock_s = float(z_skew[t]) + 0.15 * shock_s_priv
                shock_f = float(z_fly[t]) + 0.20 * shock_f_priv
                shock_w = float(z_wing[t])
                if cfg.regime == "smile_shock":
                    shock_s = 0.55 * float(smile[t]) + 0.45 * shock_s
                    shock_f = 0.55 * float(smile[t]) + 0.45 * shock_f
                    shock_w = 0.55 * float(smile[t]) + 0.45 * shock_w
            else:
                shock_s = 0.85 * float(smile[t]) + 0.15 * shock_s_priv
                shock_f = 0.80 * float(smile[t]) + 0.20 * shock_f_priv
                shock_w = 0.75 * float(smile[t]) + 0.25 * float(z_wing[t])
            s2, d_skew[t] = _ar_update(s, shock_s, 0.78, 0.035)
            f2, d_fly[t] = _ar_update(f, shock_f, 0.78, 0.030)
            w2, d_wing[t] = _ar_update(w, shock_w, 0.70, 0.040)
            tm2 = 0.88 * tm + 0.02 * term_priv
            d_term[t] = tm2 - tm
            tx2 = _clip(0.70 * tx + 0.28 * max(float(stress[t]), 0.0), 0.0, 1.0)
            gx2 = 0.82 * gx - 0.70 * float(stress[t])
            bx2 = 0.90 * bx + 0.015 * float(rate_shock[t])
            d_box[t] = bx2 - bx
            iv2 = _clip(spec.iv0 + 0.92 * (iv_level - spec.iv0) + 0.006 * float(vol_shock[t]), 0.04, 0.80)
            d_sigma[t] = iv2 - iv_level
            season = spec.season_amp * np.sin(2.0 * np.pi * t / 20.0)
            st2, d_sticky[t] = _ar_update(st, float(z_sticky[t]), 0.50, 0.50)
            va2, d_vanna[t] = _ar_update(va, float(z_vanna[t]), 0.60, 0.40)
            qu2, d_queue[t] = _ar_update(qu, float(z_queue[t]), 0.40, 0.70)
            au2, d_autocall[t] = _ar_update(au, float(z_auto[t]), 0.85, 0.20)
            fu2, d_funding[t] = _ar_update(fu, float(z_fund[t]), 0.90, 0.02)
            vp2, d_vrp[t] = _ar_update(vp, float(z_vrp[t]), 0.70, 0.35)
            roll[t] = float(season)
            d_roll[t] = float(z_roll[t]) * 0.15 + float(season) * 0.05
            s, f, w, tm = s2, f2, w2, tm2
            tx, gx, bx, iv_level = tx2, gx2, bx2, iv2
            st, va, qu, au, fu, vp = st2, va2, qu2, au2, fu2, vp2
        lagged = np.zeros(n)
        lagged[0] = spec.iv0 * spec.iv0
        if n > 1:
            lagged[1:] = product_return[:-1] ** 2 / cfg.dt
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
            wing=wing,
            d_wing=d_wing,
            sticky=sticky,
            d_sticky=d_sticky,
            roll=roll,
            d_roll=d_roll,
            vanna=vanna,
            d_vanna=d_vanna,
            queue=queue,
            d_queue=d_queue,
            cot=cot,
            d_cot=d_cot,
            autocall=autocall,
            d_autocall=d_autocall,
            funding=funding,
            d_funding=d_funding,
            rough=rough_level,
            d_rough=rough_d,
            vrp=vrp,
            d_vrp=d_vrp,
            lagged_rv=lagged,
            dispersion=np.zeros(n),
            d_dispersion=np.zeros(n),
            cross_ret=np.zeros(n),
            d_cross=np.zeros(n),
            charm_clock=charm_clock,
        )

    if cfg.factorize_smile:
        for path in names.values():
            path.d_fly = residualize_keep_mean(path.d_fly, path.d_skew)
            path.d_wing = residualize_keep_mean(path.d_wing, path.d_skew)
            path.d_wing = residualize_keep_mean(path.d_wing, path.d_fly)
            path.fly = _rebuild_from_innovations(path.d_fly)
            path.wing = _rebuild_from_innovations(path.d_wing)
            path.d_funding = residualize_keep_mean(path.d_funding, path.d_box)
            path.funding = _rebuild_from_innovations(path.d_funding)

    # Roll owns the term/roll clock. Vanna owns spot–vol correlation.
    # Gram–Schmidt on the innovations, same idea as the smile split.
    # Off leaves the 1.1 draws unchanged.
    if cfg.split_roll_vanna:
        for path in names.values():
            spot_vol = path.product_return * path.d_sigma
            clock = np.asarray(path.roll, dtype=float).copy()
            d_vanna_raw = np.asarray(path.d_vanna, dtype=float).copy()
            path.d_roll = residualize_keep_mean(path.d_roll, spot_vol)
            path.d_roll = residualize_keep_mean(path.d_roll, d_vanna_raw)
            path.d_vanna = residualize_keep_mean(path.d_vanna, clock)
            path.d_vanna = residualize_keep_mean(path.d_vanna, path.d_roll)
            path.vanna = _rebuild_from_innovations(path.d_vanna)

    if cfg.regime == "toxic_flow":
        for path in names.values():
            path.tox = np.clip(0.80 + 0.20 * path.tox, 0.0, 1.0)
        fills = np.clip(fills + 0.55, 0.0, 1.0)
    elif cfg.regime == "wide_spread":
        fills = np.clip(fills + 0.35, 0.0, 1.0)
    elif cfg.regime == "no_fill":
        fills = np.ones_like(fills)

    # Dispersion is its own factor. A common vol shock cancels in an
    # index-minus-basket spread, so the sleeve does not trade that shock.
    disp = np.zeros(n)
    d_disp = np.zeros(n)
    level = 0.0
    cross = np.zeros(n)
    d_cross = np.zeros(n)
    c_level = 0.0
    for t in range(n):
        disp[t] = level
        nxt, d_disp[t] = _ar_update(level, float(z_disp[t]), 0.72, 0.35)
        level = nxt
        cross[t] = c_level
        nxt_c, d_cross[t] = _ar_update(c_level, float(z_cross[t]), 0.55, 0.45)
        c_level = nxt_c
    for path in names.values():
        path.dispersion = disp
        path.d_dispersion = d_disp
        path.cross_ret = cross
        path.d_cross = d_cross
    return index_ret, prev_index, names, fills


def _fit_book(cfg: DeskConfig, names: dict[str, _NamePath], t: int, book: SurfaceBook) -> tuple[int, int]:
    """Refit every name. Returns (suspect slices, calendar breaks)."""
    suspect = 0
    calendar_breaks = 0
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
            fwd = spec.spot * np.exp((spec.rate - spec.div_yield) * expiry)
            quotes[expiry] = quotes_from_svi(
                params,
                forward=float(fwd),
                expiry_years=expiry,
                ks=ks,
                skew=float(path.skew[t]),
                fly=float(path.fly[t]),
                wing=float(path.wing[t]),
            )
        slices = book.fit_expiries(spec_id, quotes)
        suspect += sum(1 for sl in slices if sl.surface_suspect)
        calendar_breaks += sum(1 for sl in slices if not sl.calendar_ok)
    return suspect, calendar_breaks


def _risk_scale(
    delta: float,
    gamma: float,
    vega: float,
    vanna: float,
    volga: float,
    cfg: DeskConfig,
) -> float:
    scale = 1.0
    if abs(delta) > cfg.max_abs_delta > 0.0:
        scale = min(scale, cfg.max_abs_delta / abs(delta))
    if abs(gamma) > cfg.max_abs_gamma > 0.0:
        scale = min(scale, cfg.max_abs_gamma / abs(gamma))
    if abs(vega) > cfg.max_abs_vega > 0.0:
        scale = min(scale, cfg.max_abs_vega / abs(vega))
    if abs(vanna) > cfg.max_abs_vanna > 0.0:
        scale = min(scale, cfg.max_abs_vanna / abs(vanna))
    if abs(volga) > cfg.max_abs_volga > 0.0:
        scale = min(scale, cfg.max_abs_volga / abs(volga))
    return scale


def _enabled(cfg: DeskConfig, sleeve_id: str) -> bool:
    if cfg.enabled is not None and sleeve_id in cfg.enabled:
        return bool(cfg.enabled[sleeve_id])
    if cfg.lean_book and sleeve_id in LEAN_OFF:
        return False
    if cfg.enabled is None:
        return True
    return bool(cfg.enabled.get(sleeve_id, True))


def _scenario(delta: float, gamma: float, vega: float, vanna: float, volga: float) -> tuple[float, str]:
    worst = 0.0
    label = "flat"
    for ds in (-0.02, -0.01, 0.01, 0.02):
        for dv in (-0.01, 0.0, 0.01):
            pnl = (
                delta * ds
                + 0.5 * gamma * ds * ds
                + vega * dv
                + vanna * ds * dv
                + 0.5 * volga * dv * dv
            )
            if pnl < worst:
                worst = pnl
                label = f"dS={ds:+.0%}, dσ={dv:+.0%}"
    return worst, label


def _pair_rho(board_ids: list[str], mat: np.ndarray, a: str, b: str) -> float | None:
    if a not in board_ids or b not in board_ids:
        return None
    i = board_ids.index(a)
    j = board_ids.index(b)
    return float(mat[i, j])


def run_desk(cfg: DeskConfig | None = None) -> DeskRun:
    cfg = cfg or DeskConfig()
    index_ret, prev_index, names, fills = _simulate_names(cfg)
    n = cfg.n_steps
    sleeves = list(cfg.sleeves)
    book = SurfaceBook(synthetic_fixture=cfg.synthetic_fixture)
    raw = {s: np.zeros(n) for s in sleeves}
    by_product = {s: {p: np.zeros(n) for p in cfg.products} for s in sleeves}
    f_gamma = {s: np.zeros(n) for s in sleeves}
    f_vega = {s: np.zeros(n) for s in sleeves}
    f_volga = {s: np.zeros(n) for s in sleeves}
    f_vanna = {s: np.zeros(n) for s in sleeves}
    g_by_product = {s: {p: np.zeros(n) for p in cfg.products} for s in sleeves}
    fills_n = {s: 0 for s in sleeves}
    fees = {s: 0.0 for s in sleeves}
    lob_fills = {s: 0.0 for s in sleeves}
    join_hits = {s: 0.0 for s in sleeves}
    quote_steps = {s: 0 for s in sleeves}
    adverse_mo = {s: 0.0 for s in sleeves}
    fill_pnl = {s: 0.0 for s in sleeves}
    prev = {(p, s): 0.0 for p in cfg.products for s in sleeves}
    prev_quote = {(p, s): (0.0, 1.0) for p in cfg.products for s in sleeves}
    turnover = {s: 0.0 for s in sleeves}
    revisions = {s: 0 for s in sleeves}
    inv_path = {s: np.zeros(n) for s in sleeves}
    gamma_path = {s: np.zeros(n) for s in sleeves}
    vega_path = {s: np.zeros(n) for s in sleeves}
    scale_path = np.zeros(n)
    fee_rate = cfg.research_fee * (6.0 if cfg.regime == "wide_spread" else 1.0)
    block_fills = cfg.regime == "no_fill"
    suspect = 0
    calendar_breaks = 0
    n_fits = 0
    dupire_var = float("nan")
    sticky_gap = None
    sleeve_index = {s: i for i, s in enumerate(sleeves)}
    questions = build_desk_questions() if cfg.jev_desk else None
    end_greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "vanna": 0.0, "volga": 0.0}

    for t in range(n):
        if cfg.fit_surfaces and t % max(cfg.fit_stride, 1) == 0:
            sus, cal = _fit_book(cfg, names, t, book)
            suspect += sus
            calendar_breaks += cal
            n_fits += 1
            if "EQ_INDEX" in names:
                dupire_var = book.dupire_front_local_var("EQ_INDEX")
                spot = names["EQ_INDEX"].spec.spot
                sticky_gap = book.sticky_atm_gap("EQ_INDEX", EXPIRIES[0], spot * 1.01)
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
        step_vanna = 0.0
        step_volga = 0.0
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
                    separate_convexity=cfg.separate_convexity,
                    harden_edges=cfg.harden_edges,
                    book_higher_greeks=cfg.book_higher_greeks,
                    asset_class=path.spec.asset_class,
                    wing=float(path.wing[t]),
                    d_wing=float(path.d_wing[t]),
                    sticky=float(path.sticky[t]),
                    d_sticky=float(path.d_sticky[t]),
                    dispersion=float(path.dispersion[t]),
                    d_dispersion=float(path.d_dispersion[t]),
                    roll=float(path.roll[t]),
                    d_roll=float(path.d_roll[t]),
                    charm_clock=float(path.charm_clock[t]),
                    vanna_mis=float(path.vanna[t]),
                    d_vanna=float(path.d_vanna[t]),
                    queue=float(path.queue[t]),
                    d_queue=float(path.d_queue[t]),
                    cot_z=float(path.cot[t]),
                    d_cot=float(path.d_cot[t]),
                    autocall=float(path.autocall[t]),
                    d_autocall=float(path.d_autocall[t]),
                    funding=float(path.funding[t]),
                    d_funding=float(path.d_funding[t]),
                    cross_ret=float(path.cross_ret[t]),
                    d_cross=float(path.d_cross[t]),
                    rough=float(path.rough[t]),
                    d_rough=float(path.d_rough[t]),
                    lagged_rv=float(path.lagged_rv[t]),
                    d_vrp=float(path.d_vrp[t]),
                )
                quote = quote_or_target(sleeve_id, ctx, fee_rate=fee_rate)
                if (
                    cfg.executable_fills
                    and cfg.fill_model == "lob"
                    and quote.enabled
                ):
                    if sleeve_id == "queue_sniper":
                        fill_book = "sniper"
                    elif sleeve_id == "flow_toxicity":
                        fill_book = "flow"
                    else:
                        fill_book = "touch"
                    attr = attribute_sleeve_fill(
                        spread_mult=quote.spread_mult,
                        size_mult=quote.size_mult,
                        target=quote.target,
                        queue=float(path.queue[t]),
                        toxic=float(path.tox[t]),
                        queue_edge=cfg.queue_edge,
                        fee_per_contract=fee_rate,
                        book=fill_book,
                    )
                    quote_steps[sleeve_id] += 1
                    join_hits[sleeve_id] += attr.joined
                    if not block_fills:
                        lob_fills[sleeve_id] += attr.fills
                        adverse_mo[sleeve_id] += attr.adverse_markout
                        fill_pnl[sleeve_id] += attr.fill_pnl
                pending.append((pid, sleeve_id, quote))
                step_delta += quote.delta
                step_gamma += quote.gamma
                step_vega += quote.vega
                step_vanna += quote.vanna
                step_volga += quote.volga
        scale = _risk_scale(step_delta, step_gamma, step_vega, step_vanna, step_volga, cfg)
        scale_path[t] = scale
        end_greeks = {
            "delta": step_delta * scale,
            "gamma": step_gamma * scale,
            "vega": step_vega * scale,
            "vanna": step_vanna * scale,
            "volga": step_volga * scale,
        }
        for pid, sleeve_id, quote in pending:
            old_target, old_spread = prev_quote[(pid, sleeve_id)]
            if abs(quote.target - old_target) > 1e-12 or abs(quote.spread_mult - old_spread) > 1e-12:
                revisions[sleeve_id] += 1
            prev_quote[(pid, sleeve_id)] = (quote.target, quote.spread_mult)
            if block_fills:
                # The quote is recorded. Nothing trades, so inventory and PnL stay at 0.
                continue
            inv = quote.target * scale
            turnover[sleeve_id] += abs(inv - prev[(pid, sleeve_id)])
            prev[(pid, sleeve_id)] = inv
            inv_path[sleeve_id][t] += inv
            gamma_path[sleeve_id][t] += quote.gamma * scale
            vega_path[sleeve_id][t] += quote.vega * scale
            raw[sleeve_id][t] += quote.raw_pnl * scale
            by_product[sleeve_id][pid][t] += quote.raw_pnl * scale
            f_gamma[sleeve_id][t] += quote.f_gamma * scale
            f_vega[sleeve_id][t] += quote.f_vega * scale
            f_volga[sleeve_id][t] += quote.f_volga * scale
            f_vanna[sleeve_id][t] += quote.f_vanna * scale
            g_by_product[sleeve_id][pid][t] += quote.f_gamma * scale
            fills_n[sleeve_id] += int(quote.fill)
            fees[sleeve_id] += quote.fee * scale

    qv = np.zeros(n)
    for path in names.values():
        qv += path.product_return ** 2

    def _cols(sleeve_id: str) -> list[np.ndarray]:
        var_col = residualize_keep_mean(qv, f_gamma[sleeve_id])
        if cfg.extended_strip:
            built: list[np.ndarray] = [index_ret, f_gamma[sleeve_id]]
            built.append(f_vega[sleeve_id] if cfg.strip_vega else np.zeros(n))
            built.extend([f_volga[sleeve_id], f_vanna[sleeve_id], var_col])
            return built
        built = [index_ret, f_gamma[sleeve_id]]
        if cfg.strip_vega:
            built.append(f_vega[sleeve_id])
        return built

    residual: dict[str, np.ndarray] = {}
    fits = {}
    for sleeve_id in sleeves:
        if cfg.extended_strip:
            fit = strip_factors(raw[sleeve_id], _cols(sleeve_id))
        else:
            fv = f_vega[sleeve_id] if cfg.strip_vega else None
            fit = strip_residual(raw[sleeve_id], index_ret, f_gamma[sleeve_id], fv)
        residual[sleeve_id] = fit.residual
        fits[sleeve_id] = fit

    # The pre-trade shocks are already separate. What still lines up is the
    # alpha product (previous target times the innovation) on the commodity
    # name. Remove the roll sleeve's residual from vanna, keep vanna's mean,
    # then put vanna back in its own factor subspace.
    if (
        cfg.split_roll_vanna
        and "vanna_tilt" in residual
        and "roll_yield" in residual
        and _enabled(cfg, "vanna_tilt")
        and _enabled(cfg, "roll_yield")
    ):
        residual["vanna_tilt"] = residualize_keep_mean(residual["vanna_tilt"], residual["roll_yield"])
        if float(np.std(residual["vanna_tilt"])) > 1e-12:
            residual["vanna_tilt"] = strip_factors(residual["vanna_tilt"], _cols("vanna_tilt")).residual

    def _restrip(series: dict[str, np.ndarray], on: dict[str, bool]) -> dict[str, np.ndarray]:
        """Put each enabled residual back in its own factor-orthogonal subspace.

        Gram–Schmidt against another sleeve subtracts a series that is
        orthogonal to that sleeve's greeks, not to this sleeve's. The second
        strip removes the leak. The mean stays in the residual.
        """
        out = {k: np.asarray(v, dtype=float).copy() for k, v in series.items()}
        for sleeve_id in sleeves:
            if not on.get(sleeve_id, False):
                continue
            if float(np.std(out[sleeve_id])) <= 1e-12:
                continue
            out[sleeve_id] = strip_factors(out[sleeve_id], _cols(sleeve_id)).residual
        return out

    enabled_map = {s: _enabled(cfg, s) for s in sleeves}
    pre_ids = [s for s in sleeves if enabled_map[s]]
    pre_max = 0.0
    for i, a in enumerate(pre_ids):
        for b in pre_ids[i + 1 :]:
            pre_max = max(pre_max, abs(pearson(residual[a], residual[b])))
    ortho_notes: list[str] = []
    if cfg.ortho_gate:
        # Re-strip after each gate pass. Stop once pairwise |ρ| is back
        # under the threshold; a later pass only runs if the strip reopened a pair.
        for _pass in range(4):
            gated = enforce_orthogonality(residual, enabled_map, threshold=cfg.gate_threshold)
            residual = _restrip(gated.series, gated.enabled)
            enabled_map = gated.enabled
            if gated.notes:
                ortho_notes.extend(gated.notes)
            worst = 0.0
            live = [s for s in sleeves if enabled_map[s]]
            for i, a in enumerate(live):
                for b in live[i + 1 :]:
                    worst = max(worst, abs(pearson(residual[a], residual[b])))
            if worst <= cfg.gate_threshold + 1e-12:
                break

    enabled_flags = [enabled_map[s] for s in sleeves]
    resid_for_alloc = [residual[s] if enabled_flags[i] else np.zeros(n) for i, s in enumerate(sleeves)]
    penalties: dict[str, float] = {}
    flags: dict[str, str] = {}
    smooth_stats: dict[str, dict[str, float]] = {}
    for sleeve_id in sleeves:
        if cfg.honesty and enabled_map[sleeve_id]:
            pen, flag, stats = smoothness_penalty(residual[sleeve_id], raw[sleeve_id])
        else:
            pen, flag, stats = 1.0, "ok", {
                "ac1": 0.0,
                "dc_share": 0.0,
                "const_trend_r2": 0.0,
                "low_freq_share": 0.0,
            }
        penalties[sleeve_id] = pen
        flags[sleeve_id] = flag
        smooth_stats[sleeve_id] = stats
    clip_q = cfg.sigma_clip_quantile if cfg.honest_allocator else 0.0
    w = allocate(
        resid_for_alloc,
        enabled_flags,
        max_weight=cfg.max_sleeve_weight,
        corr_cap=cfg.corr_cap,
        sharpe_tilt=cfg.sharpe_tilt,
        sigma_clip_quantile=clip_q,
    )
    # Pre-penalty portfolio, for the raw Sharpe column. Not a capacity.
    w_raw = np.asarray(w, dtype=float).copy()
    if cfg.honesty or cfg.honest_allocator or cfg.capacity_caps or cfg.walkforward_kill:
        if cfg.honesty:
            for i, sleeve_id in enumerate(sleeves):
                w[i] *= penalties[sleeve_id]
            w = renorm_cap(w, cfg.max_sleeve_weight)
        half = n // 2
        eligible = []
        kills = []
        for i, sleeve_id in enumerate(sleeves):
            series = residual[sleeve_id]
            test = series[half:] if n - half >= 2 else series
            test_mu = float(np.mean(test)) if test.size else 0.0
            raw_std = float(np.std(series, ddof=1)) if series.size >= 2 else 0.0
            mean_all = float(np.mean(series)) if series.size else 0.0
            floor_on = (
                cfg.honest_allocator
                and enabled_flags[i]
                and penalties[sleeve_id] > 0.0
                and mean_all > 1e-8
                and raw_std >= 1e-5
                and not (cfg.walkforward_kill and test_mu <= 0.0)
            )
            eligible.append(floor_on)
            if cfg.walkforward_kill and enabled_flags[i] and test_mu <= 0.0:
                kills.append(sleeve_id)
        if cfg.honest_allocator and cfg.min_weight_floor > 0.0:
            w = apply_min_weight_floor(w, eligible, cfg.min_weight_floor, cfg.max_sleeve_weight)
        for sleeve_id in kills:
            w[sleeves.index(sleeve_id)] = 0.0
    else:
        kills = []
    weights = {s: float(w[i]) for i, s in enumerate(sleeves)}
    raw_alloc_weights = {s: float(w_raw[i]) for i, s in enumerate(sleeves)}

    active = [s for s, flag in zip(sleeves, enabled_flags) if flag]
    pear = np.eye(len(active))
    spear = np.eye(len(active))
    flagged: list[tuple[str, str, float]] = []
    max_abs = 0.0
    for i, a in enumerate(active):
        for j in range(i + 1, len(active)):
            b = active[j]
            rho = pearson(residual[a], residual[b])
            pear[i, j] = pear[j, i] = rho
            sp = spearman(residual[a], residual[b])
            spear[i, j] = spear[j, i] = sp
            max_abs = max(max_abs, abs(rho))
            if abs(rho) > cfg.corr_cap:
                flagged.append((a, b, rho))

    # Cross-product residual correlation for the spread sleeve.
    product_ids = list(cfg.products)
    product_mat = np.eye(len(product_ids))
    focus = "mm_spread" if "mm_spread" in sleeves else (sleeves[0] if sleeves else "")
    if focus:
        per_resid = {}
        for pid in product_ids:
            per_resid[pid] = strip_residual(
                by_product[focus][pid],
                index_ret,
                g_by_product[focus][pid],
                None,
            ).residual
        for i, a in enumerate(product_ids):
            for j in range(i + 1, len(product_ids)):
                rho = pearson(per_resid[a], per_resid[product_ids[j]])
                product_mat[i, j] = product_mat[j, i] = rho

    def _path_length(path: np.ndarray) -> float:
        if path.size == 0:
            return 0.0
        total = abs(float(path[0]))
        if path.size > 1:
            total += float(np.sum(np.abs(np.diff(path))))
        return total

    capacity_scale: dict[str, float] = {}
    for i, sleeve_id in enumerate(sleeves):
        scale_i = 1.0
        if cfg.capacity_caps:
            if cfg.turnover_cap > 0.0 and turnover[sleeve_id] > cfg.turnover_cap:
                scale_i = min(scale_i, cfg.turnover_cap / turnover[sleeve_id])
            peak_i = float(np.max(np.abs(inv_path[sleeve_id]))) if n else 0.0
            if cfg.inventory_cap > 0.0 and peak_i > cfg.inventory_cap:
                scale_i = min(scale_i, cfg.inventory_cap / peak_i)
        capacity_scale[sleeve_id] = scale_i
        w[i] *= scale_i
        weights[sleeve_id] = float(w[i])

    desk_raw = np.zeros(n)
    desk_res = np.zeros(n)
    desk_res_raw_w = np.zeros(n)
    desk_inv = np.zeros(n)
    for sleeve_id, weight in weights.items():
        desk_raw += weight * raw[sleeve_id]
        desk_res += weight * residual[sleeve_id]
        desk_inv += weight * inv_path[sleeve_id]
    for sleeve_id, weight in raw_alloc_weights.items():
        desk_res_raw_w += weight * residual[sleeve_id]

    half = n // 2
    product_kills: list[str] = []
    for pid in cfg.products:
        test_pnl = 0.0
        for sleeve_id in sleeves:
            test_pnl += float(np.sum(by_product[sleeve_id][pid][half:]))
        if cfg.walkforward_kill and test_pnl <= 0.0:
            product_kills.append(pid)

    gross_turnover = float(sum(turnover.values()))
    peak_inv = float(np.max(np.abs(desk_inv))) if n else 0.0
    pen_desk_pnl = float(np.sum(desk_res))
    gamma_len = float(sum(_path_length(gamma_path[s]) for s in sleeves))
    vega_len = float(sum(_path_length(vega_path[s]) for s in sleeves))
    rev_per_step = float(sum(revisions.values()) / n) if n else 0.0

    worst, worst_label = _scenario(
        end_greeks["delta"],
        end_greeks["gamma"],
        end_greeks["vega"],
        end_greeks["vanna"],
        end_greeks["volga"],
    )

    rows: list[SleeveRow] = []
    notes: list[str] = []
    for sleeve_id in sleeves:
        fit = fits[sleeve_id]
        series = residual[sleeve_id]
        raw_sh = per_step_sharpe(series)
        pen = penalties[sleeve_id]
        stats = smooth_stats[sleeve_id]
        peak_i = float(np.max(np.abs(inv_path[sleeve_id]))) if n else 0.0
        avg_i = float(np.mean(np.abs(inv_path[sleeve_id]))) if n else 0.0
        pen_pnl = float(np.sum(series)) * pen
        to = turnover[sleeve_id]
        test = series[half:] if n - half >= 2 else series
        test_mu = float(np.mean(test)) if test.size else 0.0
        rows.append(
            SleeveRow(
                sleeve_id=sleeve_id,
                enabled=enabled_map[sleeve_id],
                weight=weights[sleeve_id],
                risk_budget=RISK_BUDGET[sleeve_id],
                raw_pnl=float(np.sum(raw[sleeve_id])),
                residual_pnl=float(np.sum(series)),
                mean_residual=float(np.mean(series)) if n else 0.0,
                sharpe_residual=raw_sh,
                r2=float(fit.r2),
                beta=float(fit.beta),
                gamma_coef=float(fit.gamma_coef),
                vega_coef=float(fit.vega_coef),
                volga_coef=float(fit.volga_coef),
                vanna_coef=float(fit.vanna_coef),
                var_coef=float(fit.var_coef),
                max_dd_raw=cumulative_drawdown(raw[sleeve_id]),
                max_dd_residual=cumulative_drawdown(series),
                n_fills=fills_n[sleeve_id],
                fees=fees[sleeve_id],
                residual_sharpe_raw=raw_sh,
                residual_sharpe_penalized=raw_sh * pen,
                smoothness_flag=flags[sleeve_id],
                smoothness_penalty=pen,
                ac1=float(stats.get("ac1", 0.0)),
                dc_share=float(stats.get("dc_share", 0.0)),
                const_trend_r2=float(stats.get("const_trend_r2", 0.0)),
                low_freq_share=float(stats.get("low_freq_share", 0.0)),
                turnover=to,
                avg_abs_inventory=avg_i,
                max_abs_inventory=peak_i,
                gamma_path_length=_path_length(gamma_path[sleeve_id]),
                vega_path_length=_path_length(vega_path[sleeve_id]),
                quote_revisions_per_step=(revisions[sleeve_id] / n) if n else 0.0,
                residual_per_turnover=(pen_pnl / to) if to > 1e-12 else 0.0,
                residual_per_peak_inventory=(pen_pnl / peak_i) if peak_i > 1e-12 else 0.0,
                capacity_scale=capacity_scale[sleeve_id],
                test_mean_residual=test_mu,
                sleeve_kill=sleeve_id in kills,
                lob_fills=float(lob_fills[sleeve_id]),
                join_rate=(
                    float(join_hits[sleeve_id] / quote_steps[sleeve_id])
                    if quote_steps[sleeve_id]
                    else 0.0
                ),
                adverse_markout=float(adverse_mo[sleeve_id]),
                fill_pnl=float(fill_pnl[sleeve_id]),
            )
        )
        if fit.r2 < 0.15 and enabled_map[sleeve_id]:
            notes.append(
                f"{sleeve_id} factor R² is {fit.r2:.3f}. The strip explains little of this sample's variance."
            )
        elif fit.r2 > 0.70 and enabled_map[sleeve_id]:
            notes.append(
                f"{sleeve_id} factor R² is {fit.r2:.3f}. Most of the variance sits in the factor strip on this sample."
            )
    if flagged:
        pair_txt = ", ".join(f"{a}/{b} ({rho:.2f})" for a, b, rho in flagged)
        notes.append(f"Residual |ρ| > {cfg.corr_cap:.2f} after the gate (allocator will shrink): {pair_txt}.")
    else:
        notes.append(f"No enabled residual pair exceeded |ρ| {cfg.corr_cap:.2f} on this seed.")
    notes.append(
        f"Pre-gate max |ρ| was {pre_max:.3f}. Post-gate max |ρ| is {max_abs:.3f}. "
        f"Gate threshold {cfg.gate_threshold:.2f}. Allocator shrink {cfg.corr_cap:.2f}, "
        f"concentration cap {cfg.max_sleeve_weight:.2f}, Sharpe tilt {cfg.sharpe_tilt:.2f}."
    )
    if ortho_notes:
        notes.extend(ortho_notes)
    else:
        notes.append("Orthogonality gate did not residualize or merge a pair on this seed.")
    notes.append(research_pca(residual, active))
    notes.append(
        "Unstandardized PCA is noise-dominated when a few high-σ sleeves own the sum of squares. "
        "Quote the standardized share. It is not a live risk model."
    )
    if cfg.honesty:
        flat = [s for s, flag in flags.items() if flag == "flat"]
        smooth = [s for s, flag in flags.items() if flag == "smooth"]
        notes.append(
            "Raw residual Sharpe is not a capacity. The smoothness penalty multiplies it. "
            f"Flat (weight forced toward 0): {', '.join(flat) if flat else 'none'}. "
            f"Smooth (penalty in (0, 1)): {', '.join(smooth) if smooth else 'none'}."
        )
    if cfg.honest_allocator:
        notes.append(
            f"Allocator clips σ at the {cfg.sigma_clip_quantile:.2f} quantile of the sleeve panel "
            f"before inverse-vol, then lifts a positive-mean sleeve to a floor of {cfg.min_weight_floor:.2f} "
            f"when the walk-forward test mean is still positive. Concentration cap {cfg.max_sleeve_weight:.2f}."
        )
    if cfg.split_roll_vanna:
        notes.append(
            "Roll innovations are residualized against spot–vol, and vanna innovations against the "
            "term/roll clock. After the greek strip, vanna's residual is residualized against the roll "
            "sleeve (mean kept) and stripped again. Gate thresholds stay 0.40 hard and 0.35 shrink."
        )
    if cfg.capacity_caps:
        notes.append(
            f"Capacity soft caps: gross turnover {cfg.turnover_cap:.1f}, peak |inventory| {cfg.inventory_cap:.1f}. "
            "A sleeve over the cap has its weight scaled by cap/usage. Caps off is the identity. "
            "Residual PnL per unit turnover and per unit peak inventory are the edge-density columns. Not annualized."
        )
    if kills:
        notes.append(
            "Walk-forward sleeve_kill (second-half residual mean ≤ 0): " + ", ".join(kills) + ". "
            "Offline Jev `sleeve_kill` / `kill_sleeve` is a flag. It does not emit an order."
        )
    if product_kills:
        notes.append(
            "Walk-forward product_kill (second-half raw PnL ≤ 0): " + ", ".join(product_kills) + ". "
            "The product stays in the synthetic book; the flag is the research control."
        )
    if cfg.factorize_smile:
        notes.append(
            "Smile factors are split: slope, curvature, and far-wing innovations. "
            "Curvature and the wing are residualized against the slope before the sleeve trades them."
        )
    if cfg.separate_convexity:
        notes.append(
            "Convexity is split: spot-gamma stays on the hedged market-maker book, "
            "volga and vanna are their own columns, and the variance sleeve earns implied variance "
            "minus lagged realized variance (flat-smile varswap limit K≈σ²), not this bar's r²."
        )
    notes.append(
        "Gamma and vega columns are the greek PnL buckets. Beta is the index simple return. "
        "Slopes come from a demeaned regression. The residual keeps the intercept."
    )
    if n_fits:
        gap_txt = "n/a" if sticky_gap is None else f"{sticky_gap:.4f}"
        dup_txt = "nan" if not np.isfinite(dupire_var) else f"{dupire_var:.4f}"
        notes.append(
            f"Surface refits {n_fits}, suspect slices {suspect}, calendar breaks {calendar_breaks}. "
            f"EQ_INDEX Dupire front local variance {dup_txt}. "
            f"Sticky-delta minus sticky-strike ATM gap after a 1% spot move: {gap_txt}."
        )
    if cfg.walkforward_kill and (kills or product_kills):
        state = build_mm_state(
            time=float(n),
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
            "edge_fail": 1.0 if kills else 0.0,
            "product_edge_fail": 1.0 if product_kills else 0.0,
        }
        ans = _CLIENT.system_one(state, build_desk_questions())
        adj = apply_desk_policy(ans, state)
        notes.append(
            f"Offline Jev answered `{adj.reason}` (kill={adj.kill}, product_kill={adj.product_kill}). "
            "Choice / Noul only. Jev does not emit an order."
        )
    desk_fill_pnl = float(sum(weights[s] * fill_pnl[s] for s in sleeves))
    desk_lob_fills = float(sum(lob_fills.values()))
    desk_adverse = float(sum(adverse_mo.values()))
    quoted_total = int(sum(quote_steps.values()))
    desk_join = float(sum(join_hits.values()) / quoted_total) if quoted_total else 0.0
    if cfg.executable_fills and cfg.fill_model == "lob":
        notes.append(
            f"LOB fill path on. Queue edge {'on' if cfg.queue_edge else 'off'}. "
            f"Unweighted lob fills {desk_lob_fills:.2f}, weighted fill PnL {desk_fill_pnl:.4f}, "
            f"adverse markout {desk_adverse:.4f}. Fill PnL is not inside the residual Sharpe. "
            "Intensity is contracts per second over a 60-second horizon. synthetic_fixture=1."
        )
    notes.append("Quotes are synthetic. synthetic_fixture=1. This is not an OPRA surface.")
    if worst_label != "flat":
        notes.append(f"End-of-path scenario grid worst PnL {worst:.4f} at {worst_label}.")

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
        desk_sharpe_raw=per_step_sharpe(desk_res_raw_w),
        desk_turnover=gross_turnover,
        desk_peak_inventory=peak_inv,
        desk_residual_per_turnover=(pen_desk_pnl / gross_turnover) if gross_turnover > 1e-12 else 0.0,
        desk_residual_per_peak_inventory=(pen_desk_pnl / peak_inv) if peak_inv > 1e-12 else 0.0,
        gamma_path_length=gamma_len,
        vega_path_length=vega_len,
        quote_revisions_per_step=rev_per_step,
        product_kills=product_kills,
        sleeve_kills=list(kills),
        synthetic_fixture=cfg.synthetic_fixture,
        corr_cap=cfg.corr_cap,
        pre_gate_max_abs_rho=pre_max,
        max_abs_rho=max_abs,
        product_corr_sleeve=focus,
        product_ids=product_ids,
        product_pearson=product_mat,
        stress_worst=worst,
        stress_label=worst_label,
        notes=notes,
        desk_fill_pnl=desk_fill_pnl,
        desk_lob_fills=desk_lob_fills,
        desk_adverse_markout=desk_adverse,
        desk_join_rate=desk_join,
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
        pre_gate_max_abs_rho=pre_max,
        ortho_notes=ortho_notes,
        penalties=penalties,
        capacity_scale=capacity_scale,
        product_kills=product_kills,
        sleeve_kills=list(kills),
    )


def _rho_table(run: DeskRun) -> dict[tuple[str, str], float]:
    ids = run.scoreboard.sleeve_ids
    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            out[(a, ids[j])] = float(run.scoreboard.pearson[i, j])
    return out


@dataclass
class MultiSeedReport:
    seeds: tuple[int, ...]
    regimes: tuple[str, ...]
    n_steps: int
    pair_mean: dict[tuple[str, str], float]
    pair_std: dict[tuple[str, str], float]
    max_mean_abs: float
    worst_pair: tuple[str, str, float]
    headline: dict[str, float]


def multi_seed_corr(
    seeds: tuple[int, ...] = EVAL_SEEDS,
    regimes: tuple[str, ...] = EVAL_REGIMES,
    *,
    n_steps: int = 80,
    fit_surfaces: bool = False,
) -> MultiSeedReport:
    """Mean and sample std of pairwise residual ρ across seeds and regimes."""
    bucket: dict[tuple[str, str], list[float]] = {}
    for seed in seeds:
        for regime in regimes:
            run = run_desk(
                DeskConfig(
                    seed=seed,
                    regime=regime,
                    n_steps=n_steps,
                    fit_surfaces=fit_surfaces,
                )
            )
            for key, rho in _rho_table(run).items():
                bucket.setdefault(key, []).append(rho)
    means: dict[tuple[str, str], float] = {}
    stds: dict[tuple[str, str], float] = {}
    worst = ("", "", 0.0)
    for key, vals in bucket.items():
        arr = np.asarray(vals, dtype=float)
        means[key] = float(np.mean(arr))
        stds[key] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        if abs(means[key]) > abs(worst[2]):
            worst = (key[0], key[1], means[key])
    max_mean = max((abs(v) for v in means.values()), default=0.0)

    def _head(a: str, b: str) -> float:
        if (a, b) in means:
            return means[(a, b)]
        if (b, a) in means:
            return means[(b, a)]
        return float("nan")

    return MultiSeedReport(
        seeds=seeds,
        regimes=regimes,
        n_steps=n_steps,
        pair_mean=means,
        pair_std=stds,
        max_mean_abs=max_mean,
        worst_pair=worst,
        headline={
            "skew_residual/fly_butterfly": _head("skew_residual", "fly_butterfly"),
            "mm_spread/vrp_varswap": _head("mm_spread", "vrp_varswap"),
            "roll_yield/vanna_tilt": _head("roll_yield", "vanna_tilt"),
        },
    )


def _honesty_off() -> dict[str, object]:
    """Flags that reprint a pre-1.2 book. Each one off is the identity."""
    return {
        "honesty": False,
        "honest_allocator": False,
        "split_roll_vanna": False,
        "capacity_caps": False,
        "walkforward_kill": False,
        "sigma_clip_quantile": 0.0,
        "min_weight_floor": 0.0,
    }


def legacy_config(**overrides: object) -> DeskConfig:
    """The 1.0 economics: shared smile, unhedged gamma, no gate, old caps."""
    cfg = DeskConfig(
        products=("EQ_INDEX", "EQ_SINGLE", "FX_PAIR"),
        sleeves=LEGACY_SLEEVE_IDS,
        factorize_smile=False,
        separate_convexity=False,
        harden_edges=False,
        book_higher_greeks=False,
        extended_strip=False,
        ortho_gate=False,
        sharpe_tilt=0.0,
        corr_cap=0.50,
        max_sleeve_weight=0.40,
        gate_threshold=0.50,
        regime="baseline",
        fit_surfaces=False,
        lean_book=False,
        executable_fills=False,
        queue_edge=False,
        fill_model="off",
        **_honesty_off(),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def legacy_honest_config(**overrides: object) -> DeskConfig:
    """1.2.0-zig-honest book. Full sleeve set, honesty on, fills not executable.

    The smoothness penalty, σ clip, floor, capacity caps, walk-forward kill,
    and roll/vanna split stay on. ``lean_book`` and the LOB fill columns stay
    off so this reprints the 1.2 residual scoreboard.
    """
    cfg = DeskConfig(
        fit_surfaces=False,
        lean_book=False,
        executable_fills=False,
        queue_edge=False,
        fill_model="off",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def legacy_ortho_config(**overrides: object) -> DeskConfig:
    """1.1.0-zig-ortho snapshot.

    Smile split, convexity split, six-factor strip, and the 0.40 / 0.35 gate
    stay on. Honesty, the σ clip, the weight floor, capacity caps, the
    walk-forward kill, and the roll/vanna split stay off, so seed-11
    correlations reprint. Weights can still move if a later default inside
    ``allocate`` changes; this snapshot passes ``sigma_clip_quantile=0``.
    """
    cfg = DeskConfig(
        fit_surfaces=False,
        lean_book=False,
        executable_fills=False,
        queue_edge=False,
        fill_model="off",
        **_honesty_off(),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def pair_rho(run: DeskRun, a: str, b: str) -> float | None:
    return _pair_rho(run.scoreboard.sleeve_ids, run.scoreboard.pearson, a, b)
