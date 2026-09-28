"""Synthetic walk-forward harness and the slot for a later OPRA/NBBO tape.

No live vendor is stubbed. ``load_tape("synthetic")`` builds a GBM spot
path. A local CSV with the documented columns can be read later. Names that
look like a live OPRA/NBBO session are refused.

Fill-hazard calibration uses the seconds clock: λ(δ) = A exp(−k δ) with A
in events per second and δ in price units. That A is not Guéant's
closed-form mid-touch intensity (arXiv 1105.3115), which lives in the
units of the quoter formula. The fitted pair is applied as the Poisson
intensity and κ on the test window, and reported beside the quoter.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jev_omm.backtest.simulator import SimResult, run_simulation
from jev_omm.config import EngineConfig, MarketConfig, QuoterConfig, RiskConfig, SimConfig
from jev_omm.models.types import Side
from jev_omm.quoter.gueant_ode import IntensityFit, IntensityObs, estimate_intensity
from jev_omm.research.metrics import summarize_run

TAPE_COLUMNS = ("time_seconds", "spot", "bid", "ask", "bid_sz", "ask_sz")

_LIVE_MARKERS = ("opra://", "live-nbbo", "polygon.io", "databento", "cboe-live")


@dataclass
class Tape:
    """Spot path plus an optional top of book. ``source`` is never a vendor."""

    time_seconds: np.ndarray
    spot: np.ndarray
    source: str
    bid: np.ndarray | None = None
    ask: np.ndarray | None = None


def load_tape(spec: str, *, n_steps: int = 80, seed: int = 11, spot0: float = 100.0) -> Tape:
    """Load a synthetic path or a local fixture CSV.

    ``spec="synthetic"`` draws a GBM on the trading-time clock.
    Anything else must be a filesystem path. Live vendor URLs are refused.
    This function does not invent OPRA prints.
    """
    key = spec.strip()
    low = key.lower()
    if any(mark in low for mark in _LIVE_MARKERS):
        raise RuntimeError(
            "Live OPRA/NBBO is not wired. Use spec='synthetic' or a local CSV "
            f"with columns {','.join(TAPE_COLUMNS)}."
        )
    if key == "synthetic":
        return synthetic_spot_tape(n_steps=n_steps, seed=seed, spot0=spot0)
    path = Path(key)
    if not path.is_file():
        raise FileNotFoundError(
            f"No tape at {spec}. Historical OPRA/NBBO fixtures are not bundled; "
            "place a licensed CSV locally or use spec='synthetic'."
        )
    return _read_csv_tape(path)


def synthetic_spot_tape(*, n_steps: int, seed: int, spot0: float, dt_seconds: float = 60.0) -> Tape:
    from jev_omm.config import TRADING_SECONDS_PER_YEAR

    rng = np.random.default_rng(seed)
    dt_years = dt_seconds / TRADING_SECONDS_PER_YEAR
    spots = np.empty(n_steps, dtype=float)
    times = np.empty(n_steps, dtype=float)
    spot = float(spot0)
    for i in range(n_steps):
        z = float(rng.standard_normal())
        spot = spot * float(np.exp((-0.5 * 0.20 * 0.20) * dt_years + 0.20 * np.sqrt(dt_years) * z))
        spots[i] = spot
        times[i] = (i + 1) * dt_seconds
    return Tape(time_seconds=times, spot=spots, source="synthetic")


def _read_csv_tape(path: Path) -> Tape:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [c for c in TAPE_COLUMNS if c not in fields]
        if missing:
            raise ValueError(f"{path} missing columns {missing}; expected {list(TAPE_COLUMNS)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has a header and no rows")
    return Tape(
        time_seconds=np.array([float(r["time_seconds"]) for r in rows]),
        spot=np.array([float(r["spot"]) for r in rows]),
        bid=np.array([float(r["bid"]) for r in rows]),
        ask=np.array([float(r["ask"]) for r in rows]),
        source=f"csv:{path.name}",
    )


def _base_sim(**overrides: object) -> SimConfig:
    raw = dict(
        n_steps=48,
        seed=11,
        fill_model="poisson",
        fill_intensity_per_second=0.02,
        feature_pack="off",
        hedge_band=8.0,
    )
    raw.update(overrides)
    return SimConfig(**raw)  # type: ignore[arg-type]


def _engine(mode: str, pack: str, *, sim: SimConfig | None = None, quoter: QuoterConfig | None = None) -> EngineConfig:
    q = quoter or QuoterConfig(
        mode=mode,
        gamma=0.12,
        kappa=1.5,
        sigma=0.45,
        A=140.0,
        quote_size=1,
        min_half_spread=0.05,
        max_half_spread=2.0,
        option_grid_n=11,
        option_grid_steps=8,
        inventory_cap=6,
        ode_steps=200,
    )
    s = sim or _base_sim(feature_pack=pack)
    return EngineConfig(
        market=MarketConfig(),
        quoter=q,
        risk=RiskConfig(max_abs_inventory=40, max_abs_delta=80.0, max_loss=5000.0),
        sim=s,
    )


def ablation_specs() -> list[tuple[str, str, str]]:
    """(label, quoter mode, feature pack)."""
    return [
        ("fixed_spread", "fixed", "off"),
        ("avellaneda_stoikov", "as_finite_horizon", "off"),
        ("gueant_asymptotic", "gueant_asymptotic", "off"),
        ("option_vega", "option_vega", "off"),
        ("as_flow", "as_finite_horizon", "flow"),
        ("as_flow_gex", "as_finite_horizon", "flow_gex"),
        ("as_flow_gex_state", "as_finite_horizon", "all"),
    ]


def _quoter_for(mode: str) -> QuoterConfig:
    if mode == "fixed":
        return QuoterConfig(
            mode="as_finite_horizon",
            gamma=0.0,
            kappa=1.5,
            sigma=0.45,
            quote_size=1,
            min_half_spread=0.25,
            max_half_spread=0.25,
            T_horizon=1.0 / 252.0,
        )
    return QuoterConfig(
        mode=mode,
        gamma=0.12,
        kappa=1.5,
        sigma=0.45,
        A=140.0,
        quote_size=1,
        min_half_spread=0.05,
        max_half_spread=2.0,
        option_grid_n=11,
        option_grid_steps=8,
        contract_vega=8.0,
        vega_limit=40.0,
        inventory_cap=6,
        ode_steps=200,
    )


def run_ablation(*, n_steps: int = 48, seed: int = 11) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, mode, pack in ablation_specs():
        cfg = _engine(
            mode,
            pack,
            sim=_base_sim(n_steps=n_steps, seed=seed, feature_pack=pack),
            quoter=_quoter_for(mode),
        )
        result = run_simulation(cfg)
        metrics = summarize_run(result, n_steps=n_steps, max_abs_delta=cfg.risk.max_abs_delta)
        metrics["label"] = label
        metrics["quoter_mode"] = mode
        metrics["feature_pack"] = pack
        rows.append(metrics)
    return rows


def observations_from_result(result: SimResult, dt_seconds: float) -> list[IntensityObs]:
    """One Poisson exposure per posted side. Zeros are kept (they identify A, k)."""
    by_time: dict[float, list] = {}
    for fill in result.fills:
        by_time.setdefault(float(fill.time), []).append(fill)
    obs: list[IntensityObs] = []
    for state in result.states:
        quote = state.quote
        if quote is None:
            continue
        fills = by_time.get(float(state.time), [])
        bid_n = sum(f.size for f in fills if f.side == Side.BID)
        ask_n = sum(f.size for f in fills if f.side == Side.ASK)
        obs.append(
            IntensityObs(
                delta=max(0.0, state.option_mid - quote.bid),
                exposure=dt_seconds,
                fills=float(bid_n),
            )
        )
        obs.append(
            IntensityObs(
                delta=max(0.0, quote.ask - state.option_mid),
                exposure=dt_seconds,
                fills=float(ask_n),
            )
        )
    return obs


def hazard_from_observations(
    obs: list[IntensityObs],
    *,
    kappa_prior: float,
) -> tuple[float, float, str, IntensityFit]:
    """Fit λ(δ) = A exp(−k δ), events per second.

    When posted distances barely move, k is not identified and the joint MLE
    collapses. In that case k stays at ``kappa_prior`` and A is backed out so
    A exp(−k δ̄) matches the empirical fill rate. Guéant closed-form A
    (arXiv 1105.3115) is a different object and is not written here.
    """
    fit = estimate_intensity(obs)
    if not obs:
        return 0.0, kappa_prior, "empty", fit
    exposure = sum(o.exposure for o in obs)
    fills = sum(o.fills for o in obs)
    deltas = [o.delta for o in obs]
    span = max(deltas) - min(deltas)
    mean_d = sum(o.delta * o.exposure for o in obs) / exposure if exposure > 0.0 else 0.0
    rate = fills / exposure if exposure > 0.0 else 0.0
    identified = span >= 0.05 and fit.A > 1e-6 and fit.k > 1e-4 and math.isfinite(fit.A)
    if identified:
        return float(fit.A), float(min(fit.k, 20.0)), "mle", fit
    A = rate * math.exp(kappa_prior * mean_d)
    return float(A), float(kappa_prior), "moments_k_held", fit


def walk_forward_intensity(*, n_steps: int = 40, seed: int = 3) -> dict[str, object]:
    """Fit λ(δ) on the first window; score a second window with and without it.

    The test window is a new seed, not a slice of the same path. Both are
    synthetic. A licensed tape would replace ``load_tape("synthetic")`` and
    the same observation schema.
    """
    train_cfg = _engine(
        "as_finite_horizon",
        "off",
        sim=_base_sim(n_steps=n_steps, seed=seed, fill_intensity_per_second=0.03),
        quoter=_quoter_for("as_finite_horizon"),
    )
    train = run_simulation(train_cfg)
    obs = observations_from_result(train, train_cfg.sim.dt_seconds)
    A, k, method, fit = hazard_from_observations(obs, kappa_prior=train_cfg.quoter.kappa)
    tape_note = load_tape("synthetic", n_steps=n_steps, seed=seed).source
    base_test = _base_sim(n_steps=n_steps, seed=seed + 17, fill_intensity_per_second=0.03)
    uncal = run_simulation(
        _engine("as_finite_horizon", "off", sim=base_test, quoter=_quoter_for("as_finite_horizon"))
    )
    cal_sim = _base_sim(
        n_steps=n_steps,
        seed=seed + 17,
        fill_intensity_per_second=float(A),
    )
    cal_quoter = _quoter_for("as_finite_horizon").model_copy(update={"kappa": float(k)})
    cal = run_simulation(_engine("as_finite_horizon", "off", sim=cal_sim, quoter=cal_quoter))
    return {
        "tape_source": tape_note,
        "train_fills": len(train.fills),
        "A_per_second": float(A),
        "k_per_price": float(k),
        "fit_method": method,
        "mle_A": float(fit.A),
        "mle_k": float(fit.k),
        "loglik": float(fit.loglik),
        "n_bins": int(fit.n_bins),
        "test_uncalibrated_pnl": float(uncal.final_pnl),
        "test_calibrated_pnl": float(cal.final_pnl),
        "test_uncalibrated_fills": len(uncal.fills),
        "test_calibrated_fills": len(cal.fills),
        "note": (
            "A is events/second at δ=0 on this synthetic clock. "
            "It is not substituted into QuoterConfig.A (Guéant, arXiv 1105.3115). "
            f"Fit method: {method}."
        ),
    }


_COLUMNS = (
    "label",
    "pnl",
    "fill_rate",
    "realized_spread",
    "markout_1",
    "markout_5",
    "markout_30",
    "hedge_cost",
    "inventory_variance",
    "max_drawdown",
    "quote_uptime",
    "greek_utilization",
    "n_fills",
    "n_hedges",
)


def render_ablation_markdown(rows: list[dict[str, object]] | None = None, wf: dict[str, object] | None = None) -> str:
    rows = rows if rows is not None else run_ablation()
    wf = wf if wf is not None else walk_forward_intensity()
    header = (
        "# Synthetic ablation (checked-in)\n\n"
        "Research laboratory output. The tape is a seeded GBM plus a Poisson "
        "touch fill (`fill_model=poisson`, 48 steps, one trading minute each, "
        "seed 11). It is not OPRA, NBBO, or a production market-making result.\n\n"
        "Markout columns are 1 / 5 / 30 **simulator steps** (here, minutes), "
        "not exchange seconds. `fill_rate` is fill events per step and can "
        "exceed 1 when both sides trade. `greek_utilization` is mean absolute net delta "
        "divided by the hard delta limit. Feature packs are the existing flow, "
        "GEX, and instability scalers; `off` is the identity. `fixed_spread` "
        "is Avellaneda–Stoikov with γ = 0 and the half-spread clamped to 0.25.\n\n"
        "A later licensed OPRA/NBBO fixture should be a CSV with columns "
        "`time_seconds,spot,bid,ask,bid_sz,ask_sz` passed to `load_tape`. "
        "No such file is shipped, and live vendor URLs are refused.\n\n"
    )
    head = "| " + " | ".join(_COLUMNS) + " |\n"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |\n"
    body = ""
    for row in rows:
        cells = []
        for key in _COLUMNS:
            val = row[key]
            if isinstance(val, float):
                if abs(val) < 5e-7:
                    val = 0.0
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        body += "| " + " | ".join(cells) + " |\n"
    cal = (
        "\n## Walk-forward fill hazard (synthetic)\n\n"
        f"- Tape source: `{wf['tape_source']}`\n"
        f"- Train fills: {wf['train_fills']}\n"
        f"- Fitted A (events/second at δ=0): {float(wf['A_per_second']):.6f}\n"
        f"- Fitted k (per price unit): {float(wf['k_per_price']):.4f}\n"
        f"- Fit method: `{wf['fit_method']}` "
        f"(joint MLE A={float(wf['mle_A']):.3e}, k={float(wf['mle_k']):.4f}; "
        "rejected when δ does not move)\n"
        f"- Train log-likelihood of the joint MLE: {float(wf['loglik']):.4f} "
        f"on {wf['n_bins']} side-steps\n"
        f"- Test PnL, default intensity: {float(wf['test_uncalibrated_pnl']):.4f} "
        f"({wf['test_uncalibrated_fills']} fills)\n"
        f"- Test PnL, intensity and κ set from the train fit: "
        f"{float(wf['test_calibrated_pnl']):.4f} "
        f"({wf['test_calibrated_fills']} fills)\n\n"
        f"{wf['note']}\n\n"
        "If the calibrated test PnL is not better, that is the result. "
        "The fit is a hazard on this generator, not a claim that (A, k) "
        "transfer to listed options.\n"
    )
    return header + head + sep + body + cal
