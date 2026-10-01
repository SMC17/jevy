"""Paper sleeves. Each one is a target and a PnL stream, not an order.

The off switch is the identity: ``enabled=False`` returns a zero target and a
zero PnL. Flow, the instability gate, COT, and the autocall warehouse are
also the identity when ``gates_on`` is false.

Greek buckets go through ``greek_pnl_step``. Research units use S = 1, so
ΔS in that call is the simple return and gamma PnL is 0.5 * Γ * r^2.

``separate_convexity`` hedges the market-maker inventory gamma and moves the
variance sleeve onto an implied-minus-lagged-realized premium, with no
same-bar spot gamma. ``harden_edges`` is the 1.1 wiring into Hawkes, GEX
adjust, and the higher greek buckets. Both flags off reproduce the 1.0
sleeve formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

from jev_omm.config import QuoterConfig
from jev_omm.execution.lob import expected_fills
from jev_omm.flow.hawkes import HawkesParams, excitation
from jev_omm.flow.signals import flow_prior
from jev_omm.hedge.delta import greek_pnl_step
from jev_omm.positioning.adjust import cot_fade, gex_adjust
from jev_omm.positioning.overlays import annualized_roll
from jev_omm.pricing.black_scholes import charm_tau
from jev_omm.pricing.parity import box_theo
from jev_omm.pricing.varswap import variance_swap_vega
from jev_omm.quoter.avellaneda_stoikov import optimal_half_spread
from jev_omm.state_os.cross_impact import hub_spoke_impact
from jev_omm.state_os.gate import state_gate
from jev_omm.state_os.warehouse import autocall_flow

SLEEVE_IDS: tuple[str, ...] = (
    "mm_spread",
    "skew_residual",
    "calendar_term",
    "fly_butterfly",
    "vrp_varswap",
    "flow_toxicity",
    "gex_forced",
    "parity_box",
    "wing_kurtosis",
    "sticky_regime",
    "dispersion_index",
    "roll_yield",
    "charm_bleed",
    "vanna_tilt",
    "queue_sniper",
    "cot_fade_sleeve",
    "warehouse_autocall",
    "box_rate",
    "cross_impact",
    "rough_vol_stress",
)

LEGACY_SLEEVE_IDS: tuple[str, ...] = SLEEVE_IDS[:8]

# Static research budgets. They are labels for the allocator write-up, not live limits.
RISK_BUDGET: dict[str, float] = {
    "mm_spread": 1.0,
    "skew_residual": 0.6,
    "calendar_term": 0.5,
    "fly_butterfly": 0.45,
    "vrp_varswap": 0.7,
    "flow_toxicity": 0.4,
    "gex_forced": 0.5,
    "parity_box": 0.35,
    "wing_kurtosis": 0.40,
    "sticky_regime": 0.30,
    "dispersion_index": 0.45,
    "roll_yield": 0.35,
    "charm_bleed": 0.25,
    "vanna_tilt": 0.40,
    "queue_sniper": 0.30,
    "cot_fade_sleeve": 0.35,
    "warehouse_autocall": 0.30,
    "box_rate": 0.30,
    "cross_impact": 0.35,
    "rough_vol_stress": 0.25,
}

_AS = QuoterConfig()
_BOX_K1 = 100.0
_BOX_K2 = 110.0
_BOX_T = 0.25
_HAWKES = HawkesParams()
_ROLL_ASSETS = frozenset({"commodity", "rates"})


@dataclass
class SleeveContext:
    product_id: str
    beta_to_index: float
    product_return: float
    index_return: float
    d_sigma: float
    dt: float
    skew: float
    d_skew: float
    fly: float
    d_fly: float
    term: float
    d_term: float
    tox: float
    gex_norm: float
    instability: float
    f_signed: float
    box_edge: float
    d_box: float
    iv: float
    prev_target: float
    gates_on: bool
    enabled: bool
    jev_size_mult: float = 1.0
    jev_kill: bool = False
    fill_draw: float = 0.5
    rate: float = 0.04
    separate_convexity: bool = True
    harden_edges: bool = True
    book_higher_greeks: bool = True
    asset_class: str = "equity_index"
    wing: float = 0.0
    d_wing: float = 0.0
    sticky: float = 0.0
    d_sticky: float = 0.0
    dispersion: float = 0.0
    d_dispersion: float = 0.0
    roll: float = 0.0
    d_roll: float = 0.0
    charm_clock: float = 1.0
    vanna_mis: float = 0.0
    d_vanna: float = 0.0
    queue: float = 0.0
    d_queue: float = 0.0
    cot_z: float = 0.0
    d_cot: float = 0.0
    autocall: float = 0.0
    d_autocall: float = 0.0
    funding: float = 0.0
    d_funding: float = 0.0
    cross_ret: float = 0.0
    d_cross: float = 0.0
    rough: float = 0.0
    d_rough: float = 0.0
    lagged_rv: float = 0.0
    d_vrp: float = 0.0


@dataclass
class SleeveQuote:
    sleeve_id: str
    enabled: bool
    target: float
    spread_mult: float
    size_mult: float
    risk_budget: float
    delta: float
    gamma: float
    vega: float
    vanna: float
    volga: float
    theta: float
    raw_pnl: float
    fee: float
    fill: int
    f_beta: float
    f_gamma: float
    f_vega: float
    f_volga: float
    f_vanna: float
    note: str


def _zero(sleeve_id: str, ctx: SleeveContext, note: str) -> SleeveQuote:
    return SleeveQuote(
        sleeve_id=sleeve_id,
        enabled=ctx.enabled,
        target=0.0,
        spread_mult=1.0,
        size_mult=1.0,
        risk_budget=RISK_BUDGET[sleeve_id],
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        vanna=0.0,
        volga=0.0,
        theta=0.0,
        raw_pnl=0.0,
        fee=0.0,
        fill=0,
        f_beta=ctx.index_return,
        f_gamma=0.0,
        f_vega=0.0,
        f_volga=0.0,
        f_vanna=0.0,
        note=note,
    )


def _finish(
    sleeve_id: str,
    ctx: SleeveContext,
    *,
    target: float,
    spread_mult: float,
    size_mult: float,
    delta: float,
    gamma: float,
    vega: float,
    vanna: float,
    volga: float,
    theta: float,
    alpha: float,
    fill: int,
    fee_rate: float,
    note: str,
) -> SleeveQuote:
    """Pack a sleeve step. Callers pass the position they actually hold.

    ``alpha`` is everything that is not a greek bucket (spread, residual
    edge, variance premium). Volga and vanna cash are added only when
    ``book_higher_greeks`` is on, and those same numbers are the strip columns.
    """
    if not ctx.enabled or ctx.jev_kill:
        return _zero(sleeve_id, ctx, "identity" if not ctx.enabled else "jev_kill")
    scale = ctx.jev_size_mult
    target *= scale
    delta *= scale
    gamma *= scale
    vega *= scale
    vanna *= scale
    volga *= scale
    theta *= scale
    alpha *= scale
    step = greek_pnl_step(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        d_spot=ctx.product_return,
        d_sigma=ctx.d_sigma,
        dt=ctx.dt,
    )
    volga_pnl = 0.5 * volga * ctx.d_sigma * ctx.d_sigma
    vanna_pnl = vanna * ctx.product_return * ctx.d_sigma
    extra = (volga_pnl + vanna_pnl) if ctx.book_higher_greeks else 0.0
    fee = fee_rate * abs(target)
    raw = alpha + step.delta_pnl + step.gamma_pnl + step.vega_pnl + step.theta_pnl + extra - fee
    return SleeveQuote(
        sleeve_id=sleeve_id,
        enabled=True,
        target=target,
        spread_mult=spread_mult,
        size_mult=size_mult * ctx.jev_size_mult,
        risk_budget=RISK_BUDGET[sleeve_id],
        delta=delta,
        gamma=gamma,
        vega=vega,
        vanna=vanna,
        volga=volga,
        theta=theta,
        raw_pnl=raw,
        fee=fee,
        fill=fill,
        f_beta=ctx.index_return,
        f_gamma=step.gamma_pnl,
        f_vega=step.vega_pnl,
        f_volga=volga_pnl,
        f_vanna=vanna_pnl,
        note=note,
    )


def quote_or_target(sleeve_id: str, ctx: SleeveContext, *, fee_rate: float = 0.001) -> SleeveQuote:
    if sleeve_id not in RISK_BUDGET:
        raise KeyError(sleeve_id)
    if not ctx.enabled:
        return _zero(sleeve_id, ctx, "identity")
    fn = _SLEEVES[sleeve_id]
    return fn(ctx, fee_rate=fee_rate)


def _mm(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    half = optimal_half_spread(_AS, t_remaining=_AS.T_horizon)
    size_mult = 1.0
    spread_mult = 1.0
    filled = 1 if ctx.fill_draw < 0.45 * size_mult else 0
    spread_capture = half * 0.02 * filled
    beta = ctx.beta_to_index
    if ctx.separate_convexity:
        # Inventory gamma is hedged. What remains is a small leak plus the spread.
        delta = 0.06 * beta
        gamma = -4.0
        note = "as_half_spread_gamma_hedged"
    else:
        delta = 0.35 * beta
        gamma = -80.0
        note = "as_half_spread"
    return _finish(
        "mm_spread",
        ctx,
        target=float(filled),
        spread_mult=spread_mult,
        size_mult=size_mult,
        delta=delta,
        gamma=gamma,
        vega=0.40,
        vanna=0.10,
        volga=-0.05,
        theta=-0.02,
        alpha=spread_capture,
        fill=filled,
        fee_rate=fee_rate,
        note=note,
    )


def _skew(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    target = -4.0 * ctx.skew
    alpha = ctx.prev_target * ctx.d_skew * 1.5
    return _finish(
        "skew_residual",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.08 * ctx.beta_to_index,
        gamma=4.0,
        vega=1.20,
        vanna=0.80,
        volga=0.15,
        theta=-0.01,
        alpha=alpha,
        fill=1 if abs(target) > 1e-8 else 0,
        fee_rate=fee_rate,
        note="fade_wing_slope",
    )


def _calendar(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    target = -3.0 * ctx.term
    alpha = ctx.prev_target * ctx.d_term * 1.4
    return _finish(
        "calendar_term",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.02 * ctx.beta_to_index,
        gamma=1.0,
        vega=0.90,
        vanna=0.05,
        volga=0.02,
        theta=-0.04,
        alpha=alpha,
        fill=1 if abs(target) > 1e-8 else 0,
        fee_rate=fee_rate,
        note="fade_term_slope",
    )


def _fly(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    target = -4.0 * ctx.fly
    alpha = ctx.prev_target * ctx.d_fly * 1.6
    return _finish(
        "fly_butterfly",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.03 * ctx.beta_to_index,
        gamma=6.0,
        vega=0.30,
        vanna=0.10,
        volga=0.70,
        theta=-0.01,
        alpha=alpha,
        fill=1 if abs(target) > 1e-8 else 0,
        fee_rate=fee_rate,
        note="fade_curvature",
    )


def _vrp(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    if ctx.separate_convexity:
        # Flat-smile limit of the replication: K_var → σ². The accrual uses
        # lagged realized variance, not this bar's squared return, so the
        # premium is not the market-maker's spot gamma. Carr–Wu sign.
        k_var = ctx.iv * ctx.iv
        vega = 0.05 * variance_swap_vega(max(ctx.iv, 1e-4))
        alpha = (k_var - ctx.lagged_rv) * ctx.dt + ctx.prev_target * ctx.d_vrp * 0.25
        return _finish(
            "vrp_varswap",
            ctx,
            target=-1.0,
            spread_mult=1.0,
            size_mult=1.0,
            delta=0.0,
            gamma=0.0,
            vega=vega,
            vanna=0.0,
            volga=0.0,
            theta=0.0,
            alpha=alpha,
            fill=1,
            fee_rate=fee_rate,
            note="varswap_premium_vs_lagged_rv",
        )
    k_var = ctx.iv * ctx.iv
    return _finish(
        "vrp_varswap",
        ctx,
        target=-1.0,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.15 * ctx.beta_to_index,
        gamma=-2.0,
        vega=0.50,
        vanna=0.0,
        volga=0.20,
        theta=0.0,
        alpha=k_var * ctx.dt,
        fill=1,
        fee_rate=fee_rate,
        note="short_variance",
    )


def _flow(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    if ctx.gates_on:
        _tox, spread_mult, size_mult = flow_prior(ctx.tox, 0.0, 0.0, 0.0, 0.0)
    else:
        spread_mult, size_mult = 1.0, 1.0
    filled = 1 if ctx.fill_draw < 0.40 * size_mult else 0
    spread_earn = 0.015 * spread_mult * filled
    adverse = 0.06 * ctx.tox * filled
    if ctx.harden_edges and ctx.gates_on and ctx.tox > 0.4:
        # A short memory of toxic prints. Excitation widens the adverse bill.
        exc = excitation(_HAWKES, 1.0, [1.0 - 0.2 * ctx.tox, 0.85])
        adverse *= 1.0 + 0.15 * min(exc, 4.0)
        note = "flow_prior_hawkes"
    else:
        note = "flow_prior" if ctx.gates_on else "flow_identity"
    return _finish(
        "flow_toxicity",
        ctx,
        target=size_mult,
        spread_mult=spread_mult,
        size_mult=size_mult,
        delta=0.05 * ctx.beta_to_index * size_mult,
        gamma=-2.0 * size_mult,
        vega=0.10 * size_mult,
        vanna=0.0,
        volga=0.0,
        theta=0.0,
        alpha=spread_earn - adverse,
        fill=filled,
        fee_rate=fee_rate,
        note=note,
    )


def _gex(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    gate = state_gate(
        enabled=ctx.gates_on,
        instability_value=ctx.instability,
        constraint_active=False,
        parent_remaining=0.0,
        f_signed=ctx.f_signed,
        l_exec=1.0,
        mid=1.0,
    )
    short_gamma = 1.0 if ctx.gex_norm < 0.0 else 0.0
    direction = 1.0 if ctx.f_signed >= 0.0 else -1.0
    scale = gate.size_mult
    spread_mult = gate.spread_mult
    if ctx.harden_edges and ctx.gates_on:
        _shift, spread_g, size_g, _band, _urg = gex_adjust(
            True, ctx.gex_norm, 0.0, ctx.f_signed, 1.0
        )
        scale *= size_g
        spread_mult *= spread_g
        note = "state_gate_gex_adjust"
    else:
        note = "state_gate" if ctx.gates_on else "gex_identity"
    target = direction * short_gamma * min(abs(ctx.gex_norm), 2.0) * scale
    alpha = ctx.prev_target * ctx.index_return * 0.8
    gamma = -2.0 if ctx.separate_convexity else -10.0
    return _finish(
        "gex_forced",
        ctx,
        target=target,
        spread_mult=spread_mult,
        size_mult=scale,
        delta=0.50 * ctx.beta_to_index * short_gamma * scale,
        gamma=gamma * short_gamma * scale,
        vega=0.20 * scale,
        vanna=0.40 * short_gamma * scale,
        volga=0.0,
        theta=0.0,
        alpha=alpha,
        fill=1 if abs(target) > 1e-8 else 0,
        fee_rate=fee_rate,
        note=note,
    )


def _parity(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    theo = box_theo(_BOX_K1, _BOX_K2, _BOX_T, ctx.rate)
    target = -ctx.box_edge * theo
    alpha = ctx.prev_target * ctx.d_box * theo * 0.02
    return _finish(
        "parity_box",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.01 * ctx.beta_to_index,
        gamma=0.2,
        vega=0.05,
        vanna=0.0,
        volga=0.0,
        theta=0.0,
        alpha=alpha,
        fill=1 if abs(ctx.box_edge) > 1e-6 else 0,
        fee_rate=fee_rate,
        note="box_theo",
    )


def _wing(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    target = -3.0 * ctx.wing
    alpha = ctx.prev_target * ctx.d_wing * 1.5
    return _finish(
        "wing_kurtosis",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.02 * ctx.beta_to_index,
        gamma=1.5,
        vega=0.20,
        vanna=0.05,
        volga=0.35,
        theta=-0.005,
        alpha=alpha,
        fill=1 if abs(target) > 1e-8 else 0,
        fee_rate=fee_rate,
        note="fade_far_wing",
    )


def _sticky(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    target = -2.5 * ctx.sticky
    alpha = ctx.prev_target * ctx.d_sticky * 1.3
    return _finish(
        "sticky_regime",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.0,
        gamma=0.0,
        vega=0.10,
        vanna=0.0,
        volga=0.0,
        theta=0.0,
        alpha=alpha,
        fill=1 if abs(ctx.d_sticky) > 1e-8 else 0,
        fee_rate=fee_rate,
        note="sticky_strike_vs_delta",
    )


def _dispersion(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    if ctx.product_id != "EQ_INDEX":
        return _zero("dispersion_index", ctx, "index_only")
    # The common vol shock cancels in a spread. This sleeve trades the
    # residual index-versus-basket gap, with no spot vega of its own.
    target = -4.0 * ctx.dispersion
    alpha = ctx.prev_target * ctx.d_dispersion * 1.5
    return _finish(
        "dispersion_index",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        vanna=0.0,
        volga=0.0,
        theta=0.0,
        alpha=alpha,
        fill=1 if abs(target) > 1e-8 else 0,
        fee_rate=fee_rate,
        note="index_vs_basket_iv",
    )


def _roll(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    if ctx.asset_class not in _ROLL_ASSETS:
        return _zero("roll_yield", ctx, "not_this_asset")
    rolled = annualized_roll(1.0, 1.0 + 0.01 * ctx.roll, 30.0)
    target = -3.0 * rolled
    alpha = ctx.prev_target * ctx.d_roll * 1.1
    return _finish(
        "roll_yield",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        vanna=0.0,
        volga=0.0,
        theta=-0.01,
        alpha=alpha,
        fill=1 if abs(target) > 1e-8 else 0,
        fee_rate=fee_rate,
        note="contango_fade",
    )


def _charm(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    iv = ctx.iv if ctx.iv > 0.02 else 0.02
    ch = charm_tau(1.0, 1.0, 30.0 / 365.0, ctx.rate, 0.0, iv, True)
    clock = ctx.charm_clock if ctx.charm_clock > 0.0 else 1.0
    target = -0.4 * clock
    # The clock (weekend vs the hedge schedule) is the sleeve. Charm itself
    # is a slow level, not a second spot-gamma.
    alpha = 0.0035 * clock + 0.00005 * ch
    return _finish(
        "charm_bleed",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        vanna=0.0,
        volga=0.0,
        theta=-0.01 * clock,
        alpha=alpha,
        fill=1,
        fee_rate=fee_rate,
        note="charm_vs_hedge_clock",
    )


def _vanna(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    target = -3.0 * ctx.vanna_mis
    alpha = ctx.prev_target * ctx.d_vanna * 1.4
    return _finish(
        "vanna_tilt",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.0,
        gamma=0.0,
        vega=0.15,
        vanna=0.55,
        volga=0.0,
        theta=0.0,
        alpha=alpha,
        fill=1 if abs(target) > 1e-8 else 0,
        fee_rate=fee_rate,
        note="spot_vol_mispricing",
    )


def _queue(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    ahead = max(0.05, 1.2 - max(ctx.queue, 0.0))
    got = expected_fills(ahead, 1.0, 6.0, 1.5, 1.0, 0.20)
    size = min(max(got, 0.0), 1.0)
    target = -3.0 * ctx.queue * max(size, 0.25)
    alpha = ctx.prev_target * ctx.d_queue * 1.3
    return _finish(
        "queue_sniper",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        vanna=0.0,
        volga=0.0,
        theta=0.0,
        alpha=alpha,
        fill=1 if got > 1e-8 else 0,
        fee_rate=fee_rate,
        note="queue_value",
    )


def _cot(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    if ctx.gates_on:
        _shift, size = cot_fade(True, ctx.cot_z, 1.0)
        note = "cot_fade"
    else:
        size = 1.0
        note = "cot_identity"
    target = -ctx.cot_z * size
    alpha = ctx.prev_target * ctx.d_cot * 0.9
    return _finish(
        "cot_fade_sleeve",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=size,
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        vanna=0.0,
        volga=0.0,
        theta=0.0,
        alpha=alpha,
        fill=1 if abs(ctx.d_cot) > 1e-10 else 0,
        fee_rate=fee_rate,
        note=note,
    )


def _warehouse(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    if not ctx.gates_on:
        return _zero("warehouse_autocall", ctx, "warehouse_identity")
    spot = max(0.5, 1.0 + ctx.autocall)
    flow = autocall_flow(
        spot,
        1.0,
        0.70,
        max(ctx.iv, 0.05),
        0.25,
        [spot],
        0.30,
        0.0,
        min(max(ctx.autocall, 0.0), 1.0),
        1.0,
    )
    score = 0.002 * flow.digital + 0.15 * flow.knock_in + 0.01 * flow.duration
    target = score if ctx.autocall >= 0.0 else -score
    alpha = ctx.prev_target * ctx.d_autocall * 1.2
    gate = state_gate(
        enabled=True,
        instability_value=ctx.instability,
        constraint_active=False,
        parent_remaining=0.0,
        f_signed=0.0,
        l_exec=1.0,
        mid=1.0,
    )
    return _finish(
        "warehouse_autocall",
        ctx,
        target=target * gate.size_mult,
        spread_mult=gate.spread_mult,
        size_mult=gate.size_mult,
        delta=0.05 * gate.size_mult,
        gamma=0.0,
        vega=0.0,
        vanna=0.0,
        volga=0.0,
        theta=0.0,
        alpha=alpha * gate.size_mult,
        fill=1 if abs(target) > 1e-8 else 0,
        fee_rate=fee_rate,
        note="autocall_warehouse",
    )


def _box_rate(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    theo = box_theo(_BOX_K1, _BOX_K2, _BOX_T, ctx.rate)
    target = -ctx.funding * theo * 0.5
    alpha = ctx.prev_target * ctx.d_funding * theo * 0.02
    return _finish(
        "box_rate",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        vanna=0.0,
        volga=0.0,
        theta=0.0,
        alpha=alpha,
        fill=1 if abs(ctx.funding) > 1e-8 else 0,
        fee_rate=fee_rate,
        note="funding_vs_box",
    )


def _cross(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    # One hub book. Copying the same factor onto every name would scale it
    # by the product count without adding a new risk.
    if ctx.product_id != "EQ_INDEX":
        return _zero("cross_impact", ctx, "hub_only")
    hub, _spoke = hub_spoke_impact(ctx.cross_ret, 0.0, 1.0, 0.15, 0.05)
    target = -3.0 * hub
    alpha = ctx.prev_target * ctx.d_cross * 1.3
    return _finish(
        "cross_impact",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        vanna=0.0,
        volga=0.0,
        theta=0.0,
        alpha=alpha,
        fill=1 if abs(hub) > 1e-12 else 0,
        fee_rate=fee_rate,
        note="hub_spoke_lag",
    )


def _rough(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    target = -2.0 * ctx.rough
    alpha = ctx.prev_target * ctx.d_rough * 1.3
    return _finish(
        "rough_vol_stress",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.0,
        gamma=0.0,
        vega=0.10,
        vanna=0.0,
        volga=0.05,
        theta=0.0,
        alpha=alpha,
        fill=1 if abs(target) > 1e-8 else 0,
        fee_rate=fee_rate,
        note="rough_bergomi_vs_svi",
    )


_SLEEVES = {
    "mm_spread": _mm,
    "skew_residual": _skew,
    "calendar_term": _calendar,
    "fly_butterfly": _fly,
    "vrp_varswap": _vrp,
    "flow_toxicity": _flow,
    "gex_forced": _gex,
    "parity_box": _parity,
    "wing_kurtosis": _wing,
    "sticky_regime": _sticky,
    "dispersion_index": _dispersion,
    "roll_yield": _roll,
    "charm_bleed": _charm,
    "vanna_tilt": _vanna,
    "queue_sniper": _queue,
    "cot_fade_sleeve": _cot,
    "warehouse_autocall": _warehouse,
    "box_rate": _box_rate,
    "cross_impact": _cross,
    "rough_vol_stress": _rough,
}


def flow_size_mult(tox: float, gates_on: bool) -> float:
    if not gates_on:
        return 1.0
    _tox, _spread, size = flow_prior(tox, 0.0, 0.0, 0.0, 0.0)
    return size
