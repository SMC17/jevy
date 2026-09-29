"""Eight paper sleeves. Each one is a target and a PnL stream, not an order.

The off switch is the identity: ``enabled=False`` returns a zero target and a
zero PnL. Flow and the instability gate are also the identity when
``gates_on`` is false (spread and size multipliers stay 1).

Greek buckets go through ``greek_pnl_step``. Research units use S = 1, so
ΔS in that call is the simple return and gamma PnL is 0.5 * Γ * r^2, which
is the same expression as 0.5 * Γ * S^2 * (ΔS/S)^2 at S = 1.

Sleeves
-------
mm_spread
    A–S half-spread on a short-gamma inventory. Delta leak is proportional
    to beta. Avellaneda–Stoikov, Quantitative Finance 2008,
    https://doi.org/10.1080/14697680701381228
skew_residual
    Fade the wing residual versus the SVI fit. Small delta, keep vanna.
calendar_term
    Fade the term-structure state. Delta-light, keeps theta and term vega.
fly_butterfly
    Fade curvature versus the fit. Keeps volga. Shares the smile shock with
    the skew sleeve on purpose.
vrp_varswap
    Short a unit of variance. Payoff uses the variance-swap sign
    (implied minus realized). Carr and Wu, Review of Financial Studies 2009,
    https://doi.org/10.1093/rfs/hhn039
    The squared-return piece is the gamma factor.
flow_toxicity
    ``flow_prior`` widens and cuts size. Zeros are the identity. The sleeve
    earns spread when it quotes and pays adverse selection when it quotes
    into a toxic print.
gex_forced
    Lean with the previous spot move when dealer gamma is short, and let
    ``state_gate`` cut size. The gate is the identity when disabled.
parity_box
    Fade a box-versus-rate dislocation. ``box_theo`` is the package value.
    Equity beta on this sleeve is a small leak, not the thesis.
"""

from __future__ import annotations

from dataclasses import dataclass

from jev_omm.config import QuoterConfig
from jev_omm.flow.signals import flow_prior
from jev_omm.hedge.delta import greek_pnl_step
from jev_omm.pricing.parity import box_theo
from jev_omm.quoter.avellaneda_stoikov import optimal_half_spread
from jev_omm.state_os.gate import state_gate

SLEEVE_IDS: tuple[str, ...] = (
    "mm_spread",
    "skew_residual",
    "calendar_term",
    "fly_butterfly",
    "vrp_varswap",
    "flow_toxicity",
    "gex_forced",
    "parity_box",
)

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
}

_AS = QuoterConfig()
# Box strikes in research units. The theo is a function of rate, not a quote.
_BOX_K1 = 100.0
_BOX_K2 = 110.0
_BOX_T = 0.25


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

    ``delta`` / ``gamma`` / ``vega`` / ``theta`` are the greeks inside this
    step's PnL. ``alpha`` is everything that is not one of those buckets
    (spread, residual edge, variance premium).
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
    fee = fee_rate * abs(target)
    raw = alpha + step.delta_pnl + step.gamma_pnl + step.vega_pnl + step.theta_pnl - fee
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
    # Half-spread is in price units from the A–S formula. Scale it down so a
    # one-lot paper fill is a research PnL, not a cash equity quote.
    spread_capture = half * 0.02 * filled
    beta = ctx.beta_to_index
    return _finish(
        "mm_spread",
        ctx,
        target=float(filled),
        spread_mult=spread_mult,
        size_mult=size_mult,
        delta=0.35 * beta,
        gamma=-80.0,
        vega=0.40,
        vanna=0.10,
        volga=-0.05,
        theta=-0.02,
        alpha=spread_capture,
        fill=filled,
        fee_rate=fee_rate,
        note="as_half_spread",
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
        note="fade_wing_residual",
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
    # Short one variance unit. The premium is K * dt. The convexity
    # -r^2 is the gamma bucket (Γ = -2 ⇒ 0.5 Γ r^2 = -r^2), not a second copy
    # inside alpha. Carr–Wu sign: short variance earns K − realized.
    k_var = ctx.iv * ctx.iv
    target = -1.0
    alpha = k_var * ctx.dt
    return _finish(
        "vrp_varswap",
        ctx,
        target=target,
        spread_mult=1.0,
        size_mult=1.0,
        delta=0.15 * ctx.beta_to_index,
        gamma=-2.0,
        vega=0.50,
        vanna=0.0,
        volga=0.20,
        theta=0.0,
        alpha=alpha,
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
        note="flow_prior" if ctx.gates_on else "flow_identity",
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
    # f_signed is the previous index return. Using the current return would look ahead.
    direction = 1.0 if ctx.f_signed >= 0.0 else -1.0
    scale = gate.size_mult
    target = direction * short_gamma * min(abs(ctx.gex_norm), 2.0) * scale
    alpha = ctx.prev_target * ctx.index_return * 0.8
    return _finish(
        "gex_forced",
        ctx,
        target=target,
        spread_mult=gate.spread_mult,
        size_mult=scale,
        delta=0.50 * ctx.beta_to_index * short_gamma * scale,
        gamma=-10.0 * short_gamma * scale,
        vega=0.20 * scale,
        vanna=0.40 * short_gamma * scale,
        volga=0.0,
        theta=0.0,
        alpha=alpha,
        fill=1 if abs(target) > 1e-8 else 0,
        fee_rate=fee_rate,
        note="state_gate" if ctx.gates_on else "gex_identity",
    )


def _parity(ctx: SleeveContext, *, fee_rate: float) -> SleeveQuote:
    theo = box_theo(_BOX_K1, _BOX_K2, _BOX_T, ctx.rate)
    # box_edge is a fraction of theo. Fade it.
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


_SLEEVES = {
    "mm_spread": _mm,
    "skew_residual": _skew,
    "calendar_term": _calendar,
    "fly_butterfly": _fly,
    "vrp_varswap": _vrp,
    "flow_toxicity": _flow,
    "gex_forced": _gex,
    "parity_box": _parity,
}


def flow_size_mult(tox: float, gates_on: bool) -> float:
    if not gates_on:
        return 1.0
    _tox, _spread, size = flow_prior(tox, 0.0, 0.0, 0.0, 0.0)
    return size
