"""Avellaneda–Stoikov style reservation price + optimal spread for one options series.

Approximation notes (READ BEFORE EXTENDING):
  1. Classic A–S (2008) models a *spot* dealer with exponential utility and
     Poisson market-order arrivals. We reuse the closed forms but feed the
     *option mid* as the reference price and option contracts as inventory.
  2. Option mid variance is NOT S²σ² — we take `sigma` as an absolute $/√yr
     vol of the option mid (config), not underlying vol. Calibrate offline.
  3. gamma_penalty / vega_penalty are *ad-hoc* inventory tilts for Greeks
     risk. They do not come from A–S. They shift the reservation price so
     long gamma/vega inventory shades quotes toward selling (and vice versa).
  4. No adverse-selection / informed-flow model. No multi-strike book.
  5. Terminal horizon T_horizon is a rolling "session" horizon, not expiry.

Formulas (A–S):
  reservation r = mid - q * gamma * sigma² * (T - t)
  optimal half-spread δ = gamma * sigma² * (T-t)/2 + (1/gamma) * ln(1 + gamma/kappa)
  bid = r - δ,  ask = r + δ

WHY inventory penalty: market makers earn spread but accumulate directional
risk; reservation skews quotes to attract neutralizing flow.

DecisionClient (Jev) modulates aggressiveness/size/widen around this math —
it does NOT replace the reservation-price formula.
"""

from __future__ import annotations

import math

from jev_omm.config import QuoterConfig
from jev_omm.models.types import Greeks, Quote


def _mode(cfg: QuoterConfig) -> str:
    return getattr(cfg, "mode", "as_finite_horizon")


def _is_gueant(cfg: QuoterConfig) -> bool:
    return _mode(cfg) == "gueant_asymptotic"


def _is_gueant_ode(cfg: QuoterConfig) -> bool:
    return _mode(cfg) == "gueant_ode"


def _is_option_vega(cfg: QuoterConfig) -> bool:
    return _mode(cfg) == "option_vega"


def reservation_price(
    mid: float,
    inventory: int,
    cfg: QuoterConfig,
    t_remaining: float | None = None,
    greeks: Greeks | None = None,
) -> float:
    """Indifference / reservation price given inventory and optional Greek tilts."""
    if _is_option_vega(cfg):
        from jev_omm.quoter.option_mm import make_quote as oq

        return oq(mid, inventory, cfg, greeks).reservation
    if _is_gueant_ode(cfg):
        from jev_omm.quoter.gueant_ode import reservation_price as ode_r

        return ode_r(mid, inventory, cfg, greeks)
    if _is_gueant(cfg):
        from jev_omm.quoter.gueant import reservation_price as g_r

        return g_r(mid, inventory, cfg, greeks)
    T = cfg.T_horizon if t_remaining is None else max(t_remaining, 0.0)
    # Core A–S inventory skew: long inventory → lower reservation → attract sells
    r = mid - inventory * cfg.gamma * (cfg.sigma ** 2) * T

    # Optional Greek penalties (approximation — see module docstring)
    if greeks is not None and inventory != 0:
        # Positive inventory * positive gamma ⇒ we want to sell → lower r
        r -= inventory * cfg.gamma_penalty * abs(greeks.gamma)
        r -= inventory * cfg.vega_penalty * abs(greeks.vega) * 0.01

    return r


def optimal_half_spread(cfg: QuoterConfig, t_remaining: float | None = None) -> float:
    """A–S / Guéant optimal half-spread; clamped to [min_half_spread, max_half_spread]."""
    if _is_option_vega(cfg):
        from jev_omm.quoter.option_mm import make_quote as oq

        return oq(1.0, 0, cfg, None).half_spread
    if _is_gueant_ode(cfg):
        from jev_omm.quoter.gueant_ode import optimal_half_spread as ode_h

        return ode_h(cfg, inventory=0)
    if _is_gueant(cfg):
        from jev_omm.quoter.gueant import optimal_half_spread as g_h

        return g_h(cfg)
    T = cfg.T_horizon if t_remaining is None else max(t_remaining, 0.0)
    risk_term = 0.5 * cfg.gamma * (cfg.sigma ** 2) * T
    if cfg.gamma <= 0.0 or cfg.kappa <= 0.0:
        intensity_term = cfg.min_half_spread
    else:
        intensity_term = (1.0 / cfg.gamma) * math.log(1.0 + cfg.gamma / cfg.kappa)
    half = risk_term + intensity_term
    return float(min(cfg.max_half_spread, max(cfg.min_half_spread, half)))


def make_quote(
    mid: float,
    inventory: int,
    cfg: QuoterConfig,
    t_remaining: float | None = None,
    greeks: Greeks | None = None,
    *,
    spread_mult: float = 1.0,
    size_mult: float = 1.0,
) -> Quote:
    """Build a two-sided quote around the reservation price.

    spread_mult / size_mult come from DecisionClient (aggressiveness / widen).
    Classical A–S math is unchanged; decisions only scale the posted quote.
    """
    if _is_option_vega(cfg):
        from jev_omm.quoter.option_mm import make_quote as oq

        return oq(mid, inventory, cfg, greeks, spread_mult=spread_mult, size_mult=size_mult)
    if _is_gueant_ode(cfg):
        from jev_omm.quoter.gueant_ode import make_quote as ode_q

        return ode_q(mid, inventory, cfg, greeks, spread_mult=spread_mult, size_mult=size_mult)
    if _is_gueant(cfg):
        from jev_omm.quoter.gueant import make_quote as g_q

        return g_q(mid, inventory, cfg, greeks, spread_mult=spread_mult, size_mult=size_mult)
    if mid <= 0.0:
        mid = max(mid, 0.01)

    r = reservation_price(mid, inventory, cfg, t_remaining, greeks)
    half = optimal_half_spread(cfg, t_remaining) * max(spread_mult, 0.25)
    half = float(min(cfg.max_half_spread, max(cfg.min_half_spread, half)))
    bid = max(0.01, r - half)
    ask = max(bid + 0.01, r + half)
    size = max(1, int(round(cfg.quote_size * max(size_mult, 0.0))))
    return Quote(
        bid=bid,
        ask=ask,
        bid_size=size,
        ask_size=size,
        reservation=r,
        half_spread=half,
    )
