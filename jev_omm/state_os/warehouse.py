"""Warehouse forced-flow terms. Computable modules and fixtures.

The client is long income, protection, or callability. The warehouse is short
the embedded option, and the hedge target depends on the capital regime R.

    economic  — market greeks
    statutory — the CTE / CSV number the caller supplies
    rating    — the larger of the two, scaled by a buffer

Principles implemented here:

- P1. The hedge target is the payoff filtered by R, not a single delta.
- P2. Path-dependent books have a continuous greek and a discrete clock.
- P3. Internal netting happens before the tape. Visible GEX is the residual.
- P4. Amplification is how much notional sits on a cusp, not a personality.

What this file does not do: read a prospectus, an OPRA chain, a TBA stack,
or a live BoE LDI report. Those are deferred. The formulas are toys with
the right geometry, and the tests check the geometry.

Autocall shape: Guillaume, Journal of Derivatives 2015,
https://doi.org/10.3905/jod.2015.22.3.073
Capital wedge: Koijen & Yogo, AER 2015, https://doi.org/10.1257/aer.20121036
MBS refinance cusp: Richard & Roll, JF 1989,
https://doi.org/10.1111/j.1540-6261.1989.tb05062.x
LDI cash spiral, as a historical episode rather than a live feed:
Bank of England Financial Stability Report, December 2022,
https://www.bankofengland.co.uk/financial-stability-report/2022/december-2022
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jev_omm.pricing.black_scholes import greeks


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass
class AutocallFlow:
    digital: float
    knock_in: float
    worst_of: float
    duration: float

    @property
    def total(self) -> float:
        return self.digital + self.knock_in + self.worst_of + self.duration


def digital_density(spot: float, barrier: float, sigma: float, t: float) -> float:
    """Cash-or-nothing density in the underlying. Peaks when spot is on the barrier."""
    if spot <= 0.0 or barrier <= 0.0 or sigma <= 0.0 or t <= 0.0:
        return 0.0
    width = sigma * math.sqrt(t)
    d = math.log(spot / barrier) / width
    return _pdf(d) / (spot * width)


def knock_in_mass(spot: float, barrier: float, sigma: float) -> float:
    """Gaussian mass of (P − B) / (σ P). High on the barrier, low when far or calm."""
    if spot <= 0.0 or sigma <= 0.0:
        return 0.0
    z = (spot - barrier) / (sigma * spot)
    return math.exp(-0.5 * z * z)


def worst_of_flow(spots: list[float], corr: float, borrow_fee: float) -> float:
    """Weight on the worst name. Low correlation and a high borrow fee raise it."""
    if not spots:
        return 0.0
    worst = min(spots)
    mean = sum(spots) / len(spots)
    gap = 0.0 if mean <= 0.0 else (mean - worst) / mean
    corr_clip = min(max(corr, -1.0), 1.0)
    return (1.0 - corr_clip) * (1.0 + gap) * (1.0 + max(borrow_fee, 0.0))


def missed_ko_duration(ko_prob: float, maturity: float) -> float:
    """Years of duration that remain because the note did not knock out."""
    p = min(max(ko_prob, 0.0), 1.0)
    return (1.0 - p) * max(maturity, 0.0)


def autocall_flow(
    spot: float,
    coupon_barrier: float,
    knock_in_barrier: float,
    sigma: float,
    t: float,
    spots_worst: list[float],
    corr: float,
    borrow_fee: float,
    ko_prob: float,
    maturity: float,
) -> AutocallFlow:
    """F_AC = F_digital + F_KI + F_wo + F_duration. Components keep their own units.

    The sum is a research score, not a single dollar hedge. Scale it outside.
    """
    return AutocallFlow(
        digital=digital_density(spot, coupon_barrier, sigma, t),
        knock_in=knock_in_mass(spot, knock_in_barrier, sigma),
        worst_of=worst_of_flow(spots_worst, corr, borrow_fee),
        duration=missed_ko_duration(ko_prob, maturity),
    )


def rila_issuer_delta(
    spot: float,
    cap: float,
    buffer: float,
    t: float,
    rate: float,
    div_yield: float,
    iv: float,
) -> float:
    """Issuer delta of a segment: long the cap call, short the ATM-versus-buffer put spread.

    Investor payoff versus the underlying is long the put spread (the buffer)
    and short the cap call. The issuer has the other side. A zero buffer and
    a very high cap leave almost no option delta.
    """
    if t <= 0.0 or iv <= 0.0 or spot <= 0.0:
        return 0.0
    k_cap = spot * (1.0 + cap)
    k_buf = spot * (1.0 - buffer)
    call = greeks(spot, k_cap, t, rate, div_yield, iv, True).delta
    put_atm = greeks(spot, spot, t, rate, div_yield, iv, False).delta
    put_buf = greeks(spot, k_buf, t, rate, div_yield, iv, False).delta
    return call - put_atm + put_buf


def hedge_by_regime(
    economic_hedge: float,
    statutory_hedge: float,
    regime: str,
    rating_buffer: float = 1.15,
) -> float:
    """P1. ``economic``, ``statutory``, or ``rating``."""
    if regime == "economic":
        return economic_hedge
    if regime == "statutory":
        return statutory_hedge
    if regime == "rating":
        larger = economic_hedge if abs(economic_hedge) >= abs(statutory_hedge) else statutory_hedge
        sign = 1.0 if larger >= 0.0 else -1.0
        return sign * rating_buffer * abs(larger)
    raise ValueError(f"unknown capital regime: {regime}")


def va_dynamic(delta: float, rho: float, rate_shock: float) -> float:
    """Continuous VA hedge: delta plus the rate sensitivity times a shock."""
    return delta + rho * rate_shock


def va_tail(vega: float, gamma: float, d_sigma: float, d_spot: float) -> float:
    """One-step tail greek: ν Δσ + ½ Γ (ΔS)²."""
    return vega * d_sigma + 0.5 * gamma * d_spot * d_spot


def cusp_mass_one(coupon: float, yield_level: float, width: float = 0.0075) -> float:
    """Refinance mass. Peaks when the coupon is on the current yield."""
    if width <= 0.0:
        return 0.0
    z = (coupon - yield_level) / width
    return math.exp(-0.5 * z * z)


def mbs_duration_slope(
    coupon: float,
    yield_level: float,
    deep_discount: float = 0.015,
    width: float = 0.0075,
) -> float:
    """∂D_eff/∂y. Deep-discount pools are already extended, so the slope is off."""
    if coupon <= yield_level - deep_discount:
        return 0.0
    return -25.0 * cusp_mass_one(coupon, yield_level, width)


def mbs_hedge(coupon: float, yield_level: float, dy: float, notional: float) -> float:
    """(∂D_eff/∂y) · Δy · notional. A research slope, not an OAS engine."""
    return mbs_duration_slope(coupon, yield_level) * dy * notional


def ldi_cash_need(
    dv01: float,
    dy: float,
    repo_topup: float,
    velocity: float,
    velocity_floor: float = 0.002,
) -> float:
    """Repo top-up plus variation margin, gated on yield velocity.

    The same yield *level* with a slow move does not force the cash call.
    The 2022 gilt episode was a velocity event. This is not a live BoE feed.
    """
    if abs(velocity) < velocity_floor:
        return 0.0
    return abs(dv01 * dy) + max(repo_topup, 0.0)


def ldi_gap(cash_need: float, unencumbered: float) -> float:
    return max(cash_need - unencumbered, 0.0)


def spend_flow(balance: float, withdrawal_rate: float) -> float:
    """Scheduled sale. Negative is a sale of assets."""
    return -max(withdrawal_rate, 0.0) * balance


def tape_residual(gross: float, internal_net: float) -> float:
    """P3. What is left after the warehouse nets with itself."""
    return gross - internal_net


def visible_gex(book_gex: float, gross: float, internal_net: float) -> float:
    """GEX the street can see is the book times the residual fraction."""
    if gross == 0.0:
        return 0.0
    return book_gex * tape_residual(gross, internal_net) / gross


def mass_on_cusp(distances: list[float], bandwidth: float) -> float:
    """P4. Fraction of notionals sitting inside the bandwidth of a barrier.

    There is no personality parameter. An empty book has mass 0.
    """
    if not distances or bandwidth < 0.0:
        return 0.0
    hit = sum(1.0 for distance in distances if abs(distance) <= bandwidth)
    return hit / len(distances)
