"""European Black–Scholes–Merton pricing and analytic Greeks.

WHY: Analytic BS is the research baseline for European equity options.
We use continuous dividend yield q so the forward is S*exp((r-q)T).
All Greeks are returned in SI-ish units:
  - delta: ∂V/∂S
  - gamma: ∂²V/∂S²
  - vega:  ∂V/∂σ  (σ decimal; multiply by 0.01 for "per vol point")
  - theta: ∂V/∂t  with t in years (negative for long options usually)
  - vanna: ∂²V/∂S∂σ = −e^{−qT} n(d1) d2 / σ   (same for calls and puts)
  - volga: ∂²V/∂σ² = vega · d1 · d2 / σ       (same for calls and puts)
"""

from __future__ import annotations

import math

from scipy.stats import norm

from jev_omm.models.types import Greeks


_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _d1_d2(
    spot: float,
    strike: float,
    t: float,
    rate: float,
    div_yield: float,
    iv: float,
) -> tuple[float, float]:
    if t <= 0.0 or iv <= 0.0 or spot <= 0.0 or strike <= 0.0:
        # Degenerate: treat as intrinsic; callers handle t<=0 separately
        return 0.0, 0.0
    vol_sqrt_t = iv * math.sqrt(t)
    forward_log = math.log(spot / strike) + (rate - div_yield + 0.5 * iv * iv) * t
    d1 = forward_log / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def price(
    spot: float,
    strike: float,
    t: float,
    rate: float,
    div_yield: float,
    iv: float,
    is_call: bool,
) -> float:
    """European BS price. For t<=0 returns discounted intrinsic (undiscounted if r=0)."""
    if t <= 0.0:
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        return intrinsic
    if iv <= 0.0:
        # Zero vol → discounted forward intrinsic
        forward = spot * math.exp((rate - div_yield) * t)
        df = math.exp(-rate * t)
        if is_call:
            return df * max(forward - strike, 0.0)
        return df * max(strike - forward, 0.0)

    d1, d2 = _d1_d2(spot, strike, t, rate, div_yield, iv)
    df = math.exp(-rate * t)
    dq = math.exp(-div_yield * t)
    if is_call:
        return spot * dq * norm.cdf(d1) - strike * df * norm.cdf(d2)
    return strike * df * norm.cdf(-d2) - spot * dq * norm.cdf(-d1)


def greeks(
    spot: float,
    strike: float,
    t: float,
    rate: float,
    div_yield: float,
    iv: float,
    is_call: bool,
) -> Greeks:
    """Analytic BS Greeks. Near expiry / zero vol, use limiting values."""
    if t <= 0.0 or spot <= 0.0:
        # At expiry: delta is 0 or 1/-1; gamma/vega/theta → 0
        if is_call:
            delta = 1.0 if spot > strike else (0.5 if spot == strike else 0.0)
        else:
            delta = -1.0 if spot < strike else (-0.5 if spot == strike else 0.0)
        return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0)

    if iv <= 0.0:
        forward = spot * math.exp((rate - div_yield) * t)
        if is_call:
            delta = math.exp(-div_yield * t) if forward > strike else 0.0
        else:
            delta = -math.exp(-div_yield * t) if forward < strike else 0.0
        return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0)

    d1, d2 = _d1_d2(spot, strike, t, rate, div_yield, iv)
    dq = math.exp(-div_yield * t)
    df = math.exp(-rate * t)
    pdf_d1 = math.exp(-0.5 * d1 * d1) / _SQRT_2PI

    gamma = dq * pdf_d1 / (spot * iv * math.sqrt(t))
    vega = spot * dq * pdf_d1 * math.sqrt(t)
    vanna = -dq * pdf_d1 * d2 / iv
    volga = vega * d1 * d2 / iv

    if is_call:
        delta = dq * norm.cdf(d1)
        theta = (
            -spot * dq * pdf_d1 * iv / (2.0 * math.sqrt(t))
            - rate * strike * df * norm.cdf(d2)
            + div_yield * spot * dq * norm.cdf(d1)
        )
    else:
        delta = -dq * norm.cdf(-d1)
        theta = (
            -spot * dq * pdf_d1 * iv / (2.0 * math.sqrt(t))
            + rate * strike * df * norm.cdf(-d2)
            - div_yield * spot * dq * norm.cdf(-d1)
        )

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, vanna=vanna, volga=volga)


def price_and_greeks(
    spot: float,
    strike: float,
    t: float,
    rate: float,
    div_yield: float,
    iv: float,
    is_call: bool,
) -> tuple[float, Greeks]:
    return (
        price(spot, strike, t, rate, div_yield, iv, is_call),
        greeks(spot, strike, t, rate, div_yield, iv, is_call),
    )
