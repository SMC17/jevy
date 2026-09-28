"""Dupire local variance from total implied variance, and from call prices.

Gatheral form (total variance w(k, T), k = log(K/F)):

    σ_loc² = ∂_T w / g(k)

    g(k) = (1 − k w_k / (2w))² − (w_k² / 4) (1/w + 1/4) + w_kk / 2

A flat smile w = σ² T has g = 1 and ∂_T w = σ², so σ_loc = σ.

Call-price form (r = q = 0):

    σ_loc²(K, T) = ∂_T C / (½ K² ∂_{KK} C)

Research only. Cite Dupire, "Pricing with a Smile", Risk, 1994 (no free
canonical URL verified here) and the total-variance algebra in Gatheral,
The Volatility Surface.
"""

from __future__ import annotations

import math

from jev_omm.pricing.black_scholes import price


def local_variance_from_total(k: float, w: float, w_k: float, w_kk: float, w_T: float) -> float:
    if w <= 1e-12:
        return math.nan
    g = (1.0 - k * w_k / (2.0 * w)) ** 2 - (w_k * w_k) / 4.0 * (1.0 / w + 0.25) + 0.5 * w_kk
    if g <= 1e-12:
        return math.nan
    return w_T / g


def dupire_local_variance(
    spot: float,
    strike: float,
    t: float,
    iv: float,
    *,
    rate: float = 0.0,
    div_yield: float = 0.0,
    d_k: float | None = None,
    d_t: float | None = None,
) -> float:
    """Finite-difference Dupire on a flat (or strike-independent) Black vol.

    For a flat smile the local variance should land on iv². Rates are carried
    into the call prices; the inversion below is the r = q = 0 formula, so
    tests should pass rate = div_yield = 0.
    """
    if not (spot > 0.0 and strike > 0.0 and t > 0.0 and iv > 0.0):
        return math.nan
    dk = d_k if d_k is not None else max(0.5, 0.01 * strike)
    dt = d_t if d_t is not None else min(1.0 / 252.0, 0.05 * t)

    def call(s: float, k: float, expiry: float) -> float:
        return price(s, k, expiry, rate, div_yield, iv, True)

    c = call(spot, strike, t)
    c_t = call(spot, strike, t + dt)
    c_up = call(spot, strike + dk, t)
    c_dn = call(spot, strike - dk, t)
    d_c_d_t = (c_t - c) / dt
    d2 = (c_up - 2.0 * c + c_dn) / (dk * dk)
    denom = 0.5 * strike * strike * d2
    if denom <= 1e-14:
        return math.nan
    return d_c_d_t / denom
