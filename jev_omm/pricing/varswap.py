"""Variance- and vol-swap replication weights. Mirrors ``zig/src/varswap.zig``.

K_var = (2/T) df ∫ OTM(K)/K² dK.
Demeterfi, Derman, Kamal, Zou,
https://emanuelderman.com/wp-content/uploads/1999/02/gs-volatility_swaps.pdf.

Vol strike ≈ √K_var (1 − Var(K_var) / (8 K_var²)), the sqrt convexity in
Bossu, Strasser, Guichard,
http://docs.sbossu.com/bossu-strasser-guichard-varswap.pdf.

Flat-smile vega of the variance strike is ∂(σ²)/∂σ = 2σ.
"""

from __future__ import annotations

import math

from jev_omm.pricing.black_scholes import price


def fair_variance(strikes: list[float], otm: list[float], t: float, df: float = 1.0) -> float:
    if len(strikes) < 2 or len(strikes) != len(otm) or not (t > 0.0):
        return 0.0
    acc = 0.0
    for i in range(len(strikes) - 1):
        k0, k1 = strikes[i], strikes[i + 1]
        if not (k0 > 0.0 and k1 > k0):
            continue
        y0 = otm[i] / (k0 * k0)
        y1 = otm[i + 1] / (k1 * k1)
        acc += 0.5 * (y0 + y1) * (k1 - k0)
    disc = df if df > 0.0 else 1.0
    return (2.0 / t) * disc * acc


def variance_swap_vega(sigma: float) -> float:
    return 2.0 * sigma


def vol_swap_from_variance(k_var: float, var_of_var: float = 0.0) -> float:
    if not (k_var > 0.0):
        return 0.0
    conv = max(var_of_var, 0.0) / (8.0 * k_var * k_var)
    return math.sqrt(k_var) * max(0.0, 1.0 - conv)


def otm_strip_black(
    spot: float,
    strikes: list[float],
    t: float,
    iv: float,
    rate: float = 0.0,
    div_yield: float = 0.0,
) -> tuple[float, list[float]]:
    """Forward and OTM prices (put below F, call above) under flat Black vol."""
    df_r = math.exp(-rate * t)
    f = spot * math.exp((rate - div_yield) * t)
    otm = []
    for k in strikes:
        if k <= f:
            otm.append(price(spot, k, t, rate, div_yield, iv, False))
        else:
            otm.append(price(spot, k, t, rate, div_yield, iv, True))
    return f, otm


def fair_variance_flat_black(
    spot: float,
    t: float,
    iv: float,
    *,
    n: int = 81,
    k_lo: float = 0.4,
    k_hi: float = 2.2,
    rate: float = 0.0,
) -> float:
    """Discrete replication of a flat smile. Approaches iv² as the strip widens."""
    strikes = [spot * (k_lo + (k_hi - k_lo) * i / (n - 1)) for i in range(n)]
    _f, otm = otm_strip_black(spot, strikes, t, iv, rate=rate)
    df = math.exp(-rate * t)
    return fair_variance(strikes, otm, t, df)
