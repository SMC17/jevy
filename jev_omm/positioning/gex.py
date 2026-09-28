"""Dealer gamma, flip level, pin candidate, charm and vanna hedge proxies.

Paper book. Open interest is an input (fixture or caller). No options vendor.

Dollar gamma for a 1% spot move, per strike:

    dealer_sign * OI * multiplier * Γ_BS * S^2 * 0.01

Γ_BS is per share. Call gamma equals put gamma. ``multiplier`` is 100 for
a standard equity contract.

Postures (modeling choices, not measured dealer inventory):

- ``short_premium``: dealers short calls and short puts (sign −1, −1).
  Net gamma stays negative. There is no zero-gamma flip. This is the
  default and matches a book that sold both wings. Index puts in particular
  are the end-user long side in Garleanu, Pedersen, Poteshman, NBER w11843
  (https://doi.org/10.3386/w11843).
- ``dashboard_flip``: dealers long calls (+1) and short puts (−1). This is
  the public-dashboard sign that can cross zero as relative strike gammas
  move. It is not the Garleanu end-user book. Use it only when you want a
  flip level. Barbon & Buraschi (https://doi.org/10.2139/ssrn.3725454)
  is the citation for the hedge-feedback sign, not for this inventory split.

Pinning: Ni, Pearson, Poteshman, JFE 2005
(https://doi.org/10.1016/j.jfineco.2004.08.005) and Avellaneda & Lipkin,
Quantitative Finance 2003 (https://doi.org/10.1088/1469-7688/3/6/301).
Max pain is the strike that minimizes total holder intrinsic. It is a
pin *candidate* only when normalized dealer gamma is positive. It is not
a directional forecast, and it is ignored in a short-gamma book.
"""

from __future__ import annotations

from dataclasses import dataclass

from jev_omm.pricing.black_scholes import charm_tau, greeks


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class OptionOI:
    strike: float
    call_oi: float
    put_oi: float
    iv: float
    t: float
    call_volume: float = 0.0
    put_volume: float = 0.0


def dealer_signs(posture: str) -> tuple[float, float]:
    """Return (call_sign, put_sign) for the dealer book."""
    if posture == "short_premium":
        return -1.0, -1.0
    if posture == "dashboard_flip":
        return 1.0, -1.0
    raise ValueError(f"unknown dealer posture: {posture}")


def dollar_gex_1pct(
    spot: float,
    legs: list[OptionOI],
    posture: str,
    rate: float,
    div_yield: float,
    multiplier: float = 100.0,
) -> float:
    """Aggregate dealer dollar-delta change for a 1% spot move."""
    call_sign, put_sign = dealer_signs(posture)
    total = 0.0
    for leg in legs:
        gamma = greeks(spot, leg.strike, leg.t, rate, div_yield, leg.iv, True).gamma
        scale = gamma * multiplier * spot * spot * 0.01
        total += call_sign * leg.call_oi * scale
        total += put_sign * leg.put_oi * scale
    return total


def gex_norm_from_dollar(gex: float, scale: float) -> float:
    """Clip dollar GEX into [-1, 1] by a caller-chosen scale. Scale <= 0 → 0."""
    if scale <= 0.0:
        return 0.0
    return _clamp(gex / scale, -1.0, 1.0)


def zero_gamma_level(
    legs: list[OptionOI],
    posture: str,
    spot_min: float,
    spot_max: float,
    n: int,
    rate: float,
    div_yield: float,
    multiplier: float = 100.0,
) -> float | None:
    """Linear interpolate the first sign change of dollar GEX on a spot grid.

    ``None`` when the book does not cross zero (the short-premium case).
    """
    if n < 2 or spot_max <= spot_min:
        return None
    spots = [spot_min + (spot_max - spot_min) * i / (n - 1) for i in range(n)]
    vals = [
        dollar_gex_1pct(s, legs, posture, rate, div_yield, multiplier) for s in spots
    ]
    for i in range(len(vals) - 1):
        if vals[i] == 0.0:
            return spots[i]
        if vals[i] * vals[i + 1] < 0.0:
            w = abs(vals[i]) / (abs(vals[i]) + abs(vals[i + 1]))
            return spots[i] + w * (spots[i + 1] - spots[i])
    return None


def max_pain(legs: list[OptionOI]) -> float | None:
    """Strike on the grid that minimizes total holder intrinsic. Not a forecast."""
    if not legs:
        return None
    strikes = sorted({leg.strike for leg in legs})
    best_k = strikes[0]
    best_pay = float("inf")
    for spot in strikes:
        pay = 0.0
        for leg in legs:
            pay += leg.call_oi * max(spot - leg.strike, 0.0)
            pay += leg.put_oi * max(leg.strike - spot, 0.0)
        if pay < best_pay:
            best_pay = pay
            best_k = spot
    return best_k


def oi_wall(legs: list[OptionOI]) -> float | None:
    """Strike with the most call + put open interest."""
    if not legs:
        return None
    best = max(legs, key=lambda leg: leg.call_oi + leg.put_oi)
    return best.strike


def put_call_oi_ratio(legs: list[OptionOI]) -> float:
    calls = sum(leg.call_oi for leg in legs)
    puts = sum(leg.put_oi for leg in legs)
    if calls <= 0.0:
        return 0.0
    return puts / calls


def put_call_volume_ratio(legs: list[OptionOI]) -> float:
    calls = sum(leg.call_volume for leg in legs)
    puts = sum(leg.put_volume for leg in legs)
    if calls <= 0.0:
        return 0.0
    return puts / calls


def pin_level(
    gex_norm: float,
    max_pain_strike: float | None,
    flip: float | None,
    wall: float | None,
) -> float | None:
    """Pin candidate only in positive dealer gamma. Short gamma returns None."""
    if gex_norm <= 0.0:
        return None
    if flip is not None and max_pain_strike is not None:
        return 0.5 * flip + 0.5 * max_pain_strike
    if flip is not None:
        return flip
    if max_pain_strike is not None:
        return max_pain_strike
    return wall


def charm_vanna_hedge(
    spot: float,
    legs: list[OptionOI],
    posture: str,
    rate: float,
    div_yield: float,
    dt: float,
    d_sigma: float,
    multiplier: float = 100.0,
) -> tuple[float, float]:
    """Share-equivalents the dealer hedge would trade for charm and vanna.

    Charm uses ∂Δ/∂τ from ``charm_tau``. Calendar time falling by ``dt``
    changes delta by −(∂Δ/∂τ) dt. The returned quantity is what a hedger
    trades to stay flat (the opposite of the book delta change).

    ``d_sigma`` is a decimal vol move. Zero vol shock → vanna hedge 0.
    Both are 0 when ``dt`` is 0 and ``d_sigma`` is 0.
    """
    call_sign, put_sign = dealer_signs(posture)
    charm_h = 0.0
    vanna_h = 0.0
    for leg in legs:
        for is_call, sign, oi in (
            (True, call_sign, leg.call_oi),
            (False, put_sign, leg.put_oi),
        ):
            if oi == 0.0:
                continue
            qty = sign * oi * multiplier
            ch = charm_tau(spot, leg.strike, leg.t, rate, div_yield, leg.iv, is_call)
            d_delta = qty * (-ch) * dt
            charm_h += -d_delta
            vanna = greeks(spot, leg.strike, leg.t, rate, div_yield, leg.iv, is_call).vanna
            vanna_h += -qty * vanna * d_sigma
    return charm_h, vanna_h
