"""Multi-expiry book greeks and term-structure risk.

Conventions (σ decimal, t in years; vega = ∂V/∂σ):

* parallel vega — sum of vega across expiries
* bucket vega — vega of one expiry
* term-structure vega (slope) — Σ vega_i · (T_i − T_front), units vega·years.
  First-order PnL of a tilt dσ(T) = φ (T − T_front) is ``term_vega_slope * φ``.
* vanna — ∂²V/∂S∂σ, cross PnL ≈ vanna · dS · dσ
* volga — ∂²V/∂σ², second-order PnL ≈ ½ volga · (dσ)²

Mirrors ``zig/src/term_book.zig``. Quotes for a strip use SSVI; SABR-lite
remains the single-slice surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from jev_omm.config import QuoterConfig
from jev_omm.pricing.black_scholes import greeks, price
from jev_omm.quoter.avellaneda_stoikov import make_quote
from jev_omm.surface.svi import SsviParams, iv_from_total_var, ssvi_total_var


@dataclass
class Leg:
    expiry: float
    strike: float
    qty: float
    iv: float = 0.2
    is_call: bool = True


@dataclass
class Bucket:
    expiry: float
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    vanna: float = 0.0
    volga: float = 0.0
    n_legs: int = 0


@dataclass
class TermRisk:
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    vanna: float = 0.0
    volga: float = 0.0
    buckets: list[Bucket] = field(default_factory=list)
    t_front: float = 0.0
    term_vega_slope: float = 0.0


@dataclass
class TermLimits:
    max_abs_parallel_vega: float = 1e9
    max_abs_bucket_vega: float = 1e9
    max_abs_vanna: float = 1e9
    max_abs_volga: float = 1e9
    max_abs_term_slope: float = 1e9


def aggregate(spot: float, rate: float, div_yield: float, legs: list[Leg]) -> TermRisk:
    risk = TermRisk()
    for leg in legs:
        if leg.qty == 0.0 or leg.expiry <= 0.0 or leg.strike <= 0.0:
            continue
        g = greeks(spot, leg.strike, leg.expiry, rate, div_yield, leg.iv, leg.is_call)
        risk.delta += leg.qty * g.delta
        risk.gamma += leg.qty * g.gamma
        risk.vega += leg.qty * g.vega
        risk.theta += leg.qty * g.theta
        risk.vanna += leg.qty * g.vanna
        risk.volga += leg.qty * g.volga
        slot = next((b for b in risk.buckets if abs(b.expiry - leg.expiry) <= 1e-10), None)
        if slot is None:
            slot = Bucket(expiry=leg.expiry)
            risk.buckets.append(slot)
        slot.delta += leg.qty * g.delta
        slot.gamma += leg.qty * g.gamma
        slot.vega += leg.qty * g.vega
        slot.theta += leg.qty * g.theta
        slot.vanna += leg.qty * g.vanna
        slot.volga += leg.qty * g.volga
        slot.n_legs += 1
    risk.buckets.sort(key=lambda b: b.expiry)
    if risk.buckets:
        risk.t_front = risk.buckets[0].expiry
        risk.term_vega_slope = sum(b.vega * (b.expiry - risk.t_front) for b in risk.buckets)
    return risk


def evaluate_limits(risk: TermRisk, cfg: TermLimits) -> str:
    if abs(risk.vega) > cfg.max_abs_parallel_vega:
        return "parallel_vega"
    if abs(risk.vanna) > cfg.max_abs_vanna:
        return "vanna"
    if abs(risk.volga) > cfg.max_abs_volga:
        return "volga"
    if abs(risk.term_vega_slope) > cfg.max_abs_term_slope:
        return "term_slope"
    for b in risk.buckets:
        if abs(b.vega) > cfg.max_abs_bucket_vega:
            return "bucket_vega"
    return "none"


def scenario_pnl(
    risk: TermRisk,
    spot: float,
    d_spot_frac: float,
    d_iv_parallel: float,
    slope_phi: float,
) -> float:
    """IV at T is ``d_iv_parallel + slope_phi * (T − T_front)``."""
    pnl = 0.0
    d_s = spot * d_spot_frac
    for b in risk.buckets:
        d_iv = d_iv_parallel + slope_phi * (b.expiry - risk.t_front)
        pnl += (
            b.delta * d_s
            + 0.5 * b.gamma * d_s * d_s
            + b.vega * d_iv
            + b.vanna * d_s * d_iv
            + 0.5 * b.volga * d_iv * d_iv
        )
    return pnl


def scenario_reprice(
    spot: float,
    rate: float,
    div_yield: float,
    legs: list[Leg],
    d_spot_frac: float,
    d_iv_parallel: float,
    slope_phi: float,
    t_front: float,
) -> float:
    pnl = 0.0
    s2 = spot * (1.0 + d_spot_frac)
    for leg in legs:
        if leg.qty == 0.0:
            continue
        d_iv = d_iv_parallel + slope_phi * (leg.expiry - t_front)
        iv2 = max(leg.iv + d_iv, 1e-6)
        p0 = price(spot, leg.strike, leg.expiry, rate, div_yield, leg.iv, leg.is_call)
        p1 = price(s2, leg.strike, leg.expiry, rate, div_yield, iv2, leg.is_call)
        pnl += leg.qty * (p1 - p0)
    return pnl


def quote_expiries(
    spot: float,
    rate: float,
    div_yield: float,
    smile: SsviParams,
    atm_iv: float,
    expiries: list[float],
    strikes: list[float],
    quoter: QuoterConfig,
    is_call: bool = True,
) -> list[dict]:
    """One SSVI strip per expiry. θ(T) = atm_iv² · T."""
    out: list[dict] = []
    for t in expiries:
        theta = atm_iv * atm_iv * max(t, 0.0)
        forward = spot * math.exp((rate - div_yield) * t)
        row_quotes = []
        for strike in strikes:
            k = math.log(strike / forward) if forward > 0.0 and strike > 0.0 else 0.0
            iv = iv_from_total_var(ssvi_total_var(k, theta, smile), t)
            if iv <= 1e-6:
                iv = max(atm_iv, 1e-4)
            px = price(spot, strike, t, rate, div_yield, iv, is_call)
            g = greeks(spot, strike, t, rate, div_yield, iv, is_call)
            q = make_quote(px, 0, quoter, t_remaining=quoter.T_horizon, greeks=g)
            row_quotes.append({"strike": strike, "iv": iv, "mid": px, "quote": q, "greeks": g})
        out.append({"expiry": t, "theta": theta, "quotes": row_quotes})
    return out
