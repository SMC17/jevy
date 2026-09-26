"""Put-call parity, synthetics, and box spreads (Akuna 101 §2.2 / §3.6).

Prefers Zig ``libjev_omm.so`` via ctypes when present; pure-Python otherwise.
Package edges always use executable sides (buy ask / sell bid), never mids.

Public refs:
  https://akunacapital.com/work-with-us/options-101/
  https://blog.moontower.ai/implying-the-cost-of-carry-in-options/
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from jev_omm.pricing import _native


def discount_factor(rate: float, t: float) -> float:
    return math.exp(-rate * t)


def forward(spot: float, rate: float, div_yield: float, t: float) -> float:
    return spot * math.exp((rate - div_yield) * t)


def parity_diff(
    spot: float, strike: float, t: float, rate: float, div_yield: float = 0.0
) -> float:
    """C − P theo = S e^{-qT} − K e^{-rT}."""
    if _native.ZIG_AVAILABLE and hasattr(_native, "parity_diff"):
        try:
            return _native.parity_diff(spot, strike, t, rate, div_yield)
        except Exception:
            pass
    return spot * math.exp(-div_yield * t) - strike * math.exp(-rate * t)


def parity_residual(
    call_mid: float,
    put_mid: float,
    spot: float,
    strike: float,
    t: float,
    rate: float,
    div_yield: float = 0.0,
) -> float:
    if _native.ZIG_AVAILABLE and hasattr(_native, "parity_residual"):
        try:
            return _native.parity_residual(
                call_mid, put_mid, spot, strike, t, rate, div_yield
            )
        except Exception:
            pass
    return (call_mid - put_mid) - parity_diff(spot, strike, t, rate, div_yield)


@dataclass(frozen=True)
class SideQuotes:
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


@dataclass
class SyntheticEdge:
    package_debit: float
    theo_debit: float
    edge: float
    conversion_edge: float
    reversal_edge: float


def synthetic_forward_edge(
    call: SideQuotes,
    put: SideQuotes,
    spot: float,
    strike: float,
    t: float,
    rate: float,
    div_yield: float = 0.0,
    underlier: Optional[SideQuotes] = None,
) -> SyntheticEdge:
    if _native.ZIG_AVAILABLE and hasattr(_native, "synthetic_edge"):
        try:
            und = underlier
            d = _native.synthetic_edge(
                call.bid, call.ask, put.bid, put.ask,
                spot, strike, t, rate, div_yield,
                und.bid if und else None,
                und.ask if und else None,
            )
            return SyntheticEdge(**d)
        except Exception:
            pass
    theo = parity_diff(spot, strike, t, rate, div_yield)
    buy_synth = call.ask - put.bid
    sell_synth = call.bid - put.ask
    out = SyntheticEdge(
        package_debit=buy_synth,
        theo_debit=theo,
        edge=theo - buy_synth,
        reversal_edge=theo - buy_synth,
        conversion_edge=sell_synth - theo,
    )
    if underlier is not None:
        df = discount_factor(rate, t)
        dq = math.exp(-div_yield * t)
        conv_cash = call.bid - put.ask - underlier.ask * dq
        out.conversion_edge = conv_cash + strike * df
        rev_cash = -call.ask + put.bid + underlier.bid * dq
        out.reversal_edge = rev_cash - strike * df
        out.edge = out.reversal_edge
        out.package_debit = call.ask - put.bid - underlier.bid * dq
        out.theo_debit = -strike * df
    return out


@dataclass
class BoxResult:
    theo_pv: float
    package_debit: float
    package_credit: float
    buy_edge: float
    sell_edge: float
    implied_rate_mid: float
    implied_rate_buy: float
    implied_rate_sell: float


def _implied_rate(pv: float, width: float, t: float) -> float:
    if t <= 0.0 or width <= 0.0 or pv <= 0.0:
        return 0.0
    df = pv / width
    if df <= 0.0:
        return 0.0
    return -math.log(df) / t


def box_theo(k1: float, k2: float, t: float, rate: float) -> float:
    if _native.ZIG_AVAILABLE and hasattr(_native, "box_theo"):
        try:
            return _native.box_theo(k1, k2, t, rate)
        except Exception:
            pass
    return (k2 - k1) * discount_factor(rate, t)


def box_spread(
    call_k1: SideQuotes,
    call_k2: SideQuotes,
    put_k1: SideQuotes,
    put_k2: SideQuotes,
    k1: float,
    k2: float,
    t: float,
    rate: float,
) -> BoxResult:
    if _native.ZIG_AVAILABLE and hasattr(_native, "box_spread"):
        try:
            d = _native.box_spread(
                call_k1.bid, call_k1.ask, call_k2.bid, call_k2.ask,
                put_k1.bid, put_k1.ask, put_k2.bid, put_k2.ask,
                k1, k2, t, rate,
            )
            return BoxResult(**d)
        except Exception:
            pass
    width = k2 - k1
    theo = width * discount_factor(rate, t)
    buy_debit = call_k1.ask - call_k2.bid - put_k1.bid + put_k2.ask
    sell_credit = call_k1.bid - call_k2.ask - put_k1.ask + put_k2.bid
    mid_pv = 0.5 * (buy_debit + sell_credit)
    return BoxResult(
        theo_pv=theo,
        package_debit=buy_debit,
        package_credit=sell_credit,
        buy_edge=theo - buy_debit,
        sell_edge=sell_credit - theo,
        implied_rate_mid=_implied_rate(mid_pv, width, t),
        implied_rate_buy=_implied_rate(buy_debit, width, t),
        implied_rate_sell=_implied_rate(sell_credit, width, t),
    )
