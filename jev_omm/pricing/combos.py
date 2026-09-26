"""Combo / package theos (Akuna theos & spreads/flies).

Verticals, butterflies, straddles/strangles from BS legs; package bid/ask from
executable sides; edge vs theo. Prefers Zig when ``libjev_omm.so`` present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jev_omm.models.types import Greeks
from jev_omm.pricing import _native
from jev_omm.pricing.black_scholes import greeks as bs_greeks
from jev_omm.pricing.black_scholes import price as bs_price
from jev_omm.pricing.parity import SideQuotes


@dataclass
class PackageTheo:
    theo: float
    greeks: Greeks = field(default_factory=lambda: Greeks(delta=0, gamma=0, vega=0, theta=0))
    package_bid: float = 0.0
    package_ask: float = 0.0
    buy_edge: float = 0.0
    sell_edge: float = 0.0
    kind: str = ""


def _add(a: Greeks, b: Greeks, w: float = 1.0) -> Greeks:
    return Greeks(
        delta=a.delta + w * b.delta,
        gamma=a.gamma + w * b.gamma,
        vega=a.vega + w * b.vega,
        theta=a.theta + w * b.theta,
    )


def straddle_theo(
    spot: float, strike: float, t: float, rate: float, div_yield: float, iv: float
) -> PackageTheo:
    if _native.ZIG_AVAILABLE and hasattr(_native, "straddle_theo"):
        try:
            d = _native.straddle_theo(spot, strike, t, rate, div_yield, iv)
            return PackageTheo(
                theo=d["theo"],
                greeks=Greeks(delta=d["delta"], gamma=d["gamma"], vega=d["vega"], theta=d["theta"]),
                package_bid=d["package_bid"],
                package_ask=d["package_ask"],
                buy_edge=d["buy_edge"],
                sell_edge=d["sell_edge"],
                kind="straddle",
            )
        except Exception:
            pass
    c = bs_price(spot, strike, t, rate, div_yield, iv, True)
    p = bs_price(spot, strike, t, rate, div_yield, iv, False)
    gc = bs_greeks(spot, strike, t, rate, div_yield, iv, True)
    gp = bs_greeks(spot, strike, t, rate, div_yield, iv, False)
    return PackageTheo(theo=c + p, greeks=_add(gc, gp), kind="straddle")


def vertical_call_theo(
    spot: float, k1: float, k2: float, t: float, rate: float, div_yield: float,
    iv1: float, iv2: float,
) -> PackageTheo:
    p1 = bs_price(spot, k1, t, rate, div_yield, iv1, True)
    p2 = bs_price(spot, k2, t, rate, div_yield, iv2, True)
    g1 = bs_greeks(spot, k1, t, rate, div_yield, iv1, True)
    g2 = bs_greeks(spot, k2, t, rate, div_yield, iv2, True)
    return PackageTheo(theo=p1 - p2, greeks=_add(g1, g2, -1.0), kind="vertical_call")


def butterfly_call_theo(
    spot: float, k1: float, k2: float, k3: float, t: float, rate: float,
    div_yield: float, iv1: float, iv2: float, iv3: float,
) -> PackageTheo:
    p1 = bs_price(spot, k1, t, rate, div_yield, iv1, True)
    p2 = bs_price(spot, k2, t, rate, div_yield, iv2, True)
    p3 = bs_price(spot, k3, t, rate, div_yield, iv3, True)
    g1 = bs_greeks(spot, k1, t, rate, div_yield, iv1, True)
    g2 = bs_greeks(spot, k2, t, rate, div_yield, iv2, True)
    g3 = bs_greeks(spot, k3, t, rate, div_yield, iv3, True)
    g = _add(_add(g1, g2, -2.0), g3)
    return PackageTheo(theo=p1 - 2.0 * p2 + p3, greeks=g, kind="butterfly_call")


def attach_package_sides(
    pkg: PackageTheo, legs: list[SideQuotes], weights: list[float]
) -> PackageTheo:
    buy = 0.0
    sell = 0.0
    for leg, w in zip(legs, weights):
        if w > 0:
            buy += w * leg.ask
            sell += w * leg.bid
        elif w < 0:
            buy += w * leg.bid
            sell += w * leg.ask
    pkg.package_ask = buy
    pkg.package_bid = sell
    pkg.buy_edge = pkg.theo - pkg.package_ask
    pkg.sell_edge = pkg.package_bid - pkg.theo
    return pkg
