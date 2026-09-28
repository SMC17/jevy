"""Flow-signed gamma versus the open-interest (structural) book.

Structural GEX is the existing OI aggregator: a posture times open interest
times Black–Scholes gamma. Flow-signed GEX uses the session's signed customer
volume instead. The two can disagree. The desk treats that disagreement as a
reason to ignore a pin, not as a second price forecast.

Vanna and charm for the index book are converted to E-mini contracts only
when the contract is SPX. SPY is an American ETF; its European BS charm is
not the SPX hedge, and this module refuses the ES conversion.

SPX multiplier 100, ES point value 50.
https://www.cboe.com/tradable_products/sp_500/spx_options/
https://www.cmegroup.com/markets/equities/us-index/e-mini-sandp500.html
"""

from __future__ import annotations

from dataclasses import dataclass

from jev_omm.positioning.gex import OptionOI, charm_vanna_hedge, dollar_gex_1pct
from jev_omm.pricing.black_scholes import greeks

ES_POINT_VALUE = 50.0
SPX_MULTIPLIER = 100.0


@dataclass
class FlowLeg:
    """Signed customer volume. Positive means the customer bought."""

    strike: float
    iv: float
    t: float
    customer_call_volume: float = 0.0
    customer_put_volume: float = 0.0


def signs_disagree(structural: float, flow: float) -> bool:
    """False when either print is zero. Zero is not a side."""
    if structural == 0.0 or flow == 0.0:
        return False
    return (structural > 0.0) != (flow > 0.0)


def flow_signed_gex(
    spot: float,
    legs: list[FlowLeg],
    rate: float,
    div_yield: float,
    multiplier: float = SPX_MULTIPLIER,
) -> float:
    """Dealer dollar-gamma from customer flow. Dealer sign is minus the customer.

    Same 1% scaling as ``dollar_gex_1pct``. Volume is contracts traded today,
    not open interest.
    """
    total = 0.0
    for leg in legs:
        gamma = greeks(spot, leg.strike, leg.t, rate, div_yield, leg.iv, True).gamma
        scale = gamma * multiplier * spot * spot * 0.01
        total += -leg.customer_call_volume * scale
        total += -leg.customer_put_volume * scale
    return total


def gex_pair(
    spot: float,
    oi_legs: list[OptionOI],
    flow_legs: list[FlowLeg],
    posture: str,
    rate: float,
    div_yield: float,
    multiplier: float = SPX_MULTIPLIER,
) -> dict[str, float | bool]:
    structural = dollar_gex_1pct(spot, oi_legs, posture, rate, div_yield, multiplier)
    flow = flow_signed_gex(spot, flow_legs, rate, div_yield, multiplier)
    return {
        "structural": structural,
        "flow": flow,
        "disagree": signs_disagree(structural, flow),
    }


def spx_vanna_charm_es(
    spot: float,
    legs: list[OptionOI],
    posture: str,
    rate: float,
    div_yield: float,
    dt: float,
    d_sigma: float,
    contract: str = "SPX",
) -> dict[str, float | None]:
    """Hedge in ES contracts for an SPX book. SPY returns null ES fields.

    ``charm_vanna_hedge`` with multiplier 100 is dollars per index point.
    One ES contract is worth 50 dollars per index point.
    """
    charm, vanna = charm_vanna_hedge(
        spot, legs, posture, rate, div_yield, dt, d_sigma, SPX_MULTIPLIER
    )
    if contract != "SPX":
        return {
            "charm_index_points": charm,
            "vanna_index_points": vanna,
            "charm_es": None,
            "vanna_es": None,
            "contract": contract,
        }
    return {
        "charm_index_points": charm,
        "vanna_index_points": vanna,
        "charm_es": charm / ES_POINT_VALUE,
        "vanna_es": vanna / ES_POINT_VALUE,
        "contract": "SPX",
    }
