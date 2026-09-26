"""Exogenous fill model: Poisson arrivals against posted quotes.

WHY: The baseline research sim posts against an exogenous Poisson flow.
Intensity rises when our quote is closer to mid and falls when we widen.
Bid and ask are independent. Queue position, cancel latency, and adverse
selection live in ``lob.py`` (still synthetic — no live market data).
"""

from __future__ import annotations

import math

import numpy as np

from jev_omm.models.types import Fill, Quote, Side


def _side_intensity(
    half_away: float,
    base: float,
    kappa: float,
) -> float:
    """λ = base * exp(-kappa * distance_from_mid_in_half_spreads-ish)."""
    return base * math.exp(-kappa * max(half_away, 0.0))


def sample_fills(
    rng: np.random.Generator,
    time: float,
    mid: float,
    quote: Quote,
    dt: float,
    base_intensity: float,
    kappa: float,
) -> list[Fill]:
    """Bernoulli/Poisson fills over dt for bid and ask separately."""
    fills: list[Fill] = []
    if quote.bid_size <= 0 and quote.ask_size <= 0:
        return fills

    # Distance of quote from mid (positive = away from mid)
    bid_away = max(0.0, mid - quote.bid)
    ask_away = max(0.0, quote.ask - mid)

    lam_bid = _side_intensity(bid_away, base_intensity, kappa) * dt
    lam_ask = _side_intensity(ask_away, base_intensity, kappa) * dt

    # Poisson count; cap at posted size for simplicity
    n_bid = int(rng.poisson(lam_bid)) if quote.bid_size > 0 else 0
    n_ask = int(rng.poisson(lam_ask)) if quote.ask_size > 0 else 0
    n_bid = min(n_bid, quote.bid_size)
    n_ask = min(n_ask, quote.ask_size)

    if n_bid > 0:
        fills.append(Fill(time=time, side=Side.BID, price=quote.bid, size=n_bid, mid_at_fill=mid))
    if n_ask > 0:
        fills.append(Fill(time=time, side=Side.ASK, price=quote.ask, size=n_ask, mid_at_fill=mid))
    return fills
