"""Mark-to-model PnL helpers."""

from __future__ import annotations

from jev_omm.models.types import Position


def marked_pnl(position: Position, option_mid: float, spot: float = 0.0) -> float:
    """Cash + option inventory at the option mid + underlier at spot.

    Hedge tickets pay ``underlier_qty * fill_price`` out of ``cash``, so the
    underlier must be marked or the hedge cost is counted without the asset.
    ``spot`` defaults to 0 only for option-only books (underlier flat).
    """
    return position.cash + position.qty * option_mid + position.underlier_qty * spot
