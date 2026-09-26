"""Mark-to-model PnL helpers."""

from __future__ import annotations

from jev_omm.models.types import Position


def marked_pnl(position: Position, option_mid: float) -> float:
    """Cash + inventory marked at current option mid."""
    return position.cash + position.qty * option_mid
