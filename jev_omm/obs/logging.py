"""Lightweight observability helpers (Rich-friendly dict snapshots)."""

from __future__ import annotations

from typing import Any


def summarize_run(
    *,
    steps: int,
    fills: int,
    final_inventory: int,
    final_pnl: float,
    breaches: int,
    hedges: int,
    decision_source: str,
) -> dict[str, Any]:
    return {
        "steps": steps,
        "fills": fills,
        "final_inventory": final_inventory,
        "final_pnl": round(final_pnl, 4),
        "risk_breaches": breaches,
        "hedge_signals": hedges,
        "decision_source": decision_source,
    }
