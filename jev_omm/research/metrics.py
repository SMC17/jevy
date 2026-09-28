"""Path metrics for a paper simulation. Headline PnL is one column.

Markout horizons are the simulator's 1 / 5 / 30 step marks (one step is
``dt_seconds``, default 60s, so these are 1 / 5 / 30 minute marks — not
exchange 1 / 5 / 30 second marks). Say so when you quote them.
"""

from __future__ import annotations

from typing import Any

from jev_omm.backtest.simulator import SimResult


def _markout(result: SimResult, index: int) -> float:
    marks = result.attribution.markout
    if index < len(marks):
        return float(marks[index])
    return 0.0


def max_drawdown(path: list[float]) -> float:
    if not path:
        return 0.0
    peak = path[0]
    worst = 0.0
    for x in path:
        if x > peak:
            peak = x
        worst = max(worst, peak - x)
    return float(worst)


def inventory_variance(path: list[int] | list[float]) -> float:
    if not path:
        return 0.0
    mean = sum(path) / len(path)
    return float(sum((q - mean) ** 2 for q in path) / len(path))


def summarize_run(result: SimResult, *, n_steps: int, max_abs_delta: float) -> dict[str, Any]:
    n = max(int(n_steps), 1)
    util = 0.0
    if result.abs_delta_path and max_abs_delta > 0.0:
        util = (sum(result.abs_delta_path) / len(result.abs_delta_path)) / max_abs_delta
    return {
        "pnl": float(result.final_pnl),
        "fill_rate": len(result.fills) / n,
        "realized_spread": float(result.attribution.spread_capture),
        "markout_1": _markout(result, 0),
        "markout_5": _markout(result, 1),
        "markout_30": _markout(result, 2),
        "hedge_cost": float(-result.position.hedge_slippage),
        "inventory_variance": inventory_variance(result.inventory_path),
        "max_drawdown": max_drawdown(result.pnl_path),
        "quote_uptime": result.quoted_steps / n,
        "greek_utilization": float(util),
        "n_fills": len(result.fills),
        "n_hedges": len(result.hedges),
        "fill_model": result.fill_model,
    }
