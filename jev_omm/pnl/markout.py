"""Markout attribution for paper fills.

On each fill, record mid_at_fill and resolve mids after fixed step horizons
(default 1 / 5 / 30 steps ≈ minutes at default sim dt).

Decomposition:
  spread_capture — edge vs mid at fill (bid: mid-px; ask: px-mid) * size
  markout_h      — signed_size * (mid_h - mid0)
  adverse_h      — -markout_h  (positive ⇒ mid moved against the maker)
  inventory_mtm  — running open_qty * Δmid between steps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from jev_omm.models.types import Fill, Side

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 30)


def spread_edge(side: Side, price: float, mid: float, size: int) -> float:
    sz = float(size)
    if side == Side.BID:
        return (mid - price) * sz
    return (price - mid) * sz


def signed_size(side: Side, size: int) -> int:
    return size if side == Side.BID else -size


@dataclass
class PendingFill:
    step: int
    side: Side
    price: float
    size: int
    mid0: float
    pending: list[bool]  # parallel to horizons


@dataclass
class AttributionSummary:
    n_fills: int = 0
    total_contracts: int = 0
    spread_capture: float = 0.0
    markout: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    adverse: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    resolved: list[int] = field(default_factory=lambda: [0, 0, 0])
    inventory_mtm: float = 0.0
    horizons: tuple[int, ...] = DEFAULT_HORIZONS

    def as_rows(self) -> list[tuple[str, str]]:
        rows = [
            ("fills", str(self.n_fills)),
            ("contracts", str(self.total_contracts)),
            ("spread_capture", f"{self.spread_capture:.4f}"),
        ]
        for i, h in enumerate(self.horizons):
            rows.append(
                (
                    f"markout_{h}step",
                    f"{self.markout[i]:.4f}  (adverse {self.adverse[i]:.4f}, n={self.resolved[i]})",
                )
            )
        rows.append(("inventory_mtm", f"{self.inventory_mtm:.4f}"))
        return rows


@dataclass
class MarkoutTracker:
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    pending: list[PendingFill] = field(default_factory=list)
    summary: AttributionSummary = field(default_factory=AttributionSummary)
    last_mid: float = 0.0
    open_qty: int = 0

    def __post_init__(self) -> None:
        self.summary.horizons = self.horizons
        n = len(self.horizons)
        self.summary.markout = [0.0] * n
        self.summary.adverse = [0.0] * n
        self.summary.resolved = [0] * n

    def on_fill(self, step: int, fill: Fill) -> None:
        edge = spread_edge(fill.side, fill.price, fill.mid_at_fill, fill.size)
        self.summary.spread_capture += edge
        self.summary.n_fills += 1
        self.summary.total_contracts += fill.size
        self.open_qty += signed_size(fill.side, fill.size)
        self.pending.append(
            PendingFill(
                step=step,
                side=fill.side,
                price=fill.price,
                size=fill.size,
                mid0=fill.mid_at_fill,
                pending=[True] * len(self.horizons),
            )
        )

    def on_step(self, step: int, mid: float) -> None:
        if self.last_mid != 0.0 and self.open_qty != 0:
            self.summary.inventory_mtm += self.open_qty * (mid - self.last_mid)
        self.last_mid = mid

        still: list[PendingFill] = []
        for pf in self.pending:
            for i, h in enumerate(self.horizons):
                if not pf.pending[i]:
                    continue
                if step >= pf.step + h:
                    signed = float(signed_size(pf.side, pf.size))
                    mo = signed * (mid - pf.mid0)
                    self.summary.markout[i] += mo
                    self.summary.adverse[i] += -mo
                    self.summary.resolved[i] += 1
                    pf.pending[i] = False
            if any(pf.pending):
                still.append(pf)
        self.pending = still


def attribute_fills_offline(
    fills: Sequence[Fill],
    mids_by_step: Sequence[float],
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> AttributionSummary:
    """Replay fills against a mid path (research helper)."""
    tr = MarkoutTracker(horizons=horizons)
    fill_i = 0
    # Assume fills.time maps onto step index externally; use enumerated mids
    # and match fills whose time order aligns with step order of appearance.
    # For offline use we assign fill steps by order of appearance among fills
    # if times are unique — callers should prefer live MarkoutTracker.
    step_for_fill: dict[int, int] = {}
    for fi, f in enumerate(fills):
        # Map fill to nearest step by scanning mids length — use fi if no better info
        step_for_fill[fi] = min(fi, len(mids_by_step) - 1) if mids_by_step else 0

    for step, mid in enumerate(mids_by_step):
        tr.on_step(step, mid)
        while fill_i < len(fills) and step_for_fill.get(fill_i, -1) == step:
            tr.on_fill(step, fills[fill_i])
            fill_i += 1
    return tr.summary
