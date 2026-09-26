"""Markout attribution unit tests."""

from __future__ import annotations

from jev_omm.models.types import Fill, Side
from jev_omm.pnl.markout import MarkoutTracker, spread_edge


def test_spread_edge_signs():
    assert abs(spread_edge(Side.BID, 9.0, 10.0, 2) - 2.0) < 1e-12
    assert abs(spread_edge(Side.ASK, 11.0, 10.0, 2) - 2.0) < 1e-12


def test_markout_resolves():
    tr = MarkoutTracker(horizons=(1, 5, 30))
    fill = Fill(time=0.0, side=Side.BID, price=9.5, size=2, mid_at_fill=10.0)
    tr.on_fill(0, fill)
    assert abs(tr.summary.spread_capture - 1.0) < 1e-12

    tr.on_step(0, 10.0)
    assert tr.summary.resolved[0] == 0

    tr.on_step(1, 9.0)  # mid drop → markout = 2*(9-10) = -2
    assert tr.summary.resolved[0] == 1
    assert abs(tr.summary.markout[0] - (-2.0)) < 1e-12
    assert abs(tr.summary.adverse[0] - 2.0) < 1e-12

    for s in range(2, 6):
        tr.on_step(s, 9.0)
    assert tr.summary.resolved[1] == 1

    for s in range(6, 31):
        tr.on_step(s, 9.0)
    assert tr.summary.resolved[2] == 1
    assert len(tr.pending) == 0
