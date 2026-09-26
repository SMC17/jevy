"""Queue-aware fills: depth, cancel latency, toxic markout."""

from __future__ import annotations

import numpy as np

from jev_omm.execution.lob import LobConfig, expected_fills, fill_markout, simulate, time_to_first_fill
from jev_omm.models.types import Side


def test_deeper_queue_fills_slower():
    assert time_to_first_fill(80.0, 20.0, 0.0) > time_to_first_fill(5.0, 20.0, 0.0) * 5
    shallow = expected_fills(5.0, 10.0, 20.0, 0.0, 1.0, None)
    deep = expected_fills(80.0, 10.0, 20.0, 0.0, 1.0, None)
    assert shallow > deep == 0.0

    rng = np.random.default_rng(11)
    sim_s = simulate(rng, LobConfig(ahead=0.0, our_size=5.0, trade_intensity=30.0, horizon=1.0, dt=0.01, toxic_flow=0.0, adverse_jump=0.0))
    sim_d = simulate(rng, LobConfig(ahead=500.0, our_size=5.0, trade_intensity=30.0, horizon=1.0, dt=0.01, toxic_flow=0.0, adverse_jump=0.0))
    assert sim_s.filled > sim_d.filled
    assert sim_d.filled == 0.0
    assert sim_s.time_to_first < sim_d.time_to_first


def test_late_cancel_reduces_adverse_fills():
    stayed = expected_fills(0.0, 1000.0, 50.0, 0.0, 1.0, None)
    late = expected_fills(0.0, 1000.0, 50.0, 0.0, 1.0, 0.25)
    fast = expected_fills(0.0, 1000.0, 50.0, 0.0, 1.0, 0.02)
    assert late < stayed * 0.5
    assert fast < late

    rng = np.random.default_rng(19)
    base = dict(ahead=0.0, our_size=500.0, trade_intensity=40.0, horizon=1.0, dt=0.01, toxic_from=0.0, toxic_flow=1.0, adverse_jump=0.02)
    stay = simulate(rng, LobConfig(**base, cancel_latency=None))
    late_sim = simulate(rng, LobConfig(**base, cancel_latency=0.2))
    assert late_sim.adverse_filled < stay.adverse_filled
    kinds = [e.kind for e in late_sim.events]
    assert kinds[0] == "add"
    assert "cancel" in kinds


def test_toxic_flow_worsens_markout():
    calm = fill_markout(Side.BID, 1.0, 1.05, 2.0, 0.04, 0.0)
    toxic = fill_markout(Side.BID, 1.0, 1.05, 2.0, 0.04, 3.0)
    assert toxic < calm
    assert fill_markout(Side.ASK, 1.10, 1.05, 2.0, 0.04, 3.0) < fill_markout(Side.ASK, 1.10, 1.05, 2.0, 0.04, 0.0)

    rng = np.random.default_rng(3)
    common = dict(ahead=0.0, our_size=4.0, trade_intensity=80.0, horizon=0.5, dt=0.01, adverse_jump=0.05, price=1.0, mid=1.04, side=Side.BID)
    s0 = simulate(rng, LobConfig(**common, toxic_flow=0.0))
    s1 = simulate(rng, LobConfig(**common, toxic_flow=4.0))
    assert s0.filled > 0 and s1.filled > 0
    assert s1.markout / s1.filled < s0.markout / s0.filled


def test_partial_fill_and_cancels_ahead():
    exp = expected_fills(0.0, 10.0, 4.0, 0.0, 1.0, None)
    assert 0.0 < exp < 10.0
    assert abs(exp - 4.0) < 1e-12
    assert time_to_first_fill(20.0, 10.0, 30.0) < time_to_first_fill(20.0, 10.0, 0.0)
    fills = expected_fills(20.0, 100.0, 10.0, 30.0, 1.0, None)
    assert abs(fills - 5.0) < 1e-9

    rng = np.random.default_rng(5)
    sim = simulate(rng, LobConfig(ahead=0.0, our_size=80.0, trade_intensity=25.0, horizon=1.0, dt=0.02, toxic_flow=0.0, adverse_jump=0.0))
    assert 0.0 < sim.filled < 80.0
    assert any(e.kind == "execute" for e in sim.events)
