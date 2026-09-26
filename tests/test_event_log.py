"""Sequenced JSONL event log + deterministic replay."""

from __future__ import annotations

from pathlib import Path

from jev_omm.backtest.simulator import run_simulation
from jev_omm.config import EngineConfig, SimConfig
from jev_omm.decisions.client import DeterministicFallbackClient
from jev_omm.models.types import Fill, Side
from jev_omm.obs.event_log import EventLog, read_jsonl, replay_jsonl, sha256_file
from jev_omm.decisions.schemas import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResult,
)


def test_event_log_append_and_hash(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    log = EventLog().open(path)
    log.append_underlying_tick(0.0, 0, 100.0)
    log.append_book_top(0.0, 0, bid=3.9, ask=4.1, bid_sz=2, ask_sz=2, mid=4.0)
    fill = Fill(time=0.0, side=Side.BID, price=3.9, size=1, mid_at_fill=4.0)
    log.append_fill(0.0, 0, fill)
    result = SystemOneResult(
        model="fallback-heuristic",
        source="fallback",
        answers={
            "regime": ChoiceAnswer(
                choice="calm",
                probabilities={"calm": 0.8, "trending": 0.1, "volatile": 0.05, "stressed": 0.05},
                confidence=0.8,
            ),
            "toxicity": ScoreAnswer(
                score=0.0,
                legend={"0": "a", "1": "b", "2": "c", "3": "d"},
                probabilities={"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0},
                confidence=0.9,
            ),
            "informed_flow": NoulAnswer(noul=0.1),
        },
    )
    log.append_decision_snapshot(0.0, 0, result)
    log.close()

    events = read_jsonl(path)
    assert len(events) == 4
    assert events[0]["type"] == "UnderlyingTick"
    assert events[3]["type"] == "DecisionSnapshot"
    assert events[3]["source"] == "fallback"
    assert "regime" in events[3]["confidence"]
    assert sha256_file(path) == log.sha256_hex()


def test_replay_markout_from_jsonl(tmp_path: Path):
    path = tmp_path / "mo.jsonl"
    log = EventLog().open(path)
    log.append_book_top(0.0, 0, bid=9.5, ask=10.5, bid_sz=2, ask_sz=2, mid=10.0)
    log.append_fill(
        0.0, 0, Fill(time=0.0, side=Side.BID, price=9.5, size=2, mid_at_fill=10.0)
    )
    log.append_book_top(0.001, 1, bid=8.5, ask=9.5, bid_sz=2, ask_sz=2, mid=9.0)
    log.close()

    r = replay_jsonl(path)
    assert r.n_fills == 1
    assert r.qty == 2
    assert abs(r.attribution.spread_capture - 1.0) < 1e-9
    assert r.attribution.resolved[0] == 1
    assert abs(r.attribution.markout[0] - (-2.0)) < 1e-9


def test_sim_writes_event_log_deterministic(tmp_path: Path):
    cfg = EngineConfig(sim=SimConfig(n_steps=40, seed=7, fill_intensity_base=5.0e4))
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    r1 = run_simulation(cfg, client=DeterministicFallbackClient(), event_log_path=str(p1))
    r2 = run_simulation(cfg, client=DeterministicFallbackClient(), event_log_path=str(p2))
    assert r1.event_log_sha256 == r2.event_log_sha256
    assert sha256_file(p1) == sha256_file(p2) == r1.event_log_sha256
    events = read_jsonl(p1)
    types = {e["type"] for e in events}
    assert "DecisionSnapshot" in types
    assert "BookTop" in types
    # all decision sources are fallback (no key)
    snaps = [e for e in events if e["type"] == "DecisionSnapshot"]
    assert snaps and all(s["source"] == "fallback" for s in snaps)
    # replay agrees on fill count
    rr = replay_jsonl(p1)
    assert rr.n_fills == len(r1.fills)
