"""Flow toxicity features."""

from __future__ import annotations

from jev_omm.decisions.client import DeterministicFallbackClient
from jev_omm.decisions.schemas import build_mm_questions, build_mm_state
from jev_omm.flow.toxicity import ToxicityTracker


def test_balanced_low_vpin():
    tr = ToxicityTracker(bucket_volume=10.0, window_buckets=5)
    for _ in range(20):
        tr.on_trade(5.0)
        tr.on_trade(-5.0)
    assert tr.vpin() < 0.15


def test_onesided_high_vpin():
    tr = ToxicityTracker(bucket_volume=10.0, window_buckets=5)
    for _ in range(50):
        tr.on_trade(10.0)
    assert tr.vpin() > 0.9
    assert tr.composite() > 0.5


def test_features_feed_fallback():
    tr = ToxicityTracker(bucket_volume=5.0, window_buckets=4)
    for _ in range(40):
        tr.on_trade(5.0)
    feats = tr.feature_dict()
    state = build_mm_state(
        time=0.0, spot=100, option_mid=4, iv=0.2, inventory=0,
        delta=0, gamma=0, vega=0, cash_pnl=0, half_spread=0.1,
        quoting_allowed=True, toxicity_features=feats,
    )
    assert "flow" in state
    assert state["flow"]["vpin"] > 0.5
    r = DeterministicFallbackClient().system_one(state, build_mm_questions())
    assert r.answers["toxicity"].score >= 1.0
    assert r.answers["informed_flow"].noul > 0.3
