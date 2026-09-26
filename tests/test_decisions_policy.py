"""Decision fallback typing + confidence-gated policy."""

from __future__ import annotations

from jev_omm.decisions.client import DeterministicFallbackClient
from jev_omm.decisions.policy import (
    CONFIDENCE_FLOOR,
    apply_policy,
    composite_toxicity,
    decide_quote_adjustments,
)
from jev_omm.decisions.schemas import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResult,
    build_mm_questions,
    build_mm_state,
)


def test_fallback_answers_are_typed_system_one_shape():
    client = DeterministicFallbackClient()
    state = build_mm_state(
        time=0.0,
        spot=100.0,
        option_mid=4.0,
        iv=0.22,
        inventory=0,
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        cash_pnl=0.0,
        half_spread=0.2,
        quoting_allowed=True,
    )
    result = client.system_one(state, build_mm_questions())
    assert result.source == "fallback"
    expected = {
        "regime",
        "toxicity",
        "informed_flow",
        "widen_quotes",
        "pull_quotes",
        "hedge_now",
        "size_tier",
        "surface_suspect",
    }
    assert set(result.answers.keys()) == expected
    assert isinstance(result.answers["regime"], ChoiceAnswer)
    assert isinstance(result.answers["toxicity"], ScoreAnswer)
    assert isinstance(result.answers["informed_flow"], NoulAnswer)
    assert 0.0 <= result.answers["informed_flow"].noul <= 1.0
    assert 0.0 <= result.answers["regime"].confidence <= 1.0


def test_low_confidence_widens_and_cuts_size():
    """Confidence gate: low confidence → widen or flatten (never larger size)."""
    low = SystemOneResult(
        model="test",
        source="fallback",
        answers={
            "regime": ChoiceAnswer(
                choice="calm",
                probabilities={"calm": 0.4, "trending": 0.2, "volatile": 0.2, "stressed": 0.2},
                confidence=0.3,  # below CONFIDENCE_FLOOR
            ),
            "toxicity": ScoreAnswer(
                score=0.5,
                legend={"0": "a", "1": "b", "2": "c", "3": "d"},
                probabilities={"0": 0.4, "1": 0.3, "2": 0.2, "3": 0.1},
                confidence=0.4,
            ),
            "informed_flow": NoulAnswer(noul=0.1),
            "widen_quotes": NoulAnswer(noul=0.1),
            "pull_quotes": NoulAnswer(noul=0.05),
            "hedge_now": NoulAnswer(noul=0.1),
            "size_tier": ChoiceAnswer(
                choice="large",
                probabilities={"tiny": 0.2, "normal": 0.2, "large": 0.6},
                confidence=0.3,
            ),
            "surface_suspect": NoulAnswer(noul=0.05),
        },
    )
    assert low.answers["regime"].confidence < CONFIDENCE_FLOOR
    adj = apply_policy(low)
    assert adj.spread_mult >= 1.35
    assert adj.size_mult <= 0.5
    assert "low_confidence_gate" in adj.reason


def test_high_toxicity_can_pull():
    hot = SystemOneResult(
        model="test",
        source="fallback",
        answers={
            "regime": ChoiceAnswer(
                choice="stressed",
                probabilities={"calm": 0.05, "trending": 0.05, "volatile": 0.1, "stressed": 0.8},
                confidence=0.8,
            ),
            "toxicity": ScoreAnswer(
                score=2.8,
                legend={"0": "a", "1": "b", "2": "c", "3": "d"},
                probabilities={"0": 0.0, "1": 0.05, "2": 0.15, "3": 0.8},
                confidence=0.8,
            ),
            "informed_flow": NoulAnswer(noul=0.85),
            "widen_quotes": NoulAnswer(noul=0.9),
            "pull_quotes": NoulAnswer(noul=0.8),
            "hedge_now": NoulAnswer(noul=0.75),
            "size_tier": ChoiceAnswer(
                choice="tiny",
                probabilities={"tiny": 0.8, "normal": 0.15, "large": 0.05},
                confidence=0.8,
            ),
            "surface_suspect": NoulAnswer(noul=0.4),
        },
    )
    adj = apply_policy(hot)
    assert adj.pull is True
    assert adj.size_mult == 0.0
    assert adj.hedge_now is True
    assert composite_toxicity(hot) > 0.5


def test_decide_quote_adjustments_offline():
    adj = decide_quote_adjustments(
        DeterministicFallbackClient(),
        time=0.0,
        spot=100.0,
        option_mid=3.5,
        iv=0.2,
        inventory=12,
        delta=6.0,
        gamma=0.2,
        vega=40.0,
        cash_pnl=-20.0,
        half_spread=0.25,
        quoting_allowed=True,
        spot_return_bps=25.0,
    )
    assert adj.spread_mult >= 1.0
    assert adj.result is not None
    assert adj.result.source == "fallback"


def test_mm_questions_match_api_shape():
    q = build_mm_questions()
    assert q["regime"]["type"] == "choice"
    assert q["toxicity"]["type"] == "score"
    assert q["informed_flow"]["type"] == "noul"
    assert isinstance(q["regime"]["criteria"], dict)
    assert isinstance(q["toxicity"]["criteria"], list)
