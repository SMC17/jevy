"""TypeSafe decision client: fallback, mocked HTTP live path, skip real live without key."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from jev_omm.decisions.client import (
    DeterministicFallbackClient,
    ResilientDecisionClient,
    TypeSafeDecisionClient,
    make_decision_client,
    _parse_answers,
)
from jev_omm.decisions.policy import apply_policy, CONFIDENCE_FLOOR
from jev_omm.decisions.schemas import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResult,
    build_mm_questions,
    build_mm_state,
)


def _calm_state() -> dict[str, Any]:
    return build_mm_state(
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


def test_make_decision_client_defaults_to_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    client = make_decision_client()
    assert isinstance(client, DeterministicFallbackClient)
    result = client.system_one(_calm_state(), build_mm_questions())
    assert result.source == "fallback"


def test_typesafe_client_requires_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        TypeSafeDecisionClient(api_key="")


def test_http_live_path_mocked(monkeypatch: pytest.MonkeyPatch):
    """SDK missing → HTTP POST /v1/systemone; tag source=live."""
    canned = {
        "model": "jev-1.13.0",
        "answers": {
            "regime": {
                "type": "choice",
                "choice": "calm",
                "probabilities": {
                    "calm": 0.7,
                    "trending": 0.1,
                    "volatile": 0.1,
                    "stressed": 0.1,
                },
                "confidence": 0.7,
            },
            "toxicity": {
                "type": "score",
                "score": 0.5,
                "legend": {"0": "a", "1": "b", "2": "c", "3": "d"},
                "probabilities": {"0": 0.6, "1": 0.3, "2": 0.1, "3": 0.0},
                "confidence": 0.6,
            },
            "informed_flow": {"type": "noul", "noul": 0.2},
            "widen_quotes": {"type": "noul", "noul": 0.2},
            "pull_quotes": {"type": "noul", "noul": 0.05},
            "hedge_now": {"type": "noul", "noul": 0.1},
            "size_tier": {
                "type": "choice",
                "choice": "normal",
                "probabilities": {"tiny": 0.1, "normal": 0.8, "large": 0.1},
                "confidence": 0.75,
            },
            "surface_suspect": {"type": "noul", "noul": 0.05},
        },
    }

    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return canned

    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> FakeResp:
            assert "/v1/systemone" in url
            assert headers["Authorization"].startswith("Bearer ")
            assert json["model"] == "jev-latest"
            return FakeResp()

    fake_httpx = MagicMock()
    fake_httpx.Client = FakeClient

    client = TypeSafeDecisionClient(api_key="test-key-not-real")
    with patch.dict("sys.modules", {"httpx": fake_httpx}):
        result = client._http_system_one(
            _calm_state(), build_mm_questions(), model="jev-latest"
        )

    assert result.source == "live"
    assert result.model == "jev-1.13.0"
    assert isinstance(result.answers["regime"], ChoiceAnswer)
    assert result.answers["regime"].choice == "calm"
    adj = apply_policy(result)
    assert adj.spread_mult >= 1.0
    assert adj.size_mult > 0


def test_resilient_falls_back_on_primary_error():
    class Boom(TypeSafeDecisionClient):
        def __init__(self) -> None:
            # bypass key check
            self.api_key = "x"
            self.timeout_s = 1.0

        def system_one(self, *a: Any, **k: Any) -> SystemOneResult:  # type: ignore[override]
            raise RuntimeError("network down")

    client = ResilientDecisionClient(Boom())  # type: ignore[arg-type]
    result = client.system_one(_calm_state())
    assert result.source == "fallback"


def test_parse_answers_from_dict():
    raw = {
        "n": {"type": "noul", "noul": 0.4},
        "c": {
            "type": "choice",
            "choice": "calm",
            "probabilities": {"calm": 1.0},
            "confidence": 0.9,
        },
        "s": {
            "type": "score",
            "score": 1.0,
            "legend": {"0": "lo", "1": "hi"},
            "probabilities": {"0": 0.0, "1": 1.0},
            "confidence": 0.8,
        },
    }
    parsed = _parse_answers(raw)
    assert parsed["n"].noul == 0.4
    assert parsed["c"].choice == "calm"
    assert parsed["s"].score == 1.0


@pytest.mark.skipif(
    not os.environ.get("TYPESAFE_API_KEY", "").strip(),
    reason="TYPESAFE_API_KEY not set — export TYPESAFE_API_KEY=... for live smoke (do not invent keys)",
)
def test_live_typesafe_optional():
    """Real live call — only when key present. Never invent keys in CI."""
    client = TypeSafeDecisionClient()
    result = client.system_one(_calm_state(), build_mm_questions(), model="jev-latest")
    assert result.source == "live"
    assert "regime" in result.answers
    # policy still composes
    adj = apply_policy(result)
    assert adj.result is not None


def test_low_confidence_gate_still_covered():
    """Policy confidence gates remain covered after client hardening."""
    low = SystemOneResult(
        model="test",
        source="fallback",
        answers={
            "regime": ChoiceAnswer(
                choice="calm",
                probabilities={"calm": 0.4, "trending": 0.2, "volatile": 0.2, "stressed": 0.2},
                confidence=0.3,
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
