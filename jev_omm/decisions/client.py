"""Decision clients: live TypeSafe System One or deterministic offline fallback.

Live path (when ``TYPESAFE_API_KEY`` is set):
  1. Prefer ``typesafe-sdk`` ``TypeSafeClient.system_one(state, questions, model=jev-latest)``
  2. Else ``httpx`` / urllib ``POST https://api.typesafe.ai/v1/systemone`` with Bearer key

Offline path (default for demo/tests):
  ``DeterministicFallbackClient`` — heuristics over structured MM state so
  ``python -m jev_omm.demo`` never needs network or keys.

Do **not** invent API keys. Export your own::

    export TYPESAFE_API_KEY=...

Refs: https://docs.typesafe.ai/ (Choice / Score / Noul; system_one; jev-latest)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from jev_omm.decisions.schemas import (
    REGIME_CRITERIA,
    SIZE_TIER_CRITERIA,
    TOXICITY_LEVELS,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResult,
    build_mm_questions,
)


TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"


class DecisionClient(ABC):
    @abstractmethod
    def system_one(
        self,
        state: dict[str, Any],
        questions: Optional[dict[str, Any]] = None,
        *,
        model: str = DEFAULT_MODEL,
    ) -> SystemOneResult:
        ...


def _coerce_one_answer(ans: Any) -> Any:
    """Normalize one API/SDK answer into a pydantic Answer model."""
    if isinstance(ans, (NoulAnswer, ChoiceAnswer, ScoreAnswer)):
        return ans
    if not isinstance(ans, dict):
        t = getattr(ans, "type", None)
        if t == "noul" or hasattr(ans, "noul"):
            return NoulAnswer(noul=float(ans.noul))
        if t == "choice" or hasattr(ans, "choice"):
            return ChoiceAnswer(
                choice=str(ans.choice),
                probabilities={str(k): float(v) for k, v in dict(ans.probabilities).items()},
                confidence=float(ans.confidence),
            )
        if t == "score" or hasattr(ans, "score"):
            legend_raw = dict(getattr(ans, "legend", {}))
            probs_raw = dict(ans.probabilities)
            return ScoreAnswer(
                score=float(ans.score),
                legend={str(k): str(v) for k, v in legend_raw.items()},
                probabilities={str(k): float(v) for k, v in probs_raw.items()},
                confidence=float(ans.confidence),
            )
        raise TypeError(f"unrecognized answer object: {type(ans)!r}")

    t = ans.get("type")
    if t == "noul" or "noul" in ans and "choice" not in ans and "score" not in ans:
        return NoulAnswer(noul=float(ans["noul"]))
    if t == "choice" or "choice" in ans:
        return ChoiceAnswer(
            choice=str(ans["choice"]),
            probabilities={str(k): float(v) for k, v in ans["probabilities"].items()},
            confidence=float(ans["confidence"]),
        )
    if t == "score" or "score" in ans:
        return ScoreAnswer(
            score=float(ans["score"]),
            legend={str(k): str(v) for k, v in ans.get("legend", {}).items()},
            probabilities={str(k): float(v) for k, v in ans["probabilities"].items()},
            confidence=float(ans["confidence"]),
        )
    raise TypeError(f"unrecognized answer dict keys: {sorted(ans)}")


def _parse_answers(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize API/SDK answer dicts into pydantic Answer models."""
    return {key: _coerce_one_answer(ans) for key, ans in raw.items()}


def _answers_from_sdk_response(resp: Any) -> dict[str, Any]:
    """Prefer flat ``.answers``; else merge ``.nouls`` / ``.choices`` / ``.scores``."""
    if hasattr(resp, "answers") and resp.answers:
        return dict(resp.answers)
    merged: dict[str, Any] = {}
    for attr in ("nouls", "choices", "scores"):
        part = getattr(resp, attr, None)
        if part:
            merged.update(dict(part))
    return merged


class TypeSafeDecisionClient(DecisionClient):
    """Calls TypeSafe when TYPESAFE_API_KEY is set; raises if key missing.

    Tries official SDK first, then HTTP ``POST /v1/systemone``.
    Successful responses are tagged ``source="live"``.
    """

    def __init__(self, api_key: Optional[str] = None, timeout_s: float = 5.0) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY", "")
        self.api_key = (self.api_key or "").strip()
        self.timeout_s = timeout_s
        if not self.api_key:
            raise RuntimeError(
                "TYPESAFE_API_KEY not set — use DeterministicFallbackClient or "
                "export TYPESAFE_API_KEY=... (do not invent keys)"
            )

    def system_one(
        self,
        state: dict[str, Any],
        questions: Optional[dict[str, Any]] = None,
        *,
        model: str = DEFAULT_MODEL,
    ) -> SystemOneResult:
        q = questions if questions is not None else build_mm_questions()

        # 1) Prefer official SDK if installed
        try:
            from typesafe_sdk import TypeSafeClient  # type: ignore
        except ImportError:
            TypeSafeClient = None  # type: ignore

        if TypeSafeClient is not None:
            with TypeSafeClient(api_key=self.api_key) as client:
                resp = client.system_one(state=state, questions=q, model=model)
            raw_answers = _answers_from_sdk_response(resp)
            model_id = getattr(resp, "model", model)
            return SystemOneResult(
                model=str(model_id),
                answers=_parse_answers(raw_answers),
                source="live",
            )

        # 2) httpx (or stdlib) POST /v1/systemone
        return self._http_system_one(state, q, model=model)

    def _http_system_one(
        self,
        state: dict[str, Any],
        questions: dict[str, Any],
        *,
        model: str,
    ) -> SystemOneResult:
        payload = {"state": state, "model": model, "questions": questions}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            import httpx

            with httpx.Client(timeout=self.timeout_s) as http:
                r = http.post(TYPESAFE_URL, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
        except ImportError:
            import json
            import urllib.request

            req = urllib.request.Request(
                TYPESAFE_URL,
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                data = json.loads(resp.read().decode())

        return SystemOneResult(
            model=str(data.get("model", model)),
            answers=_parse_answers(data.get("answers", {})),
            source="live",
        )


class DeterministicFallbackClient(DecisionClient):
    """Heuristic System One stand-in so demo/tests run fully offline.

    WHY: Classical A–S still needs regime/toxicity gates in research code;
    this mirrors the same question keys/answer shapes so policy.py is identical
    whether answers come from Jev or from these rules.
    """

    def system_one(
        self,
        state: dict[str, Any],
        questions: Optional[dict[str, Any]] = None,
        *,
        model: str = "fallback-heuristic",
    ) -> SystemOneResult:
        _ = questions  # same keys always produced
        book = state.get("book", {})
        market = state.get("market", {})
        inv = int(book.get("inventory", 0))
        cash_pnl = float(book.get("cash_pnl", 0.0))
        ret_bps = abs(float(market.get("spot_return_bps", 0.0)))
        half = float(market.get("half_spread", 0.1))

        # Regime choice
        if abs(inv) > 15 or cash_pnl < -100 or ret_bps > 40:
            regime, regime_conf = "stressed", 0.75
        elif ret_bps > 15:
            regime, regime_conf = "volatile", 0.70
        elif ret_bps > 5:
            regime, regime_conf = "trending", 0.65
        else:
            regime, regime_conf = "calm", 0.80
        regime_probs = {k: 0.05 for k in REGIME_CRITERIA}
        regime_probs[regime] = max(0.55, regime_conf)
        s = sum(regime_probs.values())
        regime_probs = {k: v / s for k, v in regime_probs.items()}

        flow = state.get("flow", {})
        tox_feat = float(flow.get("toxicity_composite", 0.0))
        vpin = float(flow.get("vpin", 0.0))
        tox_idx = 0.0
        if abs(inv) > 10:
            tox_idx += 1.0
        if ret_bps > 20:
            tox_idx += 1.0
        if cash_pnl < -50:
            tox_idx += 0.5
        # Research-grade tape features (VPIN-style / imbalance) when present
        if tox_feat > 0.4 or vpin > 0.5:
            tox_idx += 1.0
        if tox_feat > 0.7:
            tox_idx += 0.5
        tox_idx = min(3.0, tox_idx)
        tox_probs = {}
        for i, _ in enumerate(TOXICITY_LEVELS):
            dist = abs(i - tox_idx)
            tox_probs[str(i)] = max(0.05, 1.0 - 0.4 * dist)
        ts = sum(tox_probs.values())
        tox_probs = {k: v / ts for k, v in tox_probs.items()}
        tox_score = sum(int(k) * p for k, p in tox_probs.items())
        tox_conf = max(tox_probs.values())

        informed = min(
            0.95,
            0.15 + 0.04 * abs(inv) + 0.01 * ret_bps + 0.35 * tox_feat + 0.25 * vpin,
        )
        widen = min(
            0.95,
            0.2 + 0.03 * abs(inv) + 0.15 * (1 if regime in ("volatile", "stressed") else 0),
        )
        pull = (
            0.85
            if (abs(inv) > 20 or cash_pnl < -200)
            else (0.55 if regime == "stressed" and tox_score > 2 else 0.08)
        )
        hedge = min(0.95, 0.1 + 0.05 * abs(inv) + (0.3 if regime == "stressed" else 0.0))
        surface_suspect = 0.25 if half > 1.0 else 0.08

        if tox_score >= 2.0 or regime == "stressed":
            size_tier, size_conf = "tiny", 0.72
        elif regime == "calm" and abs(inv) < 5:
            size_tier, size_conf = "large", 0.68
        else:
            size_tier, size_conf = "normal", 0.75
        size_probs = {k: 0.08 for k in SIZE_TIER_CRITERIA}
        size_probs[size_tier] = max(0.55, size_conf)
        ss = sum(size_probs.values())
        size_probs = {k: v / ss for k, v in size_probs.items()}

        legend = {str(i): lvl for i, lvl in enumerate(TOXICITY_LEVELS)}
        answers = {
            "regime": ChoiceAnswer(choice=regime, probabilities=regime_probs, confidence=regime_conf),
            "toxicity": ScoreAnswer(
                score=tox_score, legend=legend, probabilities=tox_probs, confidence=float(tox_conf)
            ),
            "informed_flow": NoulAnswer(noul=informed),
            "widen_quotes": NoulAnswer(noul=widen),
            "pull_quotes": NoulAnswer(noul=pull),
            "hedge_now": NoulAnswer(noul=hedge),
            "size_tier": ChoiceAnswer(choice=size_tier, probabilities=size_probs, confidence=size_conf),
            "surface_suspect": NoulAnswer(noul=surface_suspect),
        }
        return SystemOneResult(model=model, answers=answers, source="fallback")


class ResilientDecisionClient(DecisionClient):
    """Try primary (live); on any failure, fall back to DeterministicFallbackClient."""

    def __init__(
        self,
        primary: DecisionClient,
        fallback: Optional[DecisionClient] = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or DeterministicFallbackClient()

    def system_one(
        self,
        state: dict[str, Any],
        questions: Optional[dict[str, Any]] = None,
        *,
        model: str = DEFAULT_MODEL,
    ) -> SystemOneResult:
        try:
            return self.primary.system_one(state, questions, model=model)
        except Exception:
            return self.fallback.system_one(state, questions, model="fallback-heuristic")


def make_decision_client() -> DecisionClient:
    """Factory: live TypeSafe (SDK→HTTP) if TYPESAFE_API_KEY set, else offline fallback.

    Live failures degrade to DeterministicFallbackClient (fail safe, not aggressive).
    """
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if key:
        try:
            return ResilientDecisionClient(TypeSafeDecisionClient(api_key=key))
        except Exception:
            return DeterministicFallbackClient()
    return DeterministicFallbackClient()
