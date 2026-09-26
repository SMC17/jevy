"""TypeSafe System One–shaped schemas for options MM decisions.

Mirrors docs.typesafe.ai primitives (Choice / Score / Noul) so offline code
and live TypeSafeClient.system_one() share one question map.

API shape (POST /v1/systemone):
  questions[id] = {type, instructions, criteria?}
  answers[id]   = {type, noul?} | {type, choice, probabilities, confidence}
                  | {type, score, legend, probabilities, confidence}
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Question primitives (request side) — dict-serializable for HTTP / SDK
# ---------------------------------------------------------------------------


class NoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: str | dict[str, Any]
    criteria: Optional[dict[str, str]] = None  # {"true": ..., "false": ...}


class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str | dict[str, Any]
    criteria: dict[str, Optional[str]]  # option -> description | null


class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    instructions: str | dict[str, Any]
    criteria: list[str]  # ordered levels, low → high


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


# ---------------------------------------------------------------------------
# Answer primitives (response side)
# ---------------------------------------------------------------------------


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


class SystemOneResult(BaseModel):
    """Subset of TypeSafe system_one response we consume in policy."""

    model: str = "fallback"
    answers: dict[str, Answer]
    source: Literal["live", "fallback"] = "fallback"


# ---------------------------------------------------------------------------
# Quote adjustments produced by policy (code, not the model)
# ---------------------------------------------------------------------------


class QuoteAdjustments(BaseModel):
    """Policy output that scales classical A–S quotes; never replaces reservation math."""

    spread_mult: float = 1.0
    size_mult: float = 1.0
    pull: bool = False
    hedge_now: bool = False
    composite_toxicity: float = 0.0
    reason: str = ""
    result: Optional[SystemOneResult] = None


# ---------------------------------------------------------------------------
# MM state + parallel question batch
# ---------------------------------------------------------------------------

REGIME_CRITERIA: dict[str, Optional[str]] = {
    "calm": "Stable mid, normal flow, no stress signals",
    "trending": "Persistent directional mid moves",
    "volatile": "Elevated realized / quoted vol, wide swings",
    "stressed": "Gap risk, one-sided flow, or limit proximity",
}

SIZE_TIER_CRITERIA: dict[str, Optional[str]] = {
    "tiny": "Minimal size — defensive or uncertain",
    "normal": "Baseline quote size",
    "large": "Confident / low-toxicity conditions",
}

TOXICITY_LEVELS: list[str] = [
    "benign flow",
    "mild adverse selection",
    "elevated toxicity",
    "severe informed flow",
]


def build_mm_state(
    *,
    time: float,
    spot: float,
    option_mid: float,
    iv: float,
    inventory: int,
    delta: float,
    gamma: float,
    vega: float,
    cash_pnl: float,
    half_spread: float,
    quoting_allowed: bool,
    recent_fills: int = 0,
    spot_return_bps: float = 0.0,
    toxicity_features: dict[str, float] | None = None,
    underlier_pos: float = 0.0,
    net_delta: float | None = None,
) -> dict[str, Any]:
    """Structured program state for System One (not chat text).

    Multi-strike demos may additionally pass portfolio_delta / n_strikes via
    DecisionSnapshot.state (event log); not required for single-series.

    ``toxicity_features`` (research-grade VPIN-style / imbalance) feed the
    toxicity Score and informed_flow Noul — see ``jev_omm.flow.toxicity``.
    """
    flow = toxicity_features or {}
    return {
        "session": {"time_years": time, "quoting_allowed": quoting_allowed},
        "market": {
            "spot": spot,
            "option_mid": option_mid,
            "iv": iv,
            "half_spread": half_spread,
            "spot_return_bps": spot_return_bps,
        },
        "book": {
            "inventory": inventory,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "cash_pnl": cash_pnl,
            "recent_fills": recent_fills,
            "underlier_pos": underlier_pos,
            "net_delta": float(net_delta if net_delta is not None else delta + underlier_pos),
        },
        "flow": {
            "vpin": float(flow.get("vpin", 0.0)),
            "trade_imbalance": float(flow.get("trade_imbalance", 0.0)),
            "abs_imbalance": float(flow.get("abs_imbalance", 0.0)),
            "ewma_toxicity": float(flow.get("ewma_toxicity", 0.0)),
            "toxicity_composite": float(flow.get("toxicity_composite", 0.0)),
            "hawkes_intensity": float(flow.get("hawkes_intensity", 0.0)),
            "hawkes_excitation": float(flow.get("hawkes_excitation", 0.0)),
            "note": "research-grade toxicity features (not production VPIN)",
        },
        "policy_note": (
            "European single-series options MM sim. "
            "Classical Avellaneda-Stoikov sets reservation price; "
            "these questions only modulate size/spread/pull/hedge."
        ),
    }


def build_mm_questions() -> dict[str, dict[str, Any]]:
    """One parallel System One batch — atomic questions, composed in policy.py."""
    questions: dict[str, Question] = {
        "regime": ChoiceQuestion(
            instructions="Classify the current options market-making regime from state",
            criteria=REGIME_CRITERIA,
        ),
        "toxicity": ScoreQuestion(
            instructions="Score adverse-selection / toxicity of recent flow given state",
            criteria=TOXICITY_LEVELS,
        ),
        "informed_flow": NoulQuestion(
            instructions="Is current flow likely informed / predatory against our quotes?",
            criteria={
                "true": "Evidence of informed or anticipatory flow",
                "false": "Flow looks uninformed or balanced",
            },
        ),
        "widen_quotes": NoulQuestion(
            instructions="Should we widen posted bid/ask relative to the A-S half-spread?",
        ),
        "pull_quotes": NoulQuestion(
            instructions="Should we pull quotes entirely this step (stop posting)?",
        ),
        "hedge_now": NoulQuestion(
            instructions="Should we hedge delta immediately given inventory and regime?",
        ),
        "size_tier": ChoiceQuestion(
            instructions="Which quote size tier is appropriate right now?",
            criteria=SIZE_TIER_CRITERIA,
        ),
        "surface_suspect": NoulQuestion(
            instructions=(
                "Is the implied-vol / mid inconsistent enough that the "
                "parametric surface should be treated as suspect?"
            ),
        ),
    }
    # Serialize to plain dicts matching POST /v1/systemone
    return {k: v.model_dump(exclude_none=True) for k, v in questions.items()}
