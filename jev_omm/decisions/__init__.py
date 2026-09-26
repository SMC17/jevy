"""TypeSafe System One (Jev) decision layer for options MM.

Live calls optional via TYPESAFE_API_KEY; DeterministicFallbackClient always works.
Classical A–S reservation price stays pure math — this layer only modulates
spread/size/pull/hedge after atomic Choice/Score/Noul answers are composed in policy.
"""

from jev_omm.decisions.client import (
    DecisionClient,
    DeterministicFallbackClient,
    TypeSafeDecisionClient,
    ResilientDecisionClient,
    make_decision_client,
)
from jev_omm.decisions.policy import apply_policy, decide_quote_adjustments
from jev_omm.decisions.schemas import QuoteAdjustments, SystemOneResult, build_mm_questions, build_mm_state

__all__ = [
    "DecisionClient",
    "DeterministicFallbackClient",
    "TypeSafeDecisionClient",
    "ResilientDecisionClient",
    "QuoteAdjustments",
    "SystemOneResult",
    "apply_policy",
    "build_mm_questions",
    "build_mm_state",
    "decide_quote_adjustments",
    "make_decision_client",
]
