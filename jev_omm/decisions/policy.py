"""Compose System One answers into QuoteAdjustments in *code*.

WHY (TypeSafe pattern): keep atomic questions narrow; combine with explicit
thresholds you control. Confidence gates: low confidence → widen or flatten,
never lean harder into size.
"""

from __future__ import annotations

from typing import Any, Optional

from jev_omm.decisions.client import DecisionClient, DeterministicFallbackClient
from jev_omm.decisions.schemas import (
    ChoiceAnswer,
    NoulAnswer,
    QuoteAdjustments,
    ScoreAnswer,
    SystemOneResult,
    build_mm_questions,
    build_mm_state,
)


# Tunable policy thresholds (pin model version before changing these in prod)
CONFIDENCE_FLOOR = 0.55
NOUL_WIDEN = 0.55
NOUL_PULL = 0.70
NOUL_HEDGE = 0.65
NOUL_INFORMED = 0.60
NOUL_SURFACE = 0.55
TOXICITY_WIDEN = 1.5  # score on 0..3 scale
TOXICITY_PULL = 2.5
SIZE_MULT = {"tiny": 0.25, "normal": 1.0, "large": 1.5}
REGIME_SPREAD = {"calm": 1.0, "trending": 1.15, "volatile": 1.4, "stressed": 1.75}


def _noul(result: SystemOneResult, key: str, default: float = 0.0) -> float:
    ans = result.answers.get(key)
    if isinstance(ans, NoulAnswer):
        return float(ans.noul)
    return default


def _choice(result: SystemOneResult, key: str, default: str = "") -> tuple[str, float]:
    ans = result.answers.get(key)
    if isinstance(ans, ChoiceAnswer):
        return ans.choice, float(ans.confidence)
    return default, 0.0


def _score(result: SystemOneResult, key: str, default: float = 0.0) -> tuple[float, float]:
    ans = result.answers.get(key)
    if isinstance(ans, ScoreAnswer):
        return float(ans.score), float(ans.confidence)
    return default, 0.0


def composite_toxicity(result: SystemOneResult) -> float:
    """Blend Score toxicity with informed_flow Noul (weights owned by code)."""
    tox, _ = _score(result, "toxicity")
    informed = _noul(result, "informed_flow")
    # Map score 0..3 + noul 0..1 → 0..1 composite
    return float(min(1.0, 0.55 * (tox / 3.0) + 0.45 * informed))


def apply_policy(result: SystemOneResult, state: Optional[dict[str, Any]] = None) -> QuoteAdjustments:
    """Map parallel System One answers → quote adjustments + flags."""
    reasons: list[str] = []
    regime, regime_conf = _choice(result, "regime", "calm")
    size_tier, size_conf = _choice(result, "size_tier", "normal")
    tox, tox_conf = _score(result, "toxicity")
    widen_n = _noul(result, "widen_quotes")
    pull_n = _noul(result, "pull_quotes")
    hedge_n = _noul(result, "hedge_now")
    informed = _noul(result, "informed_flow")
    surface = _noul(result, "surface_suspect")
    comp = composite_toxicity(result)

    # Confidence gate: uncertain Choice/Score → defensive (widen + smaller size)
    low_conf = (
        regime_conf < CONFIDENCE_FLOOR
        or size_conf < CONFIDENCE_FLOOR
        or tox_conf < CONFIDENCE_FLOOR
    )
    if low_conf:
        reasons.append("low_confidence_gate")

    spread_mult = REGIME_SPREAD.get(regime, 1.2)
    if widen_n >= NOUL_WIDEN or tox >= TOXICITY_WIDEN or informed >= NOUL_INFORMED:
        spread_mult = max(spread_mult, 1.0 + 0.8 * max(widen_n, informed, tox / 3.0))
        reasons.append("widen")
    if surface >= NOUL_SURFACE:
        spread_mult = max(spread_mult, 1.5)
        reasons.append("surface_suspect")
    if low_conf:
        spread_mult = max(spread_mult, 1.35)

    size_mult = SIZE_MULT.get(size_tier, 1.0)
    if low_conf or comp > 0.55:
        size_mult = min(size_mult, 0.5)
        reasons.append("size_cut")

    pull = pull_n >= NOUL_PULL or tox >= TOXICITY_PULL or (low_conf and regime == "stressed")
    if pull:
        size_mult = 0.0
        reasons.append("pull")

    hedge_now = hedge_n >= NOUL_HEDGE or (abs(comp) > 0.7 and regime in ("volatile", "stressed"))
    if hedge_now:
        reasons.append("hedge_now")

    instability = 0.0
    parent_remaining = 0.0
    constraint_active = False
    if state is not None:
        latent = state.get("latent") or {}
        if float(latent.get("enabled", 0.0)) >= 0.5:
            instability = float(latent.get("instability", 0.0))
            parent_remaining = float(latent.get("parent_remaining", 0.0))
            constraint_active = float(latent.get("constraint_active", 0.0)) >= 0.5
            if instability > 0.0 or parent_remaining > 0.0 or constraint_active:
                reasons.append("latent_state")

    return QuoteAdjustments(
        spread_mult=float(spread_mult),
        size_mult=float(size_mult),
        pull=bool(pull),
        hedge_now=bool(hedge_now),
        composite_toxicity=comp,
        reason=",".join(reasons) if reasons else "neutral",
        result=result,
        instability=instability,
        parent_remaining=parent_remaining,
        constraint_active=constraint_active,
    )


class DeskAdjustments:
    """Code-side map of the desk battery. Not an order."""

    def __init__(
        self,
        weight_mult: float = 1.0,
        kill: bool = False,
        reason: str = "identity",
        merge: bool = False,
        product_kill: bool = False,
    ) -> None:
        self.weight_mult = float(weight_mult)
        self.kill = bool(kill)
        self.reason = reason
        self.merge = bool(merge)
        self.product_kill = bool(product_kill)


def apply_desk_policy(result: SystemOneResult, state: Optional[dict[str, Any]] = None) -> DeskAdjustments:
    """Map sleeve_weight / kill_sleeve onto a multiplier.

    When ``desk.enabled`` is off, the result is the identity (multiplier 1,
    kill false) even if the answers say otherwise. The model does not emit
    an order; the caller decides whether a zero weight means flat risk.
    """
    desk = (state or {}).get("desk") or {}
    if float(desk.get("enabled", 0.0)) < 0.5:
        return DeskAdjustments(1.0, False, "identity")
    kill_n = max(_noul(result, "kill_sleeve"), _noul(result, "sleeve_kill"))
    choice, _conf = _choice(result, "sleeve_weight", "hold")
    merge_choice, _mconf = _choice(result, "merge_sleeve", "keep")
    cut_pair = _noul(result, "cut_corr_pair")
    product_n = _noul(result, "product_kill")
    mult = {"cut": 0.25, "hold": 1.0, "boost": 1.25}.get(choice, 1.0)
    kill = kill_n >= 0.70
    merge = merge_choice == "merge" or cut_pair >= 0.70
    product_kill = product_n >= 0.70
    reason = "desk"
    if kill:
        mult = 0.0
        reason = "kill_sleeve"
    elif merge:
        reason = "merge_sleeve"
    elif product_kill:
        reason = "product_kill"
    elif choice != "hold":
        reason = f"sleeve_weight:{choice}"
    return DeskAdjustments(mult, kill, reason, merge=merge, product_kill=product_kill)


def decide_quote_adjustments(
    client: DecisionClient,
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
    latent: Optional[dict[str, float]] = None,
) -> QuoteAdjustments:
    """Build state, run system_one batch, compose policy."""
    state = build_mm_state(
        time=time,
        spot=spot,
        option_mid=option_mid,
        iv=iv,
        inventory=inventory,
        delta=delta,
        gamma=gamma,
        vega=vega,
        cash_pnl=cash_pnl,
        half_spread=half_spread,
        quoting_allowed=quoting_allowed,
        recent_fills=recent_fills,
        spot_return_bps=spot_return_bps,
        latent=latent,
    )
    result = client.system_one(state, build_mm_questions())
    return apply_policy(result, state=state)


def default_offline_policy(**kwargs: Any) -> QuoteAdjustments:
    """Convenience for tests: always use DeterministicFallbackClient."""
    return decide_quote_adjustments(DeterministicFallbackClient(), **kwargs)
