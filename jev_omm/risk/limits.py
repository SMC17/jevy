"""Hard risk limits that stop quoting when breached.

WHY hard stops: soft utility (A–S gamma) only *skews* quotes; inventory can
still grow unbounded under one-sided flow. Hard limits protect the book in sim
and mirror what a production kill-switch would do.
"""

from __future__ import annotations

from jev_omm.config import RiskConfig
from jev_omm.models.types import Greeks, RiskSnapshot


def evaluate_risk(
    inventory: int,
    greeks_per_contract: Greeks,
    cash_pnl: float,
    cfg: RiskConfig,
) -> RiskSnapshot:
    """Aggregate position Greeks and check hard limits."""
    delta = inventory * greeks_per_contract.delta
    gamma = inventory * greeks_per_contract.gamma
    vega = inventory * greeks_per_contract.vega

    reason: str | None = None
    if abs(inventory) > cfg.max_abs_inventory:
        reason = f"inventory |{inventory}| > {cfg.max_abs_inventory}"
    elif abs(delta) > cfg.max_abs_delta:
        reason = f"delta |{delta:.4f}| > {cfg.max_abs_delta}"
    elif abs(vega) > cfg.max_abs_vega:
        reason = f"vega |{vega:.4f}| > {cfg.max_abs_vega}"
    elif abs(gamma) > cfg.max_abs_gamma:
        reason = f"gamma |{gamma:.6f}| > {cfg.max_abs_gamma}"
    elif cash_pnl < -abs(cfg.max_loss):
        reason = f"cash_pnl {cash_pnl:.2f} below -{abs(cfg.max_loss)}"

    return RiskSnapshot(
        inventory=inventory,
        delta=delta,
        gamma=gamma,
        vega=vega,
        cash_pnl=cash_pnl,
        quoting_allowed=reason is None,
        breach_reason=reason,
    )
