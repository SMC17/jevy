"""Hard risk limits that stop quoting when breached.

WHY hard stops: soft utility (A–S gamma) only *skews* quotes; inventory can
still grow unbounded under one-sided flow. Hard limits protect the book in sim
and mirror what a production kill-switch would do.
"""

from __future__ import annotations

from jev_omm.config import RiskConfig
from jev_omm.models.types import Greeks, RiskSnapshot


def _limit_on(value: float | int | None) -> bool:
    """Unset (None or non-positive) limits are a no-op."""
    if value is None:
        return False
    return float(value) > 0.0


def evaluate_risk(
    inventory: int,
    greeks_per_contract: Greeks,
    cash_pnl: float,
    cfg: RiskConfig,
    *,
    extra_delta: float = 0.0,
    notional: float | None = None,
    per_strike_abs: int | None = None,
    quotes_outstanding: int | None = None,
) -> RiskSnapshot:
    """Aggregate position Greeks and check hard limits.

    ``extra_delta`` is the underlier hedge (shares). Net delta is
    ``inventory * per-contract delta + extra_delta``.

    Notional, per-strike inventory, and quotes outstanding are checked only
    when the caller passes the object **and** the matching limit is set.
    A missing object or an unset limit does not trip.
    """
    delta = inventory * greeks_per_contract.delta + extra_delta
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
    elif (
        _limit_on(cfg.max_abs_notional)
        and notional is not None
        and abs(notional) > float(cfg.max_abs_notional or 0.0)
    ):
        reason = f"notional |{notional:.4f}| > {cfg.max_abs_notional}"
    elif (
        _limit_on(cfg.max_abs_per_strike)
        and per_strike_abs is not None
        and abs(per_strike_abs) > int(cfg.max_abs_per_strike or 0)
    ):
        reason = f"per_strike |{per_strike_abs}| > {cfg.max_abs_per_strike}"
    elif (
        _limit_on(cfg.max_quotes_outstanding)
        and quotes_outstanding is not None
        and quotes_outstanding > int(cfg.max_quotes_outstanding or 0)
    ):
        reason = f"quotes_outstanding {quotes_outstanding} > {cfg.max_quotes_outstanding}"

    return RiskSnapshot(
        inventory=inventory,
        delta=delta,
        gamma=gamma,
        vega=vega,
        cash_pnl=cash_pnl,
        quoting_allowed=reason is None,
        breach_reason=reason,
    )
