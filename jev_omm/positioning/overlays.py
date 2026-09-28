"""ETF create/redeem pressure, futures roll, and a residual factor hedge.

These are feature builders. They do not change the graded ``etf_ap_arb`` or
``location_arb`` constants. The location case remains the oil-beta lesson;
``futures_overlay_qty`` is the same one-line hedge, reusable on a vector.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EtfFlow:
    """One create/redeem snapshot. Notionals are paper units, not a venue feed."""

    etf_mid: float
    basket_mid: float
    create_fee_bps: float
    redeem_fee_bps: float
    create_notional: float
    redeem_notional: float
    residual_vol_bps: float
    latency_steps: int


def premium_bps(flow: EtfFlow) -> float:
    if flow.basket_mid == 0.0:
        return 0.0
    return (flow.etf_mid - flow.basket_mid) / flow.basket_mid * 1e4


def ap_pressure(flow: EtfFlow) -> float:
    """+1 means creation dominates (AP buys the basket, sells the ETF)."""
    den = flow.create_notional + flow.redeem_notional
    if den <= 0.0:
        return 0.0
    return (flow.create_notional - flow.redeem_notional) / den


def edge_bps(flow: EtfFlow) -> float:
    """Tradable premium after the fee on the side the premium points to."""
    prem = premium_bps(flow)
    if prem >= 0.0:
        return prem - flow.redeem_fee_bps
    return -prem - flow.create_fee_bps


def near_risk_free(flow: EtfFlow) -> bool:
    """Edge large versus residual vol scaled by latency. Same idea as the ETF case.

    The case engine still owns its own size. This flag is a feature, not an order.
    """
    edge = edge_bps(flow)
    hurdle = 3.0 * flow.residual_vol_bps * (1.0 + max(flow.latency_steps, 0))
    return edge > hurdle and edge > 0.0


def annualized_roll(front: float, back: float, days: float) -> float:
    """(back/front - 1) * 365/days. Contango is positive."""
    if front <= 0.0 or days <= 0.0:
        return 0.0
    return (back / front - 1.0) * (365.0 / days)


def zscore(value: float, mean: float, std: float) -> float:
    if std <= 0.0:
        return 0.0
    return (value - mean) / std


def residual_beta(qtys: list[float], betas: list[float]) -> float:
    return sum(q * b for q, b in zip(qtys, betas))


def futures_overlay_qty(exposure: float, futures_beta: float) -> float:
    """Futures quantity that zeroes ``exposure + futures_beta * q`` when beta ≠ 0.

    The location-arb desk uses this with futures beta 0.85 against oil exposure.
    A zero futures beta returns 0 (no hedge invented).
    """
    if futures_beta == 0.0:
        return 0.0
    return -exposure / futures_beta
