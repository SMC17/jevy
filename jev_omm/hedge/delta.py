"""Banded delta hedge + slippage + greek PnL attribution (paper only).

Replaces the prior stub. Prefers Zig ``jev_omm_hedge_*`` / ``greek_pnl_step``
when ``libjev_omm.so`` is present.

Policy (documented): when |net_delta| > band, hedge qty = −net_delta (flatten
to zero). Set ``flatten=False`` to hedge only to the band edge.

Whalley–Wilmott-style practical bandwidth helper:
  band ≈ (1.5 · ε · S · Γ² / γ_risk)^{1/3}
See ``zig/src/hedge.zig`` for the full note.

Curriculum: Akuna delta/gamma/theta + Natenberg γ-PnL ≈ ½ Γ (ΔS)² vs theta.
"""

from __future__ import annotations


from pydantic import BaseModel

from jev_omm.pricing import _native


class HedgeOrder(BaseModel):
    """Paper hedge ticket (simulation only)."""

    delta_to_hedge: float
    underlier_qty: float  # shares / futures equiv; sign: +buy underlier
    reason: str = ""
    fire: bool = True


class HedgeConfig(BaseModel):
    delta_band: float = 5.0
    half_spread: float = 0.0
    slip_bps: float = 1.0
    flatten: bool = True


class HedgeFill(BaseModel):
    time: float = 0.0
    underlier_qty: float = 0.0
    fill_price: float = 0.0
    mid: float = 0.0
    cash_delta: float = 0.0
    slippage_cost: float = 0.0


class GreekPnlStep(BaseModel):
    spread_capture: float = 0.0
    hedge_slippage: float = 0.0
    gamma_pnl: float = 0.0
    theta_pnl: float = 0.0
    vega_pnl: float = 0.0
    inventory_mtm: float = 0.0
    delta_pnl: float = 0.0

    def total(self) -> float:
        return (
            self.spread_capture
            + self.hedge_slippage
            + self.gamma_pnl
            + self.theta_pnl
            + self.vega_pnl
            + self.inventory_mtm
        )


def propose_delta_hedge(
    net_delta: float,
    *,
    band: float = 5.0,
    half_spread: float = 0.0,
    slip_bps: float = 1.0,
    flatten: bool = True,
) -> HedgeOrder | None:
    """If |delta| exceeds band, propose underlier trade (paper only)."""
    if _native.ZIG_AVAILABLE and hasattr(_native, "hedge_propose"):
        try:
            d = _native.hedge_propose(net_delta, band, half_spread, slip_bps, flatten)
            if not d["fire"]:
                return None
            return HedgeOrder(
                delta_to_hedge=d["delta_to_hedge"],
                underlier_qty=d["underlier_qty"],
                reason=f"|delta|={abs(net_delta):.2f} >= band={band}",
                fire=True,
            )
        except Exception:
            pass
    if abs(net_delta) <= band:
        return None
    if flatten:
        qty = -net_delta
        reason = "flatten"
    else:
        edge = band if net_delta > 0 else -band
        qty = edge - net_delta
        reason = "to_band_edge"
    return HedgeOrder(
        delta_to_hedge=net_delta,
        underlier_qty=qty,
        reason=f"|delta|={abs(net_delta):.2f} >= band={band} ({reason})",
        fire=True,
    )


def fill_price(mid: float, qty: float, *, half_spread: float = 0.0, slip_bps: float = 1.0) -> float:
    if qty == 0.0:
        return mid
    buying = qty > 0.0
    if half_spread > 0.0:
        return mid + half_spread if buying else mid - half_spread
    slip = slip_bps * 1e-4
    return mid * (1.0 + slip) if buying else mid * (1.0 - slip)


def apply_hedge(
    time: float,
    mid: float,
    order: HedgeOrder,
    *,
    half_spread: float = 0.0,
    slip_bps: float = 1.0,
    band: float = 5.0,
    flatten: bool = True,
) -> HedgeFill:
    if _native.ZIG_AVAILABLE and hasattr(_native, "hedge_apply"):
        try:
            d = _native.hedge_apply(
                time, mid, order.underlier_qty, order.delta_to_hedge, True,
                band, half_spread, slip_bps, flatten,
            )
            return HedgeFill(**d)
        except Exception:
            pass
    px = fill_price(mid, order.underlier_qty, half_spread=half_spread, slip_bps=slip_bps)
    cash = -order.underlier_qty * px
    slip_cost = -abs(order.underlier_qty) * abs(px - mid)
    return HedgeFill(
        time=time,
        underlier_qty=order.underlier_qty,
        fill_price=px,
        mid=mid,
        cash_delta=cash,
        slippage_cost=slip_cost,
    )


def greek_pnl_step(
    *,
    delta: float,
    gamma: float,
    vega: float,
    theta: float,
    d_spot: float,
    d_sigma: float = 0.0,
    dt: float = 0.0,
    option_qty: float = 0.0,
    d_option_mid: float = 0.0,
    underlier_pos: float = 0.0,
    spread_capture: float = 0.0,
    hedge_slippage: float = 0.0,
) -> GreekPnlStep:
    if _native.ZIG_AVAILABLE and hasattr(_native, "greek_pnl_step"):
        try:
            d = _native.greek_pnl_step(
                delta, gamma, vega, theta, d_spot, d_sigma, dt,
                option_qty, d_option_mid, underlier_pos,
                spread_capture, hedge_slippage,
            )
            return GreekPnlStep(**d)
        except Exception:
            pass
    return GreekPnlStep(
        spread_capture=spread_capture,
        hedge_slippage=hedge_slippage,
        gamma_pnl=0.5 * gamma * d_spot * d_spot,
        theta_pnl=theta * dt,
        vega_pnl=vega * d_sigma,
        inventory_mtm=option_qty * d_option_mid + underlier_pos * d_spot,
        delta_pnl=delta * d_spot,
    )


def whalley_wilmott_band(
    spot: float,
    gamma_abs: float,
    sigma: float,
    slip_frac: float,
    risk_aversion: float,
) -> float:
    if _native.ZIG_AVAILABLE and hasattr(_native, "ww_band"):
        try:
            return _native.ww_band(spot, gamma_abs, sigma, slip_frac, risk_aversion)
        except Exception:
            pass
    g = max(gamma_abs, 1e-8)
    eps = max(slip_frac, 1e-8)
    gamma_risk = max(risk_aversion, 1e-8)
    inside = 1.5 * eps * spot * g * g / gamma_risk
    return inside ** (1.0 / 3.0)


def net_delta(option_delta: float, underlier_pos: float) -> float:
    return option_delta + underlier_pos
