"""Market state S_t and the instability functional.

S_t = (P, σ, Σ, L, I, C, F, G, τ, B, O, K_stack, D_liab, R)

The tradeable object is not a return forecast. It is

    P(forced sellers) × P(latent bids withdraw) × |F| / L_exec

with L_exec evaluated on the path F will walk. F_net removes flow that is
already positioned and flow that cannot start.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def forced_flow(parts: list[tuple[float, float]]) -> float:
    """F(S) = Σ_k w_k(S) f_k(S). Each part is ``(weight, signed flow)``."""
    return sum(weight * flow for weight, flow in parts)


def net_forced(f_forced: float, f_already_positioned: float, f_cannot_start: float) -> float:
    """Flow that still has to print. Already-positioned and cannot-start drop out."""
    return f_forced - f_already_positioned - f_cannot_start


def instability(f_value: float, l_exec: float) -> float:
    """|F| / L_exec.

    L_exec <= 0 and |F| > 0 is an air pocket: no executable bid or offer
    remains on the path, so the ratio is infinite. Both zero is a quiet book.
    """
    forced = abs(f_value)
    if l_exec > 0.0:
        return forced / l_exec
    if forced == 0.0:
        return 0.0
    return math.inf


def decision_instability(f_value: float, l_exec: float) -> float:
    """Finite scalar for Decision state. Infinity is stored as the gate clip 4."""
    value = instability(f_value, l_exec)
    if math.isinf(value):
        return 4.0
    return value


def l_exec_on_path(l_base: float, f_value: float, dL_dQ: float) -> float:
    """Executable size after F walks the book.

    ``dL_dQ`` is the slope of executable liquidity in forced size. It is
    negative when the flow itself consumes the bids it will need. The result
    is floored at zero.
    """
    return max(0.0, l_base + dL_dQ * abs(f_value))


@dataclass
class MarketState:
    """One-name research view of S_t. Vectors in the brief are length 1 here.

    ``enabled`` is the desk flag. Off means instability reports 0 and the
    quote scaler is the identity, whatever the raw F and L numbers are.
    """

    price: float
    sigma: float
    surface_atm: float = 0.0
    surface_skew: float = 0.0
    l_displayed: float = 0.0
    l_exec: float = 0.0
    inventories: dict[str, float] = field(default_factory=dict)
    constraints_active: tuple[str, ...] = ()
    f_forced: float = 0.0
    f_already_positioned: float = 0.0
    f_cannot_start: float = 0.0
    graph_loading: float = 0.0
    tau_pressure: float = 0.0
    barriers: tuple[float, ...] = ()
    observation_count: int = 0
    coupon_stack: tuple[float, ...] = ()
    liability_duration: float = 0.0
    capital_regime: str = "economic"
    enabled: bool = False

    def f_net(self) -> float:
        return net_forced(self.f_forced, self.f_already_positioned, self.f_cannot_start)

    def instability(self) -> float:
        if not self.enabled:
            return 0.0
        return instability(self.f_net(), self.l_exec)

    def latent_features(self) -> dict[str, float]:
        """Scalars the fallback client and Zig gate both consume."""
        live = self.instability()
        return {
            "enabled": 1.0 if self.enabled else 0.0,
            "instability": decision_instability(self.f_net(), self.l_exec) if self.enabled else 0.0,
            "f_net": self.f_net() if self.enabled else 0.0,
            "l_exec": self.l_exec,
            "constraint_active": 1.0 if self.constraints_active else 0.0,
            "parent_remaining": 0.0,
            "gex_disagree": 0.0,
            "tau_pressure": self.tau_pressure if self.enabled else 0.0,
            "raw_instability_infinite": 1.0 if self.enabled and math.isinf(live) else 0.0,
        }
