"""Constraint level-sets in (P, σ, t, funding).

A binding constraint is a gate on quoting, size, and hedge urgency. It is
not a forecast of the next return. Distance is signed: positive means the
coordinate is still inside the feasible side of the boundary.

VaR level-set, normal approximation:

    z · σ · √t · |position| · price = limit
    σ*(t) = limit / (z √t |position| price)

A longer horizon binds at a lower vol. The vol-target gate is the simpler
horizontal line σ = σ_cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ConstraintLevel:
    name: str
    binds: bool
    distance: float
    coordinate: str


def vol_target_level(sigma: float, cap: float) -> ConstraintLevel:
    return ConstraintLevel("vol_target", sigma > cap, cap - sigma, "sigma")


def price_floor_level(price: float, floor: float) -> ConstraintLevel:
    return ConstraintLevel("margin", price < floor, price - floor, "price")


def time_level(t: float, deadline: float) -> ConstraintLevel:
    return ConstraintLevel("clock", t >= deadline, deadline - t, "t")


def funding_level(funding: float, cap: float) -> ConstraintLevel:
    return ConstraintLevel("funding", funding > cap, cap - funding, "funding")


def borrow_level(utilization: float, cap: float) -> ConstraintLevel:
    return ConstraintLevel("borrow", utilization > cap, cap - utilization, "funding")


def var_sigma_star(var_limit: float, z: float, t: float, position: float, price: float) -> float:
    """Vol that exhausts a normal VaR limit. Non-positive inputs → infinity (no bind)."""
    denom = z * math.sqrt(t) * abs(position) * price if t > 0.0 else 0.0
    if denom <= 0.0 or var_limit <= 0.0:
        return math.inf
    return var_limit / denom


def active_constraints(levels: list[ConstraintLevel]) -> list[ConstraintLevel]:
    return [level for level in levels if level.binds]


def gate_blocks(levels: list[ConstraintLevel]) -> bool:
    """Quoting gate. Empty or all-clear level-sets do not block."""
    return any(level.binds for level in levels)
