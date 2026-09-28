"""Instability gate for reservation, spread, size, and hedge urgency.

Identity when ``enabled`` is false, including when the caller still passes a
large ratio. With the flag on and a zero ratio, no live parent, and no
binding constraint, the multipliers are also 1 and the shift is 0.

Hard pull (size 0) when the ratio reaches 2, or when a constraint level-set
is already binding. A live parent cuts size without pulling by itself.
The shift leans with the sign of net forced flow and is not a return forecast.

Constants are mirrored in ``zig/src/state_os.zig``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class StateGate:
    reservation_shift: float = 0.0
    spread_mult: float = 1.0
    size_mult: float = 1.0
    hedge_urgency: float = 0.0
    pull: bool = False
    instability: float = 0.0


def state_gate(
    enabled: bool,
    instability_value: float,
    constraint_active: bool,
    parent_remaining: float,
    f_signed: float,
    l_exec: float,
    mid: float,
) -> StateGate:
    if not enabled:
        return StateGate()
    if math.isinf(instability_value):
        u_clip = 4.0
    else:
        u_clip = max(instability_value, 0.0)
    pull = bool(constraint_active or u_clip >= 2.0)
    mild = min(u_clip, 2.0)
    spread = 1.0 + 0.80 * mild
    size = 1.0 / (1.0 + 1.10 * mild)
    parent = _clamp(parent_remaining, 0.0, 1.0)
    if parent > 0.0:
        size /= 1.0 + 0.75 * parent
    urgency = 0.45 * mild if u_clip >= 1.0 else 0.0
    if constraint_active:
        urgency = max(urgency, 0.70)
    shift = 0.0
    if mid > 0.0 and l_exec > 0.0 and u_clip > 0.0:
        pressure = _clamp(f_signed / l_exec, -1.0, 1.0)
        shift = 0.0015 * pressure * mid * min(u_clip, 1.0)
    if pull:
        size = 0.0
    stored = 4.0 if math.isinf(instability_value) else u_clip
    return StateGate(
        reservation_shift=shift,
        spread_mult=spread,
        size_mult=size,
        hedge_urgency=urgency,
        pull=pull,
        instability=stored,
    )
