"""Impact residual. Trade ε, not the raw print.

    ΔP = f(Q, σ, D, V, S) + ε
    f = Y σ √(|Q| / V)  +  spread × max(|Q| − D, 0) / D

The square-root piece is the temporary impact law. The second piece is the
extra paid once the order is larger than displayed depth. ε is what is left
after that function. A residual that still lines up with Q means the function
is wrong; on a path drawn from f plus independent noise, ε does not.

Y is a scale the caller sets. This module does not fit a live tape.
"""

from __future__ import annotations

import math

from jev_omm.state_os.metaorder import sqrt_impact


def predicted_impact(
    quantity: float,
    sigma: float,
    depth: float,
    volume: float,
    spread: float,
    y: float = 1.0,
) -> float:
    base = sqrt_impact(quantity, sigma, volume, y)
    if depth <= 0.0 or quantity == 0.0:
        return base
    extra = spread * max(abs(quantity) - depth, 0.0) / depth
    if quantity < 0.0:
        extra = -extra
    return base + extra


def impact_residual(
    dp: float,
    quantity: float,
    sigma: float,
    depth: float,
    volume: float,
    spread: float,
    y: float = 1.0,
) -> float:
    return dp - predicted_impact(quantity, sigma, depth, volume, spread, y)


def correlation(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)
