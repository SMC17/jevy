"""Sparse hub-and-spoke cross-impact, and the common-flow trap.

A regression of spoke returns on hub returns attributes the common factor to
cross-impact. Residualize both series on the common flow first. What is left
is the cross-impact coefficient this module will trust.

Benzaquen, Mastromatteo, Eisler, Bouchaud, J. Stat. Mech. 2017,
https://doi.org/10.1088/1742-5468/aa53f7

The propagator that would make G_ij a full matrix is not estimated here.
The hub-spoke product below is the research object: diagonal self-impact
plus a single hub loading.
"""

from __future__ import annotations

from jev_omm.training.scoring import lcg_next


def ols_slope(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    var = sum((a - mx) ** 2 for a in x)
    if var <= 0.0:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / var


def residualize(series: list[float], factor: list[float]) -> list[float]:
    slope = ols_slope(factor, series)
    n = len(series)
    mx = sum(factor) / n
    my = sum(series) / n
    intercept = my - slope * mx
    return [value - (intercept + slope * f) for value, f in zip(series, factor)]


def hub_spoke_impact(
    q_hub: float,
    q_spoke: float,
    g_hh: float,
    g_hs: float,
    g_ss: float,
) -> tuple[float, float]:
    """Return (hub impact, spoke impact). Spoke feels its own flow and the hub."""
    hub = g_hh * q_hub
    spoke = g_ss * q_spoke + g_hs * q_hub
    return hub, spoke


def _lcg_series(n: int, seed: int) -> tuple[list[float], list[float], list[float], int]:
    state = seed
    z: list[float] = []
    e_h: list[float] = []
    e_s: list[float] = []
    for _ in range(n):
        state, a = lcg_next(state)
        state, b = lcg_next(state)
        state, c = lcg_next(state)
        z.append(a)
        e_h.append(0.15 * b)
        e_s.append(0.15 * c)
    return z, e_h, e_s, state


def common_flow_trap(n: int = 400, seed: int = 3) -> dict[str, float]:
    """True cross-impact is zero. Naive hub beta is the common factor.

    Identified beta residualizes on that factor and should sit near zero.
    """
    z, e_h, e_s, _ = _lcg_series(n, seed)
    hub = [zi + eh for zi, eh in zip(z, e_h)]
    spoke = [zi + es for zi, es in zip(z, e_s)]
    naive = ols_slope(hub, spoke)
    identified = ols_slope(residualize(hub, z), residualize(spoke, z))
    return {"naive": naive, "identified": identified}


def identified_cross_beta(n: int = 400, seed: int = 9, true_beta: float = 0.35) -> dict[str, float]:
    """No common factor. Residualizing on an independent factor keeps the loading."""
    _z, e_h, e_s, state = _lcg_series(n, seed)
    decoy = []
    for _ in range(n):
        state, a = lcg_next(state)
        decoy.append(a)
    hub = e_h
    spoke = [true_beta * h + es for h, es in zip(hub, e_s)]
    naive = ols_slope(hub, spoke)
    identified = ols_slope(residualize(hub, decoy), residualize(spoke, decoy))
    return {"naive": naive, "identified": identified, "true": true_beta}


def beta_gap(naive: float, identified: float) -> float:
    return abs(naive) - abs(identified)
