"""Exponential Hawkes intensity. Mirrors ``zig/src/hawkes.zig``.

λ(t) = μ + Σ α exp(−β (t − t_i)). Excitation (λ−μ)/μ is a Decision feature
``flow.hawkes_excitation``. It does not emit orders.

Hawkes, Biometrika 1971, https://doi.org/10.1093/biomet/58.1.83.
Bacry, Mastromatteo, Muzy, https://arxiv.org/abs/1502.04592.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HawkesParams:
    mu: float = 1.0
    alpha: float = 0.6
    beta: float = 2.0


def intensity(p: HawkesParams, t: float, events: list[float]) -> float:
    lam = max(p.mu, 0.0)
    a = max(p.alpha, 0.0)
    for ti in events:
        if ti <= t:
            lam += a * math.exp(-p.beta * (t - ti))
    return lam


def excitation(p: HawkesParams, t: float, events: list[float]) -> float:
    mu = max(p.mu, 1e-12)
    return max(0.0, (intensity(p, t, events) - mu) / mu)


def branching_ratio(p: HawkesParams) -> float:
    if not (p.beta > 0.0):
        return math.inf
    return max(p.alpha, 0.0) / p.beta


def fill_intensity(base: float, exc: float) -> float:
    return max(base, 0.0) * (1.0 + max(exc, 0.0))


def flow_features(p: HawkesParams, t: float, events: list[float]) -> dict[str, float]:
    lam = intensity(p, t, events)
    exc = excitation(p, t, events)
    return {"hawkes_intensity": lam, "hawkes_excitation": exc}
