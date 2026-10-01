"""Inverse-vol sleeve weights with a correlation shrink, a Sharpe tilt, and a cap.

Same rule as ``zig/src/desk.zig`` ``allocateInverseVol``. Disabled sleeves
stay at weight 0 (the off switch is the identity: they do not take risk).

Rule, in order:

1. Sample standard deviation with divisor n−1. A series with σ < 1e-5 has
   no residual variance and gets raw weight 0. Otherwise floor σ at 1e-8.
2. Raw weight proportional to ``boost / σ``. ``boost`` is
   ``min(3, 1 + tilt * max(Sharpe, 0))``. A negative residual mean multiplies
   ``boost`` by 0.25. ``tilt`` defaults to 0 in this function so the
   toxic-sleeve example stays exact. The desk passes ``DEFAULT_SHARPE_TILT``.
   The 0.25 haircut still applies at tilt 0, so a legacy replay of the 1.0
   weight vector will not match: negative-mean sleeves shrink.
3. For every enabled pair with |ρ| > ``corr_cap`` (default 0.35), multiply
   both raw weights by ``corr_cap / |ρ|``. A sleeve in several pairs is
   scaled once per pair.
4. If a sleeve's residual mean is negative and another enabled sleeve has
   |ρ| above the cap and a strictly higher mean, set the worse sleeve's raw
   weight to 0.
5. Renormalize the survivors so they sum to 1.
6. Cap any weight at ``max_weight`` (default 0.35). Excess is spread across
   uncapped positive weights. If every survivor is already capped, the
   leftover stays in cash and the weights sum to less than 1.
"""

from __future__ import annotations

import math

import numpy as np

from jev_omm.pnl.residual import pearson

MAX_SLEEVES = 24
DEFAULT_CORR_CAP = 0.35
DEFAULT_MAX_WEIGHT = 0.35
DEFAULT_SHARPE_TILT = 0.25
_TILT_CAP = 3.0
_FLAT_STD = 1e-5


def _raw_std(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    m = float(np.mean(x))
    var = float(np.sum((x - m) ** 2) / (x.size - 1))
    return math.sqrt(max(var, 0.0))


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 1.0
    m = float(np.mean(x))
    if x.size < 2:
        return m, 1.0
    std = _raw_std(x)
    if std < 1e-8:
        std = 1e-8
    return m, std


def allocate(
    residual: list[np.ndarray] | list[list[float]],
    enabled: list[bool] | None = None,
    *,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    corr_cap: float = DEFAULT_CORR_CAP,
    sharpe_tilt: float = 0.0,
) -> np.ndarray:
    series = [np.asarray(s, dtype=float).reshape(-1) for s in residual]
    n = len(series)
    on = [True] * n if enabled is None else [bool(v) for v in enabled]
    if len(on) < n:
        on = on + [False] * (n - len(on))
    m = min(n, MAX_SLEEVES)
    inv = np.zeros(n)
    means = np.zeros(n)
    rho = np.eye(n)
    for i in range(m):
        means[i], std = _mean_std(series[i])
        if not on[i]:
            continue
        raw_std = _raw_std(series[i])
        if raw_std < _FLAT_STD:
            inv[i] = 0.0
            continue
        sharpe = means[i] / raw_std if raw_std > 0.0 else 0.0
        boost = 1.0 + float(sharpe_tilt) * max(sharpe, 0.0)
        if boost > _TILT_CAP:
            boost = _TILT_CAP
        # A negative residual mean is down-weighted even when it is not
        # collinear. The hard zero remains rule 4, for the correlated loser.
        if means[i] < 0.0:
            boost *= 0.25
        inv[i] = boost / std
    for i in range(m):
        for j in range(i + 1, m):
            p = pearson(series[i], series[j])
            rho[i, j] = rho[j, i] = p
    for i in range(m):
        for j in range(i + 1, m):
            if not (on[i] and on[j]):
                continue
            ar = abs(float(rho[i, j]))
            if ar > corr_cap and ar > 0.0:
                scale = corr_cap / ar
                inv[i] *= scale
                inv[j] *= scale
    for i in range(m):
        if not on[i] or means[i] >= 0.0:
            continue
        for j in range(m):
            if i == j or not on[j]:
                continue
            if abs(float(rho[i, j])) > corr_cap and means[j] > means[i]:
                inv[i] = 0.0
                break
    total = float(inv.sum())
    w = np.zeros(n)
    if not (total > 0.0):
        return w
    w[:m] = inv[:m] / total
    capped = np.zeros(n, dtype=bool)
    for _ in range(MAX_SLEEVES):
        excess = 0.0
        free_sum = 0.0
        for i in range(m):
            if w[i] > max_weight + 1e-15:
                excess += w[i] - max_weight
                w[i] = max_weight
                capped[i] = True
            elif w[i] > 0.0 and not capped[i]:
                free_sum += w[i]
        if excess <= 1e-15 or not (free_sum > 0.0):
            break
        for i in range(m):
            if w[i] > 0.0 and not capped[i]:
                w[i] += excess * (w[i] / free_sum)
    return w


# Fixed series shared with zig/src/desk.zig and the training case.
TOXIC_SLEEVE_GOOD = (0.03, 0.04, 0.02, 0.05, 0.03, 0.04)
TOXIC_SLEEVE_TOXIC = (-0.04, -0.03, -0.05, -0.02, -0.04, -0.03)


def toxic_sleeve_weights(max_weight: float = DEFAULT_MAX_WEIGHT) -> np.ndarray:
    return allocate(
        [list(TOXIC_SLEEVE_GOOD), list(TOXIC_SLEEVE_TOXIC)],
        [True, True],
        max_weight=max_weight,
        corr_cap=DEFAULT_CORR_CAP,
    )
