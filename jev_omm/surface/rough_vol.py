"""Small rough-volatility stress paths (research only).

Fractional Brownian covariance

    E[B^H_t B^H_s] = ½ (t^{2H} + s^{2H} − |t−s|^{2H})

and a one-factor rough Bergomi variance

    ξ_t = ξ_0 exp(η W^H_t − ½ η² t^{2H}).

Cite Gatheral, Jaisson, Rosenbaum, "Volatility is rough",
https://arxiv.org/abs/1410.3394.
"""

from __future__ import annotations

import math

import numpy as np


def fbm_covariance(times: np.ndarray, hurst: float) -> np.ndarray:
    h = float(hurst)
    t = np.asarray(times, dtype=float)
    tt = t[:, None]
    ss = t[None, :]
    return 0.5 * (tt ** (2.0 * h) + ss ** (2.0 * h) - np.abs(tt - ss) ** (2.0 * h))


def sample_fbm(n: int, hurst: float, seed: int, t_end: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("n >= 2")
    times = np.linspace(t_end / n, t_end, n)
    cov = fbm_covariance(times, hurst)
    jitter = 1e-10 * np.eye(n)
    chol = np.linalg.cholesky(cov + jitter)
    z = np.random.default_rng(seed).standard_normal(n)
    return times, chol @ z


def rough_bergomi_variance(
    n: int,
    hurst: float,
    eta: float,
    xi0: float,
    seed: int,
    t_end: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    times, w = sample_fbm(n, hurst, seed, t_end=t_end)
    var = xi0 * np.exp(eta * w - 0.5 * eta * eta * np.power(times, 2.0 * hurst))
    return times, var


def holder_scale(dt: float, hurst: float) -> float:
    """Typical increment scale Δt^H. Research diagnostic, not a path."""
    return math.pow(max(dt, 0.0), hurst)
