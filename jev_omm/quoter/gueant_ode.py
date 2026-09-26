"""Guéant–Lehalle–Fernandez-Tapia ODE / spectral quotes and (A, k) MLE.

Mirrors ``zig/src/gueant_ode.zig``. Cite https://arxiv.org/abs/1105.3115.

The linear system dv/dτ = M v is solved in ω-space (v = exp(−(k/γ) ω))
by RK4. ``spectral_offsets`` isolates the principal eigenmode of M.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from jev_omm.config import QuoterConfig
from jev_omm.models.types import Greeks, Quote
from jev_omm.quoter.gueant import optimal_offsets as asymptotic_offsets

MAX_Q = 32
NO_QUOTE = 1.0e6


@dataclass
class Offsets:
    delta_b: float
    delta_a: float
    half: float
    reservation_shift: float


@dataclass
class IntensityObs:
    delta: float
    exposure: float
    fills: float


@dataclass
class IntensityFit:
    A: float
    k: float
    loglik: float
    n_bins: int


def coefficient_c(gamma: float, k: float, A: float) -> float:
    if gamma <= 0.0 or k <= 0.0 or A <= 0.0:
        return 0.0
    ratio = k / (k + gamma)
    return A * (gamma / (k + gamma)) * (ratio ** (k / gamma))


def _clamp_q(q: int) -> int:
    return max(1, min(MAX_Q, int(q)))


def solve_omega(cfg: QuoterConfig, q_cap: int | None = None, horizon: float | None = None, n_steps: int | None = None) -> tuple[int, np.ndarray]:
    q_cap = _clamp_q(cfg.inventory_cap if q_cap is None else q_cap)
    horizon = cfg.T_horizon if horizon is None else horizon
    n_steps = cfg.ode_steps if n_steps is None else n_steps
    n = 2 * q_cap + 1
    if cfg.gamma <= 0.0 or cfg.kappa <= 0.0 or cfg.A <= 0.0 or horizon <= 0.0:
        return q_cap, np.zeros(n)
    steps = max(int(n_steps), 1)
    dt = horizon / steps
    alpha = 0.5 * cfg.sigma * cfg.sigma * cfg.gamma * cfg.gamma
    C = coefficient_c(cfg.gamma, cfg.kappa, cfg.A)
    coef = cfg.kappa / cfg.gamma
    w = np.zeros(n)

    def deriv(state: np.ndarray) -> np.ndarray:
        dw = np.empty(n)
        for i in range(n):
            q = i - q_cap
            benefit = 0.0
            if q + 1 <= q_cap:
                expo = max(-80.0, min(80.0, -coef * (state[i + 1] - state[i])))
                benefit += C * math.exp(expo)
            if q - 1 >= -q_cap:
                expo = max(-80.0, min(80.0, -coef * (state[i - 1] - state[i])))
                benefit += C * math.exp(expo)
            dw[i] = alpha * q * q - benefit
        return dw

    for _ in range(steps):
        k1 = deriv(w)
        k2 = deriv(w + 0.5 * dt * k1)
        k3 = deriv(w + 0.5 * dt * k2)
        k4 = deriv(w + dt * k3)
        w = w + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return q_cap, w


def offsets_at(cfg: QuoterConfig, omega: np.ndarray, q_cap: int, inventory: int) -> Offsets:
    gamma, k = cfg.gamma, cfg.kappa
    if gamma <= 0.0 or k <= 0.0:
        return Offsets(0.0, 0.0, 0.0, 0.0)
    psi = (1.0 / gamma) * math.log(1.0 + gamma / k)
    q = max(-q_cap, min(q_cap, inventory))
    i = q + q_cap
    delta_b = psi + (omega[i + 1] - omega[i]) / gamma if q + 1 <= q_cap else NO_QUOTE
    delta_a = psi + (omega[i - 1] - omega[i]) / gamma if q - 1 >= -q_cap else NO_QUOTE
    db = cfg.max_half_spread if delta_b >= NO_QUOTE * 0.5 else delta_b
    da = cfg.max_half_spread if delta_a >= NO_QUOTE * 0.5 else delta_a
    return Offsets(delta_b, delta_a, 0.5 * (da + db), 0.5 * (da - db))


def optimal_offsets(cfg: QuoterConfig, inventory: int) -> Offsets:
    q_cap, omega = solve_omega(cfg)
    return offsets_at(cfg, omega, q_cap, inventory)


def spectral_offsets(cfg: QuoterConfig, inventory: int) -> Offsets:
    """Principal eigenmode of dv/dτ = M v (RK4 + renormalization)."""
    q_cap = _clamp_q(cfg.inventory_cap)
    n = 2 * q_cap + 1
    gamma, k = cfg.gamma, cfg.kappa
    if gamma <= 0.0 or k <= 0.0 or cfg.A <= 0.0:
        return Offsets(0.0, 0.0, 0.0, 0.0)
    ratio = k / (k + gamma)
    eta = cfg.A * (ratio ** ((k + gamma) / gamma))
    diag = 0.5 * k * gamma * cfg.sigma * cfg.sigma
    v = np.ones(n)

    def mv(src: np.ndarray) -> np.ndarray:
        dst = np.empty(n)
        for i in range(n):
            q = i - q_cap
            acc = -diag * q * q * src[i]
            if q + 1 <= q_cap:
                acc += eta * src[i + 1]
            if q - 1 >= -q_cap:
                acc += eta * src[i - 1]
            dst[i] = acc
        return dst

    dt = 0.002
    for _ in range(4000):
        k1 = mv(v)
        k2 = mv(v + 0.5 * dt * k1)
        k3 = mv(v + 0.5 * dt * k2)
        k4 = mv(v + dt * k3)
        v = v + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        scale = float(np.mean(np.abs(v)))
        if scale > 0.0:
            v = v / scale
    omega = -(gamma / k) * np.log(np.maximum(v, 1e-300))
    return offsets_at(cfg, omega, q_cap, inventory)


def reservation_price(mid: float, inventory: int, cfg: QuoterConfig, greeks: Greeks | None = None) -> float:
    o = optimal_offsets(cfg, inventory)
    r = mid + o.reservation_shift
    if greeks is not None and inventory != 0:
        r -= inventory * cfg.gamma_penalty * abs(greeks.gamma)
        r -= inventory * cfg.vega_penalty * abs(greeks.vega) * 0.01
    return r


def optimal_half_spread(cfg: QuoterConfig, inventory: int = 0) -> float:
    half = optimal_offsets(cfg, inventory).half
    return float(min(cfg.max_half_spread, max(cfg.min_half_spread, half)))


def make_quote(
    mid: float,
    inventory: int,
    cfg: QuoterConfig,
    greeks: Greeks | None = None,
    *,
    spread_mult: float = 1.0,
    size_mult: float = 1.0,
) -> Quote:
    if mid <= 0.0:
        mid = max(mid, 0.01)
    o = optimal_offsets(cfg, inventory)
    half = min(cfg.max_half_spread, max(cfg.min_half_spread, o.half * max(spread_mult, 0.25)))
    r = mid + o.reservation_shift
    if greeks is not None and inventory != 0:
        r -= inventory * cfg.gamma_penalty * abs(greeks.gamma)
        r -= inventory * cfg.vega_penalty * abs(greeks.vega) * 0.01
    bid_open = o.delta_b < NO_QUOTE * 0.5
    ask_open = o.delta_a < NO_QUOTE * 0.5
    size = max(1, int(round(cfg.quote_size * max(size_mult, 0.0))))
    bid = max(0.01, r - half) if bid_open else 0.01
    ask = max(bid + 0.01, r + half)
    return Quote(
        bid=bid,
        ask=ask,
        bid_size=size if bid_open else 0,
        ask_size=size if ask_open else 0,
        reservation=r,
        half_spread=half,
    )


def _loglik(obs: list[IntensityObs], A: float, k: float) -> float:
    ll = 0.0
    for o in obs:
        if o.exposure <= 0.0:
            continue
        mean = A * math.exp(-k * o.delta) * o.exposure
        if o.fills > 0.0:
            ll += o.fills * math.log(max(mean, 1e-300))
        ll -= mean
    return ll


def estimate_intensity(obs: list[IntensityObs]) -> IntensityFit:
    """Poisson MLE for λ(δ) = A exp(−k δ), seeded by weighted log regression."""
    sw = swx = swy = swxx = swxy = 0.0
    n_used = 0
    for o in obs:
        if o.exposure <= 0.0:
            continue
        n_used += 1
        if o.fills <= 0.0:
            continue
        y = math.log(o.fills / o.exposure)
        x = -o.delta
        w = o.fills
        sw += w
        swx += w * x
        swy += w * y
        swxx += w * x * x
        swxy += w * x * y
    if n_used == 0 or sw <= 0.0:
        return IntensityFit(0.0, 0.0, 0.0, n_used)
    det = sw * swxx - swx * swx
    ln_a, k = 0.0, 1.0
    if abs(det) > 1e-12:
        ln_a = (swxx * swy - swx * swxy) / det
        k = (sw * swxy - swx * swy) / det
    if k <= 1e-6:
        k = 1e-6
    for _ in range(16):
        A = math.exp(min(20.0, max(-20.0, ln_a)))
        g1 = g2 = h11 = h12 = h22 = 0.0
        for o in obs:
            if o.exposure <= 0.0:
                continue
            mean = A * math.exp(-k * o.delta) * o.exposure
            g1 += o.fills - mean
            g2 += -o.fills * o.delta + mean * o.delta
            h11 += -mean
            h12 += mean * o.delta
            h22 += -mean * o.delta * o.delta
        hdet = h11 * h22 - h12 * h12
        if abs(hdet) <= 1e-18:
            break
        d_ln = (h22 * g1 - h12 * g2) / hdet
        d_k = (-h12 * g1 + h11 * g2) / hdet
        ln_a -= d_ln
        k -= d_k
        if k < 1e-6:
            k = 1e-6
        if abs(d_ln) < 1e-10 and abs(d_k) < 1e-10:
            break
    A = math.exp(min(20.0, max(-20.0, ln_a)))
    return IntensityFit(A, k, _loglik(obs, A, k), n_used)


def asymptotic_pair(cfg: QuoterConfig, inventory: int) -> tuple[float, float]:
    return asymptotic_offsets(cfg, inventory)
