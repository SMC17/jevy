"""Raw SVI and SSVI total variance, arbitrage checks, sticky regimes.

Mirrors ``zig/src/svi.zig``. SABR-lite stays in ``sabr.py``.

Cite: Gatheral & Jacquier, https://arxiv.org/abs/1204.0646
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class StickyRegime(str, Enum):
    STICKY_STRIKE = "sticky_strike"
    STICKY_DELTA = "sticky_delta"


@dataclass
class SviParams:
    a: float = 0.04
    b: float = 0.1
    rho: float = -0.4
    m: float = 0.0
    sigma: float = 0.2


@dataclass
class SsviParams:
    rho: float = -0.4
    eta: float = 0.8
    gamma: float = 0.4


@dataclass
class ButterflyReport:
    ok: bool
    min_g: float
    min_w: float
    lee_slope: float
    lee_ok: bool
    w_positive: bool


@dataclass
class FitResult:
    params: SviParams
    rmse: float
    max_abs_residual: float
    butterfly: ButterflyReport
    surface_suspect: bool


_K_LO = -1.5
_K_HI = 1.5
_K_STEPS = 61
_LEE_MAX = 2.0


def total_var(p: SviParams, k: float) -> float:
    km = k - p.m
    return p.a + p.b * (p.rho * km + math.sqrt(km * km + p.sigma * p.sigma))


def min_total_var(p: SviParams) -> float:
    return p.a + p.b * p.sigma * math.sqrt(max(1.0 - p.rho * p.rho, 0.0))


def _derivs(p: SviParams, k: float) -> tuple[float, float, float]:
    km = k - p.m
    sig2 = p.sigma * p.sigma
    s = math.sqrt(km * km + sig2)
    s_safe = max(s, 1e-14)
    w = p.a + p.b * (p.rho * km + s)
    wp = p.b * (p.rho + km / s_safe)
    wpp = p.b * sig2 / (s_safe**3)
    return w, wp, wpp


def density_g(p: SviParams, k: float) -> float:
    """Gatheral–Jacquier g(k). Non-negative ⇔ no butterfly arbitrage at k."""
    w, wp, wpp = _derivs(p, k)
    if w <= 1e-12:
        return -1e9
    term1 = 1.0 - k * wp / (2.0 * w)
    return term1 * term1 - (wp * wp) / 4.0 * (1.0 / w + 0.25) + wpp / 2.0


def lee_slope(p: SviParams) -> float:
    return p.b * (1.0 + abs(p.rho))


def _k_grid() -> np.ndarray:
    return np.linspace(_K_LO, _K_HI, _K_STEPS)


def butterfly_check(p: SviParams) -> ButterflyReport:
    slope = lee_slope(p)
    lee_ok = slope <= _LEE_MAX + 1e-9 and p.b >= 0.0 and abs(p.rho) < 1.0 and p.sigma > 0.0
    ks = _k_grid()
    ws = np.array([total_var(p, float(k)) for k in ks])
    gs = np.array([density_g(p, float(k)) for k in ks])
    min_w = float(min(ws.min(), min_total_var(p)))
    min_g = float(gs.min())
    w_positive = min_w > 1e-10
    ok = bool(lee_ok and w_positive and min_g >= -1e-8)
    return ButterflyReport(ok, min_g, min_w, slope, lee_ok, w_positive)


def iv_from_total_var(w: float, t: float) -> float:
    if t <= 0.0 or w <= 0.0:
        return 0.0
    return math.sqrt(w / t)


def implied_vol(p: SviParams, k: float, t: float) -> float:
    return iv_from_total_var(total_var(p, k), t)


def ssvi_phi(theta: float, p: SsviParams) -> float:
    if theta <= 0.0 or p.eta <= 0.0:
        return 0.0
    g = min(1.0, max(0.0, p.gamma))
    denom = (theta**g) * ((1.0 + theta) ** (1.0 - g))
    if denom <= 0.0:
        return 0.0
    return p.eta / denom


def ssvi_total_var(k: float, theta: float, p: SsviParams) -> float:
    if theta <= 0.0:
        return 0.0
    phi = ssvi_phi(theta, p)
    rho = max(-0.999999, min(0.999999, p.rho))
    x = phi * k + rho
    disc = x * x + (1.0 - rho * rho)
    return 0.5 * theta * (1.0 + rho * phi * k + math.sqrt(max(disc, 0.0)))


def ssvi_params_calendar_safe(p: SsviParams) -> bool:
    if p.eta < 0.0 or not (0.0 <= p.gamma <= 1.0) or abs(p.rho) >= 1.0:
        return False
    return p.eta * (1.0 + abs(p.rho)) <= _LEE_MAX + 1e-12


def ssvi_calendar_ok(theta1: float, theta2: float, p: SsviParams) -> bool:
    lo, hi = (theta1, theta2) if theta1 <= theta2 else (theta2, theta1)
    for k in _k_grid():
        if ssvi_total_var(float(k), hi, p) + 1e-9 < ssvi_total_var(float(k), lo, p):
            return False
    return True


def raw_calendar_ok(earlier: SviParams, later: SviParams) -> bool:
    for k in _k_grid():
        if total_var(later, float(k)) + 1e-9 < total_var(earlier, float(k)):
            return False
    return True


def implied_vol_after_move(
    p: SviParams,
    forward0: float,
    forward1: float,
    strike: float,
    t: float,
    regime: StickyRegime | str,
) -> float:
    if forward0 <= 0.0 or forward1 <= 0.0 or strike <= 0.0:
        return 0.0
    reg = StickyRegime(regime)
    if reg is StickyRegime.STICKY_STRIKE:
        k = math.log(strike / forward0)
    else:
        k = math.log(strike / forward1)
    return implied_vol(p, k, t)


def _unpack(y: np.ndarray) -> SviParams:
    return SviParams(
        a=float(y[0]),
        b=math.exp(min(4.0, max(-12.0, float(y[1])))),
        rho=math.tanh(float(y[2])),
        m=float(y[3]),
        sigma=math.exp(min(3.0, max(-12.0, float(y[4])))),
    )


def _loss(y: np.ndarray, ks: np.ndarray, ws: np.ndarray) -> float:
    p = _unpack(y)
    err = 0.0
    for k, w in zip(ks, ws):
        d = total_var(p, float(k)) - float(w)
        err += d * d
    slope = lee_slope(p)
    if slope > _LEE_MAX:
        err += (slope - _LEE_MAX) ** 2 * 10.0
    wmin = min_total_var(p)
    if wmin < 0.0:
        err += wmin * wmin * 100.0
    return err


def _nelder(ks: np.ndarray, ws: np.ndarray, x0: np.ndarray, iters: int = 450) -> np.ndarray:
    n = 5
    simplex = [x0.astype(float).copy()]
    for i in range(n):
        y = x0.astype(float).copy()
        bump = 0.08 * max(1.0, abs(y[i]))
        y[i] += 0.08 if abs(y[i]) < 1e-8 else bump
        simplex.append(y)
    vals = [_loss(s, ks, ws) for s in simplex]
    for _ in range(iters):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        xr = centroid + (centroid - worst)
        fr = _loss(xr, ks, ws)
        if vals[0] <= fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
            continue
        if fr < vals[0]:
            xe = centroid + 2.0 * (xr - centroid)
            fe = _loss(xe, ks, ws)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
            continue
        xc = centroid + 0.5 * (worst - centroid)
        fc = _loss(xc, ks, ws)
        if fc < vals[-1]:
            simplex[-1], vals[-1] = xc, fc
            continue
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
            vals[i] = _loss(simplex[i], ks, ws)
    best = int(np.argmin(vals))
    return simplex[best]


def calibrate(ks: list[float] | np.ndarray, ws: list[float] | np.ndarray) -> FitResult:
    ks_a = np.asarray(ks, dtype=float)
    ws_a = np.asarray(ws, dtype=float)
    imin = int(np.argmin(ws_a))
    wmin_obs = float(ws_a[imin])
    rho0 = -0.25
    y0 = np.array(
        [
            max(wmin_obs * 0.5, 1e-4),
            math.log(0.15),
            0.5 * math.log((1.0 + rho0) / (1.0 - rho0)),
            float(ks_a[imin]),
            math.log(0.2),
        ]
    )
    y = _nelder(ks_a, ws_a, y0)
    params = _unpack(y)
    resid = np.array([total_var(params, float(k)) - float(w) for k, w in zip(ks_a, ws_a)])
    rmse = float(np.sqrt(np.mean(resid * resid))) if len(resid) else 0.0
    max_abs = float(np.max(np.abs(resid))) if len(resid) else 0.0
    bf = butterfly_check(params)
    suspect = (not bf.ok) or rmse > 0.02 or max_abs > 0.05
    return FitResult(params, rmse, max_abs, bf, suspect)


def assess(
    p: SviParams,
    ks: list[float] | np.ndarray | None = None,
    ws: list[float] | np.ndarray | None = None,
    calendar_ok: bool = True,
) -> dict:
    bf = butterfly_check(p)
    rmse = 0.0
    max_abs = 0.0
    if ks is not None and ws is not None and len(ks) == len(ws) and len(ks) > 0:
        resid = np.array([total_var(p, float(k)) - float(w) for k, w in zip(ks, ws)])
        rmse = float(np.sqrt(np.mean(resid * resid)))
        max_abs = float(np.max(np.abs(resid)))
    suspect = (not bf.ok) or (not calendar_ok) or rmse > 0.02 or max_abs > 0.05
    return {
        "butterfly_ok": bf.ok,
        "calendar_ok": calendar_ok,
        "min_g": bf.min_g,
        "rmse": rmse,
        "max_abs_residual": max_abs,
        "surface_suspect": suspect,
    }
