"""Residual PnL after stripping spot beta, gamma, and optional vega.

The gamma column is the greek-PnL bucket from ``hedge.greek_pnl_step``:

    gamma_pnl = 0.5 * Γ * (ΔS)^2

The beta column is the simple return ΔS/S. The vega column, when passed, is
ν * Δσ, the same bucket as ``greek_pnl_step``.

Slopes match an intercept regression: the factors are demeaned before the
Gram matrix is formed, and a 1e-12 ridge matches ``zig/src/desk.zig``.
The stored residual subtracts ``β̂ f_β + γ̂ f_Γ + ν̂ f_ν`` and does **not**
subtract the intercept, so a constant premium stays in the residual mean.
R² is 1 − SS(resid − mean) / SS(raw − mean). A flat series has R² 0.

Paper / simulation only. A high R² means the factors explain the variance of
this sample. It is not a claim about a live book.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from jev_omm.hedge.delta import greek_pnl_step
from jev_omm.pricing import _native

RIDGE = 1e-12


@dataclass
class StripFit:
    beta: float
    gamma_coef: float
    vega_coef: float
    r2: float
    residual: np.ndarray
    n_factors: int
    volga_coef: float = 0.0
    vanna_coef: float = 0.0
    var_coef: float = 0.0
    coefs: tuple[float, ...] = ()

    @property
    def mean_residual(self) -> float:
        if self.residual.size == 0:
            return 0.0
        return float(np.mean(self.residual))


def beta_factor(d_spot: float, spot: float) -> float:
    if not (spot > 0.0):
        return 0.0
    return d_spot / spot


def gamma_factor(gamma: float, d_spot: float) -> float:
    """Match ``greek_pnl_step``'s gamma bucket."""
    return 0.5 * gamma * d_spot * d_spot


def vega_factor(vega: float, d_sigma: float) -> float:
    return vega * d_sigma


def _as_1d(x: np.ndarray | list[float], name: str, n: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if n is not None and arr.size != n:
        raise ValueError(f"{name} length {arr.size} != {n}")
    return arr


def _fit_from_coefs(raw: np.ndarray, cols: list[np.ndarray], coef: np.ndarray) -> StripFit:
    n = int(raw.size)
    k = len(cols)
    if n == 0 or k == 0:
        return StripFit(0.0, 0.0, 0.0, 0.0, np.zeros(n), 0)
    yhat = np.zeros(n)
    for j in range(k):
        yhat = yhat + float(coef[j]) * cols[j]
    resid = raw - yhat
    mean_r = float(np.mean(raw))
    centered_e = resid - float(np.mean(resid))
    centered = raw - mean_r
    ss_res = float(np.dot(centered_e, centered_e))
    ss_tot = float(np.dot(centered, centered))
    r2 = 0.0 if ss_tot <= 1e-18 else 1.0 - ss_res / ss_tot
    packed = tuple(float(c) for c in coef)

    def _at(i: int) -> float:
        return float(coef[i]) if i < k else 0.0

    return StripFit(
        beta=_at(0),
        gamma_coef=_at(1),
        vega_coef=_at(2),
        r2=float(r2),
        residual=resid,
        n_factors=k,
        volga_coef=_at(3),
        vanna_coef=_at(4),
        var_coef=_at(5),
        coefs=packed,
    )


def _solve_coefs(raw: np.ndarray, cols: list[np.ndarray]) -> np.ndarray:
    n = int(raw.size)
    k = len(cols)
    if n == 0 or k == 0:
        return np.zeros(k)
    means = [float(np.mean(c)) for c in cols]
    mean_r = float(np.mean(raw))
    xtx = np.zeros((k, k))
    xty = np.zeros(k)
    for i in range(k):
        xty[i] = float(np.dot(cols[i], raw) - n * means[i] * mean_r)
        for j in range(k):
            xtx[i, j] = float(np.dot(cols[i], cols[j]) - n * means[i] * means[j])
        xtx[i, i] += RIDGE
    try:
        return np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return np.zeros(k)


def _strip_python(
    raw: np.ndarray,
    f_beta: np.ndarray,
    f_gamma: np.ndarray,
    f_vega: np.ndarray | None,
) -> StripFit:
    n = int(raw.size)
    if n == 0:
        return StripFit(0.0, 0.0, 0.0, 0.0, np.zeros(0), 0)
    cols = [f_beta, f_gamma] if f_vega is None else [f_beta, f_gamma, f_vega]
    coef = _solve_coefs(raw, cols)
    return _fit_from_coefs(raw, cols, coef)


def _strip_zig(
    raw: np.ndarray,
    f_beta: np.ndarray,
    f_gamma: np.ndarray,
    f_vega: np.ndarray | None,
) -> StripFit | None:
    lib = _native._lib
    if not _native.ZIG_AVAILABLE or lib is None or not hasattr(lib, "jev_omm_residual_strip"):
        return None
    n = int(raw.size)
    if not hasattr(lib, "_jev_residual_bound"):
        lib.jev_omm_residual_strip.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.jev_omm_residual_strip.restype = None
        lib._jev_residual_bound = True
    out = np.zeros(n, dtype=np.float64)
    beta = ctypes.c_double()
    gamma_c = ctypes.c_double()
    vega_c = ctypes.c_double()
    r2 = ctypes.c_double()
    use = 1 if f_vega is not None else 0
    vega_arr = f_vega if f_vega is not None else f_beta
    lib.jev_omm_residual_strip(
        n,
        raw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        f_beta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        f_gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        np.ascontiguousarray(vega_arr).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        use,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(beta),
        ctypes.byref(gamma_c),
        ctypes.byref(vega_c),
        ctypes.byref(r2),
    )
    return StripFit(
        beta=float(beta.value),
        gamma_coef=float(gamma_c.value),
        vega_coef=float(vega_c.value),
        r2=float(r2.value),
        residual=out,
        n_factors=3 if use else 2,
    )


def strip_residual(
    raw: np.ndarray | list[float],
    f_beta: np.ndarray | list[float],
    f_gamma: np.ndarray | list[float],
    f_vega: np.ndarray | list[float] | None = None,
    *,
    prefer_zig: bool = True,
) -> StripFit:
    """Project ``raw`` onto beta, gamma, and optional vega. Return the residual."""
    y = _as_1d(raw, "raw")
    fb = _as_1d(f_beta, "f_beta", y.size)
    fg = _as_1d(f_gamma, "f_gamma", y.size)
    fv = None if f_vega is None else _as_1d(f_vega, "f_vega", y.size)
    y = np.ascontiguousarray(y, dtype=np.float64)
    fb = np.ascontiguousarray(fb, dtype=np.float64)
    fg = np.ascontiguousarray(fg, dtype=np.float64)
    if fv is not None:
        fv = np.ascontiguousarray(fv, dtype=np.float64)
    if prefer_zig:
        zig = _strip_zig(y, fb, fg, fv)
        if zig is not None:
            return zig
    return _strip_python(y, fb, fg, fv)


MAX_FACTORS = 6


def _strip_zig_k(raw: np.ndarray, cols: list[np.ndarray]) -> StripFit | None:
    lib = _native._lib
    if not _native.ZIG_AVAILABLE or lib is None or not hasattr(lib, "jev_omm_residual_strip_k"):
        return None
    n = int(raw.size)
    k = len(cols)
    if k < 1 or k > MAX_FACTORS:
        return None
    if not hasattr(lib, "_jev_residual_k_bound"):
        lib.jev_omm_residual_strip_k.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.jev_omm_residual_strip_k.restype = None
        lib._jev_residual_k_bound = True
    packed = np.ascontiguousarray(np.concatenate(cols), dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    coef = np.zeros(k, dtype=np.float64)
    r2 = ctypes.c_double()
    lib.jev_omm_residual_strip_k(
        n,
        k,
        raw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coef.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(r2),
    )
    fit = _fit_from_coefs(raw, cols, coef)
    # Trust the Zig residual and R². Coefs come from the same solve.
    fit.residual = out
    fit.r2 = float(r2.value)
    return fit


def strip_factors(
    raw: np.ndarray | list[float],
    factors: list[np.ndarray | list[float]],
    *,
    prefer_zig: bool = True,
) -> StripFit:
    """Project ``raw`` onto an ordered factor list. At most six columns.

    Column order used by the desk is beta, spot-gamma, vega, volga, vanna,
    and variance (quadratic variation orthogonal to spot-gamma). Fewer
    columns are allowed. The intercept stays in the residual mean.
    """
    y = np.ascontiguousarray(_as_1d(raw, "raw"), dtype=np.float64)
    cols = [np.ascontiguousarray(_as_1d(col, f"f{i}", y.size), dtype=np.float64) for i, col in enumerate(factors)]
    if len(cols) > MAX_FACTORS:
        raise ValueError(f"at most {MAX_FACTORS} factors, got {len(cols)}")
    if prefer_zig:
        zig = _strip_zig_k(y, cols)
        if zig is not None:
            return zig
    if y.size == 0:
        return StripFit(0.0, 0.0, 0.0, 0.0, np.zeros(0), 0)
    coef = _solve_coefs(y, cols)
    return _fit_from_coefs(y, cols, coef)


def factors_from_greeks(
    *,
    delta: float,
    gamma: float,
    vega: float,
    d_spot: float,
    spot: float,
    d_sigma: float = 0.0,
    theta: float = 0.0,
    dt: float = 0.0,
) -> dict[str, float]:
    """Greek buckets plus the unitless spot return used as the beta factor.

    ``gamma_pnl`` / ``vega_pnl`` / ``delta_pnl`` come from ``greek_pnl_step``
    so a residual built on these columns lines up with the event-log buckets.
    """
    step = greek_pnl_step(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        d_spot=d_spot,
        d_sigma=d_sigma,
        dt=dt,
    )
    return {
        "f_beta": beta_factor(d_spot, spot),
        "f_gamma": step.gamma_pnl,
        "f_vega": step.vega_pnl,
        "delta_pnl": step.delta_pnl,
        "gamma_pnl": step.gamma_pnl,
        "vega_pnl": step.vega_pnl,
        "theta_pnl": step.theta_pnl,
    }


def pearson(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        return 0.0
    if float(np.std(x)) <= 1e-15 or float(np.std(y)) <= 1e-15:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _average_ranks(x: np.ndarray) -> np.ndarray:
    n = int(x.size)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        return 0.0
    return pearson(_average_ranks(x), _average_ranks(y))
