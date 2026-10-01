"""Residual smoothness. A leftover that is mostly its mean is not a Sharpe.

After the greek strip, each sleeve reports:

- lag-1 autocorrelation of the demeaned residual (AC(1))
- share of uncentered energy in the lowest Fourier bin (the sample mean)
- R² of a constant-plus-trend fit against a zero baseline
- share of demeaned power in the lowest positive Fourier bin

The penalty is in ``[0, 1]``. It is 0 when the residual is a constant the
strip created (sample σ under 1e-5, or DC share at least 0.95 and at least
0.40 above the raw series). It is in ``(0, 1)`` when the residual is smooth
relative to raw PnL. It is 1 otherwise, which is the identity.

The raw per-step Sharpe is mean / sample std. It is not annualized and it
is not a capacity. Multiply it by the penalty before reading it as an edge.

Zig: ``desk.smoothnessPenalty``. Paper only.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np

from jev_omm.pricing import _native

_FLAT_STD = 1e-5

FLAG_OK = 0
FLAG_SMOOTH = 1
FLAG_FLAT = 2
FLAG_NAME = {FLAG_OK: "ok", FLAG_SMOOTH: "smooth", FLAG_FLAT: "flat"}


def _mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sum(x) / x.size)


def residual_ac1(x: np.ndarray | list[float]) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n < 3:
        return 0.0
    m = _mean(arr)
    den = 0.0
    num = 0.0
    for i in range(n):
        d = float(arr[i]) - m
        den += d * d
        if i + 1 < n:
            num += d * (float(arr[i + 1]) - m)
    if den <= 1e-18:
        return 0.0
    return num / den


def dc_share(x: np.ndarray | list[float]) -> float:
    """Fraction of ``sum(x^2)`` sitting in the sample mean (DC bin)."""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n == 0:
        return 0.0
    ss = float(np.dot(arr, arr))
    if ss <= 1e-18:
        return 0.0
    m = _mean(arr)
    return n * m * m / ss


def const_trend_r2(x: np.ndarray | list[float]) -> float:
    """R² of ``x ~ a + b t`` against a zero baseline, ``t = 0..n-1``."""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n < 3:
        return 0.0
    nf = float(n)
    sum_t = nf * (nf - 1.0) / 2.0
    sum_t2 = (nf - 1.0) * nf * (2.0 * nf - 1.0) / 6.0
    idx = np.arange(n, dtype=np.float64)
    sum_x = float(np.sum(arr))
    sum_tx = float(np.dot(idx, arr))
    det = nf * sum_t2 - sum_t * sum_t
    if abs(det) < 1e-18:
        return 0.0
    a = (sum_t2 * sum_x - sum_t * sum_tx) / det
    b = (nf * sum_tx - sum_t * sum_x) / det
    err = arr - (a + b * idx)
    ss = float(np.dot(arr, arr))
    if ss <= 1e-18:
        return 0.0
    return 1.0 - float(np.dot(err, err)) / ss


def low_freq_share(x: np.ndarray | list[float]) -> float:
    """Power in Fourier bin k=1 over bins ``1 .. n//2``, after demeaning."""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n < 4:
        return 0.0
    m = _mean(arr)
    centered = arr - m
    nbin = n // 2
    total = 0.0
    first = 0.0
    idx = np.arange(n, dtype=np.float64)
    for k in range(1, nbin + 1):
        ang = 2.0 * math.pi * k / float(n)
        theta = ang * idx
        re = float(np.dot(centered, np.cos(theta)))
        im = float(-np.dot(centered, np.sin(theta)))
        pwr = re * re + im * im
        if k == 1:
            first = pwr
        total += pwr
    if total <= 1e-18:
        return 0.0
    return first / total


def _sample_std(x: np.ndarray) -> float:
    n = int(x.size)
    if n < 2:
        return 0.0
    m = _mean(x)
    ss = float(np.dot(x - m, x - m))
    return math.sqrt(max(ss / (n - 1), 0.0))


def smoothness_penalty_python(resid: np.ndarray, raw: np.ndarray) -> tuple[float, int, dict[str, float]]:
    stats = {
        "ac1": residual_ac1(resid),
        "dc_share": dc_share(resid),
        "const_trend_r2": const_trend_r2(resid),
        "low_freq_share": low_freq_share(resid),
        "raw_ac1": residual_ac1(raw),
        "raw_dc_share": dc_share(raw),
        "raw_const_trend_r2": const_trend_r2(raw),
        "raw_low_freq_share": low_freq_share(raw),
    }
    if _sample_std(np.asarray(resid, dtype=np.float64)) < _FLAT_STD:
        return 0.0, FLAG_FLAT, stats
    if stats["dc_share"] >= 0.95 and (stats["dc_share"] - stats["raw_dc_share"]) >= 0.40:
        return 0.0, FLAG_FLAT, stats
    penalty = 1.0
    flag = FLAG_OK
    if stats["dc_share"] >= 0.85 and (stats["dc_share"] - stats["raw_dc_share"]) >= 0.25:
        span = min(max((stats["dc_share"] - 0.85) / 0.10, 0.0), 1.0)
        penalty = min(penalty, max(0.05, 1.0 - 0.95 * span))
        flag = FLAG_SMOOTH
    if stats["ac1"] >= 0.70 and (stats["ac1"] - stats["raw_ac1"]) >= 0.30:
        span = min(max((stats["ac1"] - 0.70) / 0.25, 0.0), 1.0)
        penalty = min(penalty, max(0.05, 1.0 - 0.90 * span))
        flag = FLAG_SMOOTH
    if stats["low_freq_share"] >= 0.45 and (stats["low_freq_share"] - stats["raw_low_freq_share"]) >= 0.20:
        span = min(max((stats["low_freq_share"] - 0.45) / 0.40, 0.0), 1.0)
        penalty = min(penalty, max(0.05, 1.0 - 0.80 * span))
        flag = FLAG_SMOOTH
    if stats["const_trend_r2"] >= 0.90 and (stats["const_trend_r2"] - stats["raw_const_trend_r2"]) >= 0.40:
        span = min(max((stats["const_trend_r2"] - 0.90) / 0.08, 0.0), 1.0)
        penalty = min(penalty, max(0.05, 1.0 - 0.90 * span))
        flag = FLAG_SMOOTH
    return penalty, flag, stats


def _smoothness_zig(resid: np.ndarray, raw: np.ndarray) -> tuple[float, int, dict[str, float]] | None:
    lib = _native._lib
    if not _native.ZIG_AVAILABLE or lib is None or not hasattr(lib, "jev_omm_smoothness"):
        return None
    n = int(resid.size)
    if int(raw.size) != n:
        return None
    if not hasattr(lib, "_jev_smooth_bound"):
        lib.jev_omm_smoothness.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.jev_omm_smoothness.restype = None
        lib._jev_smooth_bound = True
    ac1 = ctypes.c_double()
    dc = ctypes.c_double()
    ct = ctypes.c_double()
    lf = ctypes.c_double()
    penalty = ctypes.c_double()
    flag = ctypes.c_int32()
    lib.jev_omm_smoothness(
        n,
        np.ascontiguousarray(resid, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        np.ascontiguousarray(raw, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(ac1),
        ctypes.byref(dc),
        ctypes.byref(ct),
        ctypes.byref(lf),
        ctypes.byref(penalty),
        ctypes.byref(flag),
    )
    # Raw-series diagnostics stay on the Python side; the penalty matches Zig.
    py_pen, _py_flag, stats = smoothness_penalty_python(resid, raw)
    stats["ac1"] = float(ac1.value)
    stats["dc_share"] = float(dc.value)
    stats["const_trend_r2"] = float(ct.value)
    stats["low_freq_share"] = float(lf.value)
    _ = py_pen
    return float(penalty.value), int(flag.value), stats


def smoothness_penalty(
    resid: np.ndarray | list[float],
    raw: np.ndarray | list[float],
    *,
    prefer_zig: bool = True,
) -> tuple[float, str, dict[str, float]]:
    y = np.ascontiguousarray(np.asarray(resid, dtype=np.float64).reshape(-1))
    r = np.ascontiguousarray(np.asarray(raw, dtype=np.float64).reshape(-1))
    if y.size != r.size:
        raise ValueError(f"resid length {y.size} != raw length {r.size}")
    if prefer_zig:
        zig = _smoothness_zig(y, r)
        if zig is not None:
            penalty, flag, stats = zig
            return penalty, FLAG_NAME.get(flag, "ok"), stats
    penalty, flag, stats = smoothness_penalty_python(y, r)
    return penalty, FLAG_NAME.get(flag, "ok"), stats
