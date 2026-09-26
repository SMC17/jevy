"""Prefer Zig shared lib (`libjev_omm.so`) when present; else pure Python.

Search order for the shared library:
  1. JEV_OMM_LIB env path
  2. jev_omm/native/libjev_omm.so (installed next to package)
  3. zig/zig-out/lib/libjev_omm.so (dev build)
  4. system default loader name libjev_omm.so
"""

from __future__ import annotations

import ctypes
import os
from ctypes import Structure, c_char, c_double, c_int32, POINTER
from pathlib import Path
from typing import Optional

from jev_omm.models.types import Greeks

ZIG_AVAILABLE = False
_lib: Optional[ctypes.CDLL] = None


class _CGreeks(Structure):
    _fields_ = [
        ("delta", c_double),
        ("gamma", c_double),
        ("vega", c_double),
        ("theta", c_double),
    ]


class _CSyntheticEdge(Structure):
    _fields_ = [
        ("package_debit", c_double),
        ("theo_debit", c_double),
        ("edge", c_double),
        ("conversion_edge", c_double),
        ("reversal_edge", c_double),
    ]


class _CBoxResult(Structure):
    _fields_ = [
        ("theo_pv", c_double),
        ("package_debit", c_double),
        ("package_credit", c_double),
        ("buy_edge", c_double),
        ("sell_edge", c_double),
        ("implied_rate_mid", c_double),
        ("implied_rate_buy", c_double),
        ("implied_rate_sell", c_double),
    ]


class _CHedgeConfig(Structure):
    _fields_ = [
        ("delta_band", c_double),
        ("half_spread", c_double),
        ("slip_bps", c_double),
        ("flatten", c_int32),
    ]


class _CHedgeOrder(Structure):
    _fields_ = [
        ("underlier_qty", c_double),
        ("delta_to_hedge", c_double),
        ("fire", c_int32),
        ("_pad", c_int32),
    ]


class _CHedgeFill(Structure):
    _fields_ = [
        ("time", c_double),
        ("underlier_qty", c_double),
        ("fill_price", c_double),
        ("mid", c_double),
        ("cash_delta", c_double),
        ("slippage_cost", c_double),
    ]


class _CGreekPnl(Structure):
    _fields_ = [
        ("spread_capture", c_double),
        ("hedge_slippage", c_double),
        ("gamma_pnl", c_double),
        ("theta_pnl", c_double),
        ("vega_pnl", c_double),
        ("inventory_mtm", c_double),
        ("delta_pnl", c_double),
    ]


class _CPackageTheo(Structure):
    _fields_ = [
        ("theo", c_double),
        ("delta", c_double),
        ("gamma", c_double),
        ("vega", c_double),
        ("theta", c_double),
        ("package_bid", c_double),
        ("package_ask", c_double),
        ("buy_edge", c_double),
        ("sell_edge", c_double),
    ]


def _candidates() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("JEV_OMM_LIB")
    if env:
        out.append(Path(env))
    pkg = Path(__file__).resolve().parents[1]  # jev_omm/
    out.append(pkg / "native" / "libjev_omm.so")
    root = pkg.parent  # repo root
    out.append(root / "zig" / "zig-out" / "lib" / "libjev_omm.so")
    return out


def _load() -> Optional[ctypes.CDLL]:
    for path in _candidates():
        if path.is_file():
            try:
                return ctypes.CDLL(str(path))
            except OSError:
                continue
    try:
        return ctypes.CDLL("libjev_omm.so")
    except OSError:
        return None


def _bind(lib: ctypes.CDLL) -> None:
    lib.jev_omm_price.argtypes = [
        c_double, c_double, c_double, c_double, c_double, c_double, c_int32
    ]
    lib.jev_omm_price.restype = c_double

    lib.jev_omm_greeks.argtypes = [
        c_double, c_double, c_double, c_double, c_double, c_double, c_int32,
        POINTER(_CGreeks),
    ]
    lib.jev_omm_greeks.restype = None

    lib.jev_omm_price_and_greeks.argtypes = [
        c_double, c_double, c_double, c_double, c_double, c_double, c_int32,
        POINTER(_CGreeks),
    ]
    lib.jev_omm_price_and_greeks.restype = c_double

    lib.jev_omm_version.argtypes = []
    lib.jev_omm_version.restype = ctypes.c_char_p

    lib.jev_omm_sabr_iv.argtypes = [
        c_double, c_double, c_double, c_double, c_double, c_double, c_double
    ]
    lib.jev_omm_sabr_iv.restype = c_double

    lib.jev_omm_sabr_atm.argtypes = [
        c_double, c_double, c_double, c_double, c_double, c_double
    ]
    lib.jev_omm_sabr_atm.restype = c_double

    lib.jev_omm_spread_edge.argtypes = [c_int32, c_double, c_double, c_int32]
    lib.jev_omm_spread_edge.restype = c_double

    # --- Akuna-depth exports (0.3.0) ---
    lib.jev_omm_parity_diff.argtypes = [c_double, c_double, c_double, c_double, c_double]
    lib.jev_omm_parity_diff.restype = c_double

    lib.jev_omm_parity_residual.argtypes = [
        c_double, c_double, c_double, c_double, c_double, c_double, c_double
    ]
    lib.jev_omm_parity_residual.restype = c_double

    lib.jev_omm_box_theo.argtypes = [c_double, c_double, c_double, c_double]
    lib.jev_omm_box_theo.restype = c_double

    lib.jev_omm_ww_band.argtypes = [c_double, c_double, c_double, c_double, c_double]
    lib.jev_omm_ww_band.restype = c_double

    lib.jev_omm_scenario_taylor.argtypes = [
        c_double, c_double, c_double, c_double, c_double, c_double
    ]
    lib.jev_omm_scenario_taylor.restype = c_double

    lib.jev_omm_synthetic_edge.argtypes = [
        c_double, c_double, c_double, c_double,
        c_double, c_double, c_double, c_double, c_double,
        c_int32, c_double, c_double, POINTER(_CSyntheticEdge),
    ]
    lib.jev_omm_synthetic_edge.restype = None

    lib.jev_omm_box_spread.argtypes = [
        c_double, c_double, c_double, c_double,
        c_double, c_double, c_double, c_double,
        c_double, c_double, c_double, c_double,
        POINTER(_CBoxResult),
    ]
    lib.jev_omm_box_spread.restype = None

    lib.jev_omm_hedge_propose.argtypes = [c_double, POINTER(_CHedgeConfig), POINTER(_CHedgeOrder)]
    lib.jev_omm_hedge_propose.restype = None

    lib.jev_omm_hedge_apply.argtypes = [
        c_double, c_double, c_double, c_double, c_int32,
        POINTER(_CHedgeConfig), POINTER(_CHedgeFill),
    ]
    lib.jev_omm_hedge_apply.restype = None

    lib.jev_omm_greek_pnl_step.argtypes = [
        c_double, c_double, c_double, c_double,
        c_double, c_double, c_double, c_double, c_double, c_double,
        c_double, c_double, POINTER(_CGreekPnl),
    ]
    lib.jev_omm_greek_pnl_step.restype = None

    lib.jev_omm_straddle_theo.argtypes = [
        c_double, c_double, c_double, c_double, c_double, c_double,
        POINTER(_CPackageTheo),
    ]
    lib.jev_omm_straddle_theo.restype = None

    lib.jev_omm_vertical_call_theo.argtypes = [
        c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double,
        POINTER(_CPackageTheo),
    ]
    lib.jev_omm_vertical_call_theo.restype = None

    lib.jev_omm_butterfly_call_theo.argtypes = [
        c_double, c_double, c_double, c_double, c_double, c_double, c_double,
        c_double, c_double, c_double, POINTER(_CPackageTheo),
    ]
    lib.jev_omm_butterfly_call_theo.restype = None

    if hasattr(lib, "jev_omm_vanna"):
        lib.jev_omm_vanna.argtypes = [
            c_double, c_double, c_double, c_double, c_double, c_double, c_int32
        ]
        lib.jev_omm_vanna.restype = c_double
        lib.jev_omm_volga.argtypes = lib.jev_omm_vanna.argtypes
        lib.jev_omm_volga.restype = c_double


_lib = _load()
if _lib is not None:
    try:
        _bind(_lib)
        ZIG_AVAILABLE = True
    except AttributeError:
        _lib = None
        ZIG_AVAILABLE = False


if ZIG_AVAILABLE:

    def price(
        spot: float,
        strike: float,
        t: float,
        rate: float,
        div_yield: float,
        iv: float,
        is_call: bool,
    ) -> float:
        return float(
            _lib.jev_omm_price(spot, strike, t, rate, div_yield, iv, int(is_call))
        )

    def greeks(
        spot: float,
        strike: float,
        t: float,
        rate: float,
        div_yield: float,
        iv: float,
        is_call: bool,
    ) -> Greeks:
        out = _CGreeks()
        _lib.jev_omm_greeks(spot, strike, t, rate, div_yield, iv, int(is_call), ctypes.byref(out))
        vanna = volga = 0.0
        if hasattr(_lib, "jev_omm_vanna"):
            vanna = float(_lib.jev_omm_vanna(spot, strike, t, rate, div_yield, iv, int(is_call)))
            volga = float(_lib.jev_omm_volga(spot, strike, t, rate, div_yield, iv, int(is_call)))
        return Greeks(delta=out.delta, gamma=out.gamma, vega=out.vega, theta=out.theta, vanna=vanna, volga=volga)

    def price_and_greeks(
        spot: float,
        strike: float,
        t: float,
        rate: float,
        div_yield: float,
        iv: float,
        is_call: bool,
    ) -> tuple[float, Greeks]:
        out = _CGreeks()
        px = _lib.jev_omm_price_and_greeks(
            spot, strike, t, rate, div_yield, iv, int(is_call), ctypes.byref(out)
        )
        return float(px), Greeks(delta=out.delta, gamma=out.gamma, vega=out.vega, theta=out.theta)

    def native_version() -> str:
        return (_lib.jev_omm_version() or b"").decode("utf-8", errors="replace")

    def sabr_iv(
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        forward: float,
        strike: float,
        t: float,
    ) -> float:
        return float(_lib.jev_omm_sabr_iv(alpha, beta, rho, nu, forward, strike, t))

    def sabr_atm(
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        forward: float,
        t: float,
    ) -> float:
        return float(_lib.jev_omm_sabr_atm(alpha, beta, rho, nu, forward, t))

    def spread_edge(is_bid: bool, price: float, mid: float, size: int) -> float:
        return float(_lib.jev_omm_spread_edge(int(is_bid), price, mid, int(size)))

    def parity_diff(spot: float, strike: float, t: float, rate: float, div_yield: float) -> float:
        return float(_lib.jev_omm_parity_diff(spot, strike, t, rate, div_yield))

    def parity_residual(
        call_mid: float, put_mid: float, spot: float, strike: float,
        t: float, rate: float, div_yield: float,
    ) -> float:
        return float(_lib.jev_omm_parity_residual(call_mid, put_mid, spot, strike, t, rate, div_yield))

    def box_theo(k1: float, k2: float, t: float, rate: float) -> float:
        return float(_lib.jev_omm_box_theo(k1, k2, t, rate))

    def synthetic_edge(
        call_bid: float, call_ask: float, put_bid: float, put_ask: float,
        spot: float, strike: float, t: float, rate: float, div_yield: float,
        und_bid: float | None = None, und_ask: float | None = None,
    ) -> dict:
        out = _CSyntheticEdge()
        use = 1 if und_bid is not None and und_ask is not None else 0
        _lib.jev_omm_synthetic_edge(
            call_bid, call_ask, put_bid, put_ask, spot, strike, t, rate, div_yield,
            use, float(und_bid or 0.0), float(und_ask or 0.0), ctypes.byref(out),
        )
        return {
            "package_debit": out.package_debit,
            "theo_debit": out.theo_debit,
            "edge": out.edge,
            "conversion_edge": out.conversion_edge,
            "reversal_edge": out.reversal_edge,
        }

    def box_spread(
        c1_bid: float, c1_ask: float, c2_bid: float, c2_ask: float,
        p1_bid: float, p1_ask: float, p2_bid: float, p2_ask: float,
        k1: float, k2: float, t: float, rate: float,
    ) -> dict:
        out = _CBoxResult()
        _lib.jev_omm_box_spread(
            c1_bid, c1_ask, c2_bid, c2_ask, p1_bid, p1_ask, p2_bid, p2_ask,
            k1, k2, t, rate, ctypes.byref(out),
        )
        return {f: getattr(out, f) for f, _ in _CBoxResult._fields_}

    def hedge_propose(net_delta: float, delta_band: float = 5.0, half_spread: float = 0.0,
                      slip_bps: float = 1.0, flatten: bool = True) -> dict:
        cfg = _CHedgeConfig(delta_band, half_spread, slip_bps, int(flatten))
        out = _CHedgeOrder()
        _lib.jev_omm_hedge_propose(net_delta, ctypes.byref(cfg), ctypes.byref(out))
        return {
            "underlier_qty": out.underlier_qty,
            "delta_to_hedge": out.delta_to_hedge,
            "fire": bool(out.fire),
        }

    def hedge_apply(time: float, mid: float, underlier_qty: float, delta_to_hedge: float,
                    fire: bool, delta_band: float = 5.0, half_spread: float = 0.0,
                    slip_bps: float = 1.0, flatten: bool = True) -> dict:
        cfg = _CHedgeConfig(delta_band, half_spread, slip_bps, int(flatten))
        out = _CHedgeFill()
        _lib.jev_omm_hedge_apply(
            time, mid, underlier_qty, delta_to_hedge, int(fire),
            ctypes.byref(cfg), ctypes.byref(out),
        )
        return {f: getattr(out, f) for f, _ in _CHedgeFill._fields_}

    def greek_pnl_step(
        delta: float, gamma: float, vega: float, theta: float,
        d_spot: float, d_sigma: float, dt: float,
        option_qty: float, d_option_mid: float, underlier_pos: float,
        spread_capture: float = 0.0, hedge_slippage: float = 0.0,
    ) -> dict:
        out = _CGreekPnl()
        _lib.jev_omm_greek_pnl_step(
            delta, gamma, vega, theta, d_spot, d_sigma, dt,
            option_qty, d_option_mid, underlier_pos,
            spread_capture, hedge_slippage, ctypes.byref(out),
        )
        return {f: getattr(out, f) for f, _ in _CGreekPnl._fields_}

    def straddle_theo(spot: float, strike: float, t: float, rate: float,
                      div_yield: float, iv: float) -> dict:
        out = _CPackageTheo()
        _lib.jev_omm_straddle_theo(spot, strike, t, rate, div_yield, iv, ctypes.byref(out))
        return {f: getattr(out, f) for f, _ in _CPackageTheo._fields_}

    def ww_band(spot: float, gamma_abs: float, sigma: float,
                slip_frac: float, risk_aversion: float) -> float:
        return float(_lib.jev_omm_ww_band(spot, gamma_abs, sigma, slip_frac, risk_aversion))

    def scenario_taylor(delta: float, gamma: float, vega: float, theta: float,
                        d_spot: float, d_iv: float) -> float:
        return float(_lib.jev_omm_scenario_taylor(delta, gamma, vega, theta, d_spot, d_iv))

else:
    from jev_omm.pricing.black_scholes import (  # noqa: F401
        greeks,
        price,
        price_and_greeks,
    )

    def native_version() -> str:
        return "python"

    def sabr_iv(*_a, **_k):  # type: ignore[no-untyped-def]
        raise NotImplementedError("Zig lib required for native sabr_iv")

    def sabr_atm(*_a, **_k):  # type: ignore[no-untyped-def]
        raise NotImplementedError("Zig lib required for native sabr_atm")

    def spread_edge(*_a, **_k):  # type: ignore[no-untyped-def]
        raise NotImplementedError("Zig lib required for native spread_edge")


# Back-compat alias used briefly during Rust spike
RUST_AVAILABLE = False

__all__ = [
    "ZIG_AVAILABLE",
    "RUST_AVAILABLE",
    "price",
    "greeks",
    "price_and_greeks",
    "sabr_iv",
    "sabr_atm",
    "spread_edge",
    "native_version",
    "parity_diff",
    "parity_residual",
    "box_theo",
    "synthetic_edge",
    "box_spread",
    "hedge_propose",
    "hedge_apply",
    "greek_pnl_step",
    "straddle_theo",
    "ww_band",
    "scenario_taylor",
]
