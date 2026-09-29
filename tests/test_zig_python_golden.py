"""Randomized Zig ↔ Python comparisons for mirrored kernels.

Tolerances (absolute unless noted):
  BS price 1e-4 (scipy ``norm.cdf`` vs Zig's approximation)
  BS delta 1e-4, gamma 1e-4, vega 1e-3, theta 1e-3
  A–S / Guéant closed form 1e-8
  SVI total variance 1e-10, density and IV 1e-8
  state gate 1e-12
  Guéant ODE offsets 1e-6
  risk gate: quoting flag and the first token of the reason

CI sets JEV_OMM_REQUIRE_NATIVE=1 so a missing libjev_omm.so fails the job
instead of skipping.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from ctypes import POINTER, c_double, c_int32, c_char, Structure

from jev_omm.config import QuoterConfig, RiskConfig
from jev_omm.models.types import Greeks
from jev_omm.pricing import _native
from jev_omm.pricing.black_scholes import greeks as py_greeks
from jev_omm.pricing.black_scholes import price as py_price
from jev_omm.quoter.avellaneda_stoikov import optimal_half_spread, reservation_price
from jev_omm.quoter.gueant import inventory_scale, optimal_half_spread as gueant_half
from jev_omm.quoter.gueant_ode import optimal_offsets
from jev_omm.risk.limits import evaluate_risk
from jev_omm.state_os.gate import state_gate
from jev_omm.surface.svi import SsviParams, SviParams, density_g, implied_vol, ssvi_total_var, total_var


def _lib():
    lib = _native._lib
    if lib is None or not _native.ZIG_AVAILABLE:
        if os.environ.get("JEV_OMM_REQUIRE_NATIVE") == "1":
            pytest.fail("libjev_omm.so is required (JEV_OMM_REQUIRE_NATIVE=1) but was not loaded")
        pytest.skip("libjev_omm.so not built")
    return lib


class _CQuoter(Structure):
    _fields_ = [
        ("gamma", c_double),
        ("kappa", c_double),
        ("sigma", c_double),
        ("t_horizon", c_double),
        ("gamma_penalty", c_double),
        ("vega_penalty", c_double),
        ("min_half_spread", c_double),
        ("max_half_spread", c_double),
        ("quote_size", c_int32),
        ("mode", c_int32),
        ("A", c_double),
        ("portfolio_delta_penalty", c_double),
    ]


class _CStateGate(Structure):
    _fields_ = [
        ("reservation_shift", c_double),
        ("spread_mult", c_double),
        ("size_mult", c_double),
        ("hedge_urgency", c_double),
        ("pull", c_int32),
        ("instability", c_double),
    ]


def _bind(lib):
    lib.jev_omm_reservation_price.argtypes = [
        c_double, c_int32, POINTER(_CQuoter), c_double, c_int32, c_double, c_double, c_int32,
    ]
    lib.jev_omm_reservation_price.restype = c_double
    lib.jev_omm_optimal_half_spread.argtypes = [POINTER(_CQuoter), c_double, c_int32]
    lib.jev_omm_optimal_half_spread.restype = c_double
    lib.jev_omm_gueant_inventory_scale.argtypes = [POINTER(_CQuoter)]
    lib.jev_omm_gueant_inventory_scale.restype = c_double
    lib.jev_omm_gueant_half_spread.argtypes = [POINTER(_CQuoter)]
    lib.jev_omm_gueant_half_spread.restype = c_double
    lib.jev_omm_svi_total_var.argtypes = [c_double] * 6
    lib.jev_omm_svi_total_var.restype = c_double
    lib.jev_omm_svi_density_g.argtypes = [c_double] * 6
    lib.jev_omm_svi_density_g.restype = c_double
    lib.jev_omm_svi_iv.argtypes = [c_double] * 7
    lib.jev_omm_svi_iv.restype = c_double
    lib.jev_omm_ssvi_total_var.argtypes = [c_double] * 5
    lib.jev_omm_ssvi_total_var.restype = c_double
    lib.jev_omm_svi_butterfly_ok.argtypes = [c_double] * 5
    lib.jev_omm_svi_butterfly_ok.restype = c_int32
    lib.jev_omm_instability.argtypes = [c_double, c_double]
    lib.jev_omm_instability.restype = c_double
    lib.jev_omm_state_gate.argtypes = [
        c_int32, c_double, c_int32, c_double, c_double, c_double, c_double, POINTER(_CStateGate),
    ]
    lib.jev_omm_state_gate.restype = None
    lib.jev_omm_gueant_ode_offsets.argtypes = [
        c_double, c_double, c_double, c_double, c_int32, c_int32, c_double, c_int32,
        POINTER(c_double), POINTER(c_double),
    ]
    lib.jev_omm_gueant_ode_offsets.restype = None
    lib.jev_omm_evaluate_risk_ext.argtypes = [
        c_int32, c_double, c_double, c_double, c_double,
        c_int32, c_double, c_double, c_double, c_double,
        c_double, c_int32, c_int32,
        c_double, c_double, c_int32, c_int32,
        POINTER(c_int32), POINTER(c_char * 128),
    ]
    lib.jev_omm_evaluate_risk_ext.restype = None


def _cq(cfg: QuoterConfig, mode: int = 0) -> _CQuoter:
    return _CQuoter(
        cfg.gamma, cfg.kappa, cfg.sigma, cfg.T_horizon,
        cfg.gamma_penalty, cfg.vega_penalty, cfg.min_half_spread, cfg.max_half_spread,
        cfg.quote_size, mode, cfg.A, cfg.portfolio_delta_penalty,
    )


def test_version_string():
    lib = _lib()
    raw = lib.jev_omm_version()
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    assert text == "1.0.0-zig-desk"


def test_bs_greeks_randomized():
    lib = _lib()
    rng = np.random.default_rng(0)
    for _ in range(24):
        spot = float(rng.uniform(20.0, 200.0))
        strike = float(rng.uniform(0.7, 1.3) * spot)
        t = float(rng.uniform(1.0 / 252.0, 2.0))
        rate = float(rng.uniform(0.0, 0.08))
        q = float(rng.uniform(0.0, 0.04))
        iv = float(rng.uniform(0.05, 0.80))
        is_call = bool(rng.integers(0, 2))
        z = lib.jev_omm_price(spot, strike, t, rate, q, iv, 1 if is_call else 0)
        p = py_price(spot, strike, t, rate, q, iv, is_call)
        assert abs(z - p) < 1e-4
        out = _native._CGreeks()
        lib.jev_omm_greeks(spot, strike, t, rate, q, iv, 1 if is_call else 0, out)
        g = py_greeks(spot, strike, t, rate, q, iv, is_call)
        assert abs(out.delta - g.delta) < 1e-4
        assert abs(out.gamma - g.gamma) < 1e-4
        assert abs(out.vega - g.vega) < 1e-3
        assert abs(out.theta - g.theta) < 1e-3


def test_as_and_gueant_randomized():
    lib = _lib()
    _bind(lib)
    rng = np.random.default_rng(1)
    for _ in range(16):
        cfg = QuoterConfig(
            gamma=float(rng.uniform(0.05, 0.8)),
            kappa=float(rng.uniform(0.4, 3.0)),
            sigma=float(rng.uniform(0.1, 1.5)),
            T_horizon=float(rng.uniform(0.01, 0.2)),
            A=float(rng.uniform(20.0, 200.0)),
            min_half_spread=0.01,
            max_half_spread=10.0,
            mode="as_finite_horizon",
        )
        mid = float(rng.uniform(1.0, 20.0))
        inv = int(rng.integers(-5, 6))
        t_left = float(rng.uniform(0.0, cfg.T_horizon))
        c = _cq(cfg, 0)
        z_r = lib.jev_omm_reservation_price(mid, inv, c, t_left, 1, 0.0, 0.0, 0)
        z_h = lib.jev_omm_optimal_half_spread(c, t_left, 1)
        assert abs(z_r - reservation_price(mid, inv, cfg, t_left)) < 1e-8
        assert abs(z_h - optimal_half_spread(cfg, t_left)) < 1e-8
        gcfg = cfg.model_copy(update={"mode": "gueant_asymptotic"})
        gc = _cq(gcfg, 1)
        assert abs(lib.jev_omm_gueant_inventory_scale(gc) - inventory_scale(gcfg)) < 1e-8
        assert abs(lib.jev_omm_gueant_half_spread(gc) - gueant_half(gcfg)) < 1e-8


def test_svi_pieces_randomized():
    lib = _lib()
    _bind(lib)
    rng = np.random.default_rng(2)
    for _ in range(16):
        a = float(rng.uniform(0.01, 0.08))
        b = float(rng.uniform(0.05, 0.4))
        rho = float(rng.uniform(-0.8, 0.8))
        m = float(rng.uniform(-0.2, 0.2))
        sig = float(rng.uniform(0.05, 0.4))
        k = float(rng.uniform(-0.8, 0.8))
        t = float(rng.uniform(0.05, 1.0))
        p = SviParams(a, b, rho, m, sig)
        assert abs(lib.jev_omm_svi_total_var(a, b, rho, m, sig, k) - total_var(p, k)) < 1e-10
        assert abs(lib.jev_omm_svi_density_g(a, b, rho, m, sig, k) - density_g(p, k)) < 1e-8
        assert abs(lib.jev_omm_svi_iv(a, b, rho, m, sig, k, t) - implied_vol(p, k, t)) < 1e-8
        theta = float(rng.uniform(0.02, 0.2))
        eta = float(rng.uniform(0.2, 1.0))
        gamma = float(rng.uniform(0.2, 0.8))
        sp = SsviParams(rho=rho, eta=eta, gamma=gamma)
        z = lib.jev_omm_ssvi_total_var(theta, rho, eta, gamma, k)
        assert abs(z - ssvi_total_var(k, theta, sp)) < 1e-8


def test_state_gate_randomized():
    lib = _lib()
    _bind(lib)
    rng = np.random.default_rng(3)
    for _ in range(20):
        enabled = bool(rng.integers(0, 2))
        inst = float(rng.choice([0.0, 0.4, 0.9, 1.5, 2.5, math.inf]))
        constraint = bool(rng.integers(0, 2))
        parent = float(rng.uniform(0.0, 1.0))
        f_signed = float(rng.uniform(-2.0, 2.0))
        l_exec = float(rng.uniform(0.0, 3.0))
        mid = float(rng.uniform(0.5, 50.0))
        out = _CStateGate()
        lib.jev_omm_state_gate(
            1 if enabled else 0, inst, 1 if constraint else 0, parent, f_signed, l_exec, mid, out
        )
        py = state_gate(enabled, inst, constraint, parent, f_signed, l_exec, mid)
        assert abs(out.reservation_shift - py.reservation_shift) < 1e-12
        assert abs(out.spread_mult - py.spread_mult) < 1e-12
        assert abs(out.size_mult - py.size_mult) < 1e-12
        assert abs(out.hedge_urgency - py.hedge_urgency) < 1e-12
        assert bool(out.pull) == py.pull
        if math.isfinite(inst):
            z_inst = lib.jev_omm_instability(abs(f_signed), max(l_exec, 1e-8))
            from jev_omm.state_os.vector import instability as py_inst

            # instability() is |F|/L, not the clipped gate value.
            assert abs(z_inst - py_inst(abs(f_signed), max(l_exec, 1e-8))) < 1e-12


def test_risk_gate_matches_including_optional_limits():
    lib = _lib()
    _bind(lib)
    g = Greeks(delta=0.4, gamma=0.02, vega=3.0, theta=-1.0)
    cases = [
        (RiskConfig(max_abs_inventory=10), 11, 0.0, 0.0, None, None, None),
        (RiskConfig(), 2, 1.0, 0.0, None, None, None),
        (RiskConfig(max_loss=50.0), 0, -80.0, 0.0, None, None, None),
        (RiskConfig(max_abs_notional=1000.0), 1, 0.0, 0.0, 1500.0, None, None),
        (RiskConfig(max_abs_notional=1000.0), 1, 0.0, 0.0, None, None, None),
        (RiskConfig(max_abs_per_strike=3), 1, 0.0, 0.0, None, 5, None),
        (RiskConfig(max_quotes_outstanding=1), 0, 0.0, 0.0, None, None, 2),
        (RiskConfig(max_abs_delta=1.0), 4, 0.0, -1.5, None, None, None),
    ]
    for cfg, inv, cash, extra, notional, per_strike, quotes in cases:
        py = evaluate_risk(
            inv, g, cash, cfg,
            extra_delta=extra,
            notional=notional,
            per_strike_abs=per_strike,
            quotes_outstanding=quotes,
        )
        allowed = c_int32()
        reason = (c_char * 128)()
        lib.jev_omm_evaluate_risk_ext(
            inv, g.delta, g.gamma, g.vega, cash,
            cfg.max_abs_inventory, cfg.max_abs_delta, cfg.max_abs_vega, cfg.max_abs_gamma, cfg.max_loss,
            float(cfg.max_abs_notional or 0.0),
            int(cfg.max_abs_per_strike or 0),
            int(cfg.max_quotes_outstanding or 0),
            extra,
            float("nan") if notional is None else float(notional),
            -1 if per_strike is None else int(per_strike),
            -1 if quotes is None else int(quotes),
            allowed,
            reason,
        )
        assert bool(allowed.value) == py.quoting_allowed
        text = bytes(reason).split(b"\x00", 1)[0].decode()
        if py.breach_reason is None:
            assert text == ""
        else:
            assert text.split()[0] == py.breach_reason.split()[0]


def test_gueant_ode_offsets_match_on_a_fixed_point():
    lib = _lib()
    _bind(lib)
    cfg = QuoterConfig(
        mode="gueant_ode",
        gamma=0.1,
        kappa=1.5,
        sigma=0.4,
        A=80.0,
        inventory_cap=4,
        T_horizon=1.0,
        ode_steps=400,
        min_half_spread=1e-8,
        max_half_spread=1e6,
    )
    py = optimal_offsets(cfg, 1)
    db = c_double()
    da = c_double()
    lib.jev_omm_gueant_ode_offsets(
        cfg.gamma, cfg.kappa, cfg.sigma, cfg.A, 1, cfg.inventory_cap, cfg.T_horizon, cfg.ode_steps, db, da
    )
    assert abs(db.value - py.delta_b) < 1e-6
    assert abs(da.value - py.delta_a) < 1e-6
