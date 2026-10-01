"""Residual smoothness, allocator floor, roll/vanna split, capacity identity."""

from __future__ import annotations

import os

import numpy as np
import pytest

from jev_omm.decisions.client import DeterministicFallbackClient
from jev_omm.decisions.policy import apply_desk_policy
from jev_omm.decisions.schemas import build_desk_questions, build_mm_state
from jev_omm.desk.allocator import allocate, apply_min_weight_floor
from jev_omm.desk.falsify import research_weight_check
from jev_omm.desk.harness import (
    RESEARCH_SLEEVES,
    DeskConfig,
    legacy_config,
    legacy_ortho_config,
    pair_rho,
    run_desk,
)
from jev_omm.desk.honesty import smoothness_penalty, smoothness_penalty_python
from jev_omm.pnl.residual import pearson
from jev_omm.pricing import _native


def test_constant_leftover_is_flat_and_white_noise_is_identity():
    flat = np.array([0.20, 0.201, 0.199, 0.2005, 0.1995, 0.2002, 0.1998, 0.2001])
    noisy = np.array([0.04, -0.02, 0.03, -0.05, 0.01, -0.04, 0.02, -0.01])
    penalty, flag, stats = smoothness_penalty(flat, noisy, prefer_zig=False)
    assert flag == "flat"
    assert penalty == 0.0
    assert stats["dc_share"] > 0.95
    white = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.02, -0.015])
    penalty_w, flag_w, _ = smoothness_penalty(white, white, prefer_zig=False)
    assert flag_w == "ok"
    assert penalty_w == 1.0


def test_smoothness_matches_zig_when_the_library_is_present():
    lib = _native._lib
    if lib is None or not _native.ZIG_AVAILABLE or not hasattr(lib, "jev_omm_smoothness"):
        if os.environ.get("JEV_OMM_REQUIRE_NATIVE") == "1":
            pytest.fail("libjev_omm.so is required but jev_omm_smoothness is missing")
        pytest.skip("libjev_omm.so without smoothness")
    rng = np.random.default_rng(3)
    resid = rng.normal(0.02, 0.01, 48)
    raw = rng.normal(0.0, 0.05, 48)
    zig_pen, zig_flag, zig_stats = smoothness_penalty(resid, raw, prefer_zig=True)
    py_pen, py_flag, py_stats = smoothness_penalty_python(resid, raw)
    assert zig_flag == {0: "ok", 1: "smooth", 2: "flat"}[py_flag]
    assert abs(zig_pen - py_pen) < 1e-9
    assert abs(zig_stats["ac1"] - py_stats["ac1"]) < 1e-8
    assert abs(zig_stats["dc_share"] - py_stats["dc_share"]) < 1e-8
    assert abs(zig_stats["const_trend_r2"] - py_stats["const_trend_r2"]) < 1e-8
    assert abs(zig_stats["low_freq_share"] - py_stats["low_freq_share"]) < 1e-6
    calm = [0.02, 0.021, 0.019, 0.020, 0.022, 0.018]
    jumpy = [0.40, -0.20, 0.35, -0.30, 0.50, -0.10]
    plain = allocate([calm, jumpy], [True, True], max_weight=1.0, corr_cap=0.99, sharpe_tilt=0.0)
    clipped = allocate(
        [calm, jumpy],
        [True, True],
        max_weight=1.0,
        corr_cap=0.99,
        sharpe_tilt=0.0,
        sigma_clip_quantile=0.50,
    )
    assert clipped[1] > plain[1]
    if hasattr(lib, "jev_omm_allocate_inverse_vol"):
        import ctypes

        packed = np.ascontiguousarray(np.concatenate([calm, jumpy]), dtype=np.float64)
        enabled = np.array([1, 1], dtype=np.uint8)
        out = np.zeros(2, dtype=np.float64)
        lib.jev_omm_allocate_inverse_vol.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.jev_omm_allocate_inverse_vol.restype = None
        lib.jev_omm_allocate_inverse_vol(
            2,
            6,
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            enabled.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            1.0,
            0.99,
            0.0,
            0.50,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        assert abs(float(out[1]) - float(clipped[1])) < 1e-8


def test_floor_lifts_a_small_positive_weight_and_keeps_the_sum():
    w = np.array([0.34, 0.65, 0.01])
    out = apply_min_weight_floor(w, [False, False, True], 0.02, 0.35)
    assert abs(out[2] - 0.02) < 1e-12
    assert abs(float(out.sum()) - 1.0) < 1e-9
    assert float(out[0]) <= 0.35 + 1e-12


def test_sigma_clip_off_matches_the_unclipped_allocator():
    series = [[0.03, 0.04, 0.02, 0.05, 0.01, 0.02], [0.10, -0.04, 0.08, -0.02, 0.06, -0.01]]
    a = allocate(series, [True, True], max_weight=0.35, corr_cap=0.35, sharpe_tilt=0.25)
    b = allocate(
        series,
        [True, True],
        max_weight=0.35,
        corr_cap=0.35,
        sharpe_tilt=0.25,
        sigma_clip_quantile=0.0,
    )
    assert np.max(np.abs(a - b)) < 1e-12


def test_legacy_ortho_reprints_the_roll_vanna_failure_and_honest_shrinks_it():
    before = run_desk(legacy_ortho_config(n_steps=80, seed=11))
    rho_before = pair_rho(before, "roll_yield", "vanna_tilt")
    assert rho_before is not None and rho_before < -0.30
    flow = next(row for row in before.scoreboard.rows if row.sleeve_id == "flow_toxicity")
    assert flow.weight > 0.15
    assert flow.residual_sharpe_raw > 5.0 or flow.sharpe_residual > 5.0
    after = run_desk(DeskConfig(n_steps=80, seed=11, fit_surfaces=False))
    rho_after = pair_rho(after, "roll_yield", "vanna_tilt")
    assert rho_after is not None
    assert abs(rho_after) < abs(rho_before)
    flow_after = next(row for row in after.scoreboard.rows if row.sleeve_id == "flow_toxicity")
    assert flow_after.smoothness_flag == "flat"
    assert flow_after.smoothness_penalty == 0.0
    assert flow_after.weight == 0.0
    assert flow_after.residual_sharpe_penalized == 0.0
    assert flow_after.residual_sharpe_raw > 5.0
    means = {row.sleeve_id: row.mean_residual for row in after.scoreboard.rows}
    kept = research_weight_check(after.weights, means)
    assert len(kept) >= 2
    for row in after.scoreboard.rows:
        assert row.weight <= 0.35 + 1e-9
    assert after.scoreboard.weight_sum <= 1.0 + 1e-8
    assert after.scoreboard.desk_sharpe_residual < flow_after.residual_sharpe_raw


def test_capacity_caps_off_is_the_identity_on_weights():
    base = dict(n_steps=24, seed=5, fit_surfaces=False, honesty=False, honest_allocator=False, walkforward_kill=False)
    off = run_desk(DeskConfig(capacity_caps=False, **base))
    on_idle = run_desk(
        DeskConfig(capacity_caps=True, turnover_cap=1.0e12, inventory_cap=1.0e12, **base)
    )
    for sleeve_id in off.weights:
        assert abs(off.weights[sleeve_id] - on_idle.weights[sleeve_id]) < 1e-9
        assert off.capacity_scale[sleeve_id] == 1.0


def test_legacy_eight_sleeve_correlations_still_reprint():
    before = run_desk(legacy_config(n_steps=80, seed=11))
    skew_fly = pair_rho(before, "skew_residual", "fly_butterfly")
    mm_vrp = pair_rho(before, "mm_spread", "vrp_varswap")
    assert skew_fly is not None and skew_fly > 0.90
    assert mm_vrp is not None and mm_vrp > 0.60


def test_edge_fail_sets_sleeve_kill_without_an_order():
    state = build_mm_state(
        time=0.0,
        spot=100.0,
        option_mid=1.0,
        iv=0.2,
        inventory=0,
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        cash_pnl=0.0,
        half_spread=0.2,
        quoting_allowed=True,
    )
    state["desk"] = {"enabled": 1.0, "edge_fail": 1.0}
    ans = DeterministicFallbackClient().system_one(state, build_desk_questions())
    assert ans.answers["sleeve_kill"].noul >= 0.70
    adj = apply_desk_policy(ans, state)
    assert adj.kill is True
    assert adj.weight_mult == 0.0
    assert "order" not in adj.reason
    state["desk"] = {"enabled": 1.0, "product_edge_fail": 1.0}
    prod = DeterministicFallbackClient().system_one(state, build_desk_questions())
    adj_p = apply_desk_policy(prod, state)
    assert adj_p.product_kill is True
    assert adj_p.kill is False
    assert len(RESEARCH_SLEEVES) == 4
    # A disabled residual is orthogonal to itself; pearson of a flat pair is 0.
    assert pearson([0.0, 0.0, 0.0], [0.1, -0.2, 0.05]) == 0.0
