"""Orthogonality gate, multi-seed residual correlation, new sleeves."""

from __future__ import annotations

import numpy as np

from jev_omm.desk.allocator import DEFAULT_CORR_CAP, DEFAULT_MAX_WEIGHT, DEFAULT_SHARPE_TILT
from jev_omm.desk.fixtures import DEFAULT_PRODUCTS
from jev_omm.desk.harness import DeskConfig, legacy_config, multi_seed_corr, pair_rho, run_desk
from jev_omm.desk.orthogonal import enforce_orthogonality, residualize_keep_mean
from jev_omm.desk.sleeves import SLEEVE_IDS, SleeveContext, quote_or_target
from jev_omm.pnl.residual import pearson, strip_factors
from jev_omm.training.cases import run_pair


def test_residualize_keeps_the_mean_and_kills_the_correlation():
    x = np.array([0.2, -0.1, 0.4, 0.0, -0.3, 0.15])
    y = 0.8 * x + 0.02
    out = residualize_keep_mean(y, x)
    assert abs(float(np.mean(out)) - float(np.mean(y))) < 1e-12
    assert abs(pearson(out, x)) < 1e-8


def test_gate_merges_a_shared_factor():
    base = np.array([0.03, 0.04, 0.02, 0.05, 0.03, 0.04])
    clone = 0.93 * base - 0.01
    gated = enforce_orthogonality(
        {"skew_residual": base, "fly_butterfly": clone},
        {"skew_residual": True, "fly_butterfly": True},
        threshold=0.40,
    )
    assert gated.enabled["fly_butterfly"] is False
    assert abs(float(np.sum(gated.series["fly_butterfly"]))) < 1e-12
    assert gated.logs and gated.logs[0].action == "merge"


def test_k_factor_strip_keeps_the_constant():
    fb = np.array([0.01, -0.02, 0.015, 0.0, -0.01, 0.008])
    fg = np.array([0.001, 0.004, 0.0002, 0.003, 0.0015, 0.0004])
    fv = np.array([0.10, -0.20, 0.0, 0.05, -0.04, 0.02])
    fo = np.array([0.02, 0.01, -0.03, 0.04, -0.01, 0.0])
    raw = 0.5 * fb + 1.0 * fg + 0.25 * fv + 0.1 * fo + 0.01
    fit = strip_factors(raw, [fb, fg, fv, fo], prefer_zig=False)
    assert abs(fit.coefs[0] - 0.5) < 1e-5
    assert abs(fit.coefs[3] - 0.1) < 1e-5
    assert fit.r2 > 0.99
    assert abs(fit.mean_residual - 0.01) < 1e-5


def test_new_sleeves_are_identity_when_off():
    ctx = SleeveContext(
        product_id="EQ_INDEX",
        beta_to_index=1.0,
        product_return=0.01,
        index_return=0.01,
        d_sigma=0.0,
        dt=1.0 / 252.0,
        skew=0.02,
        d_skew=-0.005,
        fly=0.01,
        d_fly=-0.002,
        term=0.01,
        d_term=-0.001,
        tox=0.2,
        gex_norm=-0.4,
        instability=0.2,
        f_signed=0.01,
        box_edge=0.01,
        d_box=-0.002,
        iv=0.2,
        prev_target=-0.2,
        gates_on=True,
        enabled=False,
        wing=0.02,
        d_wing=-0.01,
        dispersion=0.01,
        d_dispersion=-0.004,
        autocall=0.3,
        d_autocall=-0.05,
    )
    for sleeve_id in ("wing_kurtosis", "dispersion_index", "rough_vol_stress", "queue_sniper"):
        quote = quote_or_target(sleeve_id, ctx)
        assert quote.note == "identity"
        assert quote.target == 0.0 and quote.raw_pnl == 0.0
    on = SleeveContext(**{**ctx.__dict__, "enabled": True, "gates_on": False})
    parked = quote_or_target("warehouse_autocall", on)
    assert parked.note == "warehouse_identity"
    assert parked.raw_pnl == 0.0


def test_legacy_pair_is_the_known_failure_and_the_fix_breaks_it():
    before = run_desk(legacy_config(n_steps=80, seed=11))
    skew_fly = pair_rho(before, "skew_residual", "fly_butterfly")
    mm_vrp = pair_rho(before, "mm_spread", "vrp_varswap")
    assert skew_fly is not None and skew_fly > 0.90
    assert mm_vrp is not None and mm_vrp > 0.60
    after = run_desk(
        DeskConfig(
            products=("EQ_INDEX", "EQ_SINGLE", "FX_PAIR"),
            sleeves=SLEEVE_IDS[:8],
            n_steps=80,
            seed=11,
            fit_surfaces=False,
        )
    )
    skew_fly_after = pair_rho(after, "skew_residual", "fly_butterfly")
    mm_vrp_after = pair_rho(after, "mm_spread", "vrp_varswap")
    assert skew_fly_after is not None and abs(skew_fly_after) < 0.35
    assert mm_vrp_after is not None and abs(mm_vrp_after) < 0.35
    assert after.scoreboard.max_abs_rho <= 0.40 + 1e-9
    assert after.weights["parity_box"] <= DEFAULT_MAX_WEIGHT + 1e-9


def test_multi_seed_mean_correlation_stays_under_the_gate():
    report = multi_seed_corr(
        n_steps=40,
        regimes=("baseline", "smile_shock"),
        fit_surfaces=False,
    )
    assert report.max_mean_abs <= 0.40
    assert abs(report.headline["skew_residual/fly_butterfly"]) < 0.35
    assert abs(report.headline["mm_spread/vrp_varswap"]) < 0.35
    assert len(DEFAULT_PRODUCTS) >= 8
    assert len(SLEEVE_IDS) >= 14
    assert DEFAULT_CORR_CAP == 0.35
    assert DEFAULT_MAX_WEIGHT == 0.35
    assert DEFAULT_SHARPE_TILT > 0.0


def test_ortho_and_multi_product_training_cases():
    for name in ("ortho_break", "toxic_multi"):
        naive, desk = run_pair(name)
        assert desk.score.absolute_pnl > naive.score.absolute_pnl
        assert "order" not in desk.events[0]["note"]
    naive, desk = run_pair("ortho_break")
    assert desk.events[0]["merge"] is True
    assert naive.events[0]["merge"] is False
    _naive, multi = run_pair("toxic_multi")
    assert multi.events[0]["product_kill"] is True
    assert multi.events[0]["names"] == ["EQ_INDEX", "EQ_SINGLE", "EQ_LOWBETA"]
