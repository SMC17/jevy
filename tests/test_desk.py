"""Multi-product surface book, sleeves, residual strip, desk harness."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from jev_omm.decisions.client import DeterministicFallbackClient
from jev_omm.decisions.policy import apply_desk_policy
from jev_omm.decisions.schemas import build_desk_questions, build_mm_state
from jev_omm.desk.allocator import TOXIC_SLEEVE_GOOD, TOXIC_SLEEVE_TOXIC, allocate, toxic_sleeve_weights
from jev_omm.desk.fixtures import DEFAULT_PRODUCTS, EXPIRIES, UNDERLIERS, fixture_rows, load_csv, render_csv
from jev_omm.desk.harness import DeskConfig, run_desk
from jev_omm.desk.sleeves import SLEEVE_IDS, SleeveContext, flow_size_mult, quote_or_target
from jev_omm.hedge.delta import greek_pnl_step
from jev_omm.pnl.residual import gamma_factor, strip_residual
from jev_omm.pnl.residual import _strip_python
from jev_omm.pricing import _native
from jev_omm.surface.book import SurfaceBook, quotes_from_svi
from jev_omm.surface.svi import SviParams, raw_calendar_ok
from jev_omm.training.cases import run_pair


def _ctx(**kw) -> SleeveContext:
    base = dict(
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
        tox=0.0,
        gex_norm=0.2,
        instability=0.0,
        f_signed=0.0,
        box_edge=0.01,
        d_box=-0.002,
        iv=0.2,
        prev_target=0.0,
        gates_on=True,
        enabled=True,
        fill_draw=0.1,
        rate=0.04,
    )
    base.update(kw)
    return SleeveContext(**base)


def test_fixture_is_labeled_synthetic_and_has_eight_names():
    rows = fixture_rows()
    ids = {r["underlier_id"] for r in rows}
    assert ids == set(DEFAULT_PRODUCTS)
    assert len(ids) >= 8
    assert all(r["synthetic_fixture"] == 1 for r in rows)
    assert len(EXPIRIES) == 3
    text = render_csv(rows)
    assert text.startswith("synthetic_fixture,")
    loaded = load_csv()
    assert {r["underlier_id"] for r in loaded} == ids
    assert all(r["synthetic_fixture"] == "1" for r in loaded)


def test_fixture_slices_pass_calendar_and_differ_in_beta():
    betas = [UNDERLIERS[k].beta_to_index for k in ("EQ_INDEX", "EQ_SINGLE", "FX_PAIR")]
    assert betas == [1.0, 1.35, 0.15]
    assert len({UNDERLIERS[k].beta_to_index for k in DEFAULT_PRODUCTS}) == len(DEFAULT_PRODUCTS)
    for spec in UNDERLIERS.values():
        ordered = [spec.slices[t] for t in sorted(spec.slices)]
        assert raw_calendar_ok(ordered[0], ordered[1])
        assert raw_calendar_ok(ordered[1], ordered[2])
        assert abs(ordered[0].rho - ordered[2].rho) < 1e-12


def test_surface_book_fits_three_names_and_keeps_a_residual():
    book = SurfaceBook(synthetic_fixture=1)
    ks = [-0.4, -0.2, 0.0, 0.2, 0.4]
    for spec in UNDERLIERS.values():
        book.add_underlier(
            spec.underlier_id,
            asset_class=spec.asset_class,
            beta_to_index=spec.beta_to_index,
            spot=spec.spot,
            rate=spec.rate,
            div_yield=spec.div_yield,
            synthetic_fixture=1,
        )
        quotes = {}
        for expiry, params in spec.slices.items():
            fwd = book.forward(spec.underlier_id, expiry)
            assert fwd is not None and fwd > 0.0
            quotes[expiry] = quotes_from_svi(
                params, forward=fwd, expiry_years=expiry, ks=ks
            )
        slices = book.fit_expiries(spec.underlier_id, quotes)
        assert len(slices) == 3
        assert slices[0].butterfly_ok
        assert slices[0].rmse < 1e-3
        assert slices[0].synthetic_fixture == 1
    # A call-wing bump outside a smooth SVI leaves a positive wing residual.
    spec = UNDERLIERS["EQ_INDEX"]
    expiry = EXPIRIES[0]
    fwd = book.forward(spec.underlier_id, expiry)
    fine = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
    bumped = quotes_from_svi(spec.slices[expiry], forward=fwd, expiry_years=expiry, ks=fine)
    for point in bumped:
        if point.log_moneyness >= 0.39:
            point.iv_mkt += 0.08
            point.total_var_mkt = point.iv_mkt * point.iv_mkt * expiry
    sl = book.fit_quotes(spec.underlier_id, expiry, bumped)
    far = max(sl.residuals, key=lambda p: p.log_moneyness)
    assert far.log_moneyness >= 0.39
    assert far.residual > 0.005
    assert max(abs(p.residual) for p in sl.residuals) > 0.01
    assert all(abs((p.iv_mkt - p.iv_fit) - p.residual) < 1e-12 for p in sl.residuals)


def test_missing_spot_stays_missing():
    book = SurfaceBook()
    book.add_underlier(
        "EQ_INDEX",
        asset_class="equity_index",
        beta_to_index=1.0,
        spot=None,
        rate=0.04,
        div_yield=0.01,
    )
    assert book.underliers["EQ_INDEX"].spot_missing
    assert book.forward("EQ_INDEX", 0.1) is None
    sl = book.fit_quotes("EQ_INDEX", 0.1, [])
    assert sl.forward is None
    assert sl.residuals == []
    assert book.mark_iv("EQ_INDEX", 0.1, 100.0, 100.0) is None


def test_calendar_break_is_damped():
    book = SurfaceBook()
    book.add_underlier(
        "EQ_INDEX",
        asset_class="equity_index",
        beta_to_index=1.0,
        spot=100.0,
        rate=0.0,
        div_yield=0.0,
    )
    ks = [-0.3, -0.1, 0.0, 0.1, 0.3]
    rich = SviParams(0.05, 0.08, -0.3, 0.0, 0.2)
    cheap = SviParams(0.01, 0.04, -0.2, 0.0, 0.15)
    near = quotes_from_svi(rich, forward=100.0, expiry_years=0.1, ks=ks)
    far = quotes_from_svi(cheap, forward=100.0, expiry_years=0.4, ks=ks)
    slices = book.fit_expiries("EQ_INDEX", {0.1: near, 0.4: far})
    assert slices[0].calendar_ok
    assert slices[1].calendar_ok is False
    assert slices[1].damped
    assert slices[1].surface_suspect
    marked = book.mark_iv("EQ_INDEX", 0.4, 100.0, 100.0)
    assert marked is not None and marked > 0.0


def test_gamma_factor_matches_greek_pnl_step():
    step = greek_pnl_step(delta=0.0, gamma=0.04, vega=1.0, theta=0.0, d_spot=2.0, d_sigma=0.01)
    assert abs(gamma_factor(0.04, 2.0) - step.gamma_pnl) < 1e-12
    assert abs(step.gamma_pnl - 0.08) < 1e-12


def test_strip_keeps_the_constant_and_is_orthogonal():
    fb = np.array([0.01, -0.02, 0.015, 0.0, -0.01, 0.008])
    fg = np.array([0.001, 0.004, 0.0002, 0.003, 0.0015, 0.0004])
    fv = np.array([0.10, -0.20, 0.0, 0.05, -0.04, 0.02])
    raw = 0.5 * fb + 1.0 * fg + 0.25 * fv + 0.01
    fit = strip_residual(raw, fb, fg, fv)
    py = _strip_python(raw, fb, fg, fv)
    assert abs(fit.beta - 0.5) < 1e-5
    assert abs(fit.gamma_coef - 1.0) < 1e-5
    assert abs(fit.vega_coef - 0.25) < 1e-5
    assert fit.r2 > 0.99
    assert abs(fit.mean_residual - 0.01) < 1e-5
    assert float(np.max(np.abs(fit.residual - fit.mean_residual))) < 1e-5
    assert abs(fit.beta - py.beta) < 1e-6
    assert abs(fit.r2 - py.r2) < 1e-6


def test_allocator_cuts_the_correlated_loser_and_respects_the_cap():
    w = toxic_sleeve_weights(0.40)
    assert abs(w[0] - 0.40) < 1e-12
    assert abs(w[1]) < 1e-12
    disabled = allocate(
        [list(TOXIC_SLEEVE_GOOD), list(TOXIC_SLEEVE_TOXIC)],
        [True, False],
        max_weight=1.0,
    )
    assert abs(disabled[0] - 1.0) < 1e-12
    assert abs(disabled[1]) < 1e-12


def test_sleeve_identity_and_flow_gate():
    off = quote_or_target("mm_spread", _ctx(enabled=False))
    assert off.target == 0.0 and off.raw_pnl == 0.0 and off.note == "identity"
    assert flow_size_mult(0.0, True) == 1.0
    assert flow_size_mult(0.8, False) == 1.0
    # vpin 0.8 is blended at 0.50 inside flow_prior, so size is 1/1.6.
    assert flow_size_mult(0.8, True) < 0.7
    calm = quote_or_target("flow_toxicity", _ctx(tox=0.0, gates_on=True, fill_draw=0.0))
    toxic_off = quote_or_target("flow_toxicity", _ctx(tox=0.9, gates_on=False, fill_draw=0.0))
    toxic_on = quote_or_target("flow_toxicity", _ctx(tox=0.9, gates_on=True, fill_draw=0.0))
    assert toxic_off.size_mult == 1.0
    assert toxic_on.size_mult < toxic_off.size_mult
    assert toxic_on.raw_pnl > toxic_off.raw_pnl
    assert calm.note == "flow_prior"
    gex_off = quote_or_target(
        "gex_forced",
        _ctx(gex_norm=-1.5, instability=3.0, gates_on=False, f_signed=0.01),
    )
    gex_on = quote_or_target(
        "gex_forced",
        _ctx(gex_norm=-1.5, instability=3.0, gates_on=True, f_signed=0.01),
    )
    assert gex_off.size_mult == 1.0
    assert gex_on.size_mult == 0.0
    assert gex_on.target == 0.0
    box = quote_or_target("parity_box", _ctx(box_edge=0.02, prev_target=-1.0))
    assert box.note == "box_theo"
    assert box.risk_budget > 0.0


def test_desk_questions_are_identity_when_the_gate_is_off():
    state = build_mm_state(
        time=0.0, spot=100.0, option_mid=1.0, iv=0.2, inventory=0,
        delta=0.0, gamma=0.0, vega=0.0, cash_pnl=0.0, half_spread=0.2, quoting_allowed=True,
    )
    state["desk"] = {"enabled": 0.0, "sleeve_toxic": 1.0}
    ans = DeterministicFallbackClient().system_one(state, build_desk_questions())
    assert ans.source == "fallback"
    assert ans.answers["sleeve_weight"].choice == "hold"
    assert ans.answers["kill_sleeve"].noul == 0.0
    adj = apply_desk_policy(ans, state)
    assert adj.weight_mult == 1.0 and adj.kill is False and adj.reason == "identity"
    state["desk"]["enabled"] = 1.0
    hot = DeterministicFallbackClient().system_one(state, build_desk_questions())
    assert hot.answers["kill_sleeve"].noul > 0.7
    killed = apply_desk_policy(hot, state)
    assert killed.kill and killed.weight_mult == 0.0
    # The MM battery is unchanged when the desk questions are not asked.
    plain = DeterministicFallbackClient().system_one(state)
    assert "kill_sleeve" not in plain.answers


def test_toxic_sleeve_training_case():
    naive, desk = run_pair("toxic_sleeve")
    assert desk.score.absolute_pnl > naive.score.absolute_pnl
    assert abs(naive.score.absolute_pnl) < 1e-12
    ev = next(e for e in desk.events if e["type"] == "SleeveAllocator")
    assert ev["names"] == ["EQ_INDEX", "EQ_SINGLE"]
    assert abs(ev["weight_mm_spread"] - 0.35) < 1e-9
    assert ev["weight_flow_toxicity"] == 0.0
    assert ev["source"] == "fallback"
    assert ev["kill"] is True
    assert "order" not in ev["note"]
    naive_ev = next(e for e in naive.events if e["type"] == "SleeveAllocator")
    assert naive_ev["reason"] == "identity"
    assert naive_ev["kill"] is False


def test_desk_harness_scoreboard():
    run = run_desk(DeskConfig(n_steps=48, seed=11, fit_stride=24, jev_desk=False))
    board = run.scoreboard
    assert board.synthetic_fixture == 1
    assert board.products == list(DEFAULT_PRODUCTS)
    assert len(board.rows) == len(SLEEVE_IDS) >= 14
    assert run.n_fits >= 2
    assert set(run.book.underliers) == set(board.products)
    for uid, surface in run.book.underliers.items():
        assert surface.synthetic_fixture == 1
        assert surface.spot is not None
        assert len(surface.slices) == 3
    assert board.weight_sum <= 1.0 + 1e-9
    assert board.weight_sum > 0.0
    for row in board.rows:
        assert 0.0 <= row.r2 <= 1.0 + 1e-9
        assert row.weight <= 0.35 + 1e-9
        assert math.isfinite(row.residual_pnl)
    # Residual is orthogonal to the factors that were in the regression.
    for sleeve_id in SLEEVE_IDS:
        resid = run.residual[sleeve_id]
        fg = run.f_gamma[sleeve_id]
        # A nearly constant residual is a premium. Pearson is undefined there.
        if float(np.std(resid)) > 1e-6 and float(np.std(fg)) > 1e-12:
            assert abs(float(np.corrcoef(resid, fg)[0, 1])) < 1e-3
        if float(np.std(resid)) > 1e-6 and float(np.std(run.f_beta)) > 1e-12:
            assert abs(float(np.corrcoef(resid, run.f_beta)[0, 1])) < 1e-3
    flagged = {(a, b) for a, b, _rho in board.flagged_pairs}
    for i, a in enumerate(board.sleeve_ids):
        for j in range(i + 1, len(board.sleeve_ids)):
            rho = float(board.pearson[i, j])
            assert abs(rho - float(board.pearson[j, i])) < 1e-12
            if abs(rho) > run.config.corr_cap:
                assert (a, board.sleeve_ids[j]) in flagged
    # Turning a sleeve off is the identity: no PnL, no weight.
    jev = run_desk(DeskConfig(n_steps=8, seed=11, fit_stride=8, jev_desk=True))
    assert jev.scoreboard.synthetic_fixture == 1
    assert len(jev.scoreboard.rows) == len(SLEEVE_IDS)
    off = run_desk(
        DeskConfig(
            n_steps=16,
            seed=11,
            fit_stride=16,
            enabled={s: s != "mm_spread" for s in SLEEVE_IDS},
        )
    )
    assert off.weights["mm_spread"] == 0.0
    assert abs(float(np.sum(off.raw["mm_spread"]))) < 1e-12


def test_risk_cap_scales_the_book():
    wide = run_desk(DeskConfig(n_steps=12, seed=3, fit_stride=12))
    tight = run_desk(
        DeskConfig(n_steps=12, seed=3, fit_stride=12, max_abs_delta=0.05, max_abs_gamma=0.05)
    )
    assert float(np.min(tight.scale_path)) < 1.0
    assert abs(float(np.sum(tight.raw["mm_spread"]))) < abs(float(np.sum(wide.raw["mm_spread"]))) + 1e-9


def test_native_strip_matches_python_when_the_library_is_present():
    lib = _native._lib
    if lib is None or not _native.ZIG_AVAILABLE or not hasattr(lib, "jev_omm_residual_strip"):
        if os.environ.get("JEV_OMM_REQUIRE_NATIVE") == "1":
            pytest.fail("libjev_omm.so is required (JEV_OMM_REQUIRE_NATIVE=1) but residual strip is missing")
        pytest.skip("libjev_omm.so without residual strip")
    fb = np.array([0.01, -0.02, 0.015, 0.0, -0.01, 0.008])
    fg = np.array([0.001, 0.004, 0.0002, 0.003, 0.0015, 0.0004])
    fv = np.array([0.10, -0.20, 0.0, 0.05, -0.04, 0.02])
    raw = 0.5 * fb + fg + 0.25 * fv + 0.01
    zig = strip_residual(raw, fb, fg, fv, prefer_zig=True)
    py = strip_residual(raw, fb, fg, fv, prefer_zig=False)
    assert abs(zig.beta - py.beta) < 1e-6
    assert abs(zig.gamma_coef - py.gamma_coef) < 1e-6
    assert abs(zig.vega_coef - py.vega_coef) < 1e-6
    assert abs(zig.r2 - py.r2) < 1e-6
    assert np.max(np.abs(zig.residual - py.residual)) < 1e-6
