"""Executable quotes: touch-unit κ, queue decisions, tape fills, lean desk."""

from __future__ import annotations

import math
import os

import pytest

from jev_omm.desk.harness import LEAN_OFF, SURVIVOR_SLEEVES, DeskConfig, legacy_honest_config, run_desk
from jev_omm.execution.executable import (
    attribute_sleeve_fill,
    kappa_for_touch,
    kappa_for_touch_python,
    queue_decision,
    queue_decision_python,
    queue_edge_trial,
)
from jev_omm.pricing import _native
from jev_omm.quoter.avellaneda_stoikov import optimal_half_spread
from jev_omm.config import QuoterConfig
from jev_omm.research.tape import load_local
from jev_omm.research.tape_walk import walk_forward


FIXTURE = "jev_omm/data/fixtures/tape_synthetic.csv"


def test_kappa_for_touch_round_trips_and_is_identity_when_off():
    gamma = 0.12
    target = 0.05
    kappa = kappa_for_touch_python(gamma, target, 1.5)
    half = (1.0 / gamma) * math.log(1.0 + gamma / kappa)
    assert half == pytest.approx(target, abs=1e-9)
    assert kappa_for_touch_python(0.0, target, 1.5) == 1.5
    assert kappa_for_touch_python(gamma, 0.0, 1.5) == 1.5
    cfg = QuoterConfig(gamma=gamma, kappa=kappa, sigma=0.45, min_half_spread=0.0, max_half_spread=2.0)
    # The intensity term is the touch. The tiny risk term is the finite-horizon piece.
    posted = optimal_half_spread(cfg, t_remaining=1.0 / 252.0)
    assert posted == pytest.approx(target, abs=1e-3)


def test_queue_edge_off_matches_stay_and_zig_when_present():
    off = queue_decision_python(4.0, 1.0, 8.0, 0.0, 1.0, 0.02, 0.05, 1.0, False)
    assert off.action == 0
    assert off.ahead == 4.0
    assert off.spread_mult == 1.0
    assert off.size_mult == 1.0
    assert off.cancel_latency < 0.0
    lib = _native._lib
    if lib is None or not _native.ZIG_AVAILABLE or not hasattr(lib, "jev_omm_kappa_for_touch"):
        if os.environ.get("JEV_OMM_REQUIRE_NATIVE") == "1":
            pytest.fail("libjev_omm.so is required for the fill golden")
        pytest.skip("libjev_omm.so without executable fills")
    assert kappa_for_touch(0.12, 0.05, 1.5) == pytest.approx(
        kappa_for_touch_python(0.12, 0.05, 1.5), abs=1e-12
    )
    zig = queue_decision(50.0, 1.0, 5.0, 0.0, 1.0, 0.10, 0.0, 0.0, True)
    py = queue_decision_python(50.0, 1.0, 5.0, 0.0, 1.0, 0.10, 0.0, 0.0, True)
    assert zig.action == py.action == 1
    assert zig.ahead == pytest.approx(0.0)
    toxic = queue_decision(0.0, 1.0, 8.0, 0.0, 1.0, 0.02, 0.05, 1.0, True)
    assert toxic.action == 2
    assert toxic.cancel_latency == pytest.approx(0.1, abs=1e-12)


def test_queue_edge_changes_fills_and_markout_across_seeds():
    seeds = (1, 2, 3, 4, 5, 6, 7, 8)
    toxic_off = [queue_edge_trial(s, queue_edge=False, toxic=1.0, ahead=0.0, spread_capture=0.02, intensity=8.0) for s in seeds]
    toxic_on = [queue_edge_trial(s, queue_edge=True, toxic=1.0, ahead=0.0, spread_capture=0.02, intensity=8.0) for s in seeds]
    fills_off = sum(f for f, _ in toxic_off) / len(seeds)
    fills_on = sum(f for f, _ in toxic_on) / len(seeds)
    mark_off = sum(m for _, m in toxic_off) / len(seeds)
    mark_on = sum(m for _, m in toxic_on) / len(seeds)
    assert fills_on < fills_off
    assert mark_on > mark_off

    deep_off = [queue_edge_trial(s, queue_edge=False, toxic=0.0, ahead=50.0, spread_capture=0.10, intensity=5.0) for s in seeds]
    deep_on = [queue_edge_trial(s, queue_edge=True, toxic=0.0, ahead=50.0, spread_capture=0.10, intensity=5.0) for s in seeds]
    assert sum(f for f, _ in deep_on) > sum(f for f, _ in deep_off)


def test_legacy_print_rule_still_has_zero_theory_fills_and_lob_does_not():
    tape = load_local(FIXTURE)
    assert tape.synthetic_fixture is True
    legacy = walk_forward(tape, fill_model="print", executable_units=False, queue_edge=False)
    by = {row.label: row for row in legacy["rows"]}
    assert by["fixed_spread"].n_fills > 0
    assert by["join_touch"].n_fills > 0
    assert by["avellaneda_stoikov"].n_fills == 0
    assert by["gueant_asymptotic"].n_fills == 0

    live = walk_forward(tape)
    assert live["fill_model"] == "lob"
    assert live["spot_provenance"] == "underlying_print"
    assert live["quote_mid_provenance"] == "cbbo"
    got = {row.label: row for row in live["rows"]}
    assert got["avellaneda_stoikov"].n_fills > 0
    assert got["fixed_spread"].n_fills > 0
    # Joining behind displayed size does not clear a 2-lot print.
    assert got["join_touch"].n_fills == 0
    # Wide legacy κ, same LOB book, still does not trade.
    wide = walk_forward(tape, fill_model="lob", executable_units=False, queue_edge=False)
    wide_as = {row.label: row for row in wide["rows"]}["avellaneda_stoikov"]
    assert wide_as.n_fills == 0


def test_sleeve_fill_is_identity_when_size_is_zero_and_toxicity_cuts_fills():
    calm = attribute_sleeve_fill(
        spread_mult=1.0, size_mult=1.0, target=0.0, queue=0.0, toxic=0.0, queue_edge=True
    )
    hot = attribute_sleeve_fill(
        spread_mult=1.0, size_mult=1.0, target=0.0, queue=0.0, toxic=0.95, queue_edge=True
    )
    off = attribute_sleeve_fill(
        spread_mult=1.0, size_mult=1.0, target=0.0, queue=0.0, toxic=0.95, queue_edge=False
    )
    assert calm.fills > 0.0
    assert hot.fills < calm.fills
    assert hot.fills < off.fills
    assert hot.adverse_markout > off.adverse_markout
    dead = attribute_sleeve_fill(
        spread_mult=1.0, size_mult=0.0, target=1.0, queue=0.0, toxic=0.0, queue_edge=True
    )
    assert dead.fills == 0.0 and dead.fill_pnl == 0.0


def test_lean_paper_desk_disables_killed_sleeves_and_books_lob_fills():
    paper = run_desk(DeskConfig(n_steps=12, seed=11, fit_surfaces=False))
    enabled = {row.sleeve_id: row.enabled for row in paper.scoreboard.rows}
    for sleeve_id in LEAN_OFF:
        assert enabled[sleeve_id] is False
        assert paper.weights[sleeve_id] == 0.0
    honest = run_desk(legacy_honest_config(n_steps=12, seed=11))
    assert honest.config.lean_book is False
    present = {row.sleeve_id for row in honest.scoreboard.rows}
    for sleeve_id in LEAN_OFF:
        assert sleeve_id in present
    assert honest.scoreboard.desk_lob_fills == 0.0
    assert honest.scoreboard.desk_fill_pnl == 0.0
    survivors = {row.sleeve_id: row for row in paper.scoreboard.rows}
    filled = [survivors[name].lob_fills for name in SURVIVOR_SLEEVES]
    assert sum(filled) > 0.0
    quiet = run_desk(DeskConfig(n_steps=12, seed=11, fit_surfaces=False, queue_edge=False))
    sniper_on = survivors["queue_sniper"].lob_fills
    sniper_off = next(row.lob_fills for row in quiet.scoreboard.rows if row.sleeve_id == "queue_sniper")
    assert sniper_on > sniper_off
    flow_on = survivors["flow_toxicity"].lob_fills
    mm_on = survivors["mm_spread"].lob_fills
    assert flow_on < mm_on
    assert paper.scoreboard.synthetic_fixture == 1
    for row in paper.scoreboard.rows:
        assert math.isfinite(row.fill_pnl)
        assert math.isfinite(row.residual_pnl)
