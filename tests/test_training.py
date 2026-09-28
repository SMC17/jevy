"""Citadel-style training cases: the desk policy beats the naive one on the lesson."""

from __future__ import annotations

from jev_omm.decisions.client import DeterministicFallbackClient
from jev_omm.decisions.schemas import build_mm_state
from jev_omm.training.cases import CASES, leaderboard, run_pair
from jev_omm.training.replay import replay_scores, write_jsonl
from jev_omm.training.scoring import lcg_next


def test_lcg_matches_a_fixed_step():
    state, z = lcg_next(7)
    assert state == (1664525 * 7 + 1013904223) & 0xFFFFFFFF
    assert -1.0 <= z < 1.0
    state2, z2 = lcg_next(7)
    assert state == state2 and z == z2


def test_location_arb_hedges_beta():
    naive, desk = run_pair("location_arb")
    ratio = desk.score.mean_abs_beta / naive.score.mean_abs_beta
    assert naive.score.mean_abs_beta > 10.0
    assert 0.10 < ratio < 0.20
    assert desk.score.beta_penalty < naive.score.beta_penalty
    assert desk.score.absolute_pnl > naive.score.absolute_pnl
    assert desk.score.risk_adjusted > naive.score.risk_adjusted
    meta = next(e for e in desk.events if e["type"] == "HedgeConstraint")
    assert meta["futures_in_sim"] is False
    assert meta["hedge"] == "irl_futures_basis"
    naive_meta = next(e for e in naive.events if e["type"] == "HedgeConstraint")
    assert naive_meta["hedge"] == "none"


def test_pm_fair_value_is_market_neutral():
    naive, desk = run_pair("pm_fair_value")
    assert desk.score.mean_abs_beta < 1e-9
    assert naive.score.mean_abs_beta > 1.0
    assert desk.score.risk_adjusted > naive.score.risk_adjusted
    book = next(e for e in desk.events if e["type"] == "FairValueBook")
    assert book["names"] == 3


def test_etf_sizes_before_the_edge_dies():
    naive, desk = run_pair("etf_ap_arb")
    assert desk.score.absolute_pnl > naive.score.absolute_pnl
    assert desk.score.risk_adjusted > naive.score.risk_adjusted
    snap = next(e for e in desk.events if e["type"] == "DecisionSnapshot")
    assert snap["source"] == "fallback"
    assert snap["size_tier"] == "large"
    assert "order" not in snap["note"]


def test_near_risk_free_choice_is_large_and_not_an_order():
    state = build_mm_state(
        time=0.0, spot=100.0, option_mid=1.0, iv=0.2, inventory=0,
        delta=0.0, gamma=0.0, vega=0.0, cash_pnl=0.0, half_spread=0.4, quoting_allowed=True,
    )
    state["arb"] = {"near_risk_free": True}
    ans = DeterministicFallbackClient().system_one(state)
    assert ans.answers["size_tier"].choice == "large"
    assert ans.source == "fallback"


def test_facilitator_cuts_informed_inventory():
    naive, desk = run_pair("liability_facilitator")
    assert desk.score.mean_abs_inventory < naive.score.mean_abs_inventory
    assert desk.score.absolute_pnl > 0.0
    assert desk.score.risk_adjusted > naive.score.risk_adjusted
    block = desk.events[0]
    assert block["flow_class"] == "forced"
    assert block["tox_prior"] < 0.25
    assert block["slice_gap"] == 1
    assert any(e.get("flow_class") == "discretionary" and e.get("discretionary_take") == 1.0 for e in naive.events)
    assert not any(e.get("discretionary_take") == 1.0 for e in desk.events)


def test_mm_inventory_beats_the_wave():
    from jev_omm.training.cases import run_case

    naive, desk = run_pair("mm_inventory")
    pred = run_case("mm_inventory", "predatory", 0.0)
    assert desk.score.absolute_pnl > 0.0
    assert desk.score.mean_abs_inventory < naive.score.mean_abs_inventory
    assert desk.score.risk_adjusted > naive.score.risk_adjusted
    assert pred.score.absolute_pnl > desk.score.absolute_pnl
    assert desk.score.risk_adjusted > pred.score.risk_adjusted
    race = desk.events[0]
    assert race["cancel_latency"] < 0.9
    assert race["spread_pnl"] > 0.0
    assert race["graded"] is True
    assert race["flow_classes"] == ["forced", "discretionary"]
    assert race["tox_prior_forced"] < race["tox_prior_discretionary"]
    assert pred.events[0]["graded"] is False


def test_vol_surface_refuses_arb():
    naive, desk = run_pair("vol_surface_mm")
    assert desk.score.risk_adjusted > naive.score.risk_adjusted
    assert desk.score.exec_penalty < naive.score.exec_penalty
    gate = desk.events[0]
    assert gate["butterfly_clean"] is True
    assert gate["butterfly_poison"] is False
    assert gate["calendar_ok"] is False
    assert gate["iv_sticky_strike"] != gate["iv_sticky_delta"]


def test_jsonl_replay_and_leaderboard(tmp_path):
    from jev_omm.training.replay import all_runs

    runs = all_runs()
    path = write_jsonl(tmp_path / "cases.jsonl", runs)
    checked = replay_scores(path)
    assert len(checked) == 2 * len(CASES)
    board = leaderboard([r for r in runs if r.strategy == "desk"])
    assert board[0].score.risk_adjusted >= board[-1].score.risk_adjusted
    assert {r.spec.name for r in board} == set(CASES)
