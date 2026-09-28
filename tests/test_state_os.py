"""Latent-state OS: instability gate, stand-up engines, research cores, warehouse."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

from jev_omm.decisions.client import DeterministicFallbackClient
from jev_omm.decisions.policy import apply_policy
from jev_omm.decisions.schemas import build_mm_questions, build_mm_state
from jev_omm.positioning.adjust import apply_features
from jev_omm.positioning.gex import OptionOI
from jev_omm.state_os.constraints import (
    active_constraints,
    gate_blocks,
    price_floor_level,
    time_level,
    var_sigma_star,
    vol_target_level,
)
from jev_omm.state_os.cross_impact import common_flow_trap, hub_spoke_impact, identified_cross_beta
from jev_omm.state_os.engines import (
    auction_imbalance,
    borrow_pressure,
    clock_pressure,
    cta_weight,
    gen1_roll_dates,
    gen3_coverage,
    glide_equity_weight,
    in_issuer_blackout,
    in_slr_window,
    letf_rebalance,
    lookback_vols,
    net_liquidity,
    net_liquidity_change,
    pension_equity_trade,
    realized_vol,
    risk_parity_weights,
    settlement_gap_hours,
    third_friday,
    tdf_trade,
    trend_signal,
    vol_control_boundary,
    vol_target_weight,
    withheld_buyback,
)
from jev_omm.state_os.funding import scarcity_rent
from jev_omm.state_os.gate import state_gate
from jev_omm.state_os.gex_flow import FlowLeg, flow_signed_gex, gex_pair, spx_vanna_charm_es
from jev_omm.state_os.hawkes_tv import classify_move, n_t
from jev_omm.state_os.impact import correlation, impact_residual, predicted_impact
from jev_omm.state_os.latent_book import LatentBook, d_cancel_d_sigma, expectations, liquidity_surface
from jev_omm.state_os.metaorder import (
    estimate_remaining,
    is_live_parent,
    sqrt_impact,
    synthetic_parent,
)
from jev_omm.state_os.vector import (
    MarketState,
    forced_flow,
    instability,
    l_exec_on_path,
    net_forced,
)
from jev_omm.state_os.warehouse import (
    autocall_flow,
    hedge_by_regime,
    ldi_cash_need,
    mass_on_cusp,
    mbs_hedge,
    rila_issuer_delta,
    spend_flow,
    tape_residual,
    visible_gex,
)
from jev_omm.flow.hawkes import HawkesParams
from jev_omm.training.cases import run_pair
from jev_omm.training.scoring import lcg_next

FIX = Path(__file__).resolve().parents[1] / "jev_omm" / "data" / "fixtures"


def test_instability_and_path_liquidity():
    assert forced_flow([(0.5, 10.0), (0.5, -4.0)]) == 3.0
    assert net_forced(3.0, 1.0, 0.5) == 1.5
    assert instability(80.0, 20.0) == 4.0
    assert instability(0.0, 0.0) == 0.0
    assert math.isinf(instability(5.0, 0.0))
    thinner = l_exec_on_path(40.0, 10.0, dL_dQ=-2.0)
    assert thinner == 20.0
    assert instability(10.0, thinner) > instability(10.0, 40.0)
    quiet = MarketState(price=100.0, sigma=0.2, f_forced=80.0, l_exec=20.0, enabled=False)
    live = MarketState(price=100.0, sigma=0.2, f_forced=80.0, l_exec=20.0, enabled=True)
    assert quiet.instability() == 0.0
    assert live.instability() == 4.0
    assert quiet.latent_features()["enabled"] == 0.0


def test_state_gate_is_identity_when_off_and_pulls_when_on():
    off = state_gate(False, 5.0, True, 0.9, -10.0, 1.0, 100.0)
    assert off.spread_mult == 1.0 and off.size_mult == 1.0 and off.pull is False
    calm = state_gate(True, 0.0, False, 0.0, 0.0, 10.0, 100.0)
    assert calm.spread_mult == 1.0 and calm.size_mult == 1.0 and calm.reservation_shift == 0.0
    hot = state_gate(True, 4.0, False, 0.0, -80.0, 20.0, 100.0)
    assert hot.pull is True and hot.size_mult == 0.0
    assert hot.spread_mult == 1.0 + 1.60
    parent = state_gate(True, 0.0, False, 0.5, 0.0, 1.0, 100.0)
    assert parent.pull is False
    assert abs(parent.size_mult - 1.0 / 1.375) < 1e-12
    blocked = state_gate(True, 0.0, True, 0.0, 0.0, 1.0, 100.0)
    assert blocked.pull is True and blocked.hedge_urgency >= 0.70
    same = apply_features()
    assert same.spread_mult == 1.0 and same.size_mult == 1.0
    still = apply_features(state_enabled=False, instability=4.0, constraint_active=True, parent_remaining=0.8)
    assert still.spread_mult == 1.0 and still.size_mult == 1.0
    gated = apply_features(state_enabled=True, instability=4.0, f_signed=-80.0, l_exec=20.0, mid=100.0)
    assert gated.size_mult == 0.0
    assert gated.spread_mult > 1.0


def test_fallback_maps_latent_state_into_adjustments():
    calm = build_mm_state(
        time=0.0, spot=100.0, option_mid=2.0, iv=0.2, inventory=0,
        delta=0.1, gamma=0.01, vega=4.0, cash_pnl=0.0, half_spread=0.1, quoting_allowed=True,
    )
    base = apply_policy(DeterministicFallbackClient().system_one(calm, build_mm_questions()), state=calm)
    hot_state = build_mm_state(
        time=0.0, spot=100.0, option_mid=2.0, iv=0.2, inventory=0,
        delta=0.1, gamma=0.01, vega=4.0, cash_pnl=0.0, half_spread=0.1, quoting_allowed=True,
        latent={"enabled": 1.0, "instability": 4.0, "f_net": -80.0, "l_exec": 20.0},
    )
    hot = apply_policy(DeterministicFallbackClient().system_one(hot_state, build_mm_questions()), state=hot_state)
    assert hot.pull is True
    assert hot.size_mult == 0.0
    assert hot.instability == 4.0
    assert hot.spread_mult > base.spread_mult
    parent_state = build_mm_state(
        time=0.0, spot=100.0, option_mid=2.0, iv=0.2, inventory=0,
        delta=0.1, gamma=0.01, vega=4.0, cash_pnl=0.0, half_spread=0.1, quoting_allowed=True,
        latent={"enabled": 1.0, "parent_remaining": 0.8},
    )
    parent = DeterministicFallbackClient().system_one(parent_state, build_mm_questions())
    assert parent.answers["size_tier"].choice == "tiny"
    assert parent.answers["pull_quotes"].noul < 0.70


def test_standup_engines_and_fixtures():
    assert letf_rebalance(100.0, 3.0, 0.02) == 12.0
    assert letf_rebalance(50.0, -1.0, 0.01) == 1.0
    assert letf_rebalance(100.0, 1.0, 0.05) == 0.0
    book = json.loads((FIX / "state_os_net_liq.json").read_text())
    assert net_liquidity(book["fed_assets"], book["tga"], book["rrp"]) == 6100.0
    change = net_liquidity_change(book["d_fed"], book["d_tga"], book["d_rrp"])
    assert change["delta"] == book["d_fed"] - book["d_tga"] - book["d_rrp"]
    assert abs(sum(change[k] for k in ("fed_assets", "tga", "rrp")) - 1.0) < 1e-12
    names = json.loads((FIX / "state_os_buyback.json").read_text())["names"]
    flags = []
    paces = []
    for row in names:
        asof = date(2024, 6, 20)
        qe = date.fromisoformat(row["quarter_end"])
        earn = date.fromisoformat(row["earnings"])
        flags.append(in_issuer_blackout(asof, qe, earn))
        paces.append(row["pace"])
        assert in_issuer_blackout(date(2024, 5, 1), qe, earn) is False
    assert all(flags)
    assert withheld_buyback(paces, flags) == 16.0
    exact = pension_equity_trade(1000.0, 0.60, 0.10, 0.0, exact=True)
    linear = pension_equity_trade(1000.0, 0.60, 0.10, 0.0, exact=False)
    assert exact < 0.0 and linear < 0.0
    assert abs(linear - 1000.0 * 0.6 * 0.4 * (-0.10)) < 1e-12
    lend = json.loads((FIX / "state_os_sec_lending.json").read_text())
    tight = borrow_pressure(lend["utilization"], lend["fee"], lend["days_to_cover"], lend["delta_lendable"])
    loose = borrow_pressure(0.10, 0.001, 2.0, 0.0)
    assert tight > loose
    assert auction_imbalance(80.0, 20.0) == 0.6
    assert auction_imbalance(0.0, 0.0) == 0.0
    assert third_friday(2024, 1) == date(2024, 1, 19)
    assert len(gen1_roll_dates(2024)) == 12
    assert in_slr_window(date(2024, 6, 28)) is True
    assert in_slr_window(date(2024, 6, 20)) is False
    assert settlement_gap_hours("PM") == 0.0
    assert settlement_gap_hours("AM") == 17.5
    assert clock_pressure(opex=True, slr=True) > clock_pressure()
    returns = [0.001] * 249 + [0.05]
    vols = lookback_vols(returns)
    assert set(vols) == {20, 60, 120, 250}
    assert vols[20] > realized_vol([0.001] * 250, 20)
    boundary = vol_control_boundary(0.10, 1.0)
    assert vol_target_weight(boundary - 0.01, 0.10, 1.0) == 1.0
    assert vol_target_weight(boundary + 0.05, 0.10, 1.0) < 1.0
    prices = [100.0 + i for i in range(30)]
    assert trend_signal(prices, 20) > 0.0
    assert cta_weight(0.02, 0.20, 0.10, 1.0) > 0.0
    weights = risk_parity_weights([0.10, 0.20])
    assert abs(weights[0] - 2.0 / 3.0) < 1e-12
    assert abs(sum(weights) - 1.0) < 1e-12
    assert abs(tdf_trade(0.625, 0.60, 1000.0) - (-7.5)) < 1e-9
    assert tdf_trade(0.61, 0.60, 1000.0) == 0.0
    assert abs(glide_equity_weight(40.0) - 0.90) < 1e-12
    assert abs(glide_equity_weight(0.0) - 0.30) < 1e-12
    assert abs(gen3_coverage(0.28, 0.18) - 0.70) < 1e-12
    assert abs(gen3_coverage(0.10, 0.18) - 0.34) < 1e-12


def test_flow_signed_gex_disagrees_and_spx_hedge_is_not_spy():
    oi = [OptionOI(strike=100.0, call_oi=50.0, put_oi=50.0, iv=0.20, t=30.0 / 365.0)]
    sold = [FlowLeg(strike=100.0, iv=0.20, t=30.0 / 365.0, customer_call_volume=-40.0, customer_put_volume=-40.0)]
    pair = gex_pair(100.0, oi, sold, "short_premium", 0.01, 0.0)
    assert pair["structural"] < 0.0
    assert pair["flow"] > 0.0
    assert pair["disagree"] is True
    flat = flow_signed_gex(100.0, [FlowLeg(100.0, 0.2, 0.1)], 0.01, 0.0)
    assert flat == 0.0
    es = spx_vanna_charm_es(100.0, oi, "short_premium", 0.01, 0.0, 1.0 / 252.0, -0.01, contract="SPX")
    spy = spx_vanna_charm_es(100.0, oi, "short_premium", 0.01, 0.0, 1.0 / 252.0, -0.01, contract="SPY")
    assert es["charm_es"] is not None and abs(es["charm_es"]) + abs(es["vanna_es"]) > 0.0
    assert abs(es["charm_es"] * 50.0 - es["charm_index_points"]) < 1e-9
    assert spy["charm_es"] is None and spy["vanna_es"] is None


def test_latent_book_slopes():
    book = LatentBook(
        displayed=100.0, half_life=2.0, deficit=40.0, dist=0.02, sigma=0.20,
        iceberg_prob=0.0, hidden_multiple=2.0,
    )
    base = expectations(book)
    hotter = expectations(LatentBook(**{**book.__dict__, "sigma": 0.40}))
    assert hotter["e_cancel"] > base["e_cancel"]
    assert hotter["l_exec"] < base["l_exec"]
    hidden = expectations(LatentBook(**{**book.__dict__, "iceberg_prob": 0.5}))
    assert hidden["e_hidden"] > 0.0 and hidden["l_exec"] > base["l_exec"]
    bumped = LatentBook(**{**book.__dict__, "sigma": book.sigma + 1e-4})
    numeric = (expectations(bumped)["e_cancel"] - base["e_cancel"]) / 1e-4
    assert abs(numeric - d_cancel_d_sigma(book)) / d_cancel_d_sigma(book) < 1e-3
    slopes = liquidity_surface(book)
    assert slopes["dL_dSigma"] < 0.0


def test_metaorder_is_concave_and_noise_is_not():
    i1 = sqrt_impact(100.0, 0.2, 10_000.0, 1.0)
    i4 = sqrt_impact(400.0, 0.2, 10_000.0, 1.0)
    assert abs(i4 - 2.0 * i1) < 1e-12
    assert i4 < 4.0 * i1
    path = synthetic_parent(1.0, 0.2, 10_000.0, 4.5e-5, 50.0, 12, 0.15, 0.35)
    assert is_live_parent(path["prices"], path["flows"])
    peak = max(path["prices"])
    assert path["prices"][-1] < peak
    assert path["prices"][-1] > path["permanent"]
    assert path["prices"][-1] > path["prices"][0]
    mid = 3
    remain = estimate_remaining(path["executed"][mid], path["flows"][mid], 1.0, 0.2, 10_000.0, 4.5e-5)
    assert abs(remain - path["remaining"][mid]) < 1e-6
    assert remain > 0.0
    assert estimate_remaining(path["executed"][-1], 0.0, 1.0, 0.2, 10_000.0, 4.5e-5) == 0.0
    noise_px = [math.sin(i / 2.0) for i in range(24)]
    noise_flow = [1.0] * 24
    assert is_live_parent(noise_px, noise_flow) is False


def test_impact_residual_is_not_the_raw_flow():
    y, sigma, volume, depth, spread = 0.8, 0.2, 5000.0, 100.0, 0.02
    state = 5
    qs = []
    dps = []
    eps = []
    for _ in range(40):
        state, z = lcg_next(state)
        q = 40.0 + 80.0 * (z + 1.0)
        state, noise = lcg_next(state)
        eps_i = 0.0004 * noise
        dp = predicted_impact(q, sigma, depth, volume, spread, y) + eps_i
        qs.append(q)
        dps.append(dp)
        eps.append(impact_residual(dp, q, sigma, depth, volume, spread, y))
    assert abs(correlation(eps, qs)) < abs(correlation(dps, qs))
    assert abs(correlation(eps, qs)) < 0.2


def test_constraints_funding_cross_impact_and_hawkes():
    clear = vol_target_level(0.12, 0.20)
    bound = vol_target_level(0.28, 0.20)
    assert gate_blocks([clear, price_floor_level(100.0, 90.0), time_level(0.1, 1.0)]) is False
    assert gate_blocks([bound]) is True
    assert active_constraints([clear, bound]) == [bound]
    near = var_sigma_star(1.0, 2.0, 1.0 / 252.0, 10.0, 100.0)
    far = var_sigma_star(1.0, 2.0, 20.0 / 252.0, 10.0, 100.0)
    assert far < near
    assert scarcity_rent(0.50, 0.0, 0.0, 0.0, 0.0) < scarcity_rent(0.0, 0.25, 0.0, 0.0, 0.0)
    trap = common_flow_trap()
    assert abs(trap["naive"]) > 0.45
    assert abs(trap["identified"]) < 0.15
    cross = identified_cross_beta()
    assert abs(cross["identified"] - cross["true"]) < 0.08
    hub, spoke = hub_spoke_impact(2.0, 1.0, 0.4, 0.3, 0.5)
    assert hub == 0.8 and abs(spoke - 1.1) < 1e-12
    params = HawkesParams(mu=1.0, alpha=0.8, beta=1.0)
    quiet = n_t(params, 10.0, [])
    burst = n_t(params, 5.0, [4.90, 4.92, 4.94, 4.96, 4.98])
    assert quiet < 0.05
    assert burst > 0.75
    assert classify_move(0.02, burst) == "endogenous"
    assert classify_move(0.02, quiet) == "exogenous"
    assert classify_move(0.0, burst) == "quiet"


def test_warehouse_geometry():
    near = autocall_flow(100.0, 100.0, 100.0, 0.20, 0.25, [100.0, 90.0], 0.2, 0.01, 0.3, 2.0)
    far = autocall_flow(130.0, 100.0, 70.0, 0.20, 0.25, [100.0, 90.0], 0.9, 0.0, 0.9, 2.0)
    assert near.digital > far.digital
    assert near.knock_in > far.knock_in
    assert near.worst_of > far.worst_of
    assert near.duration > far.duration
    flat = rila_issuer_delta(100.0, cap=10.0, buffer=0.0, t=1.0, rate=0.02, div_yield=0.0, iv=0.2)
    lived = rila_issuer_delta(100.0, cap=0.10, buffer=0.10, t=1.0, rate=0.02, div_yield=0.0, iv=0.2)
    assert abs(flat) < 1e-6
    assert abs(lived) > 0.01
    assert hedge_by_regime(1.0, -2.0, "economic") == 1.0
    assert hedge_by_regime(1.0, -2.0, "statutory") == -2.0
    assert hedge_by_regime(1.0, -2.0, "rating") == -2.0 * 1.15
    cusp = abs(mbs_hedge(0.05, 0.05, -0.01, 1_000_000.0))
    deep = mbs_hedge(0.02, 0.05, -0.01, 1_000_000.0)
    assert cusp > 0.0 and deep == 0.0
    assert ldi_cash_need(1_000_000.0, -0.01, 5.0, velocity=0.0) == 0.0
    assert ldi_cash_need(1_000_000.0, -0.01, 5.0, velocity=0.01) > 0.0
    assert spend_flow(1000.0, 0.04) == -40.0
    assert tape_residual(10.0, 7.0) == 3.0
    assert abs(visible_gex(-8.0, 10.0, 7.0) - (-2.4)) < 1e-12
    packed = mass_on_cusp([0.01, 0.02, 0.5, 0.9], 0.05)
    empty = mass_on_cusp([0.4, 0.5, 0.8], 0.05)
    assert packed > empty


def test_state_os_training_cases_teach_the_gate():
    names = (
        "letf_day",
        "instability_spike",
        "remaining_parent",
        "constraint_gate",
        "gex_disagree",
        "tdf_threshold",
        "overwrite_roll",
    )
    for name in names:
        naive, desk = run_pair(name)
        assert desk.score.absolute_pnl > naive.score.absolute_pnl
        assert desk.score.risk_adjusted > naive.score.risk_adjusted
    spike = run_pair("instability_spike")[1]
    snap = spike.events[0]
    assert snap["source"] == "fallback"
    assert snap["pull"] is True
    assert "order" not in snap["note"]
    gate = run_pair("constraint_gate")[1].events[0]
    assert gate["pull"] is True
    parent = run_pair("remaining_parent")[1].events[0]
    assert parent["size_tier"] == "tiny"
    assert parent["pull"] is False
    disagree = run_pair("gex_disagree")[1].events[0]
    assert disagree["size_tier"] == "tiny"
    tdf = run_pair("tdf_threshold")[1].events[0]
    assert abs(tdf["trade"] - (-7.5)) < 1e-9
    assert tdf["inside_band_trade"] == 0.0
