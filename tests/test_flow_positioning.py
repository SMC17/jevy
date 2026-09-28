"""Flow, dealer gamma, and COT features. Defaults stay the identity."""

from __future__ import annotations

import pytest

from jev_omm.decisions.client import DeterministicFallbackClient
from jev_omm.decisions.policy import apply_policy
from jev_omm.decisions.schemas import build_mm_questions, build_mm_state
from jev_omm.flow.signals import BookEvent, LeeReady, flow_prior, ofi_increment, spoof_score
from jev_omm.hedge.delta import overlay_hedge_qty, scaled_delta_band
from jev_omm.positioning.adjust import apply_features, cot_fade, gex_adjust, rr_stress
from jev_omm.positioning.cot import (
    fetch_cot_rows,
    load_fixture,
    map_underlying,
    rows_for_underlying,
    trailing_z,
    week_over_week,
)
from jev_omm.positioning.gex import (
    OptionOI,
    charm_vanna_hedge,
    dollar_gex_1pct,
    max_pain,
    pin_level,
    put_call_oi_ratio,
    zero_gamma_level,
)
from jev_omm.positioning.overlays import (
    EtfFlow,
    annualized_roll,
    ap_pressure,
    futures_overlay_qty,
    near_risk_free,
    premium_bps,
    residual_beta,
)
from jev_omm.pricing.black_scholes import charm_tau, greeks
from jev_omm.training.cases import run_pair


def _state(**kwargs):
    base = dict(
        time=0.0, spot=100.0, option_mid=2.0, iv=0.2, inventory=0,
        delta=0.2, gamma=0.01, vega=5.0, cash_pnl=0.0, half_spread=0.15, quoting_allowed=True,
    )
    base.update(kwargs)
    return build_mm_state(**base)


def test_apply_features_identity_and_golden_numbers():
    adj = apply_features()
    assert adj.reservation_shift == 0.0
    assert adj.spread_mult == 1.0
    assert adj.size_mult == 1.0
    assert adj.hedge_band_mult == 1.0
    assert adj.hedge_urgency == 0.0

    tox, spread, size = flow_prior(1.0, 0.0, 0.0, 0.0, 0.0)
    assert tox == 0.5
    assert spread == 1.0 + 1.25 * 0.5
    assert size == 1.0 / (1.0 + 1.5 * 0.5)

    shift, g_spread, g_size, band, urg = gex_adjust(True, 0.5, 0.01, 0.0, 100.0)
    assert shift == pytest.approx(0.075)
    assert g_spread == pytest.approx(0.9)
    assert g_size == pytest.approx(1.15)
    assert band == pytest.approx(1.25)
    assert urg == 0.0
    off = gex_adjust(False, 0.5, 0.01, 0.0, 100.0)
    assert off[1:] == (1.0, 1.0, 1.0, 0.0)

    c_shift, c_size = cot_fade(True, 4.0, 100.0)
    assert c_shift == pytest.approx(-0.15)
    assert c_size == pytest.approx(1.0 / 1.5)
    assert cot_fade(True, 0.0, 100.0) == (0.0, 1.0)
    assert cot_fade(False, 4.0, 100.0) == (0.0, 1.0)

    short = gex_adjust(True, -0.5, 0.0, 0.01, 100.0)
    assert short[1] == pytest.approx(1.0 + 0.70 * 0.5)
    assert short[2] < 1.0
    assert short[3] < 1.0
    assert short[4] > 0.0


def test_lee_ready_ofi_and_spoof():
    lr = LeeReady()
    assert lr.sign(100.2, 100.0, 100.2) == 1
    assert lr.sign(99.9, 99.8, 100.2) == -1
    assert lr.sign(100.0, 99.9, 100.1) == 1
    e = ofi_increment(100.0, 10.0, 100.2, 10.0, 100.1, 12.0, 100.2, 10.0)
    assert e > 0.0
    hot = spoof_score([
        BookEvent("cancel", 2, 50.0, 0.1),
        BookEvent("trade", 0, 1.0, 0.0),
    ])
    quiet = spoof_score([
        BookEvent("cancel", 2, 5.0, 0.1),
        BookEvent("trade", 0, 40.0, 0.0),
    ])
    assert hot > 0.9
    assert quiet < 0.2


def test_gex_postures_flip_and_max_pain_limit():
    legs = [
        OptionOI(90.0, call_oi=10.0, put_oi=400.0, iv=0.25, t=30.0 / 365.0),
        OptionOI(110.0, call_oi=400.0, put_oi=10.0, iv=0.25, t=30.0 / 365.0),
    ]
    short = dollar_gex_1pct(100.0, legs, "short_premium", 0.01, 0.0)
    assert short < 0.0
    assert zero_gamma_level(legs, "short_premium", 80.0, 120.0, 17, 0.01, 0.0) is None
    flip = zero_gamma_level(legs, "dashboard_flip", 80.0, 120.0, 21, 0.01, 0.0)
    assert flip is not None and 90.0 < flip < 110.0
    pain = max_pain(legs)
    assert pain in (90.0, 110.0)
    assert pin_level(-0.4, pain, flip, 110.0) is None
    assert pin_level(0.4, pain, flip, 110.0) is not None
    assert put_call_oi_ratio(legs) == pytest.approx(410.0 / 410.0)
    charm, vanna = charm_vanna_hedge(100.0, legs, "short_premium", 0.01, 0.0, 0.0, 0.0)
    assert charm == pytest.approx(0.0)
    assert vanna == pytest.approx(0.0)
    charm2, vanna2 = charm_vanna_hedge(
        100.0, legs, "short_premium", 0.01, 0.0, 1.0 / 252.0, -0.01
    )
    assert abs(charm2) + abs(vanna2) > 0.0


def test_charm_matches_a_calendar_step():
    spot, k, t, r, q, iv = 100.0, 100.0, 0.3, 0.02, 0.0, 0.2
    ch = charm_tau(spot, k, t, r, q, iv, True)
    dt = 1e-3
    d0 = greeks(spot, k, t, r, q, iv, True).delta
    d1 = greeks(spot, k, t - dt, r, q, iv, True).delta
    assert d1 == pytest.approx(d0 - ch * dt, abs=1e-5)
    assert charm_tau(spot, k, 1e-6, r, q, iv, True) == 0.0


def test_cot_fixtures_map_and_extreme_z_and_network_gate():
    dis = load_fixture("cot_disaggregated_sample.csv")
    tff = load_fixture("cot_tff_sample.csv")
    legacy = load_fixture("cot_legacy_sample.csv")
    crude = rows_for_underlying("CL", dis)
    assert len(crude) >= 5
    assert crude[-1].commercial_hedge_ratio > 1.0
    nets = [row.net_spec for row in crude]
    assert week_over_week(nets) > 0.0
    assert trailing_z(nets) > 2.0
    es = map_underlying("ES", tff)
    assert es is not None
    assert es.code == "13874A"
    assert "MICRO" not in es.market
    nq = map_underlying("NQ", tff)
    assert nq is not None and "MICRO" not in nq.market
    assert map_underlying("GC", dis) is not None
    assert legacy[0].report == "legacy"
    assert legacy[0].net_spec == 700000 - 420000
    with pytest.raises(RuntimeError, match="gated"):
        fetch_cot_rows("tff", "open_interest_all > 0", limit=1)


def test_etf_basis_and_factor_overlay():
    rich = EtfFlow(100.4, 100.0, 1.0, 1.0, 5_000_000, 500_000, residual_vol_bps=1.0, latency_steps=1)
    assert premium_bps(rich) == pytest.approx(40.0)
    assert ap_pressure(rich) > 0.5
    assert near_risk_free(rich) is True
    stale = EtfFlow(100.05, 100.0, 1.0, 1.0, 1.0, 1.0, residual_vol_bps=5.0, latency_steps=8)
    assert near_risk_free(stale) is False
    assert annualized_roll(80.0, 82.0, 30.0) > 0.0
    exposure = residual_beta([1.0, -1.0], [1.0, 0.55])
    assert exposure == pytest.approx(0.45)
    qf = futures_overlay_qty(exposure, 0.85)
    assert exposure + 0.85 * qf == pytest.approx(0.0)
    assert futures_overlay_qty(1.0, 0.0) == 0.0
    assert scaled_delta_band(5.0, 1.0) == 5.0
    assert overlay_hedge_qty(-2.0, 0.0, 0.0) == -2.0
    assert rr_stress(0.0) == 0.0
    assert rr_stress(0.05) == pytest.approx(0.5)


def test_fallback_conditions_on_flow_gex_and_cot_only_when_set():
    client = DeterministicFallbackClient()
    calm = client.system_one(_state(), build_mm_questions())
    assert calm.answers["regime"].choice == "calm"
    assert calm.answers["size_tier"].choice == "large"

    toxic = client.system_one(
        _state(toxicity_features={
            "vpin": 0.82, "ofi": 0.75, "aggr_imbalance": 0.8,
            "off_exchange_share": 0.55, "spoof": 0.7,
        }),
        build_mm_questions(),
    )
    mods = apply_policy(toxic)
    assert toxic.answers["size_tier"].choice == "tiny"
    assert toxic.answers["toxicity"].score > calm.answers["toxicity"].score
    assert mods.spread_mult > 1.0
    assert mods.size_mult < 1.0
    assert "order" not in (toxic.model or "")

    short = client.system_one(
        _state(positioning={"gex_enabled": 1.0, "gex_norm": -0.8}),
        build_mm_questions(),
    )
    short_mods = apply_policy(short)
    assert short.answers["regime"].choice == "stressed"
    assert short.answers["hedge_now"].noul >= 0.65
    assert short_mods.hedge_now is True
    assert short.answers["size_tier"].choice == "tiny"

    trending = client.system_one(_state(spot_return_bps=8.0), build_mm_questions())
    assert trending.answers["size_tier"].choice == "normal"
    long = client.system_one(
        _state(spot_return_bps=8.0, positioning={"gex_enabled": 1.0, "gex_norm": 0.8}),
        build_mm_questions(),
    )
    assert long.answers["size_tier"].choice == "large"
    assert long.answers["regime"].choice == "trending"

    crowded = client.system_one(_state(positioning={"cot_z": 3.1}), build_mm_questions())
    crowd_mods = apply_policy(crowded)
    assert crowded.answers["size_tier"].choice == "normal"
    assert crowd_mods.spread_mult > calm_spread(calm)


def calm_spread(result):
    return apply_policy(result).spread_mult


def test_training_flow_gamma_and_cot():
    naive, desk = run_pair("flow_vpin")
    assert desk.score.absolute_pnl > naive.score.absolute_pnl
    assert desk.score.risk_adjusted > naive.score.risk_adjusted
    assert desk.score.mean_abs_inventory < naive.score.mean_abs_inventory
    snap = next(e for e in desk.events if e["type"] == "DecisionSnapshot")
    assert snap["source"] == "fallback"
    assert snap["size_tier"] == "tiny"
    assert "ticket" in snap["note"]

    naive_g, desk_g = run_pair("dealer_gamma")
    assert desk_g.score.absolute_pnl > 0.0
    assert naive_g.score.absolute_pnl < 0.0
    assert desk_g.score.absolute_pnl > naive_g.score.absolute_pnl
    assert desk_g.score.risk_adjusted > naive_g.score.risk_adjusted
    gsnap = next(e for e in desk_g.events if e["type"] == "DecisionSnapshot")
    assert gsnap["hedge_flag"] is True
    assert gsnap["size_tier"] == "tiny"

    naive_c, desk_c = run_pair("cot_fade")
    assert desk_c.score.absolute_pnl > 0.0
    assert naive_c.score.absolute_pnl < 0.0
    assert desk_c.score.risk_adjusted > naive_c.score.risk_adjusted
    csnap = next(e for e in desk_c.events if e["type"] == "DecisionSnapshot")
    assert csnap["cot_shift"] < 0.0
    assert csnap["source"] == "fallback"
