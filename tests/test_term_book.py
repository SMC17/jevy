"""Multi-expiry greeks: bucket vega, term slope, vanna, volga, scenario."""

from __future__ import annotations

from jev_omm.config import QuoterConfig
from jev_omm.pricing.black_scholes import greeks
from jev_omm.risk.term import (
    Leg,
    TermLimits,
    aggregate,
    evaluate_limits,
    quote_expiries,
    scenario_pnl,
    scenario_reprice,
)
from jev_omm.surface.svi import SsviParams, ssvi_params_calendar_safe


def test_two_expiries_bucket_and_slope():
    spot = 100.0
    legs = [
        Leg(0.25, 100.0, 10.0, 0.2),
        Leg(1.0, 100.0, -4.0, 0.2),
    ]
    risk = aggregate(spot, 0.05, 0.0, legs)
    assert len(risk.buckets) == 2
    g_front = greeks(spot, 100.0, 0.25, 0.05, 0.0, 0.2, True)
    g_back = greeks(spot, 100.0, 1.0, 0.05, 0.0, 0.2, True)
    assert abs(risk.buckets[0].vega - 10.0 * g_front.vega) < 1e-8
    assert abs(risk.buckets[1].vega - (-4.0) * g_back.vega) < 1e-8
    assert abs(risk.vanna - (10 * g_front.vanna - 4 * g_back.vanna)) < 1e-6
    assert abs(risk.volga - (10 * g_front.volga - 4 * g_back.volga)) < 1e-6
    assert risk.term_vega_slope < 0.0
    assert abs(risk.term_vega_slope - risk.buckets[1].vega * 0.75) < 1e-8


def test_limits_and_vanna_volga_wing():
    risk = aggregate(100.0, 0.01, 0.0, [Leg(0.5, 100.0, 50.0, 0.22)])
    assert evaluate_limits(risk, TermLimits()) == "none"
    assert evaluate_limits(risk, TermLimits(max_abs_bucket_vega=1.0)) == "bucket_vega"
    wing = aggregate(100.0, 0.01, 0.0, [Leg(0.5, 80.0, 40.0, 0.3)])
    assert abs(wing.vanna) > 1.0 and abs(wing.volga) > 1.0
    assert evaluate_limits(wing, TermLimits(max_abs_vanna=0.5)) == "vanna"
    assert evaluate_limits(wing, TermLimits(max_abs_volga=0.5)) == "volga"


def test_scenario_tilt_and_reprice():
    legs = [
        Leg(0.25, 100.0, 1.0, 0.2),
        Leg(1.0, 100.0, 8.0, 0.2),
    ]
    risk = aggregate(100.0, 0.0, 0.0, legs)
    phi = 0.01
    pnl = scenario_pnl(risk, 100.0, 0.0, 0.0, phi)
    expected = 0.0
    for b in risk.buckets:
        d_iv = phi * (b.expiry - risk.t_front)
        expected += b.vega * d_iv + 0.5 * b.volga * d_iv * d_iv
    assert abs(pnl - expected) < 1e-8
    assert pnl > 0.0
    repriced = scenario_reprice(100.0, 0.0, 0.0, legs, 0.0, 0.01, 0.0, risk.t_front)
    taylor = scenario_pnl(risk, 100.0, 0.0, 0.01, 0.0)
    assert repriced > 0.0
    assert abs(repriced - taylor) / abs(repriced) < 0.05


def test_multi_expiry_strip_quotes():
    smile = SsviParams(-0.35, 0.9, 0.4)
    assert ssvi_params_calendar_safe(smile)
    expiries = [30 / 365.25, 120 / 365.25]
    strikes = [95.0, 100.0, 105.0]
    quoter = QuoterConfig(
        gamma=0.1, kappa=1.5, sigma=0.4, A=100.0, mode="gueant_asymptotic",
        quote_size=1, min_half_spread=0.01, max_half_spread=5.0,
    )
    rows = quote_expiries(100.0, 0.05, 0.0, smile, 0.22, expiries, strikes, quoter, True)
    assert len(rows) == 2
    assert rows[1]["theta"] > rows[0]["theta"]
    assert rows[1]["quotes"][1]["mid"] > rows[0]["quotes"][1]["mid"]
    assert rows[0]["quotes"][0]["iv"] > rows[0]["quotes"][2]["iv"]
    q = rows[0]["quotes"][1]["quote"]
    assert q.ask > q.bid


def test_vanna_finite_difference():
    g = greeks(100.0, 100.0, 0.5, 0.03, 0.01, 0.25, True)
    eps = 1e-4
    up = greeks(100.0, 100.0, 0.5, 0.03, 0.01, 0.25 + eps, True).delta
    dn = greeks(100.0, 100.0, 0.5, 0.03, 0.01, 0.25 - eps, True).delta
    assert abs(g.vanna - (up - dn) / (2 * eps)) < 1e-4
    g_put = greeks(100.0, 100.0, 0.5, 0.03, 0.01, 0.25, False)
    assert abs(g.vanna - g_put.vanna) < 1e-12
