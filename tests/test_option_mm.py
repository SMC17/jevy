"""Option-inventory MM: Baldacci grid, Stoikov–Sağlam Theorem 4, spot-vol hedge."""

from __future__ import annotations

import math

from jev_omm.config import QuoterConfig
from jev_omm.hedge.delta import spot_vol_hedge_qty
from jev_omm.quoter.avellaneda_stoikov import make_quote
from jev_omm.quoter.option_mm import (
    exponential_hamiltonian,
    logistic_intensity,
    paper_touch_probability,
    penalty_coeff,
    research_toy,
    solve_and_quote,
    stoikov_risk_scale,
    stoikov_saglam_premiums,
)


def test_exponential_hamiltonian_closed_form():
    h = exponential_hamiltonian(40.0, 2.0, 0.0, 0.0)
    assert abs(h - 20.0 / math.e) < 1e-12


def test_logistic_touch_matches_baldacci_fraction():
    frac = logistic_intensity(1.0, 0.7, 150.0, 10.0, 0.0)
    assert abs(frac - paper_touch_probability(0.7)) < 1e-12
    assert abs(paper_touch_probability(0.7) - 1.0 / (1.0 + math.exp(0.7))) < 1e-15
    assert logistic_intensity(1.0, 0.7, 150.0, 10.0, -0.1) > frac


def test_long_vega_widens_bid():
    cfg = research_toy()
    flat = solve_and_quote(cfg, 10.0, 0.0, 5.0)
    long = solve_and_quote(cfg, 10.0, 20.0, 5.0)
    short = solve_and_quote(cfg, 10.0, -20.0, 5.0)
    assert abs(flat.delta_b - flat.delta_a) < 1e-6
    assert long.delta_b > long.delta_a
    assert short.delta_a > short.delta_b
    assert long.reservation < flat.reservation < short.reservation
    assert long.bid < short.bid and long.ask < short.ask


def test_vega_limit_blocks_the_breaching_side():
    cfg = research_toy()
    cfg.vega_limit = 12.0
    q = solve_and_quote(cfg, 10.0, 10.0, 5.0)
    assert q.bid_blocked and q.bid_size == 0
    assert not q.ask_blocked and q.ask_size > 0


def test_vol_edge_buys_at_flat_vega():
    cfg = research_toy()
    cfg.vol_edge = 4.0
    q = solve_and_quote(cfg, 10.0, 0.0, 5.0)
    assert q.delta_b < q.delta_a


def test_iv_alpha_lifts_reservation():
    cfg = research_toy()
    base = solve_and_quote(cfg, 10.0, 0.0, 5.0)
    cfg.iv_alpha = 0.02
    rich = solve_and_quote(cfg, 10.0, 0.0, 5.0)
    assert abs((rich.reservation - base.reservation) - 0.10) < 1e-8
    assert rich.bid > base.bid and rich.ask > base.ask


def test_rho_shrinks_penalty():
    a = research_toy()
    b = research_toy()
    b.rho = 0.6
    assert penalty_coeff(b) < penalty_coeff(a)
    assert abs(penalty_coeff(a) - 0.5 / 8.0) < 1e-12


def test_stoikov_saglam_theorem4_toy():
    flat = stoikov_saglam_premiums(0.0, 5.0, 40.0, 200.0, 1.0)
    assert abs(flat.eps_ask - 0.1) < 1e-12 and abs(flat.eps_bid - 0.1) < 1e-12
    zero_q = stoikov_saglam_premiums(0.1, 0.0, 40.0, 200.0, 1.0)
    assert abs(zero_q.eps_ask - 0.15) < 1e-12
    long = stoikov_saglam_premiums(0.1, 5.0, 40.0, 200.0, 1.0)
    assert long.eps_ask < long.eps_bid
    assert abs(long.eps_ask) < 1e-12 and abs(long.eps_bid - 0.2) < 1e-12
    # Figure-2 style inputs: σ=0.01, overnight=1/252, α=0, Γ=0.02, S=100.
    k = stoikov_risk_scale(0.01, 1.0 / 252.0, 0.0, 0.25, 0.02, 100.0)
    assert k > 0.0


def test_mode_option_vega_skews_with_inventory():
    cfg = QuoterConfig(
        mode="option_vega",
        gamma=0.5,
        kappa=2.0,
        A=40.0,
        xi=1.0,
        contract_vega=5.0,
        vega_limit=40.0,
        T_horizon=0.25,
        option_grid_n=31,
        option_grid_steps=60,
        min_half_spread=0.0,
        max_half_spread=50.0,
        quote_size=1,
    )
    q_long = make_quote(10.0, 4, cfg)  # V = 20
    q_short = make_quote(10.0, -4, cfg)
    assert q_long.reservation < q_short.reservation
    assert q_long.bid < q_short.bid


def test_spot_vol_hedge_appendix_numbers():
    q = spot_vol_hedge_qty(0.5, rho=-0.5, xi=0.2, portfolio_vega=10.0, variance=0.04, spot=100.0)
    assert abs(q - (-0.475)) < 1e-12
    flat = spot_vol_hedge_qty(0.5, rho=0.0, xi=0.2, portfolio_vega=10.0, variance=0.04, spot=100.0)
    assert abs(flat - (-0.5)) < 1e-12
