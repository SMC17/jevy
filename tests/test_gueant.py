"""Guéant asymptotics: inventory skew monotonicity + limit behavior."""

from __future__ import annotations

from jev_omm.config import QuoterConfig
from jev_omm.models.types import Greeks
from jev_omm.quoter.gueant import (
    intensity_half,
    inventory_scale,
    make_quote,
    optimal_half_spread,
    optimal_offsets,
    reservation_price,
)


def test_inventory_scale_positive():
    cfg = QuoterConfig(gamma=0.1, kappa=1.5, sigma=0.5, A=140.0, mode="gueant_asymptotic")
    assert inventory_scale(cfg) > 0.0


def test_reservation_decreases_with_long_inventory():
    cfg = QuoterConfig(gamma=0.2, kappa=1.5, sigma=0.5, A=100.0, mode="gueant_asymptotic")
    mid = 5.0
    r_short = reservation_price(mid, -5, cfg)
    r_flat = reservation_price(mid, 0, cfg)
    r_long = reservation_price(mid, 5, cfg)
    assert r_long < r_flat < r_short


def test_quotes_skew_monotone_in_inventory():
    cfg = QuoterConfig(
        gamma=0.15, kappa=1.5, sigma=0.4, A=120.0, quote_size=1, mode="gueant_asymptotic"
    )
    q_long = make_quote(4.0, 10, cfg)
    q_short = make_quote(4.0, -10, cfg)
    assert q_long.reservation < q_short.reservation
    assert q_long.bid < q_short.bid
    assert q_long.ask < q_short.ask
    assert q_long.ask > q_long.bid


def test_offsets_long_sells_aggressively():
    cfg = QuoterConfig(gamma=0.1, kappa=1.5, sigma=0.5, A=140.0)
    db0, da0 = optimal_offsets(cfg, 0)
    db_l, da_l = optimal_offsets(cfg, 5)
    db_s, da_s = optimal_offsets(cfg, -5)
    assert da_l < da0  # long → tighter ask
    assert db_l > db0  # long → wider bid
    assert da_s > da0
    assert db_s < db0


def test_larger_A_shrinks_xi():
    lo = QuoterConfig(gamma=0.1, kappa=1.5, sigma=0.5, A=50.0)
    hi = QuoterConfig(gamma=0.1, kappa=1.5, sigma=0.5, A=500.0)
    assert inventory_scale(hi) < inventory_scale(lo)


def test_larger_sigma_grows_xi():
    lo = QuoterConfig(gamma=0.1, kappa=1.5, sigma=0.2, A=140.0)
    hi = QuoterConfig(gamma=0.1, kappa=1.5, sigma=0.8, A=140.0)
    assert inventory_scale(hi) > inventory_scale(lo)


def test_half_equals_psi_plus_half_xi():
    cfg = QuoterConfig(gamma=0.1, kappa=1.5, sigma=0.5, A=140.0, min_half_spread=0.01)
    h = optimal_half_spread(cfg)
    expected = intensity_half(cfg) + 0.5 * inventory_scale(cfg)
    assert abs(h - expected) < 1e-12


def test_greek_penalty_lowers_reservation_when_long():
    cfg = QuoterConfig(
        gamma=0.1, kappa=1.5, sigma=0.5, A=140.0, gamma_penalty=1.0, vega_penalty=1.0
    )
    g = Greeks(delta=0.5, gamma=0.02, vega=10.0, theta=-1.0)
    assert reservation_price(5.0, 5, cfg, g) < reservation_price(5.0, 5, cfg)


def test_mode_toggle_routes_via_as_module():
    from jev_omm.quoter.avellaneda_stoikov import reservation_price as as_r

    cfg = QuoterConfig(
        gamma=0.1, kappa=1.5, sigma=0.5, A=140.0, T_horizon=1.0 / 252.0, mode="gueant_asymptotic"
    )
    r = as_r(5.0, 8, cfg)
    assert r < 5.0
