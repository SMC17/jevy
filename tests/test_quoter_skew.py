"""Quoter inventory skew monotonicity."""

from __future__ import annotations

from jev_omm.config import QuoterConfig
from jev_omm.quoter.avellaneda_stoikov import make_quote, reservation_price


def test_reservation_decreases_with_long_inventory():
    cfg = QuoterConfig(gamma=0.2, sigma=0.5, T_horizon=1.0 / 252.0)
    mid = 5.0
    r_short = reservation_price(mid, inventory=-5, cfg=cfg)
    r_flat = reservation_price(mid, inventory=0, cfg=cfg)
    r_long = reservation_price(mid, inventory=5, cfg=cfg)
    assert r_long < r_flat < r_short


def test_quotes_skew_monotone_in_inventory():
    cfg = QuoterConfig(gamma=0.15, kappa=1.5, sigma=0.4, quote_size=1)
    mid = 4.0
    q_long = make_quote(mid, inventory=10, cfg=cfg)
    q_short = make_quote(mid, inventory=-10, cfg=cfg)
    # Long inventory → lower reservation → lower bid/ask to attract sells
    assert q_long.reservation < q_short.reservation
    assert q_long.bid < q_short.bid
    assert q_long.ask < q_short.ask
    assert q_long.ask > q_long.bid
