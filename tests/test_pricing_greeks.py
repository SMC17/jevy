"""Pricing Greeks sanity checks."""

from __future__ import annotations

import math

from jev_omm.pricing.black_scholes import greeks, price


def test_call_put_parity_atm():
    s, k, t, r, q, iv = 100.0, 100.0, 0.25, 0.05, 0.0, 0.2
    c = price(s, k, t, r, q, iv, True)
    p = price(s, k, t, r, q, iv, False)
    # C - P = S*e^{-qT} - K*e^{-rT}
    lhs = c - p
    rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert abs(lhs - rhs) < 1e-8


def test_call_delta_between_0_1():
    g = greeks(100.0, 100.0, 0.25, 0.05, 0.0, 0.2, True)
    assert 0.0 < g.delta < 1.0
    assert g.gamma > 0.0
    assert g.vega > 0.0


def test_put_delta_between_m1_0():
    g = greeks(100.0, 100.0, 0.25, 0.05, 0.0, 0.2, False)
    assert -1.0 < g.delta < 0.0


def test_higher_spot_higher_call_price():
    p1 = price(100.0, 100.0, 0.25, 0.05, 0.0, 0.2, True)
    p2 = price(105.0, 100.0, 0.25, 0.05, 0.0, 0.2, True)
    assert p2 > p1
