"""Combo package theos."""

from __future__ import annotations

import math

from jev_omm.pricing.combos import butterfly_call_theo, straddle_theo, vertical_call_theo


def test_forward_atm_straddle_delta_near_zero():
    s, t, r, q, iv = 100.0, 0.25, 0.05, 0.0, 0.20
    k = s * math.exp((r - q) * t)
    pkg = straddle_theo(s, k, t, r, q, iv)
    assert pkg.theo > 0
    assert abs(pkg.greeks.delta) < 0.05
    assert pkg.greeks.gamma > 0
    assert pkg.greeks.vega > 0


def test_fly_short_body_gamma():
    fly = butterfly_call_theo(100, 95, 100, 105, 5 / 365.25, 0.05, 0.0, 0.2, 0.2, 0.2)
    assert fly.theo > 0
    assert fly.theo < 5.0
    assert fly.greeks.gamma < 0


def test_vertical_bounded_by_width():
    pkg = vertical_call_theo(100, 95, 105, 0.25, 0.05, 0.0, 0.2, 0.2)
    assert 0 < pkg.theo < 10
    assert pkg.greeks.delta > 0
