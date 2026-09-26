"""Put-call parity / synthetics / boxes."""

from __future__ import annotations

import math

from jev_omm.pricing.parity import (
    SideQuotes,
    box_spread,
    box_theo,
    parity_diff,
    parity_residual,
    synthetic_forward_edge,
)


def test_parity_identity():
    s, k, t, r, q = 100.0, 100.0, 0.25, 0.05, 0.02
    a = parity_diff(s, k, t, r, q)
    f = s * math.exp((r - q) * t)
    b = math.exp(-r * t) * (f - k)
    assert abs(a - b) < 1e-10


def test_residual_zero_on_parity():
    s, k, t, r, q = 100.0, 105.0, 0.5, 0.03, 0.01
    theo = parity_diff(s, k, t, r, q)
    assert abs(parity_residual(4.0, 4.0 - theo, s, k, t, r, q)) < 1e-10


def test_no_arb_on_parity_with_spreads():
    s, k, t, r, q = 100.0, 100.0, 0.25, 0.05, 0.0
    theo = parity_diff(s, k, t, r, q)
    half = 0.05
    c_mid, p_mid = 5.0, 5.0 - theo
    call = SideQuotes(c_mid - half, c_mid + half)
    put = SideQuotes(p_mid - half, p_mid + half)
    e = synthetic_forward_edge(call, put, s, k, t, r, q)
    assert e.reversal_edge < 0.0
    assert e.conversion_edge < 0.0


def test_detectable_conversion_when_call_rich():
    s, k, t, r, q = 100.0, 100.0, 0.25, 0.05, 0.0
    theo = parity_diff(s, k, t, r, q)
    half = 0.05
    c_mid = 6.0
    p_mid = c_mid - theo - 1.0
    call = SideQuotes(c_mid - half, c_mid + half)
    put = SideQuotes(p_mid - half, p_mid + half)
    e = synthetic_forward_edge(call, put, s, k, t, r, q)
    assert e.conversion_edge > 0.5


def test_box_theo_and_cheap_edge():
    k1, k2, t, r = 100.0, 110.0, 1.0, 0.05
    theo = box_theo(k1, k2, t, r)
    assert abs(theo - 10.0 * math.exp(-r * t)) < 1e-12
    cheap = 0.50
    target = theo - cheap
    c1 = SideQuotes(4.98, 5.00)
    c2 = SideQuotes(1.00, 1.02)
    p1 = SideQuotes(1.00, 1.02)
    p2_ask = target - c1.ask + c2.bid + p1.bid
    p2 = SideQuotes(p2_ask - 0.02, p2_ask)
    box = box_spread(c1, c2, p1, p2, k1, k2, t, r)
    assert abs(box.package_debit - target) < 1e-9
    assert box.buy_edge > 0.4
    assert box.implied_rate_buy > r
