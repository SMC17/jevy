"""Banded hedge + greek PnL."""

from __future__ import annotations

from jev_omm.hedge.delta import (
    apply_hedge,
    greek_pnl_step,
    net_delta,
    propose_delta_hedge,
    whalley_wilmott_band,
)


def test_within_band_none():
    assert propose_delta_hedge(3.0, band=5.0) is None


def test_flatten_above_band():
    o = propose_delta_hedge(12.0, band=5.0, flatten=True)
    assert o is not None
    assert abs(o.underlier_qty - (-12.0)) < 1e-12


def test_to_band_edge():
    o = propose_delta_hedge(12.0, band=5.0, flatten=False)
    assert o is not None
    assert abs(o.underlier_qty - (-7.0)) < 1e-12


def test_slippage_and_apply():
    o = propose_delta_hedge(5.0, band=1.0, half_spread=0.10)
    assert o is not None
    f = apply_hedge(0.0, 100.0, o, half_spread=0.10, band=1.0)
    assert abs(f.fill_price - 99.90) < 1e-9  # sell
    assert f.slippage_cost < 0


def test_gamma_pnl_half_gamma_ds2():
    step = greek_pnl_step(delta=0, gamma=0.04, vega=0, theta=0, d_spot=2.0)
    assert abs(step.gamma_pnl - 0.08) < 1e-12


def test_ww_band_increases_with_cost():
    b1 = whalley_wilmott_band(100, 0.05, 0.2, 0.0001, 1e-3)
    b2 = whalley_wilmott_band(100, 0.05, 0.2, 0.001, 1e-3)
    assert b1 > 0 and b2 > b1


def test_net_delta():
    assert abs(net_delta(8.0, -8.0)) < 1e-12
