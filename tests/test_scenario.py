"""Scenario risk matrix."""

from __future__ import annotations

from jev_omm.models.types import Greeks
from jev_omm.risk.scenario import build_matrix, build_matrix_reprice, taylor_pnl


def test_flat_book():
    m = build_matrix(Greeks(delta=0, gamma=0, vega=0, theta=0), 100.0, soft_loss=1e9, hard_loss=1e9)
    assert abs(m.min_pnl) < 1e-12
    assert not m.soft_breach


def test_long_delta_downside():
    m = build_matrix(
        Greeks(delta=10.0, gamma=0, vega=0, theta=0),
        100.0,
        spot_shocks=(-0.05, 0.0, 0.05),
        iv_shocks=(0.0,),
        soft_loss=1e9,
        hard_loss=1e9,
    )
    assert abs(m.at(0, 0).pnl - (-50.0)) < 1e-9
    assert m.at(2, 0).pnl > 0


def test_long_gamma_convex():
    m = build_matrix(
        Greeks(delta=0, gamma=0.1, vega=0, theta=0),
        100.0,
        spot_shocks=(-0.02, 0.02),
        iv_shocks=(0.0,),
        soft_loss=1e9,
        hard_loss=1e9,
    )
    assert abs(m.at(0, 0).pnl - 0.2) < 1e-9
    assert m.at(1, 0).pnl > 0


def test_hard_breach():
    m = build_matrix(
        Greeks(delta=100.0, gamma=0, vega=0, theta=0),
        100.0,
        spot_shocks=(-0.05,),
        iv_shocks=(0.0,),
        soft_loss=10.0,
        hard_loss=100.0,
    )
    assert m.hard_breach and m.soft_breach


def test_reprice_runs():
    m = build_matrix_reprice(
        100, 100, 0.25, 0.05, 0.0, 0.2, True, -10.0,
        spot_shocks=(-0.1, 0.0, 0.1),
        iv_shocks=(0.0, 0.05),
    )
    assert m.min_pnl < 0


def test_taylor_matches_native_path():
    g = Greeks(delta=5, gamma=0.02, vega=10, theta=-1)
    assert abs(taylor_pnl(g, 1.0, 0.01) - (5 + 0.5 * 0.02 + 0.1)) < 1e-9
