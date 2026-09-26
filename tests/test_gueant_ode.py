"""Guéant ODE / spectral quotes vs asymptotics, and (A, k) recovery."""

from __future__ import annotations

import math

from jev_omm.config import QuoterConfig
from jev_omm.quoter.avellaneda_stoikov import make_quote
from jev_omm.quoter.gueant import optimal_offsets as asym_offsets
from jev_omm.quoter.gueant_ode import (
    NO_QUOTE,
    IntensityObs,
    estimate_intensity,
    make_quote as ode_quote,
    optimal_offsets,
    spectral_offsets,
)


def _cfg(**kwargs) -> QuoterConfig:
    base = dict(
        gamma=0.1,
        kappa=1.5,
        sigma=0.5,
        A=140.0,
        T_horizon=2.0,
        inventory_cap=20,
        ode_steps=2000,
        min_half_spread=1e-6,
        max_half_spread=50.0,
        mode="gueant_ode",
    )
    base.update(kwargs)
    return QuoterConfig(**base)


def test_ode_matches_asymptotic_near_flat_inventory():
    cfg = _cfg()
    ode = optimal_offsets(cfg, 0)
    db, da = asym_offsets(cfg, 0)
    assert abs(ode.delta_b - db) / db < 0.02
    assert abs(ode.delta_a - da) / da < 0.02
    ode5 = optimal_offsets(cfg, 2)
    assert ode5.delta_b > ode.delta_b
    assert ode5.delta_a < ode.delta_a


def test_ode_deep_inventory_wider_than_asymptotic():
    cfg = _cfg(inventory_cap=10, T_horizon=3.0, ode_steps=3000)
    ode = optimal_offsets(cfg, 9)
    db, _da = asym_offsets(cfg, 9)
    assert ode.delta_b > db * 1.15
    at_cap = optimal_offsets(cfg, 10)
    assert at_cap.delta_b >= NO_QUOTE * 0.5
    q = ode_quote(5.0, 10, cfg)
    assert q.bid_size == 0 and q.ask_size > 0


def test_spectral_agrees_with_asymptotic_at_flat():
    cfg = _cfg(inventory_cap=12, T_horizon=1.0)
    spec = spectral_offsets(cfg, 0)
    db, da = asym_offsets(cfg, 0)
    assert abs(spec.delta_b - db) / db < 0.05
    assert abs(spec.delta_a - da) / da < 0.05
    deep = spectral_offsets(cfg, 6)
    assert deep.delta_b > spec.delta_b
    assert deep.delta_a < spec.delta_a


def test_estimator_recovers_planted_ak():
    A, k = 40.0, 1.2
    deltas = [0.2, 0.4, 0.7, 1.0, 1.4, 1.8]
    exposure = 500.0
    obs = [
        IntensityObs(d, exposure, round(A * math.exp(-k * d) * exposure))
        for d in deltas
    ]
    fit = estimate_intensity(obs)
    assert abs(fit.A - A) / A < 0.05
    assert abs(fit.k - k) / k < 0.05


def test_mode_toggle_routes_to_ode():
    cfg = _cfg(sigma=0.4, A=120.0, T_horizon=0.5, inventory_cap=6, ode_steps=400, min_half_spread=0.01, max_half_spread=20.0, quote_size=1)
    q = make_quote(4.0, 3, cfg)
    assert q.ask > q.bid
    assert q.reservation < 4.0
