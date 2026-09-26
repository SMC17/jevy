"""Multi-strike strip quoting."""

from __future__ import annotations

from jev_omm.config import QuoterConfig
from jev_omm.models.types import Greeks
from jev_omm.pricing import price_and_greeks
from jev_omm.quoter.multi_strike import (
    StripConfig,
    StrikeSlot,
    build_strike_grid,
    portfolio_greeks,
    quote_one,
    quote_strip,
)
from jev_omm.surface.sabr import SabrIVSurface


def test_build_strike_grid_centers():
    strikes = build_strike_grid(100.4, StripConfig(half_width=2, strike_step=1.0))
    assert len(strikes) == 5
    assert strikes[2] == 100.0
    assert strikes[0] == 98.0
    assert strikes[4] == 102.0


def test_portfolio_greeks_aggregate():
    slots = [
        StrikeSlot(99.0, inventory=2),
        StrikeSlot(100.0, inventory=-1),
    ]
    g = [
        Greeks(delta=0.4, gamma=0.02, vega=5.0, theta=-0.1),
        Greeks(delta=0.5, gamma=0.03, vega=6.0, theta=-0.2),
    ]
    p = portfolio_greeks(slots, g)
    assert abs(p.delta - (0.8 - 0.5)) < 1e-12
    assert p.net_inventory == 1


def test_quote_strip_five():
    sabr = SabrIVSurface(alpha=0.22, beta=1.0, rho=-0.3, nu=0.4)
    strip = StripConfig(half_width=2, strike_step=1.0)
    strikes = build_strike_grid(100.0, strip)
    slots = [StrikeSlot(k) for k in strikes]
    cfg = QuoterConfig(
        gamma=0.12,
        kappa=1.5,
        sigma=0.45,
        A=140.0,
        mode="gueant_asymptotic",
        portfolio_delta_penalty=0.02,
        quote_size=1,
    )

    def iv_fn(forward, k, t):
        return sabr._backend_iv(forward, k, t)

    out = quote_strip(
        spot=100.0,
        t_rem=30.0 / 365.25,
        rate=0.05,
        div_yield=0.0,
        slots=slots,
        cfg=cfg,
        price_fn=price_and_greeks,
        iv_fn=iv_fn,
        strip=strip,
    )
    assert len(out) == 5
    assert out[2].strike == 100.0
    assert out[0].quote.ask > out[0].quote.bid


def test_portfolio_delta_tilt():
    cfg = QuoterConfig(
        gamma=0.1,
        kappa=1.5,
        sigma=0.4,
        A=140.0,
        portfolio_delta_penalty=0.1,
        quote_size=1,
    )
    q0 = quote_one(5.0, 0, cfg, portfolio_delta=0.0)
    q1 = quote_one(5.0, 0, cfg, portfolio_delta=20.0)
    assert q1.reservation < q0.reservation
