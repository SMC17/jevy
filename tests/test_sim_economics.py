"""Clock, hedge accounting, rolling horizon, and the LOB fill backend."""

from __future__ import annotations

from jev_omm.backtest.simulator import run_simulation
from jev_omm.config import (
    TRADING_SECONDS_PER_YEAR,
    EngineConfig,
    MarketConfig,
    QuoterConfig,
    RiskConfig,
    SimConfig,
    session_remaining,
)
from jev_omm.hedge.delta import HedgeOrder, apply_hedge, book_hedge_fill
from jev_omm.models.types import Position
from jev_omm.pnl.mark import marked_pnl


def test_intensity_uses_seconds_not_a_year_fraction_hack():
    cfg = SimConfig()
    assert abs(cfg.dt_seconds - 60.0) < 1e-12
    assert abs(cfg.dt_years - 60.0 / TRADING_SECONDS_PER_YEAR) < 1e-15
    assert abs(cfg.dt_years - 1.0 / (252.0 * 6.5 * 60.0)) < 1e-15
    # Default touch rate × one minute is O(1), not ~1e-5.
    lam = float(cfg.fill_intensity_per_second) * cfg.dt_seconds
    assert lam > 0.5
    # Legacy events-per-year alias reproduces the old 5e4 × dt_years product.
    legacy = SimConfig(fill_intensity_base=5.0e4)
    old_product = 5.0e4 / (252.0 * 6.5 * 60.0)
    new_product = float(legacy.fill_intensity_per_second) * legacy.dt_seconds
    assert abs(new_product - old_product) < 1e-9


def test_session_horizon_shrinks():
    dt = 0.05
    assert abs(session_remaining(0.30, 0.0, dt) - 0.30) < 1e-12
    assert abs(session_remaining(0.30, 0.12, dt) - 0.18) < 1e-12
    assert session_remaining(0.30, 0.40, dt) == dt


def test_simulator_passes_rolling_horizon_into_the_quote():
    cfg = EngineConfig(
        quoter=QuoterConfig(
            mode="as_finite_horizon",
            gamma=2.0,
            sigma=4.0,
            kappa=40.0,
            T_horizon=0.30,
            min_half_spread=0.001,
            max_half_spread=50.0,
            quote_size=1,
        ),
        market=MarketConfig(spot_vol=0.0),
        sim=SimConfig(
            n_steps=4,
            seed=1,
            fill_model="poisson",
            fill_intensity_per_second=0.0,
            dt_seconds=TRADING_SECONDS_PER_YEAR * 0.05,
            feature_pack="off",
        ),
    )
    result = run_simulation(cfg)
    assert result.states[0].quote is not None and result.states[-1].quote is not None
    assert result.states[-1].quote.half_spread < result.states[0].quote.half_spread - 0.5


def test_hedge_fill_books_cash_underlier_and_pnl():
    pos = Position(cash=0.0, qty=0)
    order = HedgeOrder(delta_to_hedge=10.0, underlier_qty=-10.0, reason="flatten")
    fill = apply_hedge(0.0, 100.0, order, slip_bps=10.0)
    book_hedge_fill(pos, fill)
    assert abs(pos.underlier_qty - (-10.0)) < 1e-12
    assert abs(pos.cash - fill.cash_delta) < 1e-9
    assert pos.hedge_slippage < 0.0
    # Cash paid the fill price; marking the shares at the mid recovers slippage once.
    marked_at_mid = pos.cash + pos.underlier_qty * 100.0
    assert abs(marked_at_mid - fill.slippage_cost) < 1e-8
    assert abs(marked_pnl(pos, 0.0, 100.0) - marked_at_mid) < 1e-12


def test_simulation_books_hedges_when_hedge_now_fires():
    cfg = EngineConfig(
        sim=SimConfig(
            n_steps=3,
            seed=2,
            fill_model="poisson",
            fill_intensity_per_second=0.0,
            starting_option_qty=20,
            hedge_band=5.0,
            hedge_slip_bps=10.0,
            feature_pack="off",
        ),
        risk=RiskConfig(max_abs_inventory=40, max_abs_delta=80.0, max_loss=5000.0),
    )
    result = run_simulation(cfg)
    assert result.hedges, "fallback should request a hedge at inventory 20"
    assert result.hedge_fills
    assert result.position.underlier_qty != 0.0
    assert result.position.hedge_slippage < 0.0
    last = result.states[-1]
    expect = marked_pnl(result.position, last.option_mid, last.spot)
    assert abs(result.final_pnl - expect) < 1e-8
    assert abs(result.pnl_path[-1] - result.final_pnl) < 1e-8
    # Slippage is inside cash via the fill price, not a second subtraction.
    cash_from_hedges = sum(f.cash_delta for f in result.hedge_fills)
    assert abs(result.position.cash - cash_from_hedges) < 1e-6


def test_lob_is_the_primary_backend_and_poisson_is_explicit():
    lob = run_simulation(
        EngineConfig(sim=SimConfig(n_steps=25, seed=4, fill_model="lob"))
    )
    pois = run_simulation(
        EngineConfig(
            sim=SimConfig(n_steps=40, seed=4, fill_model="poisson", fill_intensity_per_second=0.02)
        )
    )
    assert lob.fill_model == "lob"
    assert pois.fill_model == "poisson"
    assert len(lob.fills) >= 1
    assert len(pois.fills) >= 1
    # Default config is the LOB path. The demo must not pass a 5e4 intensity hack.
    assert SimConfig().fill_model == "lob"
