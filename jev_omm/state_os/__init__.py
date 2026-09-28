"""Latent-state / forced-flow operating system.

The desk does not forecast the next price change. It estimates the state that
makes the next trade obligatory, then gates quotes with

    Instability(S) = |F(S)| / L_exec(S)

where L_exec is executable liquidity on the path F itself will take.

Python builds S_t, the engines, and the research cores. Zig consumes the
precomputed scalars in ``state_gate`` and leaves every quote unchanged when
the gate is off. TypeSafe / the offline fallback see those scalars as Choice,
Score, and Noul features. Neither layer emits an order.

Simulation / paper only.
"""

from jev_omm.state_os.engines import (
    auction_imbalance,
    borrow_pressure,
    cta_weight,
    gen1_roll_dates,
    gen3_coverage,
    gen3_moneyness,
    in_issuer_blackout,
    letf_rebalance,
    lookback_vols,
    net_liquidity,
    net_liquidity_change,
    pension_equity_trade,
    realized_vol,
    risk_parity_weights,
    third_friday,
    tdf_trade,
    trend_signal,
    vol_control_boundary,
    vol_target_weight,
    weight_step,
    withheld_buyback,
)
from jev_omm.state_os.gate import StateGate, state_gate
from jev_omm.state_os.vector import (
    MarketState,
    decision_instability,
    forced_flow,
    instability,
    l_exec_on_path,
    net_forced,
)

__all__ = [
    "MarketState",
    "StateGate",
    "auction_imbalance",
    "borrow_pressure",
    "cta_weight",
    "decision_instability",
    "forced_flow",
    "gen1_roll_dates",
    "gen3_coverage",
    "gen3_moneyness",
    "in_issuer_blackout",
    "instability",
    "l_exec_on_path",
    "letf_rebalance",
    "lookback_vols",
    "net_forced",
    "net_liquidity",
    "net_liquidity_change",
    "pension_equity_trade",
    "realized_vol",
    "risk_parity_weights",
    "state_gate",
    "tdf_trade",
    "third_friday",
    "trend_signal",
    "vol_control_boundary",
    "vol_target_weight",
    "weight_step",
    "withheld_buyback",
]
