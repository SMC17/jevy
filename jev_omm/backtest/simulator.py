"""Discrete-event options MM simulator (paper / research only).

Each step:
  1. Exogenous spot mid move (GBM increment)
  2. Reprice option via BS + SABR-lite (or parametric) IV surface
  3. Risk check → may halt quoting
  4. DecisionClient (Jev / fallback) → QuoteAdjustments
  5. A–S reservation + spread, scaled by adjustments
  6. Poisson fills against posted quotes
  7. Markout tracker (spread / adverse / inventory MTM)
  8. Optional paper delta-hedge signal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Union

import numpy as np

from jev_omm.config import EngineConfig
from jev_omm.decisions.client import DecisionClient, make_decision_client
from jev_omm.decisions.policy import decide_quote_adjustments
from jev_omm.decisions.schemas import QuoteAdjustments
from jev_omm.execution.fills import sample_fills
from jev_omm.hedge.delta import HedgeOrder, propose_delta_hedge
from jev_omm.models.types import Fill, MarketState, OptionContract, OptionRight, Position, Quote
from jev_omm.pnl.mark import marked_pnl
from jev_omm.pnl.markout import AttributionSummary, MarkoutTracker
from jev_omm.obs.event_log import EventLog
from jev_omm.pricing.black_scholes import price_and_greeks
from jev_omm.quoter.avellaneda_stoikov import make_quote, optimal_half_spread
from jev_omm.risk.limits import evaluate_risk
from jev_omm.surface.parametric import ParametricIVSurface
from jev_omm.surface.sabr import SabrIVSurface


class IVSurface(Protocol):
    def iv(self, spot: float, strike: float, t: float) -> float: ...


@dataclass
class SimResult:
    states: list[MarketState] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    hedges: list[HedgeOrder] = field(default_factory=list)
    adjustments: list[QuoteAdjustments] = field(default_factory=list)
    position: Position = field(default_factory=Position)
    breach_count: int = 0
    decision_source: str = "fallback"
    final_pnl: float = 0.0
    attribution: AttributionSummary = field(default_factory=AttributionSummary)
    surface_label: str = ""
    event_log_path: str = ""
    event_log_sha256: str = ""
    event_count: int = 0


def run_simulation(
    cfg: EngineConfig | None = None,
    client: DecisionClient | None = None,
    surface: Union[SabrIVSurface, ParametricIVSurface, None] = None,
    event_log: EventLog | None = None,
    event_log_path: str | None = None,
) -> SimResult:
    cfg = cfg or EngineConfig()
    client = client or make_decision_client()
    if surface is None:
        surface = SabrIVSurface(
            alpha=cfg.market.atm_iv,
            beta=1.0,
            rho=-0.3,
            nu=0.4,
            rate=cfg.market.rate,
            div_yield=cfg.market.dividend_yield,
        )
    rng = np.random.default_rng(cfg.sim.seed)

    contract = OptionContract(
        strike=cfg.market.strike,
        expiry_years=cfg.market.expiry_years,
        right=OptionRight.CALL if cfg.market.is_call else OptionRight.PUT,
    )
    pos = Position(cash=cfg.sim.starting_cash)
    result = SimResult(position=pos)
    result.surface_label = getattr(surface, "label", type(surface).__name__)
    spot = cfg.market.spot0
    t_exp = cfg.market.expiry_years
    prev_spot = spot
    tracker = MarkoutTracker()
    own_log = False
    if event_log is None and event_log_path:
        event_log = EventLog().open(event_log_path)
        own_log = True

    for step in range(cfg.sim.n_steps):
        time = step * cfg.sim.dt_years
        # Exogenous mid move
        z = float(rng.standard_normal())
        spot = spot * math_exp_step(cfg.market.drift, cfg.market.spot_vol, cfg.sim.dt_years, z)
        spot = max(spot, 1e-6)
        ret_bps = 1e4 * (spot - prev_spot) / prev_spot if prev_spot > 0 else 0.0
        prev_spot = spot

        t_rem = max(t_exp - time, 1e-6)
        iv = surface.iv(spot, contract.strike, t_rem)
        opt_mid, greeks = price_and_greeks(
            spot,
            contract.strike,
            t_rem,
            cfg.market.rate,
            cfg.market.dividend_yield,
            iv,
            contract.is_call,
        )
        tracker.on_step(step, opt_mid)

        if event_log is not None:
            event_log.append_underlying_tick(time, step, spot)
            half_est = optimal_half_spread(cfg.quoter)
            event_log.append_book_top(
                time,
                step,
                bid=opt_mid - half_est,
                ask=opt_mid + half_est,
                bid_sz=cfg.quoter.quote_size,
                ask_sz=cfg.quoter.quote_size,
                mid=opt_mid,
            )

        cash_pnl = marked_pnl(pos, opt_mid)
        risk = evaluate_risk(pos.qty, greeks, cash_pnl, cfg.risk)
        if not risk.quoting_allowed:
            result.breach_count += 1
            if event_log is not None:
                event_log.append_risk_breach(
                    time, step, risk.breach_reason or "risk_breach", pos.qty
                )
                event_log.append_cancel(time, step, "risk_breach")

        half = optimal_half_spread(cfg.quoter)
        adj = decide_quote_adjustments(
            client,
            time=time,
            spot=spot,
            option_mid=opt_mid,
            iv=iv,
            inventory=pos.qty,
            delta=risk.delta,
            gamma=risk.gamma,
            vega=risk.vega,
            cash_pnl=cash_pnl,
            half_spread=half,
            quoting_allowed=risk.quoting_allowed,
            recent_fills=len(result.fills),
            spot_return_bps=ret_bps,
        )
        result.adjustments.append(adj)
        result.decision_source = (
            adj.result.source if adj.result is not None else result.decision_source
        )
        if event_log is not None and adj.result is not None:
            event_log.append_decision_snapshot(time, step, adj.result, adjustments=adj)

        quote: Quote | None = None
        if risk.quoting_allowed and not adj.pull:
            quote = make_quote(
                opt_mid,
                pos.qty,
                cfg.quoter,
                t_remaining=cfg.quoter.T_horizon,
                greeks=greeks,
                spread_mult=adj.spread_mult,
                size_mult=adj.size_mult,
            )
            step_fills = sample_fills(
                rng,
                time,
                opt_mid,
                quote,
                cfg.sim.dt_years,
                cfg.sim.fill_intensity_base,
                cfg.quoter.kappa,
            )
            if event_log is not None:
                event_log.append_quote(time, step, quote)
            for f in step_fills:
                pos.apply_fill(f.side, f.price, f.size)
                result.fills.append(f)
                tracker.on_fill(step, f)
                if event_log is not None:
                    event_log.append_fill(time, step, f)
        elif adj.pull or not risk.quoting_allowed:
            quote = None
            if event_log is not None and adj.pull and risk.quoting_allowed:
                event_log.append_cancel(time, step, "pull")

        if adj.hedge_now:
            # Recompute delta after fills
            cash_pnl = marked_pnl(pos, opt_mid)
            risk = evaluate_risk(pos.qty, greeks, cash_pnl, cfg.risk)
            h = propose_delta_hedge(risk.delta)
            if h is not None:
                result.hedges.append(h)

        result.states.append(
            MarketState(
                time=time,
                spot=spot,
                option_mid=opt_mid,
                iv=iv,
                greeks=greeks,
                quote=quote,
            )
        )

    result.final_pnl = marked_pnl(pos, result.states[-1].option_mid if result.states else 0.0)
    result.position = pos
    result.attribution = tracker.summary
    if event_log is not None:
        result.event_count = event_log.seq
        result.event_log_sha256 = event_log.sha256_hex()
        if event_log.path is not None:
            result.event_log_path = str(event_log.path)
        if own_log:
            event_log.close()
    return result


def math_exp_step(mu: float, sigma: float, dt: float, z: float) -> float:
    """GBM multiplicative factor."""
    import math

    return math.exp((mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z)
