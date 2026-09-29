"""Discrete-event options MM simulator (paper / research only).

Each step:
  1. Exogenous spot mid move (GBM increment on the year-fraction clock)
  2. Reprice option via BS + SABR-lite (or parametric) IV surface
  3. Risk check → may halt quoting (net delta includes the underlier hedge)
  4. DecisionClient (Jev / fallback) → QuoteAdjustments
  5. Optional feature pack (identity when ``feature_pack='off'``)
  6. A–S reservation + spread with rolling session time T − t
  7. Fills: ``fill_model='lob'`` (primary) or ``fill_model='poisson'``
  8. Markout tracker (spread / adverse / inventory MTM)
  9. If ``hedge_now`` fires, book the underlier hedge into position, cash,
     slippage, and marked PnL

``MarketState.time`` is years. Fill intensities use seconds. See ``SimConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Union

import numpy as np

from jev_omm.config import EngineConfig, session_remaining
from jev_omm.decisions.client import DecisionClient, make_decision_client
from jev_omm.decisions.policy import decide_quote_adjustments
from jev_omm.decisions.schemas import QuoteAdjustments
from jev_omm.execution.fills import sample_fills
from jev_omm.execution.lob import sample_step_fills
from jev_omm.hedge.delta import HedgeFill, HedgeOrder, book_hedge_fill, net_delta, propose_delta_hedge
from jev_omm.models.types import Fill, MarketState, OptionContract, OptionRight, Position, Quote
from jev_omm.pnl.mark import marked_pnl
from jev_omm.pnl.markout import AttributionSummary, MarkoutTracker
from jev_omm.obs.event_log import EventLog
from jev_omm.pricing.black_scholes import price_and_greeks
from jev_omm.quoter.avellaneda_stoikov import make_quote, optimal_half_spread
from jev_omm.research.features import pack_adjustment
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
    hedge_fills: list[HedgeFill] = field(default_factory=list)
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
    fill_model: str = ""
    quoted_steps: int = 0
    step_breach: list[bool] = field(default_factory=list)
    pnl_path: list[float] = field(default_factory=list)
    inventory_path: list[int] = field(default_factory=list)
    underlier_path: list[float] = field(default_factory=list)
    abs_delta_path: list[float] = field(default_factory=list)


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
    fill_model = (cfg.sim.fill_model or "lob").strip().lower()
    if fill_model not in ("lob", "poisson"):
        raise ValueError(f"fill_model must be 'lob' or 'poisson', got {cfg.sim.fill_model!r}")
    pos = Position(cash=cfg.sim.starting_cash, qty=cfg.sim.starting_option_qty)
    result = SimResult(position=pos, fill_model=fill_model)
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
        # Rolling session horizon for finite-horizon A–S. Guéant asymptotic
        # ignores t_remaining; the ODE / option-vega modes use T_horizon.
        t_left = session_remaining(cfg.quoter.T_horizon, time, cfg.sim.dt_years)
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
        step_breached = False

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

        cash_pnl = marked_pnl(pos, opt_mid, spot)
        notional = abs(pos.qty) * spot + abs(pos.underlier_qty) * spot
        risk = evaluate_risk(
            pos.qty,
            greeks,
            cash_pnl,
            cfg.risk,
            extra_delta=pos.underlier_qty,
            notional=notional,
            per_strike_abs=abs(pos.qty),
        )
        if not risk.quoting_allowed:
            result.breach_count += 1
            step_breached = True
            if event_log is not None:
                event_log.append_risk_breach(
                    time, step, risk.breach_reason or "risk_breach", pos.qty
                )
                event_log.append_cancel(time, step, "risk_breach")

        half = optimal_half_spread(cfg.quoter, t_left)
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
        pack = pack_adjustment(
            cfg.sim.feature_pack,
            ret_bps=ret_bps,
            spot=spot,
            strike=contract.strike,
            mid=opt_mid,
        )
        if pack is not None:
            adj.spread_mult *= pack.spread_mult
            adj.size_mult *= pack.size_mult
            if pack.size_mult == 0.0:
                adj.pull = True
            if pack.hedge_urgency >= 0.65:
                adj.hedge_now = True
                adj.reason = (adj.reason + ",feature_hedge").strip(",")
        result.adjustments.append(adj)
        result.decision_source = (
            adj.result.source if adj.result is not None else result.decision_source
        )
        if event_log is not None and adj.result is not None:
            event_log.append_decision_snapshot(time, step, adj.result, adjustments=adj)

        quote: Quote | None = None
        if risk.quoting_allowed and not adj.pull:
            ref_mid = opt_mid + (0.0 if pack is None else pack.reservation_shift)
            quote = make_quote(
                ref_mid,
                pos.qty,
                cfg.quoter,
                t_remaining=t_left,
                greeks=greeks,
                spread_mult=adj.spread_mult,
                size_mult=adj.size_mult,
            )
            outstanding = (1 if quote.bid_size > 0 else 0) + (1 if quote.ask_size > 0 else 0)
            risk_q = evaluate_risk(
                pos.qty,
                greeks,
                cash_pnl,
                cfg.risk,
                extra_delta=pos.underlier_qty,
                notional=notional,
                per_strike_abs=abs(pos.qty),
                quotes_outstanding=outstanding,
            )
            if not risk_q.quoting_allowed:
                risk = risk_q
                result.breach_count += 1
                step_breached = True
                quote = None
                if event_log is not None:
                    event_log.append_risk_breach(
                        time, step, risk.breach_reason or "risk_breach", pos.qty
                    )
                    event_log.append_cancel(time, step, "risk_breach")
            else:
                if fill_model == "lob":
                    step_fills = sample_step_fills(
                        rng,
                        time,
                        opt_mid,
                        quote,
                        cfg.sim.dt_seconds,
                        trade_intensity_per_second=cfg.sim.lob_trade_intensity_per_second,
                        cancel_ahead_per_second=cfg.sim.lob_cancel_ahead_per_second,
                        ahead=cfg.sim.lob_ahead,
                        adverse_jump=cfg.sim.lob_adverse_jump,
                        toxic_flow=cfg.sim.lob_toxic_flow,
                        cancel_latency_seconds=cfg.sim.lob_cancel_latency_seconds,
                    )
                else:
                    step_fills = sample_fills(
                        rng,
                        time,
                        opt_mid,
                        quote,
                        cfg.sim.dt_seconds,
                        float(cfg.sim.fill_intensity_per_second or 0.0),
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

        if quote is not None:
            result.quoted_steps += 1
        result.step_breach.append(step_breached)

        if adj.hedge_now:
            # Recompute net delta after fills. The underlier already held
            # counts, otherwise a hedged book keeps buying the same hedge.
            opt_delta = pos.qty * greeks.delta
            h = propose_delta_hedge(
                net_delta(opt_delta, pos.underlier_qty),
                band=cfg.sim.hedge_band,
                half_spread=cfg.sim.hedge_half_spread,
                slip_bps=cfg.sim.hedge_slip_bps,
                flatten=cfg.sim.hedge_flatten,
            )
            if h is not None:
                hfill = apply_and_book(time, spot, h, pos, cfg)
                result.hedges.append(h)
                result.hedge_fills.append(hfill)

        end_pnl = marked_pnl(pos, opt_mid, spot)
        result.pnl_path.append(end_pnl)
        result.inventory_path.append(pos.qty)
        result.underlier_path.append(pos.underlier_qty)
        result.abs_delta_path.append(abs(pos.qty * greeks.delta + pos.underlier_qty))

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

    last = result.states[-1] if result.states else None
    result.final_pnl = marked_pnl(
        pos,
        last.option_mid if last else 0.0,
        last.spot if last else 0.0,
    )
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


def apply_and_book(time: float, spot: float, order: HedgeOrder, pos: Position, cfg: EngineConfig):
    """Fill the paper hedge and book it. Local import keeps the cycle readable."""
    from jev_omm.hedge.delta import apply_hedge

    hfill = apply_hedge(
        time,
        spot,
        order,
        half_spread=cfg.sim.hedge_half_spread,
        slip_bps=cfg.sim.hedge_slip_bps,
        band=cfg.sim.hedge_band,
        flatten=cfg.sim.hedge_flatten,
    )
    book_hedge_fill(pos, hfill)
    return hfill


def math_exp_step(mu: float, sigma: float, dt: float, z: float) -> float:
    """GBM multiplicative factor."""
    import math

    return math.exp((mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z)
