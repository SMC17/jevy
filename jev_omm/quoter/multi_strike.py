"""Desk-shaped multi-strike quote strip (mirrors zig/src/multi_strike.zig).

Holds a small strike grid around spot with shared portfolio delta tilt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jev_omm.config import QuoterConfig
from jev_omm.models.types import Greeks, Quote
from jev_omm.quoter.avellaneda_stoikov import make_quote


@dataclass
class StripConfig:
    half_width: int = 2
    strike_step: float = 1.0
    is_call: bool = True


@dataclass
class StrikeSlot:
    strike: float
    inventory: int = 0
    active: bool = True


@dataclass
class StrikeQuote:
    strike: float
    quote: Quote
    mid: float = 0.0
    iv: float = 0.0
    greeks: Greeks = field(default_factory=Greeks)
    inventory: int = 0


@dataclass
class PortfolioGreeks:
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    net_inventory: int = 0


def build_strike_grid(spot: float, cfg: StripConfig | None = None) -> list[float]:
    cfg = cfg or StripConfig()
    max_hw = 3  # MAX_STRIP=8 → half 3
    hw = min(cfg.half_width, max_hw)
    step = max(cfg.strike_step, 1e-9)
    atm = round(spot / step) * step
    return [atm + (i - hw) * step for i in range(2 * hw + 1)]


def portfolio_greeks(slots: list[StrikeSlot], per_contract: list[Greeks]) -> PortfolioGreeks:
    out = PortfolioGreeks()
    for slot, g in zip(slots, per_contract):
        if not slot.active:
            continue
        q = float(slot.inventory)
        out.delta += q * g.delta
        out.gamma += q * g.gamma
        out.vega += q * g.vega
        out.theta += q * g.theta
        out.net_inventory += slot.inventory
    return out


def apply_portfolio_delta_tilt(q: Quote, portfolio_delta: float, cfg: QuoterConfig) -> Quote:
    tilt = portfolio_delta * cfg.portfolio_delta_penalty
    r = q.reservation - tilt
    half = q.half_spread
    bid = max(0.01, r - half)
    ask = max(bid + 0.01, r + half)
    return Quote(
        bid=bid,
        ask=ask,
        bid_size=q.bid_size,
        ask_size=q.ask_size,
        reservation=r,
        half_spread=half,
    )


def quote_one(
    mid: float,
    inventory: int,
    cfg: QuoterConfig,
    *,
    t_remaining: float | None = None,
    greeks: Greeks | None = None,
    portfolio_delta: float = 0.0,
    spread_mult: float = 1.0,
    size_mult: float = 1.0,
) -> Quote:
    q = make_quote(
        mid,
        inventory,
        cfg,
        t_remaining,
        greeks,
        spread_mult=spread_mult,
        size_mult=size_mult,
    )
    if cfg.portfolio_delta_penalty != 0.0 and portfolio_delta != 0.0:
        q = apply_portfolio_delta_tilt(q, portfolio_delta, cfg)
    return q


def quote_strip(
    *,
    spot: float,
    t_rem: float,
    rate: float,
    div_yield: float,
    slots: list[StrikeSlot],
    cfg: QuoterConfig,
    price_fn,  # (spot,K,t,rate,div,iv,is_call) -> (mid, Greeks)
    iv_fn,  # (forward, K, t) -> iv
    strip: StripConfig | None = None,
    spread_mult: float = 1.0,
    size_mult: float = 1.0,
) -> list[StrikeQuote]:
    """Price + quote each active strike with shared portfolio-delta tilt."""
    strip = strip or StripConfig()
    mids: list[float] = []
    ivs: list[float] = []
    greeks_list: list[Greeks] = []
    for slot in slots:
        if not slot.active:
            mids.append(0.0)
            ivs.append(0.0)
            greeks_list.append(Greeks())
            continue
        forward = spot * __import__("math").exp((rate - div_yield) * t_rem)
        iv = float(iv_fn(forward, slot.strike, t_rem))
        mid, g = price_fn(spot, slot.strike, t_rem, rate, div_yield, iv, strip.is_call)
        mids.append(float(mid))
        ivs.append(iv)
        greeks_list.append(g)

    port = portfolio_greeks(slots, greeks_list)
    out: list[StrikeQuote] = []
    for i, slot in enumerate(slots):
        if not slot.active:
            continue
        q = quote_one(
            mids[i],
            slot.inventory,
            cfg,
            t_remaining=cfg.T_horizon,
            greeks=greeks_list[i],
            portfolio_delta=port.delta,
            spread_mult=spread_mult,
            size_mult=size_mult,
        )
        out.append(
            StrikeQuote(
                strike=slot.strike,
                quote=q,
                mid=mids[i],
                iv=ivs[i],
                greeks=greeks_list[i],
                inventory=slot.inventory,
            )
        )
    return out
