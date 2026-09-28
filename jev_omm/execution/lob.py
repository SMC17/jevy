"""Queue / LOB-aware synthetic fills. Mirrors ``zig/src/lob.zig``.

Poisson touch fills in ``fills.py`` remain available. This model adds depth,
queue position, cancel latency, partial fills, and an adverse-selection jump.
No live market data.

The stepper is unit-agnostic: ``trade_intensity`` and ``cancel_ahead`` are
counts per unit of ``horizon``. ``run_simulation`` passes contracts per
second and ``horizon=dt_seconds``. Standalone tests use a horizon of 1 in
whatever unit ``trade_intensity`` was written in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from jev_omm.models.types import Fill, Quote, Side


@dataclass
class LobEvent:
    time: float
    kind: str  # add | cancel | execute
    side: Side
    price: float
    size: float
    ahead: float
    partial: bool = False
    adverse: bool = False


@dataclass
class LobConfig:
    trade_intensity: float = 40.0
    cancel_ahead: float = 0.0
    adverse_jump: float = 0.05
    toxic_flow: float = 1.0
    toxic_from: float = 0.0
    dt: float = 0.005
    horizon: float = 1.0
    cancel_latency: float | None = None
    ahead: float = 0.0
    our_size: float = 1.0
    price: float = 1.0
    mid: float = 1.05
    side: Side = Side.BID


@dataclass
class LobResult:
    filled: float = 0.0
    adverse_filled: float = 0.0
    markout: float = 0.0
    time_to_first: float = math.inf
    events: list[LobEvent] = field(default_factory=list)


def time_to_first_fill(ahead: float, trade_intensity: float, cancel_ahead: float) -> float:
    if ahead <= 0.0:
        return 0.0
    rate = trade_intensity + cancel_ahead
    if rate <= 0.0:
        return math.inf
    return ahead / rate


def expected_fills(
    ahead: float,
    our_size: float,
    trade_intensity: float,
    cancel_ahead: float,
    horizon: float,
    cancel_latency: float | None = None,
) -> float:
    """Fluid limit. ``cancel_latency`` caps exposure (None = never cancel)."""
    if our_size <= 0.0 or horizon <= 0.0:
        return 0.0
    exposure = horizon if cancel_latency is None else min(horizon, max(cancel_latency, 0.0))
    if exposure <= 0.0:
        return 0.0
    t_clear = time_to_first_fill(max(ahead, 0.0), trade_intensity, cancel_ahead)
    if t_clear >= exposure:
        return 0.0
    traded = max(trade_intensity, 0.0) * (exposure - t_clear)
    return min(our_size, traded)


def queue_value(
    spread_capture: float,
    adverse_per_fill: float,
    ahead: float,
    our_size: float,
    trade_intensity: float,
    cancel_ahead: float,
    horizon: float,
    cancel_latency: float | None = None,
) -> float:
    """Fills × (spread − adverse). Deeper ahead shrinks the absolute value."""
    fills = expected_fills(
        ahead, our_size, trade_intensity, cancel_ahead, horizon, cancel_latency
    )
    return fills * (spread_capture - adverse_per_fill)


def depth_ahead(levels: list[tuple[float, float]], our_index: int) -> float:
    """Sum of depth strictly in front of our level. ``levels`` are (price, depth)."""
    total = 0.0
    for i, (_px, depth) in enumerate(levels):
        if i >= our_index:
            break
        total += max(depth, 0.0)
    return total


def fill_markout(
    side: Side,
    price: float,
    mid: float,
    size: float,
    adverse_jump: float,
    toxic_flow: float,
) -> float:
    """LP markout after an adverse mid jump scaled by ``toxic_flow``."""
    jump = max(adverse_jump, 0.0) * max(toxic_flow, 0.0) * max(size, 0.0)
    if side == Side.BID:
        return (mid - jump - price) * size
    return (price - (mid + jump)) * size


def sample_step_fills(
    rng: np.random.Generator,
    time: float,
    mid: float,
    quote: Quote,
    dt_seconds: float,
    *,
    trade_intensity_per_second: float,
    cancel_ahead_per_second: float = 0.0,
    ahead: float = 0.0,
    adverse_jump: float = 0.0,
    toxic_flow: float = 0.0,
    cancel_latency_seconds: float | None = None,
) -> list[Fill]:
    """One quote cycle of the queue model, both sides, on the seconds clock.

    Each side is an independent resting order of the posted size, behind
    ``ahead`` contracts. Filled size is the integer part of the stochastic
    fill (the stepper's trade counts are already integral).
    """
    fills: list[Fill] = []
    horizon = max(dt_seconds, 1e-6)
    for side, price, size in (
        (Side.BID, quote.bid, quote.bid_size),
        (Side.ASK, quote.ask, quote.ask_size),
    ):
        if size <= 0:
            continue
        res = simulate(
            rng,
            LobConfig(
                trade_intensity=trade_intensity_per_second,
                cancel_ahead=cancel_ahead_per_second,
                adverse_jump=adverse_jump,
                toxic_flow=toxic_flow,
                toxic_from=0.0,
                dt=max(horizon / 5.0, 1e-3),
                horizon=horizon,
                cancel_latency=cancel_latency_seconds,
                ahead=ahead,
                our_size=float(size),
                price=price,
                mid=mid,
                side=side,
            ),
        )
        n = int(math.floor(res.filled + 1e-6))
        if n > 0:
            fills.append(
                Fill(time=time, side=side, price=price, size=n, mid_at_fill=mid)
            )
    return fills


def _poisson(rng: np.random.Generator, lam: float) -> int:
    if lam <= 0.0:
        return 0
    return int(rng.poisson(lam))


def simulate(rng: np.random.Generator, cfg: LobConfig) -> LobResult:
    res = LobResult()
    ahead = max(cfg.ahead, 0.0)
    remaining = max(cfg.our_size, 0.0)
    t = 0.0
    cancel_at = math.inf if cfg.cancel_latency is None else max(cfg.cancel_latency, 0.0)
    res.events.append(
        LobEvent(0.0, "add", cfg.side, cfg.price, remaining, ahead, False, False)
    )
    dt = max(cfg.dt, 1e-6)
    while t < cfg.horizon and remaining > 0.0:
        t_next = min(t + dt, cfg.horizon)
        if t < cancel_at <= t_next:
            span = max(cancel_at - t, 0.0)
            if span > 0.0:
                ahead = _consume(res, rng, cfg, ahead, remaining, t, span)
                remaining = cfg.our_size - res.filled
            res.events.append(
                LobEvent(
                    cancel_at,
                    "cancel",
                    cfg.side,
                    cfg.price,
                    max(remaining, 0.0),
                    ahead,
                    0.0 < remaining < cfg.our_size,
                    False,
                )
            )
            break
        ahead = _consume(res, rng, cfg, ahead, remaining, t, t_next - t)
        remaining = cfg.our_size - res.filled
        t = t_next
    return res


def _consume(
    res: LobResult,
    rng: np.random.Generator,
    cfg: LobConfig,
    ahead: float,
    remaining: float,
    t: float,
    span: float,
) -> float:
    if span <= 0.0 or remaining <= 0.0:
        return ahead
    n_cancel = _poisson(rng, max(cfg.cancel_ahead, 0.0) * span)
    if n_cancel > 0:
        ahead = max(0.0, ahead - float(n_cancel))
    n_trade = _poisson(rng, max(cfg.trade_intensity, 0.0) * span)
    left = float(max(n_trade, 0))
    if ahead > 0.0 and left > 0.0:
        eat = min(ahead, left)
        ahead -= eat
        left -= eat
    if left > 0.0 and remaining > 0.0:
        fill = min(remaining, left)
        toxic = (t + span) >= cfg.toxic_from and cfg.toxic_flow > 0.0 and cfg.adverse_jump > 0.0
        flow = cfg.toxic_flow if toxic else 0.0
        res.markout += fill_markout(cfg.side, cfg.price, cfg.mid, fill, cfg.adverse_jump, flow)
        res.filled += fill
        if toxic:
            res.adverse_filled += fill
        if math.isinf(res.time_to_first):
            res.time_to_first = t + span
        partial = fill + 1e-9 < cfg.our_size and (remaining - fill) > 1e-12
        res.events.append(
            LobEvent(t + span, "execute", cfg.side, cfg.price, fill, ahead, partial, toxic)
        )
    return ahead
