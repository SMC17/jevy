"""Executable quotes on the LOB path. Paper only. No live order.

The 0.9 tape replay posted Avellaneda–Stoikov at γ = 0.12 and κ = 1.5.
That indifference half-spread is about 0.64 price units. The synthetic
fixture's touch is about 0.05, and no test-window print trades through
0.64, so the model quoters filled zero contracts. This module does not
clamp those quotes to the touch. It maps the same formula into the
train-window touch, and it decides whether to stay, improve, or cancel
from the existing queue value.

Units, all per second unless noted:

- ``trade_intensity`` and ``cancel_ahead`` are contracts per second.
- ``horizon`` and ``cancel_latency`` are seconds.
- ``kappa`` is per price unit. It is not an events-per-year number.
- ``target_half`` is a price distance, the same unit as the NBBO.

``queue_edge=False`` returns the stay decision unchanged. That is the
identity. A negative cancel latency means "do not cancel".
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass

import numpy as np

from jev_omm.execution.lob import expected_fills, fill_markout, queue_value
from jev_omm.models.types import Quote, Side
from jev_omm.pricing import _native

# One minute of a resting quote inside a daily desk step. Not a session.
RESEARCH_TOUCH = 0.02
RESEARCH_HORIZON_SECONDS = 60.0
RESEARCH_INTENSITY_PER_SECOND = 0.05
RESEARCH_ADVERSE_JUMP = 0.04
RESEARCH_DEPTH = 2.0

ACTION_STAY = 0
ACTION_IMPROVE = 1
ACTION_CANCEL = 2
ACTION_NAME = {ACTION_STAY: "stay", ACTION_IMPROVE: "improve", ACTION_CANCEL: "cancel"}


def kappa_for_touch_python(gamma: float, target_half: float, kappa_prior: float) -> float:
    """κ = γ / (exp(γ δ) − 1), with δ = ``target_half``. Prior when γ or δ ≤ 0."""
    if not (gamma > 0.0 and target_half > 0.0):
        return float(kappa_prior)
    denom = math.exp(gamma * target_half) - 1.0
    if not (denom > 1e-18) or not math.isfinite(denom):
        return float(kappa_prior)
    return float(gamma / denom)


def _kappa_zig(gamma: float, target_half: float, kappa_prior: float) -> float | None:
    lib = _native._lib
    if not _native.ZIG_AVAILABLE or lib is None or not hasattr(lib, "jev_omm_kappa_for_touch"):
        return None
    if not hasattr(lib, "_jev_kappa_touch_bound"):
        lib.jev_omm_kappa_for_touch.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.jev_omm_kappa_for_touch.restype = ctypes.c_double
        lib._jev_kappa_touch_bound = True
    return float(lib.jev_omm_kappa_for_touch(float(gamma), float(target_half), float(kappa_prior)))


def kappa_for_touch(
    gamma: float,
    target_half: float,
    kappa_prior: float,
    *,
    prefer_zig: bool = True,
) -> float:
    if prefer_zig:
        zig = _kappa_zig(gamma, target_half, kappa_prior)
        if zig is not None:
            return zig
    return kappa_for_touch_python(gamma, target_half, kappa_prior)


@dataclass(frozen=True)
class QueueDecision:
    action: int
    ahead: float
    cancel_latency: float
    spread_mult: float
    size_mult: float

    @property
    def name(self) -> str:
        return ACTION_NAME.get(self.action, "stay")

    @property
    def latency(self) -> float | None:
        if self.cancel_latency < 0.0:
            return None
        return float(self.cancel_latency)


def queue_decision_python(
    ahead: float,
    our_size: float,
    trade_intensity: float,
    cancel_ahead: float,
    horizon: float,
    spread_capture: float,
    adverse_per_fill: float,
    toxic_flow: float,
    queue_edge: bool,
) -> QueueDecision:
    parked = max(float(ahead), 0.0)
    if not queue_edge:
        return QueueDecision(ACTION_STAY, parked, -1.0, 1.0, 1.0)
    stay_v = queue_value(
        spread_capture,
        adverse_per_fill,
        parked,
        our_size,
        trade_intensity,
        cancel_ahead,
        horizon,
        None,
    )
    imp_v = queue_value(
        spread_capture * 0.5,
        adverse_per_fill,
        0.0,
        our_size,
        trade_intensity,
        cancel_ahead,
        horizon,
        None,
    )
    if toxic_flow >= 0.85 or (stay_v < 0.0 and imp_v <= 0.0):
        lat = min(max(horizon, 0.0) * 0.1, max(horizon, 0.0))
        return QueueDecision(ACTION_CANCEL, parked, lat, 1.0, 0.0)
    if imp_v > stay_v + 1e-12 and imp_v > 0.0:
        return QueueDecision(ACTION_IMPROVE, 0.0, -1.0, 0.5, 1.0)
    return QueueDecision(ACTION_STAY, parked, -1.0, 1.0, 1.0)


def _decision_zig(
    ahead: float,
    our_size: float,
    trade_intensity: float,
    cancel_ahead: float,
    horizon: float,
    spread_capture: float,
    adverse_per_fill: float,
    toxic_flow: float,
    queue_edge: bool,
) -> QueueDecision | None:
    lib = _native._lib
    if not _native.ZIG_AVAILABLE or lib is None or not hasattr(lib, "jev_omm_queue_decision"):
        return None
    if not hasattr(lib, "_jev_queue_decision_bound"):
        lib.jev_omm_queue_decision.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.jev_omm_queue_decision.restype = None
        lib._jev_queue_decision_bound = True
    action = ctypes.c_uint8()
    out_ahead = ctypes.c_double()
    latency = ctypes.c_double()
    spread = ctypes.c_double()
    size = ctypes.c_double()
    lib.jev_omm_queue_decision(
        float(ahead),
        float(our_size),
        float(trade_intensity),
        float(cancel_ahead),
        float(horizon),
        float(spread_capture),
        float(adverse_per_fill),
        float(toxic_flow),
        1 if queue_edge else 0,
        ctypes.byref(action),
        ctypes.byref(out_ahead),
        ctypes.byref(latency),
        ctypes.byref(spread),
        ctypes.byref(size),
    )
    return QueueDecision(
        int(action.value),
        float(out_ahead.value),
        float(latency.value),
        float(spread.value),
        float(size.value),
    )


def queue_decision(
    ahead: float,
    our_size: float,
    trade_intensity: float,
    cancel_ahead: float,
    horizon: float,
    spread_capture: float,
    adverse_per_fill: float,
    toxic_flow: float,
    queue_edge: bool,
    *,
    prefer_zig: bool = True,
) -> QueueDecision:
    if prefer_zig:
        zig = _decision_zig(
            ahead,
            our_size,
            trade_intensity,
            cancel_ahead,
            horizon,
            spread_capture,
            adverse_per_fill,
            toxic_flow,
            queue_edge,
        )
        if zig is not None:
            return zig
    return queue_decision_python(
        ahead,
        our_size,
        trade_intensity,
        cancel_ahead,
        horizon,
        spread_capture,
        adverse_per_fill,
        toxic_flow,
        queue_edge,
    )


@dataclass(frozen=True)
class FillAttribution:
    """One sleeve step on the research LOB. Not an order and not a tape print."""

    fills: float
    joined: float
    adverse_markout: float
    fill_pnl: float
    action: str
    bid: float
    ask: float


def attribute_sleeve_fill(
    *,
    spread_mult: float,
    size_mult: float,
    target: float,
    queue: float,
    toxic: float,
    queue_edge: bool,
    fee_per_contract: float = 0.0,
    book: str = "touch",
    touch: float = RESEARCH_TOUCH,
    horizon_seconds: float = RESEARCH_HORIZON_SECONDS,
    intensity_per_second: float = RESEARCH_INTENSITY_PER_SECOND,
    adverse_jump: float = RESEARCH_ADVERSE_JUMP,
) -> FillAttribution:
    """Turn a sleeve's spread and size into a two-sided quote and a fluid fill.

    The research touch is 0.02 price units. A sleeve with ``spread_mult`` of 1
    posts that touch. A wider multiplier sits behind it and does not improve
    into the touch.

    ``book="touch"`` uses 2 contracts of depth. ``book="sniper"`` uses
    ``max(0, 12 − 6 queue)``, so a typical queue does not clear in 60 seconds
    unless the queue edge improves. ``book="flow"`` keeps the 2-contract depth
    and lets toxicity cancel. Toxicity always scales the adverse jump.
    Depth is contracts. Intensity is contracts per second. Horizon is seconds.
    """
    if size_mult <= 0.0 or touch <= 0.0 or horizon_seconds <= 0.0:
        return FillAttribution(0.0, 0.0, 0.0, 0.0, "stay", 0.0, 0.0)
    half = touch * max(float(spread_mult), 0.25)
    tilt = math.tanh(float(target) / 4.0) * half * 0.25
    mid = 1.0
    bid = mid - half + tilt
    ask = mid + half + tilt
    behind = half > touch + 1e-12
    if book == "sniper":
        ahead = max(0.0, 12.0 - 6.0 * float(queue))
    else:
        ahead = max(0.0, RESEARCH_DEPTH - 1.5 * float(queue))
    if behind:
        ahead = ahead + 20.0 * (half / touch - 1.0)
    our_size = max(0.25, min(2.0, abs(float(size_mult))))
    tox = max(float(toxic), 0.0)
    adverse_per = adverse_jump * tox
    dec = queue_decision(
        ahead,
        our_size,
        intensity_per_second,
        0.0,
        horizon_seconds,
        half,
        adverse_per,
        tox if queue_edge else 0.0,
        queue_edge and not behind,
    )
    if dec.action == ACTION_IMPROVE and not behind:
        half = half * dec.spread_mult
        bid = mid - half + tilt
        ask = mid + half + tilt
        ahead_used = dec.ahead
        latency = None
        joined = 1.0
    elif dec.action == ACTION_CANCEL and not behind:
        ahead_used = dec.ahead
        latency = dec.latency
        joined = 0.0
    elif behind:
        ahead_used = ahead
        latency = None
        joined = 0.0
    else:
        ahead_used = dec.ahead
        latency = None
        joined = 1.0 if half <= touch + 1e-9 else 0.0
    per_side = expected_fills(
        ahead_used,
        our_size,
        intensity_per_second,
        0.0,
        horizon_seconds,
        latency,
    )
    fills = per_side * 2.0
    earned = half * fills
    mark_bid = fill_markout(Side.BID, bid, mid, per_side, adverse_jump, tox)
    mark_ask = fill_markout(Side.ASK, ask, mid, per_side, adverse_jump, tox)
    # fill_markout includes the spread (mid − price). The adverse part is the jump.
    adverse = -(adverse_jump * tox * fills)
    fee = fee_per_contract * fills
    # Cash is the posted half-spread, minus the toxic jump, minus the research fee.
    # The spread inside fill_markout is the same half; we do not add it twice.
    _ = (mark_bid, mark_ask, earned)
    fill_pnl = half * fills + adverse - fee
    return FillAttribution(
        fills=float(fills),
        joined=float(joined),
        adverse_markout=float(adverse),
        fill_pnl=float(fill_pnl),
        action=dec.name if not behind else "behind",
        bid=float(bid),
        ask=float(ask),
    )


def sleeve_quote(spread_mult: float, size_mult: float, target: float, *, touch: float = RESEARCH_TOUCH) -> Quote:
    """Bid/ask a sleeve would hand the LOB. Mid is the research unit 1."""
    half = touch * max(float(spread_mult), 0.25)
    tilt = math.tanh(float(target) / 4.0) * half * 0.25
    bid = 1.0 - half + tilt
    ask = 1.0 + half + tilt
    size = max(1, int(round(max(size_mult, 0.0))))
    return Quote(
        bid=bid,
        ask=ask,
        bid_size=size,
        ask_size=size,
        reservation=1.0 + tilt,
        half_spread=half,
    )


def queue_edge_trial(
    seed: int,
    *,
    queue_edge: bool,
    toxic: float,
    ahead: float,
    spread_capture: float,
    intensity: float,
    horizon: float = 1.0,
    our_size: float = 1.0,
    adverse_jump: float = 0.05,
) -> tuple[float, float]:
    """One stochastic resting bid. Returns (filled contracts, markout)."""
    from jev_omm.execution.lob import LobConfig, simulate

    adverse_per = adverse_jump * max(toxic, 0.0)
    dec = queue_decision(
        ahead,
        our_size,
        intensity,
        0.0,
        horizon,
        spread_capture,
        adverse_per,
        toxic,
        queue_edge,
    )
    rng = np.random.default_rng(seed)
    res = simulate(
        rng,
        LobConfig(
            trade_intensity=intensity,
            cancel_ahead=0.0,
            adverse_jump=adverse_jump,
            toxic_flow=max(toxic, 0.0),
            toxic_from=0.0,
            dt=max(horizon / 20.0, 1e-3),
            horizon=horizon,
            cancel_latency=dec.latency,
            ahead=dec.ahead,
            our_size=our_size,
            price=1.0 - spread_capture * dec.spread_mult,
            mid=1.0,
            side=Side.BID,
        ),
    )
    return float(res.filled), float(res.markout)
