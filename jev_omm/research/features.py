"""Synthetic feature packs for ablations.

Every pack other than ``off`` composes the existing flow / GEX / state
scalers in ``positioning.adjust.apply_features``. Those scalers are the
identity at zero inputs and when their flags are off. This module does not
add a new quoting model.

Signals here are functions of the simulated tape (spot return, distance to
the strike). They are not OPRA open interest, dealer positioning, or a
live instability estimate.
"""

from __future__ import annotations

from jev_omm.positioning.adjust import PositioningAdjust, apply_features


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def pack_adjustment(
    pack: str,
    *,
    ret_bps: float,
    spot: float,
    strike: float,
    mid: float,
) -> PositioningAdjust | None:
    """Return quote scalers for ``pack``, or None when the pack is off."""
    name = (pack or "off").strip().lower()
    if name in ("", "off", "none", "identity"):
        return None
    flow = name in ("flow", "flow_gex", "all", "progressive")
    gex = name in ("gex", "flow_gex", "all", "progressive")
    state = name in ("state", "all", "progressive")
    if name not in ("flow", "gex", "state", "flow_gex", "all", "progressive"):
        raise ValueError(f"unknown feature_pack: {pack}")
    ofi = _clip(ret_bps / 25.0, -1.0, 1.0) if flow else 0.0
    gex_norm = 0.0
    pin_gap = 0.0
    if gex and strike > 0.0:
        # Distance from the strike, signed short-gamma when we leave the pin.
        # A research scalar, not measured dealer gamma.
        gex_norm = _clip(-abs(spot - strike) / strike * 20.0, -1.0, 1.0)
        pin_gap = (strike - spot) / max(spot, 1e-6)
    inst = min(3.0, abs(ret_bps) / 30.0) if state else 0.0
    return apply_features(
        ofi_norm=ofi,
        gex_enabled=gex,
        gex_norm=gex_norm,
        pin_gap=pin_gap,
        spot_return=ret_bps * 1e-4,
        mid=mid,
        state_enabled=state,
        instability=inst,
        f_signed=ret_bps,
        l_exec=30.0 if state else 0.0,
    )
