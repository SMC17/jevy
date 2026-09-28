"""Hot-path quote and hedge scalers for precomputed positioning features.

Classical A–S / Guéant reservation math is not replaced. These multipliers
and a reservation *shift* are applied by the caller. Every input at its
default (features off, or a zero z-score / zero GEX with the flag off)
returns shift 0, spread 1, size 1, hedge-band 1, urgency 0.

GEX sign follows Barbon & Buraschi, *Gamma Fragility*
(https://doi.org/10.2139/ssrn.3725454): negative dealer gamma chases,
positive dealer gamma damps. The shift toward a pin level is only used
when normalized GEX is positive. Max-pain is not a forecast; see
``jev_omm.positioning.gex``.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def gex_adjust(
    enabled: bool,
    gex_norm: float,
    pin_gap: float,
    spot_return: float,
    mid: float,
) -> tuple[float, float, float, float, float]:
    """Return shift, spread_mult, size_mult, hedge_band_mult, hedge_urgency.

    ``gex_norm`` is clipped to [-1, 1]. Positive = dealer long gamma.
    ``pin_gap`` is (pin_level - mid) / mid. ``spot_return`` is a simple return.
    """
    if not enabled:
        return 0.0, 1.0, 1.0, 1.0, 0.0
    g = _clamp(gex_norm, -1.0, 1.0)
    if g >= 0.0:
        spread = 1.0 - 0.20 * g
        size = 1.0 + 0.30 * g
        band = 1.0 + 0.50 * g
        shift = 0.15 * g * pin_gap * mid
    else:
        a = -g
        spread = 1.0 + 0.70 * a
        size = 1.0 / (1.0 + 0.80 * a)
        band = 1.0 / (1.0 + 1.0 * a)
        shift = 0.10 * a * spot_return * mid
    if abs(g) < 0.25:
        near = (0.25 - abs(g)) / 0.25
        spread *= 1.0 + 0.35 * near
    urgency = 0.40 * (-g) if g < -0.35 else 0.0
    return shift, spread, size, band, urgency


def cot_fade(enabled: bool, cot_z: float, mid: float) -> tuple[float, float]:
    """Fade a crowded speculative z-score. Disabled or z = 0 → (0, 1).

    Reservation moves against the sign of z (longs crowded → shift down).
    Size shrinks as |z| grows. At |z| = 4 the fade saturates.
    """
    if not enabled:
        return 0.0, 1.0
    fade = _clamp(cot_z / 4.0, -1.0, 1.0)
    shift = -fade * 0.0015 * mid
    size = 1.0 / (1.0 + 0.50 * abs(fade))
    return shift, size


def basis_spread_mult(basis_z: float) -> float:
    """Widen when the annualized roll is far from its own history. z = 0 → 1."""
    return 1.0 + 0.20 * _clamp(abs(basis_z) / 2.0, 0.0, 1.0)


def rr_spread_mult(rr_stress: float) -> float:
    """``rr_stress`` in [0, 1] from |25-delta risk reversal|. 0 → 1."""
    return 1.0 + 0.15 * _clamp(rr_stress, 0.0, 1.0)


def rr_stress(rr_25d: float, scale: float = 0.10) -> float:
    """|call_25d IV − put_25d IV| / scale, clipped to [0, 1]. Decimal vols."""
    if scale <= 0.0:
        return 0.0
    return _clamp(abs(rr_25d) / scale, 0.0, 1.0)


def pcr_spread_mult(put_call_oi: float) -> float:
    """Rich put open interest widens. 0 means the ratio was not supplied."""
    if put_call_oi <= 0.0:
        return 1.0
    return 1.0 + 0.10 * _clamp(put_call_oi - 1.0, 0.0, 1.0)


@dataclass
class PositioningAdjust:
    reservation_shift: float = 0.0
    spread_mult: float = 1.0
    size_mult: float = 1.0
    hedge_band_mult: float = 1.0
    hedge_urgency: float = 0.0


def apply_features(
    *,
    vpin: float = 0.0,
    ofi_norm: float = 0.0,
    aggr_imbalance: float = 0.0,
    off_exchange_share: float = 0.0,
    spoof: float = 0.0,
    gex_enabled: bool = False,
    gex_norm: float = 0.0,
    pin_gap: float = 0.0,
    spot_return: float = 0.0,
    mid: float = 0.0,
    cot_enabled: bool = False,
    cot_z: float = 0.0,
    basis_z: float = 0.0,
    rr_stress_value: float = 0.0,
    put_call_oi: float = 0.0,
    state_enabled: bool = False,
    instability: float = 0.0,
    constraint_active: bool = False,
    parent_remaining: float = 0.0,
    f_signed: float = 0.0,
    l_exec: float = 0.0,
) -> PositioningAdjust:
    """Compose flow, GEX, COT, basis, skew, put/call, and the instability gate.

    The state-gate arguments default to off. A disabled gate is the identity,
    so existing callers are unchanged.
    """
    from jev_omm.flow.signals import flow_prior
    from jev_omm.state_os.gate import state_gate

    tox, flow_spread, flow_size = flow_prior(
        vpin, ofi_norm, aggr_imbalance, off_exchange_share, spoof
    )
    g_shift, g_spread, g_size, band, urgency = gex_adjust(
        gex_enabled, gex_norm, pin_gap, spot_return, mid
    )
    c_shift, c_size = cot_fade(cot_enabled, cot_z, mid)
    gate = state_gate(
        state_enabled,
        instability,
        constraint_active,
        parent_remaining,
        f_signed,
        l_exec,
        mid,
    )
    spread = (
        flow_spread
        * g_spread
        * basis_spread_mult(basis_z)
        * rr_spread_mult(rr_stress_value)
        * pcr_spread_mult(put_call_oi)
        * gate.spread_mult
    )
    size = flow_size * g_size * c_size * gate.size_mult
    spread = _clamp(spread, 0.70, 3.50)
    size = _clamp(size, 0.20, 1.80)
    if gate.pull:
        size = 0.0
    if tox > 0.60:
        urgency = max(urgency, 0.30)
    urgency = max(urgency, gate.hedge_urgency)
    return PositioningAdjust(
        reservation_shift=g_shift + c_shift + gate.reservation_shift,
        spread_mult=spread,
        size_mult=size,
        hedge_band_mult=band,
        hedge_urgency=urgency,
    )
