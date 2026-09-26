"""SABR-lite implied vol surface (Hagan et al. 2002).

Prefers Zig `jev_omm_sabr_iv` when `libjev_omm.so` is available; otherwise a
labeled pure-Python parametric fallback of the same Hagan expansion.

Approximations / edge cases (mirrored from zig/src/surface.zig):
  * ATM limit |ln(f/K)| < 1e-8 uses the closed-form ATM expansion.
  * Invalid inputs (f,K,alpha ≤ 0 or T < 0) → 0.0.
  * |rho| clamped to (-0.999999, 0.999999).
  * Single expiry slice — no term structure (SABR-lite).
"""

from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field

from jev_omm.pricing import _native as native


def _clamp_rho(rho: float) -> float:
    return max(-0.999999, min(0.999999, rho))


def hagan_sabr_iv(
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    forward: float,
    strike: float,
    t: float,
) -> float:
    """Pure-Python Hagan SABR Black IV (fallback when Zig is unavailable)."""
    if forward <= 0.0 or strike <= 0.0 or alpha <= 0.0 or t < 0.0:
        return 0.0
    rho = _clamp_rho(rho)
    one_m_beta = 1.0 - beta
    log_fk = math.log(forward / strike)
    abs_log = abs(log_fk)
    fk = forward * strike
    fk_pow = 1.0 if abs(one_m_beta) < 1e-14 else fk ** (0.5 * one_m_beta)
    log2 = log_fk * log_fk
    log4 = log2 * log2
    denom_series = (
        1.0
        + (one_m_beta * one_m_beta / 24.0) * log2
        + (one_m_beta**4 / 1920.0) * log4
    )
    fk_pow_full = 1.0 if abs(one_m_beta) < 1e-14 else fk**one_m_beta
    term1 = (one_m_beta * one_m_beta / 24.0) * (alpha * alpha) / fk_pow_full
    term2 = 0.25 * rho * beta * nu * alpha / fk_pow
    term3 = ((2.0 - 3.0 * rho * rho) / 24.0) * nu * nu
    time_factor = 1.0 + (term1 + term2 + term3) * t

    if abs_log < 1e-8:
        f_pow = 1.0 if abs(one_m_beta) < 1e-14 else forward**one_m_beta
        atm = (alpha / f_pow) * time_factor
        return max(1e-8, atm)

    z = (nu / alpha) * fk_pow * log_fk
    zx = 1.0
    if abs(z) > 1e-12 and abs(nu) > 1e-14:
        disc = max(1.0 - 2.0 * rho * z + z * z, 0.0)
        numer = math.sqrt(disc) + z - rho
        denom = 1.0 - rho
        if numer > 0.0 and abs(denom) > 1e-14:
            xz = math.log(numer / denom)
            if abs(xz) > 1e-14:
                zx = z / xz

    prefactor = alpha / (fk_pow * denom_series)
    iv = prefactor * zx * time_factor
    if iv != iv:  # NaN
        return 1e-8
    return max(1e-8, iv)


class SabrIVSurface(BaseModel):
    """Hagan SABR slice. Prefer Zig hot path when available."""

    alpha: float = 0.22
    beta: float = 1.0
    rho: float = -0.3
    nu: float = 0.4
    rate: float = 0.05
    div_yield: float = 0.0
    floor_iv: float = 0.05
    cap_iv: float = 1.50
    prefer_zig: bool = True
    label: str = Field(
        default="SABR_HAGAN",
        description="SABR-lite (Hagan); Zig when available else Python fallback",
    )

    def _backend_iv(self, forward: float, strike: float, t: float) -> float:
        if self.prefer_zig and native.ZIG_AVAILABLE:
            return native.sabr_iv(
                self.alpha, self.beta, self.rho, self.nu, forward, strike, t
            )
        return hagan_sabr_iv(
            self.alpha, self.beta, self.rho, self.nu, forward, strike, t
        )

    def backend_name(self) -> str:
        if self.prefer_zig and native.ZIG_AVAILABLE:
            return f"zig:{native.native_version()}"
        return "python_hagan_fallback"

    def forward(self, spot: float, t: float) -> float:
        return spot * math.exp((self.rate - self.div_yield) * max(t, 0.0))

    def iv(self, spot: float, strike: float, t: float) -> float:
        if spot <= 0.0 or strike <= 0.0:
            return max(self.floor_iv, min(self.cap_iv, self.alpha))
        fwd = self.forward(spot, t)
        raw = self._backend_iv(fwd, strike, max(t, 0.0))
        if raw <= 0.0:
            raw = self.alpha
        return float(min(self.cap_iv, max(self.floor_iv, raw)))

    def atm_iv(self, spot: float, t: float) -> float:
        fwd = self.forward(spot, t)
        if self.prefer_zig and native.ZIG_AVAILABLE:
            raw = native.sabr_atm(self.alpha, self.beta, self.rho, self.nu, fwd, max(t, 0.0))
        else:
            raw = hagan_sabr_iv(
                self.alpha, self.beta, self.rho, self.nu, fwd, fwd, max(t, 0.0)
            )
        return float(min(self.cap_iv, max(self.floor_iv, raw if raw > 0 else self.alpha)))


# Alias for callers that want an explicit fallback label
class SabrIVSurfacePythonFallback(SabrIVSurface):
    prefer_zig: bool = False
    label: str = "PLACEHOLDER_SABR_PYTHON_FALLBACK"
