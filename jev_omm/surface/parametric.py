"""PLACEHOLDER parametric IV surface.

WHY this is labeled placeholder:
  Real market-making needs a fitted vol surface (SVI/SSVI, spline, or local vol)
  calibrated to a chain and arbitrage-free. Here we use a simple smile toy:

      σ(K, T) = σ_atm + skew * log-moneyness + smile * log-moneyness²

  so demos/tests can run without market data. Do NOT use for live trading.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field


class ParametricIVSurface(BaseModel):
    """Toy smile: ATM IV + linear skew + quadratic smile in log-moneyness."""

    atm_iv: float = 0.22
    skew: float = -0.05  # typically negative for equities (puts richer)
    smile: float = 0.10
    floor_iv: float = 0.05
    cap_iv: float = 1.50
    label: str = Field(
        default="PLACEHOLDER_PARAMETRIC",
        description="Marker so callers know this is not a calibrated surface",
    )

    def iv(self, spot: float, strike: float, t: float) -> float:
        """Return implied vol for (K, T). Independent of calendar (no term structure)."""
        if spot <= 0.0 or strike <= 0.0:
            return self.atm_iv
        # Use forward-ish moneyness; for toy surface we ignore rates
        m = math.log(strike / spot)
        # Mild term dampening so short-dated smile not explosive
        t_scale = math.sqrt(max(t, 1e-6))
        raw = self.atm_iv + self.skew * m / t_scale + self.smile * (m * m) / max(t, 1e-4)
        return float(min(self.cap_iv, max(self.floor_iv, raw)))
