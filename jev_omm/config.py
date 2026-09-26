"""Central configuration for the sim engine.

All knobs live here so demos/tests share one source of truth.
No broker endpoints or API keys — simulation/paper only.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MarketConfig(BaseModel):
    """Exogenous mid-process and series definitions."""

    spot0: float = 100.0
    rate: float = 0.05
    dividend_yield: float = 0.0
    # GBM-like mid moves: dS = mu*S*dt + sigma*S*sqrt(dt)*Z each step
    drift: float = 0.0
    spot_vol: float = 0.20
    strike: float = 100.0
    expiry_years: float = 30.0 / 365.25  # ~30 DTE
    is_call: bool = True
    # Placeholder ATM IV used when surface is not fitted
    atm_iv: float = 0.22


class QuoterConfig(BaseModel):
    """Avellaneda–Stoikov / Guéant quoter parameters.

    Classic A–S is for spot market making. We adapt it to a single options
    series by treating the option mid as the "reference price" and inventory
    as option contracts. Gamma/vega penalties are research approximations —
    not from the original A–S paper.

    ``mode`` toggles:
      - ``as_finite_horizon``: classic A–S reservation/spread with T−t
      - ``gueant_asymptotic``: Guéant–Lehalle–Fernandez-Tapia stationary
        closed form (arXiv 1105.3115); uses ``A`` mid-touch intensity.
      - ``gueant_ode``: finite-horizon / spectral ODE on the same intensity.
    """

    gamma: float = Field(default=0.1, description="Inventory risk aversion")
    kappa: float = Field(default=1.5, description="Order-arrival intensity decay")
    sigma: float = Field(default=0.5, description="Option mid vol (abs $/√yr)")
    T_horizon: float = Field(default=1.0 / 252.0, description="Terminal horizon (years)")
    A: float = Field(default=140.0, description="Guéant mid-touch arrival intensity A")
    inventory_cap: int = Field(default=10, description="Guéant ODE inventory bound Q")
    ode_steps: int = Field(default=800, description="RK4 steps for gueant_ode over T_horizon")
    # Extra inventory penalties for options Greeks (approx; see quoter comments)
    gamma_penalty: float = 0.0
    vega_penalty: float = 0.0
    portfolio_delta_penalty: float = Field(
        default=0.0, description="Reservation tilt per unit portfolio delta (multi-strike)"
    )
    min_half_spread: float = 0.05
    max_half_spread: float = 5.0
    quote_size: int = 1
    mode: str = Field(
        default="as_finite_horizon",
        description="as_finite_horizon | gueant_asymptotic | gueant_ode",
    )


class RiskConfig(BaseModel):
    """Hard limits that halt quoting when breached."""

    max_abs_inventory: int = 25
    max_abs_delta: float = 50.0
    max_abs_vega: float = 200.0
    max_abs_gamma: float = 5.0
    max_loss: float = 500.0  # cash PnL floor (negative = loss)


class SimConfig(BaseModel):
    """Discrete-event simulation controls."""

    n_steps: int = 200
    dt_years: float = 1.0 / (252.0 * 6.5 * 60.0)  # ~1 minute of trading
    fill_intensity_base: float = 2.0  # Poisson λ scale at mid
    seed: int = 42
    starting_cash: float = 0.0


class EngineConfig(BaseModel):
    market: MarketConfig = Field(default_factory=MarketConfig)
    quoter: QuoterConfig = Field(default_factory=QuoterConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
