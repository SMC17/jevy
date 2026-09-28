"""Central configuration for the sim engine.

All knobs live here so demos/tests share one source of truth.
No broker endpoints or API keys — simulation/paper only.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field, model_validator


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
      - ``option_vega``: Baldacci–Bergault–Guéant constant-vega grid
        (arXiv 1907.12433). Portfolio vega = inventory × per-contract vega.
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
        description="as_finite_horizon | gueant_asymptotic | gueant_ode | option_vega",
    )
    contract_vega: float = Field(default=10.0, description="Per-contract vega for option_vega")
    xi: float = Field(default=1.0, description="Vol-of-vol in the quadratic vega penalty")
    vega_limit: float = Field(default=40.0, description="Hard |portfolio vega| cap")
    vol_edge: float = Field(default=0.0, description="(a_P - a_Q) / (2 sqrt(nu))")
    option_rho: float = Field(default=0.0, description="Spot-vol correlation; penalty scales by (1-rho^2)")
    iv_alpha: float = Field(default=0.0, description="Theo IV minus market IV (decimal)")
    intensity_kind: str = Field(default="exponential", description="exponential | logistic")
    logistic_lambda: float = 100.0
    logistic_alpha: float = 0.7
    logistic_beta: float = 150.0
    option_grid_n: int = 31
    option_grid_steps: int = 60


class RiskConfig(BaseModel):
    """Hard limits that halt quoting when breached.

    ``max_abs_notional``, ``max_abs_per_strike``, and ``max_quotes_outstanding``
    are optional. ``None`` (or a non-positive value) means the limit is unset
    and the check is a no-op even if the caller passes the corresponding
    object. Inventory, delta, gamma, vega, and loss stay active.
    """

    max_abs_inventory: int = 25
    max_abs_delta: float = 50.0
    max_abs_vega: float = 200.0
    max_abs_gamma: float = 5.0
    max_loss: float = 500.0  # cash PnL floor (negative = loss)
    max_abs_notional: float | None = None
    max_abs_per_strike: int | None = None
    max_quotes_outstanding: int | None = None


# 252 sessions × 6.5 hours × 3600 seconds. One trading minute is
# dt_seconds=60 and dt_years = 60 / TRADING_SECONDS_PER_YEAR
# = 1 / (252 × 6.5 × 60), the historical year-fraction step.
TRADING_SECONDS_PER_YEAR: float = 252.0 * 6.5 * 3600.0


def session_remaining(horizon_years: float, elapsed_years: float, dt_years: float) -> float:
    """Rolling finite-horizon time left, floored at one step.

    Classic Avellaneda–Stoikov uses T − t inside a session of length
    ``horizon_years`` (``QuoterConfig.T_horizon``), not option expiry.
    Callers that pass ``None`` for ``t_remaining`` keep a fixed receding
    horizon equal to ``T_horizon``. The simulator passes this function.
    """
    floor = max(dt_years, 1e-8)
    return max(horizon_years - elapsed_years, floor)


class SimConfig(BaseModel):
    """Discrete-event simulation controls.

    Clocks:
      - ``dt_seconds`` is the execution step (default one trading minute).
      - ``dt_years`` is that step in year-fractions for GBM and Black–Scholes.
      - ``MarketState.time`` stays in years from the session start so expiry
        subtraction and A–S (T − t) share one pricing clock.

    Fills:
      - ``fill_intensity_per_second`` is the Poisson rate at zero distance
        from mid, in events per second. ``sample_fills`` multiplies it by
        ``dt_seconds``. The default 0.015 is about 0.9 expected touch
        arrivals per minute before κ-decay, so a 200-step run is O(10–100)
        fills. It is not an events-per-year number stuffed through ``dt_years``.
      - ``fill_intensity_base``, if set and ``fill_intensity_per_second`` is
        omitted, is the legacy events-per-year alias (the old ``5e4`` demo
        hack). Prefer the per-second field.
      - ``fill_model`` selects the execution backend. ``lob`` is the primary
        path (queue depth, cancel-ahead, partials). ``poisson`` is the
        explicit touch model for unit demos.
    """

    n_steps: int = 200
    dt_seconds: float = 60.0
    fill_intensity_per_second: float | None = None
    fill_intensity_base: float | None = Field(
        default=None,
        description="Deprecated events-per-year alias. See SimConfig docstring.",
    )
    seed: int = 42
    starting_cash: float = 0.0
    starting_option_qty: int = 0
    fill_model: str = Field(default="lob", description="lob | poisson")
    # LOB rates use the same second clock as dt_seconds. The LOB module
    # itself is unit-agnostic (rate × horizon); these defaults are per second.
    lob_trade_intensity_per_second: float = 0.08
    lob_cancel_ahead_per_second: float = 0.0
    lob_ahead: float = 2.0
    lob_adverse_jump: float = 0.02
    lob_toxic_flow: float = 0.0
    lob_cancel_latency_seconds: float | None = None
    hedge_band: float = 5.0
    hedge_slip_bps: float = 1.0
    hedge_half_spread: float = 0.0
    hedge_flatten: bool = True
    # off | flow | gex | state | flow_gex | all
    # "off" leaves quote adjustments unchanged (features are the identity).
    feature_pack: str = "off"

    @model_validator(mode="before")
    @classmethod
    def _legacy_intensity(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        per_sec = out.get("fill_intensity_per_second", None)
        legacy = out.get("fill_intensity_base", None)
        if per_sec is None and legacy is not None:
            out["fill_intensity_per_second"] = float(legacy) / TRADING_SECONDS_PER_YEAR
        if out.get("fill_intensity_per_second", None) is None:
            out["fill_intensity_per_second"] = 1.0 / 60.0 * 0.9  # 0.015
        return out

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dt_years(self) -> float:
        """Year-fraction of ``dt_seconds`` on the 252 × 6.5h trading clock."""
        return float(self.dt_seconds) / TRADING_SECONDS_PER_YEAR


class EngineConfig(BaseModel):
    market: MarketConfig = Field(default_factory=MarketConfig)
    quoter: QuoterConfig = Field(default_factory=QuoterConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
