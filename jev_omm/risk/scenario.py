"""Scenario risk matrix (Akuna Options 201 — analyze risk).

Shock grid over spot moves × IV moves → portfolio PnL via greek Taylor
(or optional BS reprice). Soft vs hard loss hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jev_omm.models.types import Greeks
from jev_omm.pricing import _native
from jev_omm.pricing.black_scholes import price as bs_price


DEFAULT_SPOT_SHOCKS = (-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05)
DEFAULT_IV_SHOCKS = (-0.05, -0.02, 0.0, 0.02, 0.05)


@dataclass
class ScenarioCell:
    d_spot_frac: float
    d_iv: float
    pnl: float


@dataclass
class ScenarioMatrix:
    cells: list[list[ScenarioCell]] = field(default_factory=list)
    min_pnl: float = 0.0
    max_pnl: float = 0.0
    soft_breach: bool = False
    hard_breach: bool = False
    worst: tuple[int, int] = (0, 0)

    def at(self, i: int, j: int) -> ScenarioCell:
        return self.cells[i][j]


def taylor_pnl(g: Greeks, d_spot: float, d_iv: float) -> float:
    if _native.ZIG_AVAILABLE and hasattr(_native, "scenario_taylor"):
        try:
            return _native.scenario_taylor(g.delta, g.gamma, g.vega, g.theta, d_spot, d_iv)
        except Exception:
            pass
    return g.delta * d_spot + 0.5 * g.gamma * d_spot * d_spot + g.vega * d_iv


def build_matrix(
    g: Greeks,
    spot: float,
    *,
    spot_shocks: tuple[float, ...] = DEFAULT_SPOT_SHOCKS,
    iv_shocks: tuple[float, ...] = DEFAULT_IV_SHOCKS,
    soft_loss: float = 100.0,
    hard_loss: float = 400.0,
) -> ScenarioMatrix:
    cells: list[list[ScenarioCell]] = []
    min_pnl = float("inf")
    max_pnl = float("-inf")
    soft = hard = False
    worst = (0, 0)
    for i, ds_frac in enumerate(spot_shocks):
        row: list[ScenarioCell] = []
        d_spot = spot * ds_frac
        for j, d_iv in enumerate(iv_shocks):
            pnl = taylor_pnl(g, d_spot, d_iv)
            row.append(ScenarioCell(ds_frac, d_iv, pnl))
            if pnl < min_pnl:
                min_pnl = pnl
                worst = (i, j)
            if pnl > max_pnl:
                max_pnl = pnl
            if abs(pnl) >= soft_loss:
                soft = True
            if abs(pnl) >= hard_loss:
                hard = True
        cells.append(row)
    return ScenarioMatrix(
        cells=cells,
        min_pnl=min_pnl,
        max_pnl=max_pnl,
        soft_breach=soft,
        hard_breach=hard,
        worst=worst,
    )


def build_matrix_reprice(
    spot: float,
    strike: float,
    t: float,
    rate: float,
    div_yield: float,
    iv: float,
    is_call: bool,
    qty: float,
    *,
    spot_shocks: tuple[float, ...] = DEFAULT_SPOT_SHOCKS,
    iv_shocks: tuple[float, ...] = DEFAULT_IV_SHOCKS,
    soft_loss: float = 100.0,
    hard_loss: float = 400.0,
) -> ScenarioMatrix:
    base = bs_price(spot, strike, t, rate, div_yield, iv, is_call)
    cells: list[list[ScenarioCell]] = []
    min_pnl = float("inf")
    max_pnl = float("-inf")
    soft = hard = False
    worst = (0, 0)
    for i, ds_frac in enumerate(spot_shocks):
        row: list[ScenarioCell] = []
        s2 = spot * (1.0 + ds_frac)
        for j, d_iv in enumerate(iv_shocks):
            iv2 = max(iv + d_iv, 1e-6)
            px = bs_price(s2, strike, t, rate, div_yield, iv2, is_call)
            pnl = qty * (px - base)
            row.append(ScenarioCell(ds_frac, d_iv, pnl))
            if pnl < min_pnl:
                min_pnl = pnl
                worst = (i, j)
            if pnl > max_pnl:
                max_pnl = pnl
            if abs(pnl) >= soft_loss:
                soft = True
            if abs(pnl) >= hard_loss:
                hard = True
        cells.append(row)
    return ScenarioMatrix(
        cells=cells, min_pnl=min_pnl, max_pnl=max_pnl,
        soft_breach=soft, hard_breach=hard, worst=worst,
    )
