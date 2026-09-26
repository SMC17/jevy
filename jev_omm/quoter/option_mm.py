"""Option-inventory quotes under a constant-vega approximation.

Mirrors ``zig/src/option_mm.zig``.

Baldacci, Bergault, Guéant, https://arxiv.org/abs/1907.12433 :
portfolio vega V^π = Σ q_i V^i with V^i frozen, objective

    spread income + V^π (a_P − a_Q) / (2 √ν) − (γ ξ² / 8) (1−ρ²) (V^π)²

Intensity is exponential Λ(δ) = A e^{−k δ} or logistic
Λ(δ) = λ / (1 + exp(α + β δ / V^i)). Premiums come from a small explicit-Euler
HJB grid in V^π. The hard set is |V^π| ≤ V̄.

Stoikov–Sağlam Theorem 4 (linear intensity), https://doi.org/10.1007/s11147-009-9036-3.
Lucic–Tse IV view, https://ssrn.com/abstract=4729290 : ``iv_alpha`` shifts the
reservation by V^i * (theo IV − market IV). The decision layer does not emit orders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from jev_omm.config import QuoterConfig
from jev_omm.models.types import Greeks, Quote

MAX_GRID = 65


@dataclass
class OptionMmConfig:
    gamma: float = 0.5
    xi: float = 1.0
    A: float = 40.0
    kappa: float = 2.0
    lambda0: float = 100.0
    alpha: float = 0.7
    beta: float = 150.0
    intensity: str = "exponential"  # exponential | logistic
    vega_limit: float = 40.0
    horizon: float = 0.25
    grid_n: int = 31
    n_steps: int = 60
    delta_inf: float = 0.0
    vol_edge: float = 0.0
    rho: float = 0.0
    trade_size: float = 1.0
    iv_alpha: float = 0.0
    min_premium: float = 0.0
    max_premium: float = 50.0
    quote_size: int = 1


@dataclass
class OptionQuote:
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    reservation: float
    half_spread: float
    delta_b: float
    delta_a: float
    portfolio_vega: float
    bid_blocked: bool
    ask_blocked: bool


@dataclass
class ValueGrid:
    n: int = 0
    vega_limit: float = 0.0
    w: list[float] = field(default_factory=list)


@dataclass
class StoikovPremiums:
    eps_ask: float
    eps_bid: float


def research_toy() -> OptionMmConfig:
    """Scaled toy where the vega skew is visible. Not the euro grid in Baldacci §4."""
    return OptionMmConfig()


def penalty_coeff(cfg: OptionMmConfig) -> float:
    rho = min(0.999999, max(-0.999999, cfg.rho))
    return cfg.gamma * cfg.xi * cfg.xi * (1.0 - rho * rho) / 8.0


def exponential_intensity(A: float, k: float, delta: float) -> float:
    if not (A > 0.0 and k > 0.0):
        return 0.0
    return A * math.exp(-k * delta)


def exponential_hamiltonian(A: float, k: float, p: float, delta_inf: float) -> float:
    if not (A > 0.0 and k > 0.0):
        return 0.0
    d_star = p + 1.0 / k
    if d_star >= delta_inf:
        return (A / k) * math.exp(-1.0 - k * p)
    lam = exponential_intensity(A, k, delta_inf)
    return lam * (delta_inf - p)


def exponential_premium(k: float, p: float, delta_inf: float) -> float:
    if not (k > 0.0):
        return delta_inf
    return max(delta_inf, p + 1.0 / k)


def logistic_intensity(lam: float, alpha: float, beta: float, contract_vega: float, delta: float) -> float:
    if not (lam > 0.0):
        return 0.0
    nu = max(abs(contract_vega), 1e-8)
    theta = max(beta, 1e-12) / nu
    u = alpha + theta * delta
    return lam / (1.0 + math.exp(u))


def logistic_premium(
    lam: float, alpha: float, beta: float, contract_vega: float, p: float, delta_inf: float
) -> float:
    del lam
    nu = max(abs(contract_vega), 1e-8)
    theta = max(beta, 1e-12) / nu
    d = p + 1.0 / theta
    for _ in range(16):
        e = math.exp(-(alpha + theta * d))
        g = d - p - (1.0 / theta) * (1.0 + e)
        gp = 1.0 + e
        d -= g / gp
    return max(delta_inf, d)


def logistic_hamiltonian(
    lam: float, alpha: float, beta: float, contract_vega: float, p: float, delta_inf: float
) -> float:
    d = logistic_premium(lam, alpha, beta, contract_vega, p, delta_inf)
    return logistic_intensity(lam, alpha, beta, contract_vega, d) * (d - p)


def _grid_count(cfg: OptionMmConfig) -> int:
    n = int(min(MAX_GRID, max(5, cfg.grid_n)))
    if n % 2 == 0:
        n -= 1
    return n


def _interp(limit: float, w: list[float], x: float) -> float:
    n = len(w)
    if n == 0:
        return 0.0
    if n == 1 or x <= -limit:
        return w[0]
    if x >= limit:
        return w[n - 1]
    d_v = (2.0 * limit) / float(n - 1)
    if not (d_v > 0.0):
        return w[0]
    pos = (x + limit) / d_v
    i = int(math.floor(pos))
    if i >= n - 1:
        return w[n - 1]
    frac = pos - float(i)
    return w[i] * (1.0 - frac) + w[i + 1] * frac


def _v_at(limit: float, n: int, j: int) -> float:
    if n <= 1:
        return 0.0
    d_v = (2.0 * limit) / float(n - 1)
    return -limit + d_v * float(j)


def _side_premium(cfg: OptionMmConfig, contract_vega: float, p: float) -> float:
    if cfg.intensity == "logistic":
        return logistic_premium(cfg.lambda0, cfg.alpha, cfg.beta, contract_vega, p, cfg.delta_inf)
    return exponential_premium(cfg.kappa, p, cfg.delta_inf)


def _side_h(cfg: OptionMmConfig, contract_vega: float, p: float) -> float:
    if cfg.intensity == "logistic":
        return logistic_hamiltonian(cfg.lambda0, cfg.alpha, cfg.beta, contract_vega, p, cfg.delta_inf)
    return exponential_hamiltonian(cfg.A, cfg.kappa, p, cfg.delta_inf)


def solve_grid(cfg: OptionMmConfig, vegas: list[float]) -> ValueGrid:
    n = _grid_count(cfg)
    limit = max(cfg.vega_limit, 1e-8)
    steps = max(int(cfg.n_steps), 1)
    horizon = max(cfg.horizon, 1e-8)
    dtau = horizon / float(steps)
    z = max(cfg.trade_size, 1e-8)
    pen = penalty_coeff(cfg)
    w = [0.0] * n
    for _step in range(steps):
        wn = [0.0] * n
        for j in range(n):
            v = _v_at(limit, n, j)
            ham = 0.0
            for nu in vegas:
                if nu == 0.0:
                    continue
                buy = v + z * nu
                if abs(buy) <= limit + 1e-9:
                    p_b = (w[j] - _interp(limit, w, buy)) / z
                    ham += z * _side_h(cfg, nu, p_b)
                sell = v - z * nu
                if abs(sell) <= limit + 1e-9:
                    p_a = (w[j] - _interp(limit, w, sell)) / z
                    ham += z * _side_h(cfg, nu, p_a)
            growth = ham + cfg.vol_edge * v - pen * v * v
            wn[j] = w[j] + dtau * growth
        w = wn
    return ValueGrid(n=n, vega_limit=limit, w=w)


def value_at(grid: ValueGrid, portfolio_vega: float) -> float:
    if grid.n == 0:
        return 0.0
    return _interp(grid.vega_limit, grid.w, portfolio_vega)


def quote_on_grid(
    cfg: OptionMmConfig,
    grid: ValueGrid,
    mid: float,
    portfolio_vega: float,
    contract_vega: float,
    spread_mult: float = 1.0,
    size_mult: float = 1.0,
) -> OptionQuote:
    if mid <= 0.0:
        mid = max(mid, 0.01)
    limit = grid.vega_limit if grid.vega_limit > 0.0 else max(cfg.vega_limit, 1e-8)
    z = max(cfg.trade_size, 1e-8)
    nu = contract_vega if contract_vega != 0.0 else 1e-8
    v = min(limit, max(-limit, portfolio_vega))
    w_here = value_at(grid, v)
    buy = v + z * nu
    sell = v - z * nu
    bid_blocked = abs(buy) > limit + 1e-8
    ask_blocked = abs(sell) > limit + 1e-8
    p_b = 0.0 if bid_blocked else (w_here - value_at(grid, buy)) / z
    p_a = 0.0 if ask_blocked else (w_here - value_at(grid, sell)) / z
    d_b = cfg.max_premium if bid_blocked else _side_premium(cfg, nu, p_b)
    d_a = cfg.max_premium if ask_blocked else _side_premium(cfg, nu, p_a)
    sm = max(spread_mult, 0.0)
    d_b *= sm
    d_a *= sm
    d_b = min(cfg.max_premium, max(cfg.min_premium, d_b))
    d_a = min(cfg.max_premium, max(cfg.min_premium, d_a))
    shift = nu * cfg.iv_alpha
    center = mid + shift + 0.5 * (d_a - d_b)
    half = 0.5 * (d_a + d_b)
    bid = max(0.01, mid + shift - d_b)
    ask = max(bid + 0.01, mid + shift + d_a)
    size = int(round(cfg.quote_size * max(size_mult, 0.0)))
    if size < 0:
        size = 0
    if size_mult > 0.0 and size < 1 and not bid_blocked and not ask_blocked:
        size = 1
    return OptionQuote(
        bid=bid,
        ask=ask,
        bid_size=0 if bid_blocked or size_mult <= 0.0 else size,
        ask_size=0 if ask_blocked or size_mult <= 0.0 else size,
        reservation=center,
        half_spread=half,
        delta_b=d_b,
        delta_a=d_a,
        portfolio_vega=v,
        bid_blocked=bid_blocked,
        ask_blocked=ask_blocked,
    )


def solve_and_quote(
    cfg: OptionMmConfig,
    mid: float,
    portfolio_vega: float,
    contract_vega: float,
    spread_mult: float = 1.0,
    size_mult: float = 1.0,
) -> OptionQuote:
    grid = solve_grid(cfg, [contract_vega])
    return quote_on_grid(cfg, grid, mid, portfolio_vega, contract_vega, spread_mult, size_mult)


def config_from_quoter(cfg: QuoterConfig) -> OptionMmConfig:
    kind = getattr(cfg, "intensity_kind", "exponential")
    return OptionMmConfig(
        gamma=cfg.gamma,
        xi=getattr(cfg, "xi", 1.0),
        A=cfg.A,
        kappa=cfg.kappa,
        lambda0=getattr(cfg, "logistic_lambda", 100.0),
        alpha=getattr(cfg, "logistic_alpha", 0.7),
        beta=getattr(cfg, "logistic_beta", 150.0),
        intensity="logistic" if kind in (1, "logistic") else "exponential",
        vega_limit=getattr(cfg, "vega_limit", 40.0),
        horizon=cfg.T_horizon,
        grid_n=getattr(cfg, "option_grid_n", 31),
        n_steps=getattr(cfg, "option_grid_steps", 60),
        vol_edge=getattr(cfg, "vol_edge", 0.0),
        rho=getattr(cfg, "option_rho", 0.0),
        iv_alpha=getattr(cfg, "iv_alpha", 0.0),
        min_premium=cfg.min_half_spread,
        max_premium=cfg.max_half_spread,
        quote_size=cfg.quote_size,
    )


def make_quote(
    mid: float,
    inventory: int,
    cfg: QuoterConfig,
    greeks: Greeks | None = None,
    spread_mult: float = 1.0,
    size_mult: float = 1.0,
) -> Quote:
    """Constant-vega quote. Portfolio vega = inventory × per-contract vega."""
    per = cfg.contract_vega if greeks is None or greeks.vega == 0.0 else greeks.vega
    ocfg = config_from_quoter(cfg)
    q = solve_and_quote(ocfg, mid, inventory * per, per, spread_mult, size_mult)
    return Quote(
        bid=q.bid,
        ask=q.ask,
        bid_size=q.bid_size,
        ask_size=q.ask_size,
        reservation=q.reservation,
        half_spread=q.half_spread,
    )


def stoikov_saglam_premiums(gamma: float, q: float, C: float, D: float, k_risk: float) -> StoikovPremiums:
    """Theorem 4. Premiums are distances from the option mid."""
    if not (D > 0.0):
        return StoikovPremiums(0.0, 0.0)
    cap = C / D
    rev = C / (2.0 * D)
    tilt = gamma * k_risk
    return StoikovPremiums(
        eps_ask=min(cap, max(0.0, rev - tilt * (q - 0.5))),
        eps_bid=min(cap, max(0.0, rev + tilt * (q + 0.5))),
    )


def stoikov_risk_scale(
    sigma: float, overnight: float, alpha: float, t_mat: float, gamma_greek: float, spot: float
) -> float:
    """Printed grouping of Theorem 4's k."""
    disc = 0.5 * sigma * sigma * overnight + alpha * alpha * t_mat * t_mat
    return disc * gamma_greek * gamma_greek * (spot**4) * sigma * sigma * overnight


def paper_touch_probability(alpha: float = 0.7) -> float:
    """Baldacci §4: fraction of λ filled at δ=0 is 1/(1+e^α) ≈ 33% for α=0.7."""
    return 1.0 / (1.0 + math.exp(alpha))
