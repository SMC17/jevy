"""Guéant–Lehalle–Fernandez-Tapia asymptotic quotes (arXiv 1105.3115).

Stationary / asymptotic closed-form approximations under inventory risk.
Mirrors ``zig/src/gueant.zig``. Toggle vs classic A–S via
``QuoterConfig.mode == "gueant_asymptotic"``.

Cite: https://arxiv.org/abs/1105.3115
"""

from __future__ import annotations

import math

from jev_omm.config import QuoterConfig
from jev_omm.models.types import Greeks, Quote


def inventory_scale(cfg: QuoterConfig) -> float:
    """ξ = √( (σ² γ)/(2 k A) · (1 + γ/k)^{1 + k/γ} )."""
    gamma = cfg.gamma
    k = cfg.kappa
    A = cfg.A
    sigma = cfg.sigma
    if gamma <= 0.0 or k <= 0.0 or A <= 0.0 or sigma < 0.0:
        return 0.0
    ratio = 1.0 + gamma / k
    exp = 1.0 + k / gamma
    inside = (sigma * sigma * gamma) / (2.0 * k * A) * (ratio**exp)
    return math.sqrt(max(inside, 0.0))


def intensity_half(cfg: QuoterConfig) -> float:
    if cfg.gamma <= 0.0 or cfg.kappa <= 0.0:
        return cfg.min_half_spread
    return (1.0 / cfg.gamma) * math.log(1.0 + cfg.gamma / cfg.kappa)


def reservation_price(
    mid: float,
    inventory: int,
    cfg: QuoterConfig,
    greeks: Greeks | None = None,
) -> float:
    xi = inventory_scale(cfg)
    r = mid - inventory * xi
    if greeks is not None and inventory != 0:
        r -= inventory * cfg.gamma_penalty * abs(greeks.gamma)
        r -= inventory * cfg.vega_penalty * abs(greeks.vega) * 0.01
    return r


def optimal_half_spread(cfg: QuoterConfig) -> float:
    half = intensity_half(cfg) + 0.5 * inventory_scale(cfg)
    return float(min(cfg.max_half_spread, max(cfg.min_half_spread, half)))


def optimal_offsets(cfg: QuoterConfig, inventory: int) -> tuple[float, float]:
    """Return (δ^b*, δ^a*) paper offsets."""
    psi = intensity_half(cfg)
    xi = inventory_scale(cfg)
    q = float(inventory)
    delta_b = psi + 0.5 * (2.0 * q + 1.0) * xi
    delta_a = psi - 0.5 * (2.0 * q - 1.0) * xi
    return delta_b, delta_a


def make_quote(
    mid: float,
    inventory: int,
    cfg: QuoterConfig,
    greeks: Greeks | None = None,
    *,
    spread_mult: float = 1.0,
    size_mult: float = 1.0,
) -> Quote:
    if mid <= 0.0:
        mid = max(mid, 0.01)
    r = reservation_price(mid, inventory, cfg, greeks)
    half = optimal_half_spread(cfg) * max(spread_mult, 0.25)
    half = float(min(cfg.max_half_spread, max(cfg.min_half_spread, half)))
    bid = max(0.01, r - half)
    ask = max(bid + 0.01, r + half)
    size = max(1, int(round(cfg.quote_size * max(size_mult, 0.0))))
    return Quote(
        bid=bid,
        ask=ask,
        bid_size=size,
        ask_size=size,
        reservation=r,
        half_spread=half,
    )
