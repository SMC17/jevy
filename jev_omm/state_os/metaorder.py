"""Remaining metaorder quantity on a synthetic parent.

Square-root temporary impact plus a smaller linear permanent piece. While the
parent prints at a constant rate the price is concave in executed size. After
the parent stops, the temporary piece decays and the permanent piece stays:
partial reversion.

The remaining-size estimator is identified by a public stop rule, not by
reading the parent's order id. The parent stops when marginal square-root
impact equals a cost λ. During the print, remaining is that terminal size
minus what has already printed. When the print stops, remaining is zero.
A path that is not concave in cumulative size is noise and is not a parent.

- Tóth, Lempérière, Deremble, de Lataillade, Kockelkoren, Bouchaud, Phys. Rev. X
  2011, https://doi.org/10.1103/PhysRevX.1.021006
- Bacry, Iuga, Lasnier, Lehalle, Market Microstructure and Liquidity 2015,
  https://doi.org/10.1142/S2382626615500094
- Almgren, Thum, Hauptmann, Li, *Direct Estimation of Equity Market Impact*,
  Risk, July 2005.
"""

from __future__ import annotations

import math


def sqrt_impact(quantity: float, sigma: float, volume: float, y: float = 1.0) -> float:
    """I(Q) ≈ Y σ √(|Q| / V), signed."""
    if volume <= 0.0 or quantity == 0.0 or sigma < 0.0 or y == 0.0:
        return 0.0
    return y * sigma * math.sqrt(abs(quantity) / volume) * (1.0 if quantity > 0.0 else -1.0)


def marginal_sqrt(quantity: float, sigma: float, volume: float, y: float = 1.0) -> float:
    """dI/dQ for Q > 0."""
    if quantity <= 0.0 or volume <= 0.0 or sigma <= 0.0:
        return math.inf
    return y * sigma / (2.0 * math.sqrt(quantity * volume))


def parent_size_from_stop(y: float, sigma: float, volume: float, lambda_stop: float) -> float:
    """Size at which marginal square-root impact equals λ.

    Y σ / (2 √(Q V)) = λ  ⇒  Q = (Y σ / (2 λ))² / V.
    """
    if lambda_stop <= 0.0 or volume <= 0.0 or sigma <= 0.0 or y <= 0.0:
        return 0.0
    return (y * sigma / (2.0 * lambda_stop)) ** 2 / volume


def permanent_impact(quantity: float, sigma: float, volume: float, y: float, perm_frac: float) -> float:
    if volume <= 0.0 or quantity == 0.0:
        return 0.0
    return perm_frac * y * sigma * (quantity / volume)


def synthetic_parent(
    y: float,
    sigma: float,
    volume: float,
    lambda_stop: float,
    rate: float,
    n_after: int,
    perm_frac: float,
    decay: float,
) -> dict[str, list[float] | float]:
    """Price path of one buy parent, then a quiet revert window.

    ``rate`` is the child size per bar. The path stops adding once Q(λ) is done.
    """
    total = parent_size_from_stop(y, sigma, volume, lambda_stop)
    prices = [0.0]
    flows = [0.0]
    remaining = [total]
    executed = [0.0]
    q = 0.0
    while q < total - 1e-12:
        dq = min(rate, total - q)
        q += dq
        price = sqrt_impact(q, sigma, volume, y) + permanent_impact(q, sigma, volume, y, perm_frac)
        prices.append(price)
        flows.append(dq)
        executed.append(q)
        remaining.append(total - q)
    temp_done = sqrt_impact(q, sigma, volume, y)
    perm_done = permanent_impact(q, sigma, volume, y, perm_frac)
    for k in range(n_after):
        temp = temp_done * math.exp(-decay * (k + 1))
        prices.append(temp + perm_done)
        flows.append(0.0)
        executed.append(q)
        remaining.append(0.0)
    return {
        "prices": prices,
        "flows": flows,
        "executed": executed,
        "remaining": remaining,
        "total": total,
        "permanent": perm_done,
    }


def _corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def concavity_on_execution(prices: list[float], flows: list[float]) -> dict[str, float]:
    """Fit the printing window. A parent correlates with √q and has a negative second difference."""
    q = 0.0
    qs: list[float] = []
    px: list[float] = []
    for price, flow in zip(prices, flows):
        if flow > 0.0:
            q += flow
            qs.append(q)
            px.append(price)
    if len(qs) < 4:
        return {"corr_sqrt": 0.0, "corr_linear": 0.0, "mean_second": 0.0}
    sqrt_q = [math.sqrt(value) for value in qs]
    # Equal-q second difference is not available; use consecutive price increments.
    first = [px[i] - px[i - 1] for i in range(1, len(px))]
    second = [first[i] - first[i - 1] for i in range(1, len(first))]
    mean_second = sum(second) / len(second)
    return {
        "corr_sqrt": _corr(sqrt_q, px),
        "corr_linear": _corr(qs, px),
        "mean_second": mean_second,
    }


def is_live_parent(prices: list[float], flows: list[float], min_corr: float = 0.98) -> bool:
    """True only when the printing window is square-root concave. Noise fails."""
    stats = concavity_on_execution(prices, flows)
    return stats["corr_sqrt"] >= min_corr and stats["mean_second"] < 0.0 and stats["corr_sqrt"] > stats["corr_linear"]


def estimate_remaining(
    executed: float,
    flow_now: float,
    y: float,
    sigma: float,
    volume: float,
    lambda_stop: float,
) -> float:
    """Remaining size while the child is still printing. A flat tape is not a live parent."""
    if abs(flow_now) <= 1e-12:
        return 0.0
    total = parent_size_from_stop(y, sigma, volume, lambda_stop)
    return max(total - executed, 0.0)
