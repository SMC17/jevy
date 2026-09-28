"""Latent executable book.

Displayed size is not the size that will be there when F arrives.

    L_exec = L_disp + E[R] − E[C] + E[H]

R replenishes with a half-life. C cancels with a hazard that rises in
distance from the touch and in volatility. H is the iceberg: a probability
times a hidden multiple of the displayed size. Queue survival is the
probability the displayed order is still there after dt.

The slope of L in price, vol, and size is a finite difference of that
identity. Citations for the cancel/impact phenomenology, not a claim that
these elasticities are estimated from a live feed:

- Eisler, Bouchaud, Kockelkoren, Quantitative Finance 2012,
  https://doi.org/10.1080/14697688.2010.528444
- Cont, Kukanov, Stoikov, https://doi.org/10.1093/jjfinec/nbt003
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass
class LatentBook:
    displayed: float
    half_life: float
    deficit: float
    dist: float
    sigma: float
    iceberg_prob: float
    hidden_multiple: float
    base_hazard: float = 0.40
    elast_price: float = 1.50
    elast_sigma: float = 2.00
    dt: float = 0.10


def cancel_hazard(book: LatentBook) -> float:
    return book.base_hazard * math.exp(book.elast_price * abs(book.dist) + book.elast_sigma * book.sigma)


def expectations(book: LatentBook) -> dict[str, float]:
    hazard = cancel_hazard(book)
    e_cancel = book.displayed * (1.0 - math.exp(-hazard * book.dt))
    if book.half_life > 0.0:
        e_replenish = book.deficit * (1.0 - math.exp(-book.dt / book.half_life))
    else:
        e_replenish = 0.0
    e_hidden = book.iceberg_prob * book.hidden_multiple * book.displayed
    survival = math.exp(-hazard * book.dt)
    l_exec = book.displayed + e_replenish - e_cancel + e_hidden
    return {
        "hazard": hazard,
        "e_replenish": e_replenish,
        "e_cancel": e_cancel,
        "e_hidden": e_hidden,
        "survival": survival,
        "l_exec": l_exec,
    }


def d_cancel_d_sigma(book: LatentBook) -> float:
    """Analytic ∂E[C]/∂σ. Hazard is log-linear in σ, so the slope is positive."""
    hazard = cancel_hazard(book)
    return book.displayed * math.exp(-hazard * book.dt) * book.dt * hazard * book.elast_sigma


def d_cancel_d_distance(book: LatentBook) -> float:
    """Analytic ∂E[C]/∂|P−p| on the positive side of distance."""
    hazard = cancel_hazard(book)
    return book.displayed * math.exp(-hazard * book.dt) * book.dt * hazard * book.elast_price


def liquidity_surface(book: LatentBook, step: float = 1e-3) -> dict[str, float]:
    """Finite-difference slopes of L_exec. ∂L/∂Q uses displayed size as Q."""
    base = expectations(book)["l_exec"]
    up_p = expectations(replace(book, dist=book.dist + step))["l_exec"]
    up_s = expectations(replace(book, sigma=book.sigma + step))["l_exec"]
    up_q = expectations(replace(book, displayed=book.displayed + step))["l_exec"]
    return {
        "dL_dP": (up_p - base) / step,
        "dL_dSigma": (up_s - base) / step,
        "dL_dQ": (up_q - base) / step,
    }
