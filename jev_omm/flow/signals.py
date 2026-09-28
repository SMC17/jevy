"""Signed-flow features and the quote prior they feed.

Research / paper only. No OPRA, TRF, or venue session.

- Lee–Ready quote test, tick test at the midpoint
  (Lee & Ready, Journal of Finance 1991, https://doi.org/10.1111/j.1540-6261.1991.tb02683.x).
- One-level order-flow imbalance
  (Cont, Kukanov, Stoikov, https://doi.org/10.1093/jjfinec/nbt003,
  https://arxiv.org/abs/1011.6402).
- A synthetic off-exchange flag. Prints do not update the lit book.
- Layered cancel score: size cancelled behind the touch, before a trade,
  inside a short life. Phenomenology from Cartea, Jaimungal, Wang,
  Applied Mathematical Finance 2020, https://doi.org/10.1080/1350486X.2020.1726783.
  This is not their control problem and it does not place orders.

``flow_prior`` is the identity when every input is zero: spread multiplier 1,
size multiplier 1. Callers that never pass a tape are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class LeeReady:
    """Quote test, then the tick test when the print is on the midpoint."""

    prev_price: float | None = None
    prev_sign: int = 0

    def sign(self, price: float, bid: float, ask: float) -> int:
        mid = 0.5 * (bid + ask)
        if price > mid:
            s = 1
        elif price < mid:
            s = -1
        elif self.prev_price is None:
            s = self.prev_sign
        elif price > self.prev_price:
            s = 1
        elif price < self.prev_price:
            s = -1
        else:
            s = self.prev_sign
        self.prev_price = price
        if s != 0:
            self.prev_sign = s
        return s


def ofi_increment(
    prev_bid: float,
    prev_bid_size: float,
    prev_ask: float,
    prev_ask_size: float,
    bid: float,
    bid_size: float,
    ask: float,
    ask_size: float,
) -> float:
    """Cont–Kukanov–Stoikov one-level contribution e_n."""
    e_b = 0.0
    if bid >= prev_bid:
        e_b += bid_size
    if bid <= prev_bid:
        e_b -= prev_bid_size
    e_a = 0.0
    if ask <= prev_ask:
        e_a += ask_size
    if ask >= prev_ask:
        e_a -= prev_ask_size
    return e_b - e_a


def normalize_ofi(ofi: float, depth: float) -> float:
    """Map a raw OFI sum into [-1, 1]. ``depth`` <= 0 leaves the signal at 0."""
    if depth <= 0.0:
        return 0.0
    return _clamp(ofi / depth, -1.0, 1.0)


@dataclass
class BookEvent:
    kind: str  # "add" | "cancel" | "trade"
    level: int
    size: float
    age: float = 0.0


def spoof_score(events: list[BookEvent], *, flicker_age: float = 0.5, min_size: float = 1.0) -> float:
    """Fraction of flicker cancels behind the touch versus cancels plus trades.

    Level 0 is the touch. A cancel at level >= 1, younger than ``flicker_age``,
    counts as layered. Trades in the same window pull the score down.
    """
    layered = 0.0
    traded = 0.0
    for event in events:
        if event.kind == "trade":
            traded += event.size
        elif (
            event.kind == "cancel"
            and event.level >= 1
            and event.age <= flicker_age
            and event.size >= min_size
        ):
            layered += event.size
    return layered / (layered + traded + 1e-12)


def flow_toxicity(
    vpin: float,
    ofi_norm: float,
    aggr_imbalance: float,
    off_exchange_share: float,
    spoof: float,
) -> float:
    """Blend in [0, 1]. Zero inputs → 0."""
    tox = (
        0.50 * max(vpin, 0.0)
        + 0.20 * abs(ofi_norm)
        + 0.15 * abs(aggr_imbalance) * max(off_exchange_share, 0.0)
        + 0.35 * max(spoof, 0.0)
    )
    return _clamp(tox, 0.0, 1.0)


def flow_prior(
    vpin: float,
    ofi_norm: float,
    aggr_imbalance: float,
    off_exchange_share: float,
    spoof: float,
) -> tuple[float, float, float]:
    """Return ``(toxicity, spread_mult, size_mult)``. Identity at the origin."""
    tox = flow_toxicity(vpin, ofi_norm, aggr_imbalance, off_exchange_share, spoof)
    spread_mult = 1.0 + 1.25 * tox
    size_mult = 1.0 / (1.0 + 1.50 * tox)
    return tox, spread_mult, size_mult
