"""Rolling trade imbalance / simplified VPIN-style bucket toxicity.

Research-grade only — not production VPIN. Feeds Decision state so toxicity
Score / informed_flow Noul have real numeric features.

Ref framing: Easley, López de Prado, O'Hara (RFS); intro
https://www.quantresearch.org/From%20PIN%20to%20VPIN.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToxicityTracker:
    bucket_volume: float = 20.0
    window_buckets: int = 10
    buy_vol: float = 0.0
    sell_vol: float = 0.0
    imbalances: list[float] = field(default_factory=list)
    cum_buy: float = 0.0
    cum_sell: float = 0.0
    ewma_signed: float = 0.0
    ewma_alpha: float = 0.2

    def _close_bucket(self) -> None:
        tot = self.buy_vol + self.sell_vol
        imb = abs(self.buy_vol - self.sell_vol) / tot if tot > 0 else 0.0
        self.imbalances.append(imb)
        if len(self.imbalances) > self.window_buckets:
            self.imbalances = self.imbalances[-self.window_buckets :]
        self.buy_vol = 0.0
        self.sell_vol = 0.0

    def on_trade(self, signed_volume: float) -> None:
        if signed_volume == 0.0:
            return
        abs_v = abs(signed_volume)
        is_buy = signed_volume > 0.0
        if is_buy:
            self.cum_buy += abs_v
        else:
            self.cum_sell += abs_v
        self.ewma_signed = (
            self.ewma_alpha * signed_volume + (1.0 - self.ewma_alpha) * self.ewma_signed
        )
        remaining = abs_v
        guard = 0
        while remaining > 0.0 and guard < 64:
            guard += 1
            room = self.bucket_volume - (self.buy_vol + self.sell_vol)
            take = min(remaining, room)
            if is_buy:
                self.buy_vol += take
            else:
                self.sell_vol += take
            remaining -= take
            if self.buy_vol + self.sell_vol + 1e-12 >= self.bucket_volume:
                self._close_bucket()

    def on_print(self, price: float, size: int, mid: float) -> None:
        if size == 0:
            return
        abs_sz = float(abs(size))
        signed = abs_sz if price >= mid else -abs_sz
        self.on_trade(signed)

    def vpin(self) -> float:
        if not self.imbalances:
            return 0.0
        return sum(self.imbalances) / len(self.imbalances)

    def trade_imbalance(self) -> float:
        tot = self.cum_buy + self.cum_sell
        if tot <= 0:
            return 0.0
        return (self.cum_buy - self.cum_sell) / tot

    def abs_imbalance(self) -> float:
        return abs(self.trade_imbalance())

    def ewma_toxicity(self) -> float:
        scale = max(self.bucket_volume * 0.25, 1.0)
        return min(1.0, abs(self.ewma_signed) / scale)

    def composite(self) -> float:
        return min(1.0, 0.5 * self.vpin() + 0.3 * self.abs_imbalance() + 0.2 * self.ewma_toxicity())

    def feature_dict(self) -> dict[str, float]:
        """Keys suitable for ``build_mm_state`` / DecisionSnapshot."""
        return {
            "vpin": self.vpin(),
            "trade_imbalance": self.trade_imbalance(),
            "abs_imbalance": self.abs_imbalance(),
            "ewma_toxicity": self.ewma_toxicity(),
            "toxicity_composite": self.composite(),
        }
