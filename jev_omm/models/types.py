"""Core domain types for the options MM engine."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Side(str, Enum):
    BID = "bid"
    ASK = "ask"


class OptionRight(str, Enum):
    CALL = "call"
    PUT = "put"


class Greeks(BaseModel):
    delta: float
    gamma: float
    vega: float  # per 1.0 vol point (i.e. ∂V/∂σ, σ in decimal)
    theta: float  # per year (calendar)


class OptionContract(BaseModel):
    """Single European option series identifier."""

    underlying: str = "SIM"
    strike: float
    expiry_years: float  # time to expiry in years
    right: OptionRight = OptionRight.CALL

    @property
    def is_call(self) -> bool:
        return self.right == OptionRight.CALL


class Quote(BaseModel):
    bid: float
    ask: float
    bid_size: int = 1
    ask_size: int = 1
    reservation: float = 0.0
    half_spread: float = 0.0


class Fill(BaseModel):
    time: float
    side: Side
    price: float
    size: int
    mid_at_fill: float


class Position(BaseModel):
    qty: int = 0  # + long options
    avg_price: float = 0.0
    cash: float = 0.0

    def apply_fill(self, side: Side, price: float, size: int) -> None:
        """Update inventory and cash. Bid fill ⇒ we buy; ask fill ⇒ we sell."""
        signed = size if side == Side.BID else -size
        notional = price * size
        if side == Side.BID:
            # We bought: cash out, inventory up
            new_qty = self.qty + size
            if new_qty != 0 and self.qty >= 0:
                # average up/down long
                total_cost = self.avg_price * self.qty + notional
                self.avg_price = total_cost / new_qty if new_qty else 0.0
            elif new_qty == 0:
                self.avg_price = 0.0
            else:
                # flipped short through zero — reset avg to fill price
                self.avg_price = price
            self.qty = new_qty
            self.cash -= notional
        else:
            # We sold
            new_qty = self.qty - size
            if new_qty != 0 and self.qty <= 0:
                total_cost = self.avg_price * abs(self.qty) + notional
                self.avg_price = total_cost / abs(new_qty) if new_qty else 0.0
            elif new_qty == 0:
                self.avg_price = 0.0
            else:
                self.avg_price = price if self.qty <= 0 else self.avg_price
            self.qty = new_qty
            self.cash += notional


class RiskSnapshot(BaseModel):
    inventory: int
    delta: float
    gamma: float
    vega: float
    cash_pnl: float
    quoting_allowed: bool = True
    breach_reason: Optional[str] = None


class MarketState(BaseModel):
    time: float
    spot: float
    option_mid: float
    iv: float
    greeks: Greeks
    quote: Optional[Quote] = None
