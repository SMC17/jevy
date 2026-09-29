"""Schema-compatible synthetic NBBO paths.

Every price here is invented. ``synthetic_fixture`` is 1. These rows are
not OPRA prints and must not be cited as vendor data.
"""

from __future__ import annotations

import math

import numpy as np

from jev_omm.config import TRADING_SECONDS_PER_YEAR
from jev_omm.pricing.black_scholes import price_and_greeks
from jev_omm.research.tape import Tape, annotate_book

REGIMES = (
    "fixture",
    "baseline",
    "crossed_locked",
    "stale",
    "one_sided",
    "fees",
    "thin",
    "jumps",
)


def synthetic_rows(
    *,
    n: int = 96,
    seed: int = 11,
    regime: str = "fixture",
    spot0: float = 100.0,
    strike: float = 100.0,
    dt_seconds: float = 60.0,
) -> list[dict[str, object]]:
    """One invented option NBBO plus the underlying spot used to price it.

    The market half-spread is a few cents. Trades are a mix of touch prints
    and occasional sweeps so a wide theoretical quote and a touch quote do
    not see the same fills. Regime flags distort the book after the path
    is drawn. ``fixture`` is the checked-in file: mostly baseline, with a
    short crossed, locked, and stale patch so the handler has rows to count.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime}")
    rng = np.random.default_rng(seed)
    dt_years = dt_seconds / TRADING_SECONDS_PER_YEAR
    spot = float(spot0)
    rows: list[dict[str, object]] = []
    expiry_years = 30.0 / 365.25
    iv = 0.22
    frozen_book: tuple[float, float, float, float] | None = None
    for i in range(n):
        z = float(rng.standard_normal())
        jump = 1.0
        if regime == "jumps" and i in {n // 3, 2 * n // 3}:
            jump = 0.94 if i == n // 3 else 1.06
        spot = max(1.0, spot * math.exp((-0.5 * 0.20 * 0.20) * dt_years + 0.20 * math.sqrt(dt_years) * z) * jump)
        t_rem = max(expiry_years - i * dt_years, 1.0 / 365.25)
        mid, _greeks = price_and_greeks(spot, strike, t_rem, 0.05, 0.0, iv, True)
        mid = max(0.05, float(mid))
        if regime == "thin":
            half = 0.20
            bsz, asz = 1.0, 1.0
        else:
            half = 0.04 + 0.01 * abs(float(rng.standard_normal()))
            bsz = float(rng.integers(5, 40))
            asz = float(rng.integers(5, 40))
        bid = max(0.01, mid - half)
        ask = bid + 2.0 * half
        u = float(rng.random())
        if regime == "one_sided" or u < 0.55:
            trade_px = bid
            trade_sz = 2.0
        elif u < 0.85:
            trade_px = ask
            trade_sz = 2.0
        else:
            # Sweep through a quote that sits well behind the touch.
            trade_px = bid - 0.45 if u < 0.92 else ask + 0.45
            trade_sz = 3.0
            trade_px = max(0.01, trade_px)
        if regime == "stale" and n // 5 <= i < n // 5 + 6:
            if frozen_book is None:
                frozen_book = (bid, ask, bsz, asz)
            bid, ask, bsz, asz = frozen_book
            trade_px = max(0.01, bid - 0.30)
        if regime == "crossed_locked":
            if i % 7 == 0:
                bid, ask = ask + 0.05, ask
            elif i % 7 == 1:
                bid = ask
        if regime == "fixture":
            if i in {12, 13}:
                bid = ask
            elif i in {24, 25}:
                bid, ask = ask + 0.03, ask
            elif 36 <= i <= 39:
                if i == 36:
                    frozen_book = (bid, ask, bsz, asz)
                else:
                    bid, ask, bsz, asz = frozen_book  # type: ignore[misc]
                    trade_px = max(0.01, bid - 0.25)
        rows.append(
            {
                "time_seconds": f"{(i + 1) * dt_seconds:.1f}",
                "spot": f"{spot:.6f}",
                "bid": f"{bid:.6f}",
                "ask": f"{ask:.6f}",
                "bid_sz": f"{bsz:.0f}",
                "ask_sz": f"{asz:.0f}",
                "trade_px": f"{trade_px:.6f}",
                "trade_sz": f"{trade_sz:.0f}",
                "symbol": "SYNTH 260101C00100000",
                "expiry": "2026-01-01",
                "strike": f"{strike:.4f}",
                "right": "C",
                "exchange": "SYNTH",
                "seq": str(i + 1),
                "synthetic_fixture": "1",
            }
        )
    return rows


def rows_to_tape(rows: list[dict[str, object]], *, source: str) -> Tape:
    from jev_omm.research.tape import tape_from_rows

    as_str = [{k: str(v) for k, v in row.items()} for row in rows]
    tape = tape_from_rows(as_str, source=source)
    tape.synthetic_fixture = True
    tape.notes = "SYNTHETIC_FIXTURE=1. Invented prices. Not an OPRA print."
    return annotate_book(tape)


def synthetic_tape(*, n: int = 96, seed: int = 11, regime: str = "fixture", spot0: float = 100.0) -> Tape:
    rows = synthetic_rows(n=n, seed=seed, regime=regime, spot0=spot0)
    return rows_to_tape(rows, source=f"synthetic:{regime}:seed={seed}")
