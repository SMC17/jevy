"""Local CSV/Parquet tapes and crossed/locked/stale handling.

The checked-in fixture is synthetic. This module does not invent OPRA
prints and does not open a live session. A Databento historical pull, when
a key is present, lives in ``databento_hist`` and writes under ``data/local``.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Legacy six columns. ``spot`` may be absent on an option NBBO file.
CORE_COLUMNS = ("time_seconds", "spot", "bid", "ask", "bid_sz", "ask_sz")
OPTION_COLUMNS = (
    "symbol",
    "expiry",
    "strike",
    "right",
    "trade_px",
    "trade_sz",
    "exchange",
    "seq",
    "synthetic_fixture",
)

# Substrings that mean "open a live vendor session". A local path that
# merely contains the word databento is a file, not a session.
LIVE_MARKERS = (
    "opra://",
    "live-nbbo",
    "polygon.io",
    "cboe-live",
    "live.databento.com",
    "ws://",
    "wss://",
    "databento://",
    "hist.databento.com",
)

_OSI = re.compile(r"^(?P<root>[A-Z0-9 ]+?)(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


@dataclass
class TapeFeeSchedule:
    """Per-contract research costs applied on the tape replay.

    These are not an OPRA or equity exchange fee card. ``fee`` is paid on
    every assumed fill. ``rebate`` is subtracted from that fee (a maker
    credit). Net cost is ``fee - rebate`` and can be negative.
    """

    fee_per_contract: float = 0.05
    rebate_per_contract: float = 0.0

    @property
    def net_per_contract(self) -> float:
        return float(self.fee_per_contract) - float(self.rebate_per_contract)


@dataclass
class Tape:
    """One path. ``spot`` is NaN where the file has no underlying print."""

    time_seconds: np.ndarray
    spot: np.ndarray
    source: str
    bid: np.ndarray | None = None
    ask: np.ndarray | None = None
    bid_sz: np.ndarray | None = None
    ask_sz: np.ndarray | None = None
    trade_px: np.ndarray | None = None
    trade_sz: np.ndarray | None = None
    symbol: list[str] = field(default_factory=list)
    expiry: list[str] = field(default_factory=list)
    strike: np.ndarray | None = None
    right: list[str] = field(default_factory=list)
    exchange: list[str] = field(default_factory=list)
    seq: np.ndarray | None = None
    synthetic_fixture: bool = False
    book_state: list[str] = field(default_factory=list)
    notes: str = ""

    def __len__(self) -> int:
        return int(self.time_seconds.shape[0])


def refuse_live(spec: str) -> None:
    """Reject live session URLs. Local files and ``synthetic`` pass."""
    low = spec.strip().lower()
    if any(mark in low for mark in LIVE_MARKERS):
        raise RuntimeError(
            "Live OPRA/NBBO streaming is not wired. Use spec='synthetic' or a "
            "local CSV/Parquet. The no-account sample downloader is "
            "scripts/fetch_databento_sample.py. No API key is required."
        )


def parse_osi(symbol: str) -> tuple[str, str, str, float] | None:
    """Parse an OSI option symbol. Returns root, YYYY-MM-DD, C/P, strike.

    ``TSLA  230901C00250000`` → (``TSLA``, ``2023-09-01``, ``C``, 250).
    Returns None when the string is not OSI. Does not invent a contract.
    """
    compact = re.sub(r"\s+", "", symbol.strip().upper())
    match = _OSI.match(compact)
    if match is None:
        return None
    ymd = match.group("ymd")
    year = 2000 + int(ymd[:2])
    expiry = f"{year:04d}-{int(ymd[2:4]):02d}-{int(ymd[4:6]):02d}"
    strike = int(match.group("strike")) / 1000.0
    return match.group("root").strip(), expiry, match.group("cp"), strike


def _f(text: str | None) -> float:
    if text is None:
        return float("nan")
    raw = str(text).strip()
    if raw == "" or raw.lower() in {"nan", "none", "null"}:
        return float("nan")
    return float(raw)


def _book_label(bid: float, ask: float) -> str:
    bid_ok = math.isfinite(bid) and bid > 0.0
    ask_ok = math.isfinite(ask) and ask > 0.0
    if bid_ok and ask_ok:
        if bid > ask + 1e-9:
            return "crossed"
        if abs(bid - ask) <= 1e-9:
            return "locked"
        return "two_sided"
    if bid_ok or ask_ok:
        return "one_sided"
    return "missing"


def annotate_book(tape: Tape, *, stale_repeat: int = 3, stale_gap_seconds: float = 300.0) -> Tape:
    """Label each row. Rows are kept; the replay pulls on a bad label.

    Stale: the two-sided book is unchanged for ``stale_repeat`` consecutive
    rows while a trade (if any) prints outside that book, or the clock jumps
    by more than ``stale_gap_seconds``. The first row of a frozen book stays
    ``two_sided``; later repeats are ``stale``.
    """
    n = len(tape)
    bids = tape.bid if tape.bid is not None else np.full(n, np.nan)
    asks = tape.ask if tape.ask is not None else np.full(n, np.nan)
    trades = tape.trade_px if tape.trade_px is not None else np.full(n, np.nan)
    labels = [_book_label(float(bids[i]), float(asks[i])) for i in range(n)]
    run = 0
    prev: tuple[float, float] | None = None
    for i in range(n):
        if i > 0 and float(tape.time_seconds[i] - tape.time_seconds[i - 1]) > stale_gap_seconds:
            if labels[i] == "two_sided":
                labels[i] = "stale"
        if labels[i] != "two_sided":
            run = 0
            prev = None
            continue
        key = (round(float(bids[i]), 6), round(float(asks[i]), 6))
        if prev is not None and key == prev:
            run += 1
        else:
            run = 1
            prev = key
        trade = float(trades[i])
        outside = math.isfinite(trade) and (trade < key[0] - 1e-9 or trade > key[1] + 1e-9)
        if run >= stale_repeat and outside:
            labels[i] = "stale"
    tape.book_state = labels
    return tape


def _col(rows: list[dict[str, str]], name: str) -> np.ndarray | None:
    if not rows or name not in rows[0]:
        return None
    return np.array([_f(r.get(name)) for r in rows], dtype=float)


def _str_col(rows: list[dict[str, str]], name: str) -> list[str]:
    if not rows or name not in rows[0]:
        return []
    return [str(r.get(name, "") or "").strip() for r in rows]


def tape_from_rows(rows: list[dict[str, str]], *, source: str) -> Tape:
    """Build a tape from string rows. Missing ``spot`` stays NaN."""
    if not rows:
        raise ValueError(f"{source} has a header and no rows")
    fields = set(rows[0])
    has_core = all(c in fields for c in CORE_COLUMNS)
    has_book = "time_seconds" in fields and "bid" in fields and "ask" in fields
    has_vendor_book = "bid_px_00" in fields or "ask_px_00" in fields
    if not has_core and not has_book and not has_vendor_book:
        raise ValueError(
            f"{source} missing price columns; expected {list(CORE_COLUMNS)} "
            "or time_seconds,bid,ask or a Databento bid_px_00/ask_px_00 schema"
        )
    if has_vendor_book and "time_seconds" not in fields:
        rows = [_with_canonical_clock(r, i, rows) for i, r in enumerate(rows)]
    times = _col(rows, "time_seconds")
    assert times is not None
    spot = _col(rows, "spot")
    if spot is None:
        spot = np.full(len(rows), np.nan)
    bid = _col(rows, "bid")
    ask = _col(rows, "ask")
    if bid is None and "bid_px_00" in fields:
        bid = _col(rows, "bid_px_00")
    if ask is None and "ask_px_00" in fields:
        ask = _col(rows, "ask_px_00")
    bid_sz = _col(rows, "bid_sz")
    ask_sz = _col(rows, "ask_sz")
    if bid_sz is None and "bid_sz_00" in fields:
        bid_sz = _col(rows, "bid_sz_00")
    if ask_sz is None and "ask_sz_00" in fields:
        ask_sz = _col(rows, "ask_sz_00")
    trade_px = _col(rows, "trade_px")
    if trade_px is None and "price" in fields:
        trade_px = _col(rows, "price")
    trade_sz = _col(rows, "trade_sz")
    if trade_sz is None and "size" in fields:
        trade_sz = _col(rows, "size")
    symbols = _str_col(rows, "symbol")
    if symbols and "raw_symbol" in fields:
        symbols = [s or raw for s, raw in zip(symbols, _str_col(rows, "raw_symbol"))]
    elif not symbols and "raw_symbol" in fields:
        symbols = _str_col(rows, "raw_symbol")
    expiry = _str_col(rows, "expiry")
    strike = _col(rows, "strike")
    right = _str_col(rows, "right")
    if symbols and (not expiry or strike is None or not right):
        expiry, strike, right = _fill_osi(symbols, expiry, strike, right, len(rows))
    flag_vals = _str_col(rows, "synthetic_fixture")
    synthetic = bool(flag_vals) and all(v in {"1", "true", "True"} for v in flag_vals)
    seq = _col(rows, "seq")
    if seq is None and "sequence" in fields:
        seq = _col(rows, "sequence")
    exchange = _str_col(rows, "exchange")
    if not exchange and "publisher_id" in fields:
        exchange = _str_col(rows, "publisher_id")
    tape = Tape(
        time_seconds=times,
        spot=spot,
        source=source,
        bid=bid,
        ask=ask,
        bid_sz=bid_sz,
        ask_sz=ask_sz,
        trade_px=trade_px,
        trade_sz=trade_sz,
        symbol=symbols,
        expiry=expiry,
        strike=strike,
        right=right,
        exchange=exchange,
        seq=seq,
        synthetic_fixture=synthetic,
    )
    return annotate_book(tape)


def _fill_osi(
    symbols: list[str],
    expiry: list[str],
    strike: np.ndarray | None,
    right: list[str],
    n: int,
) -> tuple[list[str], np.ndarray, list[str]]:
    exp = list(expiry) if expiry else [""] * n
    rght = list(right) if right else [""] * n
    strikes = strike.copy() if strike is not None else np.full(n, np.nan)
    if len(exp) < n:
        exp.extend([""] * (n - len(exp)))
    if len(rght) < n:
        rght.extend([""] * (n - len(rght)))
    for i, sym in enumerate(symbols):
        parsed = parse_osi(sym) if sym else None
        if parsed is None:
            continue
        _root, ymd, cp, k = parsed
        if not exp[i]:
            exp[i] = ymd
        if not rght[i]:
            rght[i] = cp
        if not math.isfinite(float(strikes[i])):
            strikes[i] = k
    return exp, strikes, rght


def _with_canonical_clock(row: dict[str, str], index: int, rows: list[dict[str, str]]) -> dict[str, str]:
    """Map a vendor ISO timestamp onto seconds from the first row."""
    out = dict(row)
    stamp = row.get("ts_event") or row.get("ts_recv") or ""
    out["time_seconds"] = str(_iso_to_seconds(stamp, rows[0].get("ts_event") or rows[0].get("ts_recv") or "", index))
    return out


def _iso_to_seconds(stamp: str, origin: str, index: int) -> float:
    if not stamp:
        return float(index)
    try:
        return _unix(stamp) - _unix(origin)
    except ValueError:
        return float(index)


def _unix(stamp: str) -> float:
    """Parse a Databento ISO-8601 timestamp with a fractional part. UTC assumed."""
    text = stamp.strip().replace("Z", "")
    if "T" not in text:
        raise ValueError(stamp)
    day, clock = text.split("T", 1)
    year, month, dom = (int(p) for p in day.split("-"))
    frac = 0.0
    if "." in clock:
        clock, frac_s = clock.split(".", 1)
        frac = float("0." + frac_s)
    hour, minute, second = (int(p) for p in clock.split(":"))
    # Civil UTC to unix without importing datetime's timezone edge cases.
    # Days from 1970-01-01 via a proleptic Gregorian count.
    z = _days_from_civil(year, month, dom)
    return ((z * 24 + hour) * 60 + minute) * 60 + second + frac


def _days_from_civil(year: int, month: int, day: int) -> int:
    y = year - (1 if month <= 2 else 0)
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (month + (-3 if month > 2 else 9)) + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def load_local(path: str | Path) -> Tape:
    """Load a CSV or Parquet file. Live URLs are refused before the open."""
    spec = str(path)
    refuse_live(spec)
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(
            f"No tape at {spec}. Licensed OPRA/NBBO history is not bundled. "
            "A synthetic schema fixture is at jev_omm/data/fixtures/tape_synthetic.csv "
            "(synthetic_fixture=1). It is not an OPRA print."
        )
    suffix = file.suffix.lower()
    if suffix == ".parquet":
        return _read_parquet(file)
    return _read_csv(file)


def _read_csv(path: Path) -> Tape:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(r) for r in reader]
    if reader.fieldnames is None:
        raise ValueError(f"{path} has no header")
    return tape_from_rows(rows, source=f"csv:{path.name}")


def _read_parquet(path: Path) -> Tape:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to read Parquet") from exc
    try:
        frame = pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet needs pyarrow (pip install pyarrow). CSV does not."
        ) from exc
    rows = [{k: "" if v is None else str(v) for k, v in rec.items()} for rec in frame.to_dict(orient="records")]
    # pandas may stringify nan as 'nan', which _f already accepts.
    return tape_from_rows(rows, source=f"parquet:{path.name}")


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    fields = list(rows[0].keys())
    with dest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def count_states(tape: Tape) -> dict[str, int]:
    labels = tape.book_state or []
    keys = ("two_sided", "crossed", "locked", "stale", "one_sided", "missing")
    return {k: sum(1 for x in labels if x == k) for k in keys}
