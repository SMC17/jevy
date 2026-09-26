"""Sequenced event log (JSONL) — Jane Street / exchange-inspired.

Zig writes the canonical sim log (`zig/src/event_log.zig`). Python can:
  - append DecisionSnapshot / market events during paper sims
  - read JSONL for notebooks
  - replay markout / PnL from Fill + BookTop.mid (deterministic)

Event types: BookTop, UnderlyingTick, Quote, Fill, Cancel,
DecisionSnapshot, RiskBreach.

Each record: ``{"seq": N, "ts": ..., "type": "...", ...}`` with monotonic seq.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, TextIO, Union

from jev_omm.decisions.schemas import (
    ChoiceAnswer,
    NoulAnswer,
    QuoteAdjustments,
    ScoreAnswer,
    SystemOneResult,
)
from jev_omm.models.types import Fill, Quote, Side
from jev_omm.pnl.markout import MarkoutTracker


PathLike = Union[str, Path]


def answers_to_jsonable(result: SystemOneResult) -> tuple[dict[str, Any], dict[str, float]]:
    """Serialize System One answers + Choice/Score confidences for DecisionSnapshot."""
    answers: dict[str, Any] = {}
    confidence: dict[str, float] = {}
    for key, ans in result.answers.items():
        if isinstance(ans, NoulAnswer):
            answers[key] = {"type": "noul", "noul": float(ans.noul)}
        elif isinstance(ans, ChoiceAnswer):
            answers[key] = {
                "type": "choice",
                "choice": ans.choice,
                "probabilities": dict(ans.probabilities),
                "confidence": float(ans.confidence),
            }
            confidence[key] = float(ans.confidence)
        elif isinstance(ans, ScoreAnswer):
            answers[key] = {
                "type": "score",
                "score": float(ans.score),
                "legend": dict(ans.legend),
                "probabilities": dict(ans.probabilities),
                "confidence": float(ans.confidence),
            }
            confidence[key] = float(ans.confidence)
        else:
            answers[key] = {"type": "unknown", "raw": str(ans)}
    return answers, confidence


@dataclass
class EventLog:
    """Append-only JSONL writer with monotonic seq + running SHA-256."""

    path: Optional[Path] = None
    seq: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    _hasher: Any = field(default_factory=hashlib.sha256, repr=False)
    _fp: Optional[TextIO] = field(default=None, repr=False)

    def open(self, path: PathLike) -> "EventLog":
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("w", encoding="utf-8")
        return self

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _append(self, ts: float, type_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.seq += 1
        rec = {"seq": self.seq, "ts": float(ts), "type": type_name, **payload}
        line = json.dumps(rec, separators=(",", ":"), sort_keys=False) + "\n"
        self._hasher.update(line.encode("utf-8"))
        self.records.append(rec)
        if self._fp is not None:
            self._fp.write(line)
            self._fp.flush()
        return rec

    def sha256_hex(self) -> str:
        return self._hasher.hexdigest()

    def append_underlying_tick(self, ts: float, step: int, spot: float) -> dict[str, Any]:
        return self._append(ts, "UnderlyingTick", {"step": int(step), "spot": float(spot)})

    def append_book_top(
        self,
        ts: float,
        step: int,
        *,
        bid: float,
        ask: float,
        bid_sz: int,
        ask_sz: int,
        mid: float,
    ) -> dict[str, Any]:
        return self._append(
            ts,
            "BookTop",
            {
                "step": int(step),
                "bid": float(bid),
                "ask": float(ask),
                "bid_sz": int(bid_sz),
                "ask_sz": int(ask_sz),
                "mid": float(mid),
            },
        )

    def append_quote(self, ts: float, step: int, quote: Quote) -> dict[str, Any]:
        return self._append(
            ts,
            "Quote",
            {
                "step": int(step),
                "bid": float(quote.bid),
                "ask": float(quote.ask),
                "bid_sz": int(quote.bid_size),
                "ask_sz": int(quote.ask_size),
                "reservation": float(quote.reservation),
                "half_spread": float(quote.half_spread),
            },
        )

    def append_quote_strike(
        self, ts: float, step: int, strike: float, quote: Quote
    ) -> dict[str, Any]:
        """Quote with strike tag (multi-strike strip; backward-compatible extra field)."""
        return self._append(
            ts,
            "Quote",
            {
                "step": int(step),
                "strike": float(strike),
                "bid": float(quote.bid),
                "ask": float(quote.ask),
                "bid_sz": int(quote.bid_size),
                "ask_sz": int(quote.ask_size),
                "reservation": float(quote.reservation),
                "half_spread": float(quote.half_spread),
            },
        )

    def append_fill(self, ts: float, step: int, fill: Fill) -> dict[str, Any]:
        return self._append(
            ts,
            "Fill",
            {
                "step": int(step),
                "side": fill.side.value if isinstance(fill.side, Side) else str(fill.side),
                "price": float(fill.price),
                "size": int(fill.size),
                "mid": float(fill.mid_at_fill),
            },
        )

    def append_cancel(self, ts: float, step: int, reason: str) -> dict[str, Any]:
        return self._append(ts, "Cancel", {"step": int(step), "reason": reason})

    def append_risk_breach(
        self, ts: float, step: int, reason: str, inventory: int
    ) -> dict[str, Any]:
        return self._append(
            ts,
            "RiskBreach",
            {"step": int(step), "reason": reason, "inventory": int(inventory)},
        )

    def append_decision_snapshot_raw(
        self,
        ts: float,
        step: int,
        *,
        source: str,
        model: str,
        answers: dict[str, Any],
        confidence: dict[str, float],
        state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """DecisionSnapshot from already-serialized answers (+ optional multi-strike state)."""
        payload: dict[str, Any] = {
            "step": int(step),
            "source": source,
            "model": model,
            "answers": answers,
            "confidence": confidence,
        }
        if state is not None:
            payload["state"] = state
        return self._append(ts, "DecisionSnapshot", payload)

    def append_decision_snapshot(
        self,
        ts: float,
        step: int,
        result: SystemOneResult,
        *,
        adjustments: Optional[QuoteAdjustments] = None,
    ) -> dict[str, Any]:
        answers, confidence = answers_to_jsonable(result)
        # Normalize legacy "typesafe" → "live"
        source = result.source
        if source == "typesafe":  # type: ignore[comparison-overlap]
            source = "live"
        payload: dict[str, Any] = {
            "step": int(step),
            "source": source,
            "model": result.model,
            "answers": answers,
            "confidence": confidence,
        }
        if adjustments is not None:
            payload["spread_mult"] = float(adjustments.spread_mult)
            payload["size_mult"] = float(adjustments.size_mult)
            payload["pull"] = bool(adjustments.pull)
            payload["hedge_now"] = bool(adjustments.hedge_now)
            payload["reason"] = adjustments.reason
        return self._append(ts, "DecisionSnapshot", payload)


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    """Load all events from a JSONL file (notebook / research friendly)."""
    out: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def iter_jsonl(path: PathLike) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def sha256_file(path: PathLike) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ReplayResult:
    n_events: int = 0
    n_fills: int = 0
    n_breaches: int = 0
    n_decisions: int = 0
    last_seq: int = 0
    cash: float = 0.0
    qty: int = 0
    last_mid: float = 0.0
    marked_pnl: float = 0.0
    attribution: Any = None
    log_sha256: str = ""
    decision_sources: list[str] = field(default_factory=list)


def replay_events(events: Iterable[dict[str, Any]], *, sha256: str = "") -> ReplayResult:
    """Recompute markout / PnL from BookTop.mid + Fill events (deterministic)."""
    from jev_omm.pnl.mark import marked_pnl
    from jev_omm.models.types import Position

    tracker = MarkoutTracker()
    pos = Position()
    result = ReplayResult(log_sha256=sha256)
    last_mid = 0.0

    for ev in events:
        result.n_events += 1
        result.last_seq = int(ev.get("seq", result.last_seq))
        et = ev.get("type")
        if et == "BookTop":
            step = int(ev.get("step", 0))
            mid = float(ev["mid"])
            tracker.on_step(step, mid)
            last_mid = mid
        elif et == "Fill":
            step = int(ev.get("step", 0))
            side = Side(ev["side"])
            fill = Fill(
                time=float(ev.get("ts", 0.0)),
                side=side,
                price=float(ev["price"]),
                size=int(ev["size"]),
                mid_at_fill=float(ev["mid"]),
            )
            pos.apply_fill(side, fill.price, fill.size)
            tracker.on_fill(step, fill)
            result.n_fills += 1
            last_mid = fill.mid_at_fill
        elif et == "RiskBreach":
            result.n_breaches += 1
        elif et == "DecisionSnapshot":
            result.n_decisions += 1
            src = str(ev.get("source", ""))
            if src:
                result.decision_sources.append(src)

    result.cash = pos.cash
    result.qty = pos.qty
    result.last_mid = last_mid
    result.marked_pnl = marked_pnl(pos, last_mid)
    result.attribution = tracker.summary
    return result


def replay_jsonl(path: PathLike) -> ReplayResult:
    p = Path(path)
    return replay_events(read_jsonl(p), sha256=sha256_file(p))
