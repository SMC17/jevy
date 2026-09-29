"""Paper scoreboard for Choice / Score style probabilities.

Persists `(state, answer, confidence, outcome)` and scores Brier, log loss,
and expected calibration error. Nothing here emits an order. The synthetic
ablation compares a deterministic always-quote rule with a policy that is
allowed to see a probability. If the probability is the true conditional
frequency, the assisted rule can look good; that is an upper bound, not
evidence about live Jev. A noisy probability is reported next to it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class ScoreTuple:
    state: str
    question: str
    answer: float
    confidence: float
    outcome: float
    regime: str


def _clip_prob(p: float) -> float:
    return min(1.0 - 1e-12, max(1e-12, float(p)))


def brier(rows: list[ScoreTuple]) -> float:
    if not rows:
        return 0.0
    return float(sum((r.answer - r.outcome) ** 2 for r in rows) / len(rows))


def log_loss(rows: list[ScoreTuple]) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        p = _clip_prob(row.answer)
        y = row.outcome
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return float(total / len(rows))


def ece(rows: list[ScoreTuple], n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins on [0, 1]. Empty bins are skipped."""
    if not rows:
        return 0.0
    edges = [i / n_bins for i in range(n_bins + 1)]
    total = 0.0
    n = len(rows)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        bucket = [
            r
            for r in rows
            if (r.answer >= lo and (r.answer < hi or (i == n_bins - 1 and r.answer <= hi)))
        ]
        if not bucket:
            continue
        conf = sum(r.answer for r in bucket) / len(bucket)
        freq = sum(r.outcome for r in bucket) / len(bucket)
        total += (len(bucket) / n) * abs(freq - conf)
    return float(total)


def ece_by_regime(rows: list[ScoreTuple], n_bins: int = 10) -> dict[str, float]:
    regimes = sorted({r.regime for r in rows})
    return {name: ece([r for r in rows if r.regime == name], n_bins=n_bins) for name in regimes}


def persist_scores(path: str | Path, rows: list[ScoreTuple]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def load_scores(path: str | Path) -> list[ScoreTuple]:
    rows: list[ScoreTuple] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rows.append(ScoreTuple(**raw))
    return rows


def _pnl(quote: np.ndarray, adverse: np.ndarray, spread: float, loss: float) -> float:
    gain = np.where(quote, np.where(adverse, -loss, spread), 0.0)
    return float(np.mean(gain))


def synthetic_jev_ablation(*, n: int = 4000, seed: int = 0, spread: float = 0.05, loss: float = 0.20) -> dict[str, float | str]:
    """Binary adverse-fill game. x is the true P(adverse | state).

    Deterministic policy quotes every round. The oracle assisted policy
    pulls when x >= 0.55 (it is handed the truth). The noisy policy pulls
    when x + N(0, 0.25) clipped to (0, 1) is >= 0.55.
    """
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    adverse = rng.random(n) < x
    regime = np.where(x >= 0.55, "toxic", "calm")
    det = np.ones(n, dtype=bool)
    oracle = x < 0.55
    p_hat = np.clip(x + rng.normal(0.0, 0.25, n), 1e-3, 1.0 - 1e-3)
    noisy = p_hat < 0.55
    flat = np.full(n, 0.5)

    def rows_for(prob: np.ndarray, question: str) -> list[ScoreTuple]:
        out = []
        for i in range(n):
            out.append(
                ScoreTuple(
                    state=json.dumps({"p_adverse": round(float(x[i]), 6), "i": i}),
                    question=question,
                    answer=float(prob[i]),
                    confidence=float(max(prob[i], 1.0 - prob[i])),
                    outcome=1.0 if bool(adverse[i]) else 0.0,
                    regime=str(regime[i]),
                )
            )
        return out

    oracle_rows = rows_for(x, "adverse_fill")
    noisy_rows = rows_for(p_hat, "adverse_fill")
    flat_rows = rows_for(flat, "adverse_fill")
    det_pnl = _pnl(det, adverse, spread, loss)
    oracle_pnl = _pnl(oracle, adverse, spread, loss)
    noisy_pnl = _pnl(noisy, adverse, spread, loss)
    return {
        "n": float(n),
        "deterministic_pnl_per_round": det_pnl,
        "oracle_jev_pnl_per_round": oracle_pnl,
        "noisy_jev_pnl_per_round": noisy_pnl,
        "oracle_minus_deterministic": oracle_pnl - det_pnl,
        "noisy_minus_deterministic": noisy_pnl - det_pnl,
        "brier_oracle": brier(oracle_rows),
        "brier_noisy": brier(noisy_rows),
        "brier_flat": brier(flat_rows),
        "log_loss_oracle": log_loss(oracle_rows),
        "log_loss_noisy": log_loss(noisy_rows),
        "log_loss_flat": log_loss(flat_rows),
        "ece_oracle": ece(oracle_rows),
        "ece_noisy": ece(noisy_rows),
        "ece_flat": ece(flat_rows),
        "ece_noisy_calm": ece_by_regime(noisy_rows).get("calm", 0.0),
        "ece_noisy_toxic": ece_by_regime(noisy_rows).get("toxic", 0.0),
        "note": (
            "Oracle Jev is handed P(adverse). A higher oracle PnL is not live alpha. "
            "Read noisy_minus_deterministic before claiming the assisted policy helps."
        ),
    }


def render_score_markdown(summary: dict[str, float | str] | None = None) -> str:
    s = summary if summary is not None else synthetic_jev_ablation()
    noisy_gap = float(s["noisy_minus_deterministic"])
    if noisy_gap > 0.0:
        verdict = (
            f"On this draw the noisy assisted policy beats always-quote by {noisy_gap:.4f} "
            "per round after the stylized spread and adverse cost. That is still a "
            "synthetic Bernoulli tape, not a market."
        )
    else:
        verdict = (
            f"On this draw the noisy assisted policy does not beat always-quote "
            f"(gap {noisy_gap:.4f} per round). The interface stays; it is not "
            "evidence that Jev improves out-of-sample economics after costs."
        )
    return (
        "# Jev paper scoreboard (synthetic)\n\n"
        "Each round has a visible state `x = P(adverse fill)`. Quoting earns "
        f"+0.05 if the fill is not adverse and −0.20 if it is. Pulling earns 0. "
        "Deterministic policy always quotes. Oracle policy pulls when `x ≥ 0.55` "
        "and is given the true probability. Noisy policy uses `clip(x + N(0, 0.25))`. "
        "No orders are emitted. No TypeSafe key is used.\n\n"
        f"- n = {int(float(s['n']))}\n"
        f"- deterministic PnL / round: {float(s['deterministic_pnl_per_round']):.4f}\n"
        f"- oracle-assisted PnL / round: {float(s['oracle_jev_pnl_per_round']):.4f} "
        f"(gap {float(s['oracle_minus_deterministic']):.4f})\n"
        f"- noisy-assisted PnL / round: {float(s['noisy_jev_pnl_per_round']):.4f} "
        f"(gap {noisy_gap:.4f})\n\n"
        "| forecast | Brier | log loss | ECE |\n"
        "| --- | --- | --- | --- |\n"
        f"| oracle (the true x) | {float(s['brier_oracle']):.4f} | {float(s['log_loss_oracle']):.4f} | {float(s['ece_oracle']):.4f} |\n"
        f"| noisy | {float(s['brier_noisy']):.4f} | {float(s['log_loss_noisy']):.4f} | {float(s['ece_noisy']):.4f} |\n"
        f"| flat 0.5 | {float(s['brier_flat']):.4f} | {float(s['log_loss_flat']):.4f} | {float(s['ece_flat']):.4f} |\n\n"
        f"Noisy ECE by regime: calm {float(s['ece_noisy_calm']):.4f}, "
        f"toxic {float(s['ece_noisy_toxic']):.4f}.\n\n"
        f"{verdict}\n\n"
        f"{s['note']}\n\n"
        + _sim_loop_section()
    )


def _sim_loop_section() -> str:
    from jev_omm.research.jev_loop import render_sim_loop_markdown

    return render_sim_loop_markdown()
