"""Falsify the paper book. Adversarial regimes and a local tape.

No licensed OPRA history is invented. A missing Databento file is reported
as missing. ``CRYPTO_BETA`` stays a synthetic name inside the desk fixture.
Jev answers are Choice / Noul. This module does not emit an order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from jev_omm.desk.harness import (
    ADVERSARIAL_REGIMES,
    EVAL_SEEDS,
    RESEARCH_SLEEVES,
    DeskConfig,
    legacy_ortho_config,
    run_desk,
)
from jev_omm.desk.sleeves import LEGACY_SLEEVE_IDS, SLEEVE_IDS
from jev_omm.research.tape_walk import walk_forward
from jev_omm.research.walkforward import load_tape

_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "tape_synthetic.csv"


@dataclass
class SleeveFate:
    sleeve_id: str
    mean_penalized_sharpe: float
    mean_residual_mean: float
    frac_positive_test: float
    mean_weight: float
    mean_penalty: float
    killed_paths: int
    paths: int

    @property
    def survives(self) -> bool:
        return self.mean_residual_mean > 0.0 and self.mean_weight > 1e-8 and self.mean_penalty > 0.0


@dataclass
class AdversarialReport:
    seeds: tuple[int, ...]
    regimes: tuple[str, ...]
    n_steps: int
    by_regime: dict[str, list[SleeveFate]] = field(default_factory=dict)


def _fate(rows: list[dict[str, float]], sleeve_id: str) -> SleeveFate:
    sub = [r for r in rows if r["sleeve_id"] == sleeve_id]
    n = len(sub) or 1
    return SleeveFate(
        sleeve_id=sleeve_id,
        mean_penalized_sharpe=float(np.mean([r["pen_sharpe"] for r in sub])) if sub else 0.0,
        mean_residual_mean=float(np.mean([r["mean"] for r in sub])) if sub else 0.0,
        frac_positive_test=float(np.mean([r["test_pos"] for r in sub])) if sub else 0.0,
        mean_weight=float(np.mean([r["weight"] for r in sub])) if sub else 0.0,
        mean_penalty=float(np.mean([r["penalty"] for r in sub])) if sub else 0.0,
        killed_paths=int(sum(r["killed"] for r in sub)),
        paths=len(sub),
    )


def run_adversarial_desk(
    seeds: tuple[int, ...] = EVAL_SEEDS,
    regimes: tuple[str, ...] = ("baseline",) + ADVERSARIAL_REGIMES,
    *,
    n_steps: int = 40,
) -> AdversarialReport:
    """Multi-seed residual means after the honesty penalty and the capacity scale."""
    report = AdversarialReport(seeds=seeds, regimes=regimes, n_steps=n_steps)
    for regime in regimes:
        bag: list[dict[str, float]] = []
        for seed in seeds:
            run = run_desk(
                DeskConfig(
                    seed=int(seed),
                    regime=regime,
                    n_steps=n_steps,
                    fit_surfaces=False,
                )
            )
            killed = set(run.sleeve_kills)
            for row in run.scoreboard.rows:
                bag.append(
                    {
                        "sleeve_id": row.sleeve_id,
                        "pen_sharpe": row.residual_sharpe_penalized,
                        "mean": row.mean_residual * row.smoothness_penalty,
                        "test_pos": 1.0 if row.test_mean_residual > 0.0 else 0.0,
                        "weight": row.weight,
                        "penalty": row.smoothness_penalty,
                        "killed": 1.0 if row.sleeve_id in killed else 0.0,
                    }
                )
        report.by_regime[regime] = [_fate(bag, sleeve_id) for sleeve_id in SLEEVE_IDS]
    return report


def control_config(**overrides: object) -> DeskConfig:
    """Three products, the original eight sleeves, honesty on."""
    cfg = DeskConfig(
        products=("EQ_INDEX", "EQ_SINGLE", "FX_PAIR"),
        sleeves=LEGACY_SLEEVE_IDS,
        fit_surfaces=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def render_adversarial_markdown(report: AdversarialReport) -> str:
    lines = [
        "# Adversarial desk",
        "",
        "Synthetic paths. `synthetic_fixture=1`. Residual mean is multiplied by the smoothness penalty. "
        "Weight is the post-penalty, post-floor, post-kill, post-capacity number. "
        "A sleeve survives a regime when that penalized mean is positive and the mean weight is not zero. "
        "Not annualized. Not a capacity.",
        "",
        f"Seeds {list(report.seeds)}. Steps {report.n_steps}.",
        "",
    ]
    for regime, fates in report.by_regime.items():
        survivors = [f.sleeve_id for f in fates if f.survives]
        dead = [f.sleeve_id for f in fates if not f.survives]
        lines.append(f"## {regime}")
        lines.append("")
        lines.append(f"Survivors ({len(survivors)}): {', '.join(survivors) if survivors else 'none'}.")
        lines.append("")
        lines.append(f"Dead ({len(dead)}): {', '.join(dead) if dead else 'none'}.")
        lines.append("")
        lines.append("| sleeve | pen. mean | pen. Sharpe | P(test mean>0) | mean weight | mean penalty | kills |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for fate in fates:
            lines.append(
                f"| {fate.sleeve_id} | {_fmt(fate.mean_residual_mean)} | {_fmt(fate.mean_penalized_sharpe)} | "
                f"{_fmt(fate.frac_positive_test)} | {_fmt(fate.mean_weight)} | {_fmt(fate.mean_penalty)} | "
                f"{fate.killed_paths}/{fate.paths} |"
            )
        lines.append("")
    return "\n".join(lines)


def _tape_from_databento_json(path: Path):
    """The public preview is a JSON list of CSV lines, not a CSV file."""
    import json

    from jev_omm.research.tape import tape_from_rows

    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"{path} is not a Databento sample payload")
    header, *body = payload
    names = str(header).split(",")
    rows = []
    for line in body:
        parts = str(line).split(",")
        if len(parts) < len(names):
            parts = parts + [""] * (len(names) - len(parts))
        rows.append(dict(zip(names, parts)))
    return tape_from_rows(rows, source=f"databento-sample:{path.name}")


def tape_replay_summary(path: str | Path) -> dict[str, object]:
    """Walk-forward the existing quoters on a local file. No invented prints."""
    file = Path(path)
    if file.suffix.lower() == ".json":
        tape = _tape_from_databento_json(file)
    else:
        tape = load_tape(str(file))
    walked = walk_forward(tape)
    rows = walked["rows"]
    fills = {r.label: r.n_fills for r in rows}
    return {
        "source": walked["source"],
        "synthetic_fixture": walked["synthetic_fixture"],
        "n": walked["n"],
        "symbols": walked["symbols"],
        "counts": walked["counts"],
        "fills": fills,
        "zero_fill_labels": [label for label, n in fills.items() if n <= 0.0],
        "notes": list(walked.get("notes") or []),
        "spot_known": walked["spot_known"],
    }


def render_tape_markdown(summaries: list[dict[str, object]]) -> str:
    lines = [
        "# Local tape",
        "",
        "The quoters are the 0.9 walk-forward set. Many of them post wider than a short book and receive zero fills. "
        "A zero is a zero. This is not an OPRA backtest and it does not invent prints.",
        "",
    ]
    if not summaries:
        lines.append("No local tape was loaded.")
        lines.append("")
        return "\n".join(lines)
    for summary in summaries:
        lines.append(f"## {summary['source']}")
        lines.append("")
        synth = summary["synthetic_fixture"]
        lines.append(
            f"synthetic_fixture={synth}. Rows {summary['n']}. Symbols {summary['symbols']}. "
            f"Spot known: {summary['spot_known']}."
        )
        lines.append("")
        lines.append(f"Book labels: {summary['counts']}.")
        lines.append("")
        lines.append("| quoter | fills |")
        lines.append("| --- | --- |")
        fills = summary["fills"]
        assert isinstance(fills, dict)
        for label, nfill in fills.items():
            lines.append(f"| {label} | {float(nfill):.0f} |")
        lines.append("")
        zeros = summary["zero_fill_labels"]
        assert isinstance(zeros, list)
        lines.append(
            "Zero fills: " + (", ".join(str(z) for z in zeros) if zeros else "none") + "."
        )
        lines.append("")
    return "\n".join(lines)


def default_tapes() -> list[Path]:
    found = []
    if _FIXTURE.is_file():
        found.append(_FIXTURE)
    local = Path("data/local")
    if local.is_dir():
        for path in sorted(local.glob("*")):
            if path.suffix.lower() in {".csv", ".parquet", ".json"} and path.is_file():
                found.append(path)
    return found


def research_weight_check(run_weights: dict[str, float], run_means: dict[str, float]) -> list[str]:
    """Sleeves in the starved set with weight above 0.02 and a positive residual mean."""
    kept = []
    for name in RESEARCH_SLEEVES:
        if run_means.get(name, 0.0) > 0.0 and run_weights.get(name, 0.0) > 0.02:
            kept.append(name)
    return kept


def ortho_weight_vector(seed: int = 11, n_steps: int = 80) -> dict[str, float]:
    """1.1 allocator on the 1.1 economy. Used as the before column."""
    run = run_desk(legacy_ortho_config(seed=seed, n_steps=n_steps, fit_surfaces=False))
    return dict(run.weights)
