"""JSONL case log. Replay recomputes the score from the stored paths."""

from __future__ import annotations

import json
from pathlib import Path

from jev_omm.training.cases import CASES, run_pair
from jev_omm.training.scoring import SPECS, CaseRun, score_path


def run_to_records(run: CaseRun) -> list[dict]:
    spec = run.spec
    head = {
        "type": "CaseMeta",
        "case": spec.name,
        "strategy": run.strategy,
        "role": spec.role,
        "information_set": spec.information_set,
        "constraints": spec.constraints,
        "lesson": spec.lesson,
        "citation": spec.citation,
        "decision_source": run.decision_source,
    }
    score = {
        "type": "CaseScore",
        "case": spec.name,
        "strategy": run.strategy,
        "inventory_path": run.inventory_path,
        "beta_path": run.beta_path,
        **run.score.as_dict(),
    }
    return [head, score, *run.events]


def write_jsonl(path: str | Path, runs: list[CaseRun]) -> Path:
    p = Path(path)
    lines = []
    for run in runs:
        for rec in run_to_records(run):
            lines.append(json.dumps(rec, separators=(",", ":")))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def replay_scores(path: str | Path) -> list[dict]:
    """Recompute risk-adjusted PnL from stored paths and compare to the log."""
    checked = []
    for rec in read_jsonl(path):
        if rec.get("type") != "CaseScore":
            continue
        inv_lambda = 0.0
        beta_lambda = 0.0
        name = rec["case"]
        if name == "location_arb":
            inv_lambda, beta_lambda = 0.01, 2.0
        elif name == "pm_fair_value":
            inv_lambda, beta_lambda = 0.02, 1.5
        elif name == "liability_facilitator":
            inv_lambda = 0.05
        elif name == "mm_inventory":
            inv_lambda = 0.08
        elif name == "flow_vpin":
            inv_lambda = 0.02
        elif name == "dealer_gamma":
            inv_lambda = 0.001
        elif name == "cot_fade":
            inv_lambda = 0.02
        sc = score_path(
            rec["absolute_pnl"],
            rec["inventory_path"],
            rec["beta_path"],
            inv_lambda=inv_lambda,
            beta_lambda=beta_lambda,
            exec_penalty=rec["exec_penalty"],
            peer_pnl=rec["peer_pnl"],
        )
        if abs(sc.risk_adjusted - rec["risk_adjusted"]) > 1e-9:
            raise ValueError(f"replay mismatch for {name}/{rec['strategy']}")
        if abs(sc.relative_score - rec["relative_score"]) > 1e-9:
            raise ValueError(f"relative mismatch for {name}/{rec['strategy']}")
        checked.append({"case": name, "strategy": rec["strategy"], "risk_adjusted": sc.risk_adjusted})
    return checked


def all_runs() -> list[CaseRun]:
    runs: list[CaseRun] = []
    for name in CASES:
        naive, desk = run_pair(name)
        runs.extend((naive, desk))
    return runs


def spec_names() -> list[str]:
    return [SPECS[n].name for n in CASES]
