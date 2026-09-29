"""Desk scoreboard. Per-step Sharpe is mean / sample std of residual PnL.

It is not annualized and it is not a live track record. Research fees are
already inside the residual series.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from jev_omm.research.metrics import max_drawdown


def per_step_sharpe(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size < 2:
        return 0.0
    sd = float(np.std(arr, ddof=1))
    # A numerical leftover after a strip is not a Sharpe. 1e-8 matches the allocator floor.
    if sd < 1e-8:
        return 0.0
    return float(np.mean(arr) / sd)


@dataclass
class SleeveRow:
    sleeve_id: str
    enabled: bool
    weight: float
    risk_budget: float
    raw_pnl: float
    residual_pnl: float
    mean_residual: float
    sharpe_residual: float
    r2: float
    beta: float
    gamma_coef: float
    vega_coef: float
    volga_coef: float
    vanna_coef: float
    var_coef: float
    max_dd_raw: float
    max_dd_residual: float
    n_fills: int
    fees: float


@dataclass
class DeskScoreboard:
    rows: list[SleeveRow]
    products: list[str]
    flagged_pairs: list[tuple[str, str, float]]
    sleeve_ids: list[str]
    pearson: np.ndarray
    spearman: np.ndarray
    desk_raw_pnl: float
    desk_residual_pnl: float
    desk_sharpe_residual: float
    desk_max_dd_residual: float
    weight_sum: float
    synthetic_fixture: int = 1
    corr_cap: float = 0.35
    pre_gate_max_abs_rho: float = 0.0
    max_abs_rho: float = 0.0
    product_corr_sleeve: str = ""
    product_ids: list[str] = field(default_factory=list)
    product_pearson: np.ndarray | None = None
    stress_worst: float = 0.0
    stress_label: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "products": list(self.products),
            "synthetic_fixture": self.synthetic_fixture,
            "desk_raw_pnl": self.desk_raw_pnl,
            "desk_residual_pnl": self.desk_residual_pnl,
            "desk_sharpe_residual": self.desk_sharpe_residual,
            "desk_max_dd_residual": self.desk_max_dd_residual,
            "weight_sum": self.weight_sum,
            "flagged_pairs": [
                {"a": a, "b": b, "pearson": rho} for a, b, rho in self.flagged_pairs
            ],
            "sleeves": [row.__dict__.copy() for row in self.rows],
            "notes": list(self.notes),
        }


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def to_markdown(board: DeskScoreboard, *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Synthetic paper desk. `synthetic_fixture=1`. Per-step residual Sharpe is mean / sample std after the research fee. It is not annualized.",
        "",
        f"Products: {', '.join(board.products)}.",
        "",
        f"Desk raw PnL {_fmt(board.desk_raw_pnl)}, residual PnL {_fmt(board.desk_residual_pnl)}, residual Sharpe {_fmt(board.desk_sharpe_residual)}, residual max drawdown {_fmt(board.desk_max_dd_residual)}, weight sum {_fmt(board.weight_sum)}.",
        "",
        "| sleeve | weight | raw | residual | mean resid | resid Sharpe | R² | β | γ | ν | volga | vanna | var | fills |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in board.rows:
        lines.append(
            "| {id} | {w} | {raw} | {res} | {mu} | {sh} | {r2} | {b} | {g} | {v} | {vo} | {va} | {var} | {n} |".format(
                id=row.sleeve_id,
                w=_fmt(row.weight),
                raw=_fmt(row.raw_pnl),
                res=_fmt(row.residual_pnl),
                mu=_fmt(row.mean_residual),
                sh=_fmt(row.sharpe_residual),
                r2=_fmt(row.r2),
                b=_fmt(row.beta),
                g=_fmt(row.gamma_coef),
                v=_fmt(row.vega_coef),
                vo=_fmt(row.volga_coef),
                va=_fmt(row.vanna_coef),
                var=_fmt(row.var_coef),
                n=row.n_fills,
            )
        )
    lines.append("")
    lines.append("## Pairwise residual Pearson")
    lines.append("")
    ids = board.sleeve_ids
    header = "| | " + " | ".join(ids) + " |"
    sep = "| --- | " + " | ".join("---" for _ in ids) + " |"
    lines.extend([header, sep])
    for i, name in enumerate(ids):
        cells = " | ".join(_fmt(float(board.pearson[i, j])) for j in range(len(ids)))
        lines.append(f"| {name} | {cells} |")
    lines.append("")
    cap = board.corr_cap
    lines.append(f"## Pairs with |ρ| > {cap:.2f}")
    lines.append("")
    if not board.flagged_pairs:
        lines.append(f"No enabled pair exceeded {cap:.2f} on this sample.")
    else:
        for a, b, rho in board.flagged_pairs:
            lines.append(f"- `{a}` / `{b}`: Pearson {_fmt(rho)}")
    lines.append("")
    lines.append(
        f"Pre-gate max |ρ| {_fmt(board.pre_gate_max_abs_rho)}. "
        f"Post-gate max |ρ| {_fmt(board.max_abs_rho)}."
    )
    lines.append("")
    if board.product_pearson is not None and board.product_ids:
        lines.append(f"## {board.product_corr_sleeve} residual correlation across products")
        lines.append("")
        pids = board.product_ids
        lines.append("| | " + " | ".join(pids) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in pids) + " |")
        for i, name in enumerate(pids):
            cells = " | ".join(_fmt(float(board.product_pearson[i, j])) for j in range(len(pids)))
            lines.append(f"| {name} | {cells} |")
        lines.append("")
    if board.stress_label:
        lines.append(
            f"Scenario grid worst {_fmt(board.stress_worst)} at {board.stress_label}. Research units, end-of-path greeks."
        )
        lines.append("")
    if board.notes:
        lines.append("## Notes")
        lines.append("")
        for note in board.notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def cumulative_drawdown(path: np.ndarray) -> float:
    if path.size == 0:
        return 0.0
    return max_drawdown(np.cumsum(path).tolist())
