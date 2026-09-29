"""Multi-seed synthetic regimes. Prices are invented.

Each regime is a distortion of the same generator. The table is mean ±
sample std across seeds, paired against fixed-spread on that seed. A model
"earns" the row only when its mean PnL beats fixed-spread by more than the
seed-to-seed std of the paired gap. Otherwise it does not.
"""

from __future__ import annotations

import math

from jev_omm.research.synthetic_tape import REGIMES, synthetic_tape
from jev_omm.research.tape import TapeFeeSchedule
from jev_omm.research.tape_walk import ReplayRow, replay
from jev_omm.research.walkforward import ablation_specs

ADVERSARIAL_REGIMES = (
    "baseline",
    "crossed_locked",
    "stale",
    "one_sided",
    "fees",
    "thin",
    "jumps",
)
DEFAULT_SEEDS = (7, 11, 19, 23, 29)


def _specs() -> list[tuple[str, str, str]]:
    return list(ablation_specs()) + [("join_touch", "join_touch", "off")]


def run_regime(
    regime: str,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n: int = 64,
) -> dict[str, object]:
    if regime not in REGIMES:
        raise ValueError(regime)
    # ``fees`` keeps the baseline book and only changes the research charge,
    # so the gap versus other regimes is the fee rather than a second book.
    if regime == "fees":
        book_regime = "baseline"
        fees = TapeFeeSchedule(fee_per_contract=0.20, rebate_per_contract=0.05)
    else:
        book_regime = regime
        fees = TapeFeeSchedule(fee_per_contract=0.02, rebate_per_contract=0.0)
    by_label: dict[str, list[ReplayRow]] = {label: [] for label, _, _ in _specs()}
    for seed in seeds:
        tape = synthetic_tape(n=n, seed=seed, regime=book_regime)
        for label, mode, pack in _specs():
            by_label[label].append(
                replay(
                    tape,
                    label=label,
                    mode=mode,
                    pack=pack,
                    fees=fees,
                    kappa=1.5,
                    sigma=0.45,
                    start=0,
                    end=len(tape),
                )
            )
    fixed = by_label["fixed_spread"]
    summary: list[dict[str, object]] = []
    for label, rows in by_label.items():
        pnls = [r.pnl for r in rows]
        gaps = [r.pnl - f.pnl for r, f in zip(rows, fixed)]
        markouts = [r.markout_1 for r in rows]
        fills = [r.n_fills for r in rows]
        mean_pnl = sum(pnls) / len(pnls)
        mean_gap = sum(gaps) / len(gaps)
        std_gap = _std(gaps)
        std_pnl = _std(pnls)
        if label == "fixed_spread":
            verdict = "reference"
        elif mean_gap > std_gap and mean_gap > 0.0:
            verdict = "earns"
        elif mean_gap < -std_gap and mean_gap < 0.0:
            verdict = "behind"
        else:
            verdict = "does not earn"
        summary.append(
            {
                "label": label,
                "mean_pnl": mean_pnl,
                "std_pnl": std_pnl,
                "mean_gap_vs_fixed": mean_gap,
                "std_gap": std_gap,
                "mean_markout_1": sum(markouts) / len(markouts),
                "mean_fills": sum(fills) / len(fills),
                "verdict": verdict,
            }
        )
    return {
        "regime": regime,
        "book_regime": book_regime,
        "n_seeds": len(seeds),
        "n_steps": n,
        "fee": fees.fee_per_contract,
        "rebate": fees.rebate_per_contract,
        "rows": summary,
    }


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def run_adversarial(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n: int = 64,
) -> list[dict[str, object]]:
    return [run_regime(name, seeds=seeds, n=n) for name in ADVERSARIAL_REGIMES]


def _fmt(val: float) -> str:
    if abs(val) < 5e-7:
        val = 0.0
    return f"{val:.4f}"


def render_adversarial_markdown(blocks: list[dict[str, object]] | None = None) -> str:
    blocks = blocks if blocks is not None else run_adversarial()
    head = (
        "# Adversarial synthetic ablations\n\n"
        "Research laboratory output. Every price is invented "
        "(`SYNTHETIC_FIXTURE=1`). Five seeds "
        f"{DEFAULT_SEEDS}. Each path is {int(blocks[0]['n_steps'])} one-minute "
        "bars. The book is a Black–Scholes call on a GBM spot, then a regime "
        "distortion. This is not OPRA and not a walk-forward on a licensed tape.\n\n"
        "Posting rule matches the tape replay: join the touch when the model "
        "is tighter, sit behind when it is wider, never cross. The next bar's "
        "trade fills us with no queue ahead, which overstates touch fills. "
        "Crossed, locked, and stale rows are not quoted. Fee is a research "
        "charge, not a venue card. Baseline fee is 0.02 per contract. The "
        "`fees` regime uses the baseline book with fee 0.20 and rebate 0.05.\n\n"
        "`mean_gap_vs_fixed` is the paired PnL gap against fixed-spread on "
        "the same seed. A row **earns** only when that mean gap is positive "
        "and larger than the seed std of the gap. **behind** means the mean "
        "gap is negative and larger in magnitude than the seed std. "
        "Anything else **does not earn** — including a higher mean that is "
        "inside the seed noise.\n\n"
        "σ = 0.45 and κ = 1.5 are the 0.8 ablation knobs. Feature pack `off` "
        "is the identity. GEX uses the invented spot; it is not dealer gamma.\n\n"
    )
    cols = (
        "label",
        "mean_pnl",
        "std_pnl",
        "mean_gap_vs_fixed",
        "std_gap",
        "mean_markout_1",
        "mean_fills",
        "verdict",
    )
    body = ""
    for block in blocks:
        body += f"## {block['regime']}\n\n"
        body += (
            f"Book `{block['book_regime']}`, fee {float(block['fee']):.2f}, "
            f"rebate {float(block['rebate']):.2f}, seeds {block['n_seeds']}.\n\n"
        )
        body += "| " + " | ".join(cols) + " |\n"
        body += "| " + " | ".join("---" for _ in cols) + " |\n"
        for row in block["rows"]:  # type: ignore[index]
            cells = []
            for key in cols:
                val = row[key]
                cells.append(_fmt(val) if isinstance(val, float) else str(val))
            body += "| " + " | ".join(cells) + " |\n"
        earned = [r["label"] for r in block["rows"] if r["verdict"] == "earns"]  # type: ignore[index]
        behind = [r["label"] for r in block["rows"] if r["verdict"] == "behind"]  # type: ignore[index]
        idle = [r["label"] for r in block["rows"] if r["verdict"] == "does not earn"]  # type: ignore[index]
        if earned:
            body += "\nEarns versus fixed-spread: " + ", ".join(earned) + ".\n"
        else:
            body += "\nNo model earns a gap over fixed-spread outside seed noise.\n"
        if behind:
            body += "Behind fixed-spread: " + ", ".join(behind) + ".\n"
        if idle:
            body += "Does not earn (gap inside seed noise, or tied): " + ", ".join(idle) + ".\n"
        body += "\n"
    body += (
        "Join-touch is a reference posting rule. A loss there means the "
        "invented flow was toxic at the touch after the research fee. "
        "Beating it is not live alpha.\n"
    )
    return head + body
