"""Paper demo of the five training cases plus a small option-vega quote.

    python -m jev_omm.demo_training
"""

from __future__ import annotations

from jev_omm import __version__
from jev_omm.hedge.delta import spot_vol_hedge_qty
from jev_omm.quoter.option_mm import research_toy, solve_and_quote
from jev_omm.training.cases import CASES, leaderboard, run_pair
from jev_omm.training.replay import replay_scores, write_jsonl


def main() -> None:
    print(f"jevy training desk {__version__} — simulation / paper only")
    print("Decision snapshots use DeterministicFallbackClient. No live key. No orders from the model.\n")
    runs = []
    for name in CASES:
        naive, desk = run_pair(name)
        runs.extend((naive, desk))
        print(f"=== {name} ===")
        print(f"  role: {desk.spec.role}")
        print(f"  lesson: {desk.spec.lesson}")
        print(
            f"  naive  pnl={naive.score.absolute_pnl:.4f}  risk_adj={naive.score.risk_adjusted:.4f}"
            f"  |beta|={naive.score.mean_abs_beta:.4f}  |q|={naive.score.mean_abs_inventory:.4f}"
        )
        print(
            f"  desk   pnl={desk.score.absolute_pnl:.4f}  risk_adj={desk.score.risk_adjusted:.4f}"
            f"  relative={desk.score.relative_score:.4f}"
            f"  |beta|={desk.score.mean_abs_beta:.4f}  |q|={desk.score.mean_abs_inventory:.4f}"
        )
        snap = next((e for e in desk.events if e.get("type") == "DecisionSnapshot"), None)
        if snap:
            print(
                f"  decision source={snap['source']} size_tier={snap['size_tier']}"
                f"  policy_size_mult={snap['size_mult']}  engine_size={snap['engine_size']}"
            )
        print()
    board = leaderboard([r for r in runs if r.strategy == "desk"])
    print("desk leaderboard (risk-adjusted):")
    for i, r in enumerate(board, 1):
        print(f"  {i}. {r.spec.name:24s} {r.score.risk_adjusted:.4f}")
    path = write_jsonl("training_cases.jsonl", runs)
    checked = replay_scores(path)
    print(f"\nwrote {path}  replayed {len(checked)} scores")

    cfg = research_toy()
    flat = solve_and_quote(cfg, 10.0, 0.0, 5.0)
    long = solve_and_quote(cfg, 10.0, 20.0, 5.0)
    print("\n=== option vega MM (research toy) ===")
    print(f"  flat  δb={flat.delta_b:.4f} δa={flat.delta_a:.4f} reservation={flat.reservation:.4f}")
    print(f"  long  δb={long.delta_b:.4f} δa={long.delta_a:.4f} reservation={long.reservation:.4f}")
    q_s = spot_vol_hedge_qty(0.5, rho=-0.5, xi=0.2, portfolio_vega=10.0, variance=0.04, spot=100.0)
    print(f"  spot-vol hedge qS*={q_s:.4f} (Baldacci appendix, toy inputs)")


if __name__ == "__main__":
    main()
