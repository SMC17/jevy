"""Paper multi-product sleeve desk. Synthetic fixtures only.

    python -m jev_omm.demo_surface_desk
"""

from __future__ import annotations

from jev_omm import __version__
from jev_omm.desk.harness import EVAL_REGIMES, EVAL_SEEDS, DeskConfig, multi_seed_corr, run_desk


def main() -> None:
    print(f"jevy surface desk {__version__} — simulation / paper only")
    print("synthetic_fixture=1. No live tape. No orders.")
    run = run_desk(DeskConfig())
    print(run.markdown)
    board = run.scoreboard
    print(
        f"fits={run.n_fits} suspect_slices={run.suspect_fits} "
        f"products={len(board.products)} sleeves={len(board.rows)} "
        f"pre_gate_max_|ρ|={board.pre_gate_max_abs_rho:.4f} "
        f"post_gate_max_|ρ|={board.max_abs_rho:.4f} "
        f"raw_resid_sharpe={board.desk_sharpe_raw:.4f} "
        f"penalized_resid_sharpe={board.desk_sharpe_residual:.4f} "
        f"lob_fills={board.desk_lob_fills:.1f} "
        f"fill_pnl={board.desk_fill_pnl:.4f} "
        f"adverse_markout={board.desk_adverse_markout:.4f}"
    )
    report = multi_seed_corr()
    print(
        f"multi-seed seeds={list(report.seeds)} regimes={list(report.regimes)} "
        f"steps={report.n_steps}"
    )
    print(
        "mean ρ skew/fly "
        f"{report.headline['skew_residual/fly_butterfly']:+.4f}  "
        "mm/vrp "
        f"{report.headline['mm_spread/vrp_varswap']:+.4f}  "
        f"max mean |ρ| {report.max_mean_abs:.4f} "
        f"({report.worst_pair[0]}/{report.worst_pair[1]})"
    )
    _ = (EVAL_SEEDS, EVAL_REGIMES)


if __name__ == "__main__":
    main()
