"""Paper multi-product sleeve desk. Synthetic fixtures only.

    python -m jev_omm.demo_surface_desk
"""

from __future__ import annotations

from jev_omm import __version__
from jev_omm.desk.harness import DeskConfig, run_desk


def main() -> None:
    print(f"jevy surface desk {__version__} — simulation / paper only")
    print("synthetic_fixture=1. No live tape. No orders.")
    run = run_desk(DeskConfig())
    print(run.markdown)
    board = run.scoreboard
    print(
        f"fits={run.n_fits} suspect_slices={run.suspect_fits} "
        f"products={len(board.products)} sleeves={len(board.rows)}"
    )


if __name__ == "__main__":
    main()
