"""Citadel-style paper training desk. Cases, scores, JSONL replay.

The decision client returns Choice / Score / Noul. Case code places the sim trades.
"""

from jev_omm.training.cases import CASES, leaderboard, run_case, run_pair
from jev_omm.training.replay import all_runs, read_jsonl, replay_scores, write_jsonl
from jev_omm.training.scoring import SPECS, CaseScore, lcg_next, score_path

__all__ = [
    "CASES",
    "SPECS",
    "CaseScore",
    "all_runs",
    "lcg_next",
    "leaderboard",
    "read_jsonl",
    "replay_scores",
    "run_case",
    "run_pair",
    "score_path",
    "write_jsonl",
]
