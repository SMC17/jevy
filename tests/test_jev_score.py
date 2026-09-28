"""Brier, log loss, ECE, and the deterministic vs assisted ablation."""

from __future__ import annotations

from pathlib import Path

from jev_omm.decisions.scoreboard import (
    ScoreTuple,
    brier,
    ece,
    load_scores,
    log_loss,
    persist_scores,
    render_score_markdown,
    synthetic_jev_ablation,
)


def test_proper_scores_on_a_hand_case():
    perfect = [
        ScoreTuple("{}", "q", 1.0, 1.0, 1.0, "calm"),
        ScoreTuple("{}", "q", 0.0, 1.0, 0.0, "calm"),
    ]
    # Clip inside the scorers keeps log loss finite at 0 and 1 via _clip_prob
    # only in log_loss. Brier of exact 0/1 is 0.
    assert brier(perfect) == 0.0
    assert log_loss(perfect) < 1e-6
    assert ece(perfect) < 1e-9
    wrong = [ScoreTuple("{}", "q", 0.9, 0.9, 0.0, "toxic")]
    assert brier(wrong) == 0.81
    assert ece(wrong) == 0.9


def test_persist_roundtrip(tmp_path: Path):
    rows = [ScoreTuple('{"x":1}', "adverse_fill", 0.4, 0.6, 1.0, "calm")]
    path = tmp_path / "scores.jsonl"
    persist_scores(path, rows)
    loaded = load_scores(path)
    assert loaded == rows


def test_oracle_beats_always_quote_and_doc_matches():
    summary = synthetic_jev_ablation()
    assert summary["oracle_minus_deterministic"] > 0.0
    assert summary["brier_oracle"] < summary["brier_flat"]
    path = Path("docs/JEV_SCORE.md")
    assert path.is_file()
    assert path.read_text() == render_score_markdown()
