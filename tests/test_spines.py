"""Adversarial tables, tape report, and the sim-loop scoreboard section."""

from __future__ import annotations

from pathlib import Path

from jev_omm.research.adversarial import render_adversarial_markdown
from jev_omm.research.real_report import render_real_or_fixture_markdown
from jev_omm.research.walkforward import render_ablation_markdown


def test_synthetic_ablation_doc_matches_renderer():
    path = Path("docs/ablation_synthetic.md")
    assert path.read_text() == render_ablation_markdown()


def test_adversarial_doc_is_multi_seed_and_honest():
    text = render_adversarial_markdown()
    assert Path("docs/ablation_adversarial.md").read_text() == text
    assert "mean ±" in text or "mean_pnl" in text
    assert "does not earn" in text or "behind" in text
    assert "SYNTHETIC" in text
    assert "OPRA" in text  # named as what this file is not


def test_real_or_fixture_doc_names_both_tapes():
    text = render_real_or_fixture_markdown()
    assert Path("docs/ablation_real_or_fixture.md").read_text() == text
    assert "synthetic_fixture" in text or "Synthetic fixture" in text
    assert "OPRA.PILLAR" in text
    assert "not an OPRA print" in text or "not OPRA" in text
    assert "sha256" in text
    assert "Invented" in text or "invented" in text
