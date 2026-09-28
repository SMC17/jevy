"""Synthetic walk-forward, ablation table, and the OPRA/NBBO fixture slot."""

from __future__ import annotations

from pathlib import Path

import pytest

from jev_omm.research.features import pack_adjustment
from jev_omm.research.walkforward import TAPE_COLUMNS, load_tape, render_ablation_markdown, run_ablation


def test_feature_pack_off_is_identity():
    assert pack_adjustment("off", ret_bps=40.0, spot=110.0, strike=100.0, mid=3.0) is None


def test_feature_pack_on_moves_the_spread():
    adj = pack_adjustment("flow", ret_bps=40.0, spot=100.0, strike=100.0, mid=3.0)
    assert adj is not None
    assert adj.spread_mult > 1.0


def test_live_vendor_tape_is_refused_and_csv_slot_is_explicit(tmp_path: Path):
    tape = load_tape("synthetic", n_steps=8, seed=1)
    assert tape.source == "synthetic"
    assert len(tape.spot) == 8
    with pytest.raises(RuntimeError, match="OPRA"):
        load_tape("opra://SPX")
    with pytest.raises(FileNotFoundError, match="not bundled"):
        load_tape(str(tmp_path / "missing.csv"))
    good = tmp_path / "book.csv"
    good.write_text(
        ",".join(TAPE_COLUMNS) + "\n"
        "60,100,1.0,1.2,2,2\n"
        "120,101,1.1,1.3,2,2\n"
    )
    loaded = load_tape(str(good))
    assert loaded.source.startswith("csv:")
    assert loaded.spot[1] == pytest.approx(101.0)


def test_ablation_table_matches_checked_in_doc():
    text = render_ablation_markdown()
    path = Path("docs/ablation_synthetic.md")
    assert path.is_file(), "checked-in ablation table is missing"
    assert path.read_text() == text
    rows = run_ablation()
    labels = [r["label"] for r in rows]
    assert labels[:4] == [
        "fixed_spread",
        "avellaneda_stoikov",
        "gueant_asymptotic",
        "option_vega",
    ]
    assert "as_flow_gex_state" in labels
    for row in rows:
        assert row["n_fills"] >= 0
        assert 0.0 <= row["quote_uptime"] <= 1.0 + 1e-9
