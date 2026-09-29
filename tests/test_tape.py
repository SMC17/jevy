"""Local tape loader, book labels, and the Databento batch gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from jev_omm.research.databento_hist import DatabentoError, _get, fetch_historical_slice
from jev_omm.research.features import pack_adjustment
from jev_omm.research.synthetic_tape import synthetic_rows
from jev_omm.research.tape import (
    count_states,
    load_local,
    parse_osi,
    refuse_live,
    tape_from_rows,
    write_csv,
)
from jev_omm.research.walkforward import TAPE_COLUMNS, load_tape


def test_osi_parses_a_padded_symbol_and_rejects_garbage():
    assert parse_osi("TSLA  230901C00250000") == ("TSLA", "2023-09-01", "C", 250.0)
    assert parse_osi("not a symbol") is None


def test_live_urls_refused_and_local_name_is_a_file(tmp_path: Path):
    with pytest.raises(RuntimeError, match="OPRA"):
        refuse_live("opra://SPX")
    with pytest.raises(RuntimeError, match="streaming"):
        load_tape("https://live.databento.com/v0/stream")
    with pytest.raises(FileNotFoundError, match="not bundled"):
        load_tape(str(tmp_path / "databento_local.csv"))


def test_legacy_csv_and_checked_in_fixture_are_labeled():
    tape = load_tape("synthetic", n_steps=4, seed=2)
    assert tape.source == "synthetic"
    fixture = load_local("jev_omm/data/fixtures/tape_synthetic.csv")
    assert fixture.synthetic_fixture is True
    assert fixture.source.startswith("csv:")
    counts = count_states(fixture)
    assert counts["crossed"] >= 1
    assert counts["locked"] >= 1
    assert counts["stale"] >= 1
    # The generator and the checked-in file are the same invented path.
    fresh = synthetic_rows(n=96, seed=11, regime="fixture")
    assert len(fresh) == len(fixture)
    regenerated = Path("jev_omm/data/fixtures/tape_synthetic.csv").read_text()
    assert "synthetic_fixture" in regenerated
    assert regenerated.splitlines()[1].endswith(",1")


def test_crossed_locked_and_missing_labels():
    rows = [
        {
            "time_seconds": "0",
            "bid": "1.10",
            "ask": "1.00",
            "trade_px": "1.05",
            "symbol": "TEST  240102C00100000",
            "synthetic_fixture": "1",
        },
        {
            "time_seconds": "60",
            "bid": "1.20",
            "ask": "1.20",
            "trade_px": "1.20",
            "symbol": "TEST  240102C00100000",
            "synthetic_fixture": "1",
        },
        {
            "time_seconds": "120",
            "bid": "",
            "ask": "",
            "trade_px": "",
            "symbol": "TEST  240102C00100000",
            "synthetic_fixture": "1",
        },
    ]
    tape = tape_from_rows(rows, source="unit")
    assert tape.book_state == ["crossed", "locked", "missing"]
    assert tape.synthetic_fixture is True
    assert tape.expiry[0] == "2024-01-02"
    assert tape.strike[0] == pytest.approx(100.0)
    assert tape.spot[0] != tape.spot[0]  # NaN: no invented underlying


def test_gex_channel_is_identity_without_permission():
    flow = pack_adjustment("flow", ret_bps=40.0, spot=110.0, strike=100.0, mid=3.0)
    dropped = pack_adjustment(
        "flow_gex", ret_bps=40.0, spot=110.0, strike=100.0, mid=3.0, allow_gex=False
    )
    assert flow is not None and dropped is not None
    assert dropped.spread_mult == flow.spread_mult
    assert dropped.size_mult == flow.size_mult


def test_roundtrip_csv_columns(tmp_path: Path):
    path = tmp_path / "book.csv"
    write_csv(
        path,
        [
            {
                "time_seconds": "60",
                "spot": "100",
                "bid": "1.0",
                "ask": "1.2",
                "bid_sz": "2",
                "ask_sz": "2",
            }
        ],
    )
    text_header = path.read_text().splitlines()[0]
    assert text_header.split(",")[:6] == list(TAPE_COLUMNS)
    loaded = load_tape(str(path))
    assert loaded.spot[0] == pytest.approx(100.0)
    assert loaded.book_state == ["two_sided"]


def test_keyed_historical_client_is_gated_off(monkeypatch, tmp_path: Path):
    """A key in the environment must not place an HTTP call. The request path is untested."""
    monkeypatch.setenv("DATABENTO_API_KEY", "db-not-a-real-key")
    monkeypatch.delenv("DATABENTO_HISTORICAL", raising=False)

    def _boom(url: str, *, auth_key: str | None = None, timeout: float = 60.0) -> bytes:
        raise AssertionError(f"keyed client called the network: {url}")

    monkeypatch.setattr("jev_omm.research.databento_hist._get", _boom)
    with pytest.raises(DatabentoError, match="gated off"):
        fetch_historical_slice(
            dataset="OPRA.PILLAR",
            schema="cbbo-1m",
            symbols="TEST",
            start="2024-01-02T15:00",
            end="2024-01-02T15:05",
            dest_dir=tmp_path,
        )
    # ``_get`` here is the function object imported above, not the patched
    # module attribute. Live hosts are rejected before any socket open.
    with pytest.raises(DatabentoError, match="live"):
        _get("https://live.databento.com/v0/timeseries.get_range")


def test_parquet_roundtrip(tmp_path: Path):
    pytest.importorskip("pyarrow")
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "time_seconds": 60.0,
                "spot": 100.0,
                "bid": 1.0,
                "ask": 1.2,
                "bid_sz": 2.0,
                "ask_sz": 2.0,
                "synthetic_fixture": "1",
            }
        ]
    )
    path = tmp_path / "book.parquet"
    frame.to_parquet(path, index=False)
    loaded = load_local(path)
    assert loaded.source.startswith("parquet:")
    assert loaded.bid[0] == pytest.approx(1.0)
    assert loaded.synthetic_fixture is True
