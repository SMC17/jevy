"""CFTC Commitments of Traders → a slow positioning feature.

Public files only. Tests read the CSV fixtures under ``jev_omm/data/fixtures``.
A live pull of the CFTC Socrata API is gated: set ``JEV_COT_NETWORK=1`` or
pass ``allow_network=True``. The weekly files are not required to run tests.

Report families (https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm):

- Legacy: non-commercial vs commercial.
  Dataset ``6dca-aqww``.
- Disaggregated (physicals): managed money vs producer/merchant.
  Dataset ``72hh-3qpy``. Swap dealers are not in the speculative net.
- Traders in Financial Futures: leveraged funds vs dealer/intermediary.
  Dataset ``gpe5-46if``.
  Notes: https://www.cftc.gov/sites/default/files/idc/groups/public/@commitmentsoftraders/documents/file/tfmexplanatorynotes.pdf

Column names below are the Socrata field names (lowercase). The annual
history zips live under https://www.cftc.gov/files/dea/history/
(``fut_fin_txt_YYYY.zip``, ``fut_disagg_txt_YYYY.zip``).

Speculative net is long minus short. Spreading is excluded.
Commercial hedge ratio is commercial short / commercial long
(producers net short when the ratio is above 1).
The z-score uses history strictly before the last print, population std.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"

SODA = {
    "legacy": "6dca-aqww",
    "disaggregated": "72hh-3qpy",
    "tff": "gpe5-46if",
}
SODA_ROOT = "https://publicreporting.cftc.gov/resource"

# (spec_long, spec_short, commercial_long, commercial_short)
_COLUMNS = {
    "tff": (
        "lev_money_positions_long",
        "lev_money_positions_short",
        "dealer_positions_long_all",
        "dealer_positions_short_all",
    ),
    "disaggregated": (
        "m_money_positions_long_all",
        "m_money_positions_short_all",
        "prod_merc_positions_long",
        "prod_merc_positions_short",
    ),
    "legacy": (
        "noncomm_positions_long_all",
        "noncomm_positions_short_all",
        "comm_positions_long_all",
        "comm_positions_short_all",
    ),
}

# Sim underlyings → report family and a name prefix on Market_and_Exchange_Names.
# Codes are the CFTC contract market codes published with those names.
UNDERLYINGS: dict[str, dict[str, str]] = {
    "SPX": {"report": "tff", "match": "E-MINI S&P 500 -", "code": "13874A"},
    "ES": {"report": "tff", "match": "E-MINI S&P 500 -", "code": "13874A"},
    "NDX": {"report": "tff", "match": "E-MINI NASDAQ-100", "code": ""},
    "NQ": {"report": "tff", "match": "E-MINI NASDAQ-100", "code": ""},
    "RUT": {"report": "tff", "match": "RUSSELL 2000", "code": ""},
    "RTY": {"report": "tff", "match": "RUSSELL 2000", "code": ""},
    "CL": {"report": "disaggregated", "match": "CRUDE OIL, LIGHT SWEET", "code": ""},
    "GC": {"report": "disaggregated", "match": "GOLD - COMMODITY EXCHANGE", "code": ""},
    "NG": {"report": "disaggregated", "match": "HENRY HUB", "code": ""},
}


@dataclass
class CotObservation:
    report_date: str
    market: str
    code: str
    report: str
    open_interest: float
    spec_long: float
    spec_short: float
    commercial_long: float
    commercial_short: float
    nonreportable_long: float
    nonreportable_short: float

    @property
    def net_spec(self) -> float:
        return self.spec_long - self.spec_short

    @property
    def spec_pct_oi(self) -> float:
        if self.open_interest <= 0.0:
            return 0.0
        return self.net_spec / self.open_interest

    @property
    def commercial_hedge_ratio(self) -> float:
        if self.commercial_long <= 0.0:
            return 0.0
        return self.commercial_short / self.commercial_long


def _f(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return 0.0
    return float(raw)


def _detect_report(row: dict[str, str]) -> str:
    if "spec_long" in row and row.get("spec_long", "") != "":
        return "normalized"
    if "lev_money_positions_long" in row:
        return "tff"
    if "m_money_positions_long_all" in row:
        return "disaggregated"
    if "noncomm_positions_long_all" in row:
        return "legacy"
    raise ValueError("unrecognized COT header")


def observation_from_row(row: dict[str, str], report: str | None = None) -> CotObservation:
    """One weekly row. Keys are matched case-insensitively."""
    lowered = {k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
    kind = report or _detect_report(lowered)
    if kind == "normalized":
        spec_l, spec_s = _f(lowered, "spec_long"), _f(lowered, "spec_short")
        comm_l, comm_s = _f(lowered, "commercial_long"), _f(lowered, "commercial_short")
        family = lowered.get("report", "normalized")
    else:
        spec_l_k, spec_s_k, comm_l_k, comm_s_k = _COLUMNS[kind]
        spec_l, spec_s = _f(lowered, spec_l_k), _f(lowered, spec_s_k)
        comm_l, comm_s = _f(lowered, comm_l_k), _f(lowered, comm_s_k)
        family = kind
    date = lowered.get("report_date_as_yyyy_mm_dd") or lowered.get("report_date") or ""
    market = lowered.get("market_and_exchange_names") or lowered.get("market") or ""
    code = lowered.get("cftc_contract_market_code") or lowered.get("cftc_code") or ""
    return CotObservation(
        report_date=date[:10],
        market=market,
        code=code,
        report=family,
        open_interest=_f(lowered, "open_interest_all") or _f(lowered, "open_interest"),
        spec_long=spec_l,
        spec_short=spec_s,
        commercial_long=comm_l,
        commercial_short=comm_s,
        nonreportable_long=_f(lowered, "nonrept_positions_long_all")
        or _f(lowered, "nonreportable_long"),
        nonreportable_short=_f(lowered, "nonrept_positions_short_all")
        or _f(lowered, "nonreportable_short"),
    )


def load_csv(path: str | Path) -> list[CotObservation]:
    with open(path, newline="") as handle:
        return [observation_from_row(row) for row in csv.DictReader(handle)]


def load_fixture(name: str) -> list[CotObservation]:
    return load_csv(FIXTURE_DIR / name)


def trailing_z(nets: list[float]) -> float:
    """Z-score of the last net versus the preceding sample. Short samples → 0."""
    if len(nets) < 3:
        return 0.0
    hist = nets[:-1]
    mu = sum(hist) / len(hist)
    var = sum((x - mu) ** 2 for x in hist) / len(hist)
    std = var**0.5
    if std < 1e-12:
        return 0.0
    return (nets[-1] - mu) / std


def week_over_week(nets: list[float]) -> float:
    if len(nets) < 2:
        return 0.0
    return nets[-1] - nets[-2]


def _prefer_row(rows: list[CotObservation], spec: dict[str, str]) -> CotObservation | None:
    match = spec["match"].upper()
    code = spec.get("code", "")
    hits = [row for row in rows if match in row.market.upper()]
    if code:
        coded = [row for row in hits if row.code == code]
        if coded:
            hits = coded
    if not hits:
        return None
    # Prefer the non-micro contract when several names share a stem.
    plain = [row for row in hits if "MICRO" not in row.market.upper()]
    pool = plain or hits
    return sorted(pool, key=lambda row: row.report_date)[-1]


def rows_for_underlying(symbol: str, rows: list[CotObservation]) -> list[CotObservation]:
    """Every weekly print for the contract ``map_underlying`` picks, oldest first."""
    chosen = map_underlying(symbol, rows)
    if chosen is None:
        return []
    series = [
        row
        for row in rows
        if row.market == chosen.market and row.code == chosen.code and row.report == chosen.report
    ]
    return sorted(series, key=lambda row: row.report_date)


def map_underlying(symbol: str, rows: list[CotObservation]) -> CotObservation | None:
    """Pick the futures COT row for a sim underlying (SPX, CL, …)."""
    spec = UNDERLYINGS.get(symbol.upper())
    if spec is None:
        return None
    family_rows = [row for row in rows if row.report == spec["report"] or row.report == "normalized"]
    # Normalized fixtures carry an explicit report label equal to the family.
    found = _prefer_row(family_rows, spec)
    if found is not None:
        return found
    return _prefer_row(rows, spec)


def fetch_cot_rows(
    report: str,
    where: str,
    limit: int = 20,
    *,
    allow_network: bool = False,
) -> list[dict]:
    """GET one Socrata COT dataset. Refuses unless the gate is open.

    ``where`` is a SoQL predicate, for example
    ``upper(market_and_exchange_names) like '%E-MINI S&P 500%'``.
    """
    if not allow_network and os.environ.get("JEV_COT_NETWORK") != "1":
        raise RuntimeError(
            "COT network fetch is gated. Set JEV_COT_NETWORK=1 or pass "
            "allow_network=True. Tests use jev_omm/data/fixtures."
        )
    dataset = SODA[report]
    url = (
        f"{SODA_ROOT}/{dataset}.json?$limit={int(limit)}&$where="
        + quote(where, safe="")
    )
    import json

    with urlopen(url, timeout=20) as resp:  # noqa: S310 — gated public CFTC endpoint
        payload = json.loads(resp.read().decode())
    if not isinstance(payload, list):
        raise RuntimeError("unexpected CFTC payload")
    return payload
