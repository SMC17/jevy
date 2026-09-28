"""Databento batch historical client and the public sample endpoint.

No live gateway. The API key is read from ``DATABENTO_API_KEY`` and is never
written to disk by this module. Historical pulls are refused when the
estimated cost is above ``DATABENTO_MAX_COST_USD`` (default 0, so a key
alone does not spend). Public sample files need no key and no account;
they are a vendor preview, not a trading day.

Raw bytes go to ``data/local/`` which is gitignored. Do not commit them.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from jev_omm.research.tape import Tape, tape_from_rows

HIST_HOST = "https://hist.databento.com/v0/"
SAMPLE_URL = "https://api.databento.com/v0/dataset/sample"
LIVE_HOSTS = ("live.databento.com", "gateway.databento.com")


class DatabentoError(RuntimeError):
    pass


def _reject_live(url: str) -> None:
    low = url.lower()
    if any(host in low for host in LIVE_HOSTS) or low.startswith(("ws://", "wss://")):
        raise DatabentoError(
            f"Refusing live streaming URL {url}. Batch historical and the "
            "public sample endpoint only."
        )


def _get(url: str, *, auth_key: str | None = None, timeout: float = 60.0) -> bytes:
    _reject_live(url)
    headers = {"user-agent": "jevy-research/0.9"}
    if auth_key:
        token = base64.b64encode(f"{auth_key}:".encode()).decode()
        headers["authorization"] = f"Basic {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise DatabentoError(f"HTTP {exc.code} from {url}: {body}") from exc


def fetch_public_sample(
    *,
    dataset: str = "OPRA.PILLAR",
    schema: str = "cbbo-1m",
    encoding: str = "csv",
    stype_in: str = "raw_symbol",
    symbol: str | None = None,
    dest_dir: str | Path = "data/local",
) -> tuple[Path, Tape, bytes]:
    """Download one no-account sample preview and map it to the tape schema.

    The response is a JSON list of CSV lines (header plus a short preview).
    It is not a full session. ``cbbo-1s`` on OPRA.PILLAR is an empty preview
    as of the 2026-09-28 check; ``cbbo-1m`` is the option NBBO sample that
    returned rows.
    """
    if encoding not in {"csv", "json"}:
        raise DatabentoError("public sample loader accepts encoding csv or json only")
    query = {
        "dataset": dataset,
        "encoding": encoding,
        "schema": schema,
        "stype_in": stype_in,
    }
    if symbol:
        query["symbol"] = symbol
    url = SAMPLE_URL + "?" + urllib.parse.urlencode(query)
    raw = _get(url)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list) or not payload:
        raise DatabentoError(
            f"Public sample {dataset} schema={schema} returned no rows. "
            "That schema may be an empty preview. Try cbbo-1m, tcbbo, or trades."
        )
    header, *body = payload
    if not isinstance(header, str) or "," not in header:
        raise DatabentoError("public sample payload is not a CSV-line list")
    names = header.split(",")
    rows: list[dict[str, str]] = []
    for line in body:
        parts = str(line).split(",")
        if len(parts) < len(names):
            parts = parts + [""] * (len(names) - len(parts))
        rows.append(dict(zip(names, parts)))
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    safe_schema = schema.replace("/", "-")
    path = dest / f"databento_{dataset.replace('.', '_')}_{safe_schema}.json"
    path.write_bytes(raw)
    tape = tape_from_rows(rows, source=f"databento-sample:{dataset}:{schema}")
    tape.notes = (
        "Public Databento sample preview (no account). Not a full session. "
        "Underlying spot is not in the OPRA preview and is left missing."
    )
    return path, tape, raw


def api_key() -> str:
    return os.environ.get("DATABENTO_API_KEY", "").strip()


def max_cost_usd() -> float:
    raw = os.environ.get("DATABENTO_MAX_COST_USD", "0").strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise DatabentoError("DATABENTO_MAX_COST_USD must be a number") from exc


def _hist_url(method: str, params: dict[str, str]) -> str:
    return HIST_HOST + method + "?" + urllib.parse.urlencode(params)


def estimate_cost(params: dict[str, str], *, key: str) -> float:
    raw = _get(_hist_url("metadata.get_cost", params), auth_key=key)
    text = raw.decode("utf-8").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatabentoError(f"cost response was not JSON: {text[:200]}") from exc
    if isinstance(parsed, (int, float)):
        return float(parsed)
    if isinstance(parsed, dict):
        for name in ("cost", "usd", "estimated_cost"):
            if name in parsed:
                return float(parsed[name])
    raise DatabentoError(f"unrecognized cost payload: {text[:200]}")


def fetch_historical_slice(
    *,
    dataset: str,
    schema: str,
    symbols: str,
    start: str,
    end: str,
    stype_in: str = "raw_symbol",
    dest_dir: str | Path = "data/local",
) -> tuple[Path, Tape]:
    """Bounded historical CSV via ``timeseries.get_range``. Not a live stream.

    Requires ``DATABENTO_API_KEY``. Refuses the call when
    ``metadata.get_cost`` is above ``DATABENTO_MAX_COST_USD`` (default 0).
    """
    key = api_key()
    if not key:
        raise DatabentoError(
            "DATABENTO_API_KEY is not set. Refusing to call the historical API. "
            "The public sample endpoint does not need a key."
        )
    if not start or not end:
        raise DatabentoError("historical slice needs an explicit start and end")
    low = f"{dataset} {schema}".lower()
    if "live" in low:
        raise DatabentoError("live schemas and datasets are refused")
    params = {
        "dataset": dataset,
        "schema": schema,
        "symbols": symbols,
        "stype_in": stype_in,
        "start": start,
        "end": end,
        "encoding": "csv",
    }
    cost = estimate_cost(params, key=key)
    cap = max_cost_usd()
    if cost > cap + 1e-12:
        raise DatabentoError(
            f"Estimated cost ${cost:.4f} exceeds DATABENTO_MAX_COST_USD={cap}. "
            "Raise the cap explicitly to spend. This process will not."
        )
    raw = _get(_hist_url("timeseries.get_range", params), auth_key=key)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"databento_hist_{dataset.replace('.', '_')}_{schema}.csv"
    path.write_bytes(raw)
    text = raw.decode("utf-8")
    import csv
    from io import StringIO

    rows = [dict(r) for r in csv.DictReader(StringIO(text))]
    tape = tape_from_rows(rows, source=f"databento-hist:{dataset}:{schema}")
    tape.notes = f"Historical batch slice. Estimated cost ${cost:.4f}. Not live."
    return path, tape
