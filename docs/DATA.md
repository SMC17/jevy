# Data

Simulation and paper only. This tree does not ship a licensed OPRA or NBBO
history, and it does not open a live session.

## What you can run today

| Path | What it is | In git? |
| --- | --- | --- |
| `load_tape("synthetic")` | GBM spot path inside the 0.8 harness | generated |
| `jev_omm/data/fixtures/tape_synthetic.csv` | Schema-compatible option NBBO, `synthetic_fixture=1`, invented prices, symbol `SYNTH` | yes |
| Databento public sample | Short vendor preview, no account | raw file no, aggregates in `docs/ablation_real_or_fixture.md` |
| `DATABENTO_API_KEY` historical slice | Bounded `timeseries.get_range` | raw file no |

The fixture is not an OPRA print. Do not cite its prices as market data.

## Public sample (no account)

Databento documents sample flat files that download without a signup. The
endpoint the site uses is:

```text
GET https://api.databento.com/v0/dataset/sample
    ?dataset=OPRA.PILLAR&encoding=csv&schema=cbbo-1m&stype_in=raw_symbol
```

Checked 2026-09-28 from this environment:

- `OPRA.PILLAR` / `cbbo-1s` returned `[]`.
- `OPRA.PILLAR` / `cbbo-1m` returned a header plus 20 CSV lines: TSLA call
  `TSLA  230901C00250000`, expiry 2023-09-01, strike 250, about 13:31Z–13:50Z
  on 2023-08-28. Consolidated bid, ask, sizes, and last trade.
- `OPRA.PILLAR` / `tcbbo` returned 20 trade-plus-NBBO lines for the same
  contract, including locked markets (bid equal to ask).
- Equity previews exist (`EQUS.MINI`, `XNAS.ITCH`, `DBEQ.BASIC`, symbol MSFT
  on 2023-09-29). They were not joined to the TSLA option. Different product,
  different day.

```bash
python scripts/fetch_databento_sample.py
python scripts/fetch_databento_sample.py --schema tcbbo
```

Files land in `data/local/`, which is gitignored. The walk-forward table in
`docs/ablation_real_or_fixture.md` records the sha256 of the JSON body that
was replayed. If a fresh download hashes differently, the table is stale.

Full OPRA history is licensed. A 20-row preview is not a trading day and is
not a backtest of a market-making strategy. Cost: the preview is the free
sample; a real day is on Databento's usage or subscription price, not free.
See their pricing page. This repo does not spend that money.

### Redistribution

The preview is served so a visitor can look at the shape of the feed. OPRA's
venue license is a separate document, and it is not obvious that a git
checkout may republish the rows. Raw bytes stay out of git. Aggregates
(row counts, book-label counts, replay PnL, sha256) are what this repo keeps.

### Schema map

Databento columns used, when present:

| Vendor | Tape |
| --- | --- |
| `ts_event` (else `ts_recv`) | `time_seconds` from the first stamp |
| `bid_px_00`, `ask_px_00` | `bid`, `ask` |
| `bid_sz_00`, `ask_sz_00` | `bid_sz`, `ask_sz` |
| `price`, `size` | `trade_px`, `trade_sz` |
| `symbol` or `raw_symbol` | `symbol`, plus OSI expiry / strike / right when the symbol parses |
| `publisher_id` | `exchange` (publisher id, not a fee schedule) |
| `sequence` | `seq` |

`spot` is not in the OPRA preview. The loader leaves it NaN. It does not
invent an underlying print. GEX on that tape is the identity.

OSI example that is in the preview, not a reconstructed price: the symbol
`TSLA  230901C00250000` is root TSLA, 2023-09-01, call, strike 250.
The definition preview (2 rows) names the same instrument and expiration.
Prices in this file are not copied here.

### Crossed, locked, stale

- **Crossed:** bid > ask. Not quoted.
- **Locked:** bid = ask, both positive. Not quoted. The tcbbo preview has locked rows.
- **One-sided / missing:** a zero or blank price. Not quoted.
- **Stale:** the same two-sided book repeats for three or more rows while a trade prints outside it, or the clock jumps by more than five minutes. Later repeats are not quoted.

Rows are kept. The replay pulls.

### Fees

The tape replay charges `TapeFeeSchedule`. The checked-in table uses 0.05
per contract and rebate 0. That is not an OPRA fee card, not a maker-taker
schedule, and not a pass-through of `publisher_id`. Change it in the caller
if you want a different research assumption. The adversarial `fees` regime
uses 0.20 with a 0.05 rebate on an invented book.

## Local CSV and Parquet

```python
from jev_omm.research.walkforward import load_tape
tape = load_tape("path/to/day.csv")      # or .parquet
```

Required shape is either the legacy six columns
`time_seconds,spot,bid,ask,bid_sz,ask_sz`, or `time_seconds` plus `bid` and
`ask`. Optional: `trade_px,trade_sz,symbol,expiry,strike,right,exchange,seq,synthetic_fixture`.
A Databento CSV with `bid_px_00` / `ask_px_00` is accepted by the same reader.

Parquet goes through pandas and needs `pyarrow` (`pip install pyarrow`).
CSV does not. There is no new warehouse engine.

Live URLs are refused (`opra://`, `live.databento.com`, `hist.databento.com`
as a tape path, `ws://`, `wss://`). A local filename that contains the word
`databento` is a file, not a session.

## Historical API (key in the environment only)

```bash
export DATABENTO_API_KEY=...          # never commit this
export DATABENTO_MAX_COST_USD=0       # default. A key alone does not spend.
```

`jev_omm.research.databento_hist.fetch_historical_slice` calls
`https://hist.databento.com/v0/metadata.get_cost` and then, only if the
estimate is within the cap, `timeseries.get_range` with an explicit `start`
and `end` and `encoding=csv`. The live gateway host is refused. The response
is written under `data/local/` and mapped with the same loader.

Example parameters for a short OPRA minute slice, once you accept the cost
by raising `DATABENTO_MAX_COST_USD`:

```text
dataset=OPRA.PILLAR
schema=cbbo-1m
symbols=<OSI symbol>
stype_in=raw_symbol
start=2023-08-28T13:30
end=2023-08-28T14:00
```

This environment had no `DATABENTO_API_KEY`, so no paid slice was requested.
Do not point this client at `live.databento.com`.

## Walk-forward

```python
from jev_omm.research.tape import load_local
from jev_omm.research.tape_walk import walk_forward
print(walk_forward(load_local("jev_omm/data/fixtures/tape_synthetic.csv")))
```

Train window fits a touch hazard λ(δ) = A exp(−k δ). k replaces the quoter
kappa only when that fit is identified (posted distances have to move). A
is events per second and is not written into `QuoterConfig.A`, which is
Guéant's closed-form intensity (arXiv 1105.3115). The test window replays
fixed, Avellaneda–Stoikov, Guéant asymptotic, option-vega, flow, GEX, and
state. `join_touch` is a reference posting rule, not a new model.

Results: `docs/ablation_real_or_fixture.md`. Multi-seed invented regimes:
`docs/ablation_adversarial.md`.
