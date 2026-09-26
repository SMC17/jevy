# Performance — Zig hot path

**Measured:** 2026-09-25 (America/New_York)  
**Host:** Linux x86_64 agent box  
**Toolchain:** Zig 0.16.0 (`/home/box/bin/zig`)  
**Command:** `cd zig && zig build bench -Doptimize=ReleaseFast`  
**Clock:** `CLOCK_MONOTONIC`  
**Note:** Inputs varied each iteration (volatile) to defeat dead-code elimination.  
**Version:** `0.3.0-zig-akuna-depth`

## Summary

| Benchmark | Per-iter | Throughput |
| --- | --- | --- |
| `bs_greeks/price_and_greeks_atm_call` | **~89 ns** | **~11.2 Melem/s** |
| `bs_greeks/price_and_greeks_batch_64` | **~5.7 µs** / 64 | ~89 ns/option effective |
| `quote_cycle/bs_risk_as_quote` | **~118 ns** | **~8.5 Melem/s** |
| `quote_cycle/quote_cycle_x256` | **~29 µs** / 256 | ~113 ns/cycle effective |
| `surface/sabr_iv_single` | **~30 ns** | **~33 Melem/s** |
| `surface/sabr_iv_batch_64` | **~1.9 µs** / 64 | ~30 ns/strike effective |
| `gueant/make_quote` | **~130 ns** | **~7.7 Melem/s** |
| `multi_strike/quote_strip_5` | **~1.0 µs** / 5 strikes | **~1.0 M strips/s** |
| `parity/box_spread` | **~15 ns** | **~67 Melem/s** |
| `combos/straddle_theo` | **~144 ns** | **~7.0 Melem/s** |
| `hedge/propose_apply_greek_pnl` | **~3.1 ns** | **~320 Melem/s** |
| `scenario/matrix_7x5` | **~311 ns** | **~3.2 Melem/s** |

### Raw output (representative run)

```
Jev Options MM — Zig benches (ReleaseFast)
host=linux  CLOCK_MONOTONIC  inputs varied to defeat DCE

bs_greeks/price_and_greeks_atm_call
  iters=2000000  total_ns=178779067  per=89.390 ns  thrpt=11.187 Melem/s
bs_greeks/price_and_greeks_batch_64
  iters=50000  total_ns=283053114  per=5661.062 ns  thrpt=0.177 Melem/s
quote_cycle/bs_risk_as_quote
  iters=1000000  total_ns=117960213  per=117.960 ns  thrpt=8.477 Melem/s
quote_cycle/quote_cycle_x256
  iters=10000  total_ns=290031102  per=29003.110 ns  thrpt=0.034 Melem/s
surface/sabr_iv_single
  iters=2000000  total_ns=60682481  per=30.341 ns  thrpt=32.958 Melem/s
surface/sabr_iv_batch_64
  iters=50000  total_ns=96405789  per=1928.116 ns  thrpt=0.519 Melem/s
gueant/make_quote
  iters=1000000  total_ns=129978931  per=129.979 ns  thrpt=7.694 Melem/s
multi_strike/quote_strip_5
  iters=100000  total_ns=103637118  per=1036.371 ns  thrpt=0.965 Melem/s
parity/box_spread
  iters=1000000  total_ns=14958201  per=14.958 ns  thrpt=66.853 Melem/s
combos/straddle_theo
  iters=1000000  total_ns=143859305  per=143.859 ns  thrpt=6.951 Melem/s
hedge/propose_apply_greek_pnl
  iters=1000000  total_ns=3127197  per=3.127 ns  thrpt=319.775 Melem/s
scenario/matrix_7x5
  iters=200000  total_ns=62229673  per=311.148 ns  thrpt=3.214 Melem/s
```

## What is measured

- **`price_and_greeks_atm_call`:** European call BS price + analytic δ/Γ/ν/θ.
- **`price_and_greeks_batch_64`:** 64 spots across a strike grid.
- **`bs_risk_as_quote`:** BS → marked PnL → hard risk → A–S quote (no RNG fills).
- **`quote_cycle_x256`:** 256 varied spot/inventory cycles.
- **`sabr_iv_single`:** Hagan SABR-lite Black IV for one (f,K,T).
- **`sabr_iv_batch_64`:** 64 strikes across a wing grid.
- **`gueant/make_quote`:** Guéant asymptotic reservation + half-spread (arXiv 1105.3115).
- **`multi_strike/quote_strip_5`:** SABR+BS+Guéant quote for a 5-strike desk strip with portfolio-Δ tilt.
- **`parity/box_spread`:** four-leg box theo + executable edges + implied rate.
- **`combos/straddle_theo`:** ATM straddle package theo + greeks from BS legs.
- **`hedge/propose_apply_greek_pnl`:** banded hedge propose/apply + greek PnL step.
- **`scenario/matrix_7x5`:** default spot×IV Taylor scenario grid.

## Reproduce

```bash
export PATH="/home/box/bin:$PATH"
cd /workspace/jev-options-mm/zig
zig build bench -Doptimize=ReleaseFast
```

## Architecture note

Python = research glue (System One / TypeSafe decisions, config, demos).  
Zig = hot path (pricing, parity/combos, quoter AS/Guéant, multi-strike, hedge, scenario, toxicity, risk, fills, SABR, markout, event log).
