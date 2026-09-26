# Performance — Zig hot path

**Measured:** 2026-09-26  
**Host:** Linux x86_64 (cloud agent VM)  
**Toolchain:** Zig 0.16.0 (`$HOME/zig-0.16/zig`, official tarball https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz)  
**Command:** `cd zig && zig build bench -Doptimize=ReleaseFast`  
**Clock:** `CLOCK_MONOTONIC`  
**Note:** Inputs varied each iteration (volatile) to defeat dead-code elimination.  
**Version:** `0.4.0-zig-frontiers-1-4`

Numbers move with the machine. The previous `0.3.0` table was a different host; do not compare absolute nanoseconds across boxes.

## Summary

| Benchmark | Per-iter | Throughput |
| --- | --- | --- |
| `bs_greeks/price_and_greeks_atm_call` | **73 ns** | **13.8 Melem/s** |
| `bs_greeks/price_and_greeks_batch_64` | **4.5 µs** / 64 | ~70 ns/option effective |
| `quote_cycle/bs_risk_as_quote` | **90 ns** | **11.1 Melem/s** |
| `quote_cycle/quote_cycle_x256` | **22 µs** / 256 | ~87 ns/cycle effective |
| `surface/sabr_iv_single` | **24 ns** | **42 Melem/s** |
| `surface/sabr_iv_batch_64` | **1.5 µs** / 64 | ~24 ns/strike effective |
| `surface/svi_iv_and_density_g` | **7.7 ns** | **130 Melem/s** |
| `gueant/make_quote` | **104 ns** | **9.6 Melem/s** |
| `gueant/ode_offsets_q8_200steps` | **131 µs** | ~7.6k solves/s |
| `multi_strike/quote_strip_5` | **0.90 µs** / 5 strikes | **1.1 M strips/s** |
| `term/aggregate_3_expiries` | **132 ns** | **7.6 Melem/s** |
| `parity/box_spread` | **12 ns** | **82 Melem/s** |
| `combos/straddle_theo` | **120 ns** | **8.4 Melem/s** |
| `hedge/propose_apply_greek_pnl` | **2.6 ns** | **385 Melem/s** |
| `scenario/matrix_7x5` | **255 ns** | **3.9 Melem/s** |

### Raw output

```
Jev Options MM — Zig benches (ReleaseFast)
host=linux  CLOCK_MONOTONIC  inputs varied to defeat DCE

bs_greeks/price_and_greeks_atm_call
  iters=2000000  total_ns=145156529  per=72.578 ns  thrpt=13.778 Melem/s
bs_greeks/price_and_greeks_batch_64
  iters=50000  total_ns=223767560  per=4475.351 ns  thrpt=0.223 Melem/s
quote_cycle/bs_risk_as_quote
  iters=1000000  total_ns=89700606  per=89.701 ns  thrpt=11.148 Melem/s
quote_cycle/quote_cycle_x256
  iters=10000  total_ns=223291037  per=22329.104 ns  thrpt=0.045 Melem/s
surface/sabr_iv_single
  iters=2000000  total_ns=47742591  per=23.871 ns  thrpt=41.891 Melem/s
surface/sabr_iv_batch_64
  iters=50000  total_ns=75516186  per=1510.324 ns  thrpt=0.662 Melem/s
gueant/make_quote
  iters=1000000  total_ns=103784187  per=103.784 ns  thrpt=9.635 Melem/s
multi_strike/quote_strip_5
  iters=100000  total_ns=90219361  per=902.194 ns  thrpt=1.108 Melem/s
parity/box_spread
  iters=1000000  total_ns=12214618  per=12.215 ns  thrpt=81.869 Melem/s
combos/straddle_theo
  iters=1000000  total_ns=119759408  per=119.759 ns  thrpt=8.350 Melem/s
hedge/propose_apply_greek_pnl
  iters=1000000  total_ns=2595754  per=2.596 ns  thrpt=385.245 Melem/s
scenario/matrix_7x5
  iters=200000  total_ns=50944987  per=254.725 ns  thrpt=3.926 Melem/s
surface/svi_iv_and_density_g
  iters=1000000  total_ns=7700597  per=7.701 ns  thrpt=129.860 Melem/s
gueant/ode_offsets_q8_200steps
  iters=2000  total_ns=262259554  per=131129.777 ns  thrpt=0.008 Melem/s
term/aggregate_3_expiries
  iters=200000  total_ns=26474621  per=132.373 ns  thrpt=7.554 Melem/s
```

## What is measured

- **`svi_iv_and_density_g`:** raw SVI total variance → Black IV plus Gatheral–Jacquier \(g(k)\).
- **`ode_offsets_q8_200steps`:** one Guéant ODE solve, inventory cap 8, 200 RK4 steps, then offsets at a varying inventory. This is the expensive quote mode; the asymptotic closed form stays ~100 ns.
- **`aggregate_3_expiries`:** three-expiry book → delta/gamma/vega/theta/vanna/volga and term-structure slope.

Earlier rows match the 0.3 hot path (BS, A–S, SABR, Guéant asymptotic, strip, parity, hedge, scenario).

## Reproduce

```bash
# Zig 0.16+
curl -fL -o /tmp/zig.tar.xz https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz
mkdir -p "$HOME/zig-0.16" && tar -xJf /tmp/zig.tar.xz -C "$HOME/zig-0.16" --strip-components=1
export PATH="$HOME/zig-0.16:$PATH"
cd zig && zig build bench -Doptimize=ReleaseFast
```

## Architecture note

Python = research glue (System One / Jev decisions, config, demos, mirrors).  
Zig = hot path (pricing including vanna/volga, SVI eval, quoters AS / Guéant asymptotic / ODE, multi-expiry term risk, hedge, scenario, toxicity, risk, Poisson and queue fills, SABR, markout, event log).
