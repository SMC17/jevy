# Performance — Zig hot path

**Measured:** 2026-09-26  
**Host:** Linux x86_64 (cloud agent VM)  
**Toolchain:** Zig 0.16.0 (`$HOME/zig-0.16/zig`, official tarball https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz)  
**Command:** `cd zig && zig build bench -Doptimize=ReleaseFast`  
**Clock:** `CLOCK_MONOTONIC`  
**Note:** Inputs varied each iteration (volatile) to defeat dead-code elimination.  
**Version:** `0.5.0-zig-oom-citadel-lit`

Numbers move with the machine. The previous `0.3.0` table was a different host; do not compare absolute nanoseconds across boxes. The rows below, including the 0.4 hot path, were measured together on this VM after the option-MM, Hawkes, and training benches were added.

## Summary

| Benchmark | Per-iter | Throughput |
| --- | --- | --- |
| `bs_greeks/price_and_greeks_atm_call` | **72 ns** | **13.8 Melem/s** |
| `bs_greeks/price_and_greeks_batch_64` | **4.5 µs** / 64 | ~70 ns/option effective |
| `quote_cycle/bs_risk_as_quote` | **90 ns** | **11.2 Melem/s** |
| `quote_cycle/quote_cycle_x256` | **22 µs** / 256 | ~87 ns/cycle effective |
| `surface/sabr_iv_single` | **24 ns** | **42 Melem/s** |
| `surface/sabr_iv_batch_64` | **1.5 µs** / 64 | ~24 ns/strike effective |
| `surface/svi_iv_and_density_g` | **7.7 ns** | **130 Melem/s** |
| `gueant/make_quote` | **104 ns** | **9.7 Melem/s** |
| `gueant/ode_offsets_q8_200steps` | **131 µs** | ~7.7k solves/s |
| `multi_strike/quote_strip_5` | **0.89 µs** / 5 strikes | **1.1 M strips/s** |
| `term/aggregate_3_expiries` | **132 ns** | **7.6 Melem/s** |
| `parity/box_spread` | **12 ns** | **82 Melem/s** |
| `combos/straddle_theo` | **119 ns** | **8.4 Melem/s** |
| `hedge/propose_apply_greek_pnl` | **2.6 ns** | **385 Melem/s** |
| `scenario/matrix_7x5` | **252 ns** | **4.0 Melem/s** |
| `option_mm/solve_and_quote_grid21x20` | **11.8 µs** | ~85k solves/s |
| `hawkes/intensity_8_events` | **128 ns** | **7.8 Melem/s** |
| `training/location_arb_pair` | **334 ns** | **3.0 Melem/s** |

### Raw output

```
Jev Options MM — Zig benches (ReleaseFast)
host=linux  CLOCK_MONOTONIC  inputs varied to defeat DCE

bs_greeks/price_and_greeks_atm_call
  iters=2000000  total_ns=144514095  per=72.257 ns  thrpt=13.839 Melem/s
bs_greeks/price_and_greeks_batch_64
  iters=50000  total_ns=223268165  per=4465.363 ns  thrpt=0.224 Melem/s
quote_cycle/bs_risk_as_quote
  iters=1000000  total_ns=89607133  per=89.607 ns  thrpt=11.160 Melem/s
quote_cycle/quote_cycle_x256
  iters=10000  total_ns=222494167  per=22249.417 ns  thrpt=0.045 Melem/s
surface/sabr_iv_single
  iters=2000000  total_ns=47635712  per=23.818 ns  thrpt=41.985 Melem/s
surface/sabr_iv_batch_64
  iters=50000  total_ns=75209275  per=1504.186 ns  thrpt=0.665 Melem/s
gueant/make_quote
  iters=1000000  total_ns=103623653  per=103.624 ns  thrpt=9.650 Melem/s
multi_strike/quote_strip_5
  iters=100000  total_ns=89398863  per=893.989 ns  thrpt=1.119 Melem/s
parity/box_spread
  iters=1000000  total_ns=12224844  per=12.225 ns  thrpt=81.801 Melem/s
combos/straddle_theo
  iters=1000000  total_ns=119217192  per=119.217 ns  thrpt=8.388 Melem/s
hedge/propose_apply_greek_pnl
  iters=1000000  total_ns=2599656  per=2.600 ns  thrpt=384.666 Melem/s
scenario/matrix_7x5
  iters=200000  total_ns=50496728  per=252.484 ns  thrpt=3.961 Melem/s
surface/svi_iv_and_density_g
  iters=1000000  total_ns=7690055  per=7.690 ns  thrpt=130.038 Melem/s
gueant/ode_offsets_q8_200steps
  iters=2000  total_ns=261022448  per=130511.224 ns  thrpt=0.008 Melem/s
term/aggregate_3_expiries
  iters=200000  total_ns=26352737  per=131.764 ns  thrpt=7.589 Melem/s
option_mm/solve_and_quote_grid21x20
  iters=2000  total_ns=23537762  per=11768.881 ns  thrpt=0.085 Melem/s
hawkes/intensity_8_events
  iters=500000  total_ns=63894881  per=127.790 ns  thrpt=7.825 Melem/s
training/location_arb_pair
  iters=2000  total_ns=668445  per=334.223 ns  thrpt=2.992 Melem/s
```

## What is measured

- **`svi_iv_and_density_g`:** raw SVI total variance → Black IV plus Gatheral–Jacquier \(g(k)\).
- **`ode_offsets_q8_200steps`:** one Guéant ODE solve, inventory cap 8, 200 RK4 steps, then offsets at a varying inventory. This is the expensive quote mode; the asymptotic closed form stays ~100 ns.
- **`aggregate_3_expiries`:** three-expiry book → delta/gamma/vega/theta/vanna/volga and term-structure slope.
- **`option_mm/solve_and_quote_grid21x20`:** one constant-vega HJB solve plus a quote. Bench grid is 21 nodes × 20 Euler steps (the unit-test toy is 31 × 60 and is slower). About 11.8 µs, ~85k solves/s.
- **`hawkes/intensity_8_events`:** exponential kernel intensity plus excitation on eight event times.
- **`training/location_arb_pair`:** naive and desk `location_arb` paths (40 steps, shared LCG) plus the score.

Earlier rows are the 0.3/0.4 hot path (BS, A–S, SABR, Guéant asymptotic and ODE, SVI, strip, term book, parity, hedge, scenario), remeasured on this same run.

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
Zig = hot path (pricing including vanna/volga, SVI eval, quoters AS / Guéant asymptotic / ODE / option-vega HJB, multi-expiry term risk, hedge, scenario, toxicity, Hawkes, training kernels, variance-swap weights, risk, Poisson and queue fills, SABR, markout, event log).
