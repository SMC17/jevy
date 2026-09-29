# Performance — Zig hot path

## 2026-09-29 — `1.0.0-zig-desk` (one kernel, one run)

`desk/residual_strip_n256` is new. The number below is a single `zig build bench -Doptimize=ReleaseFast` invocation. It is not a six-run median or p95, and it does not replace the 2026-09-28 table.

| Item | Recorded value |
| --- | --- |
| Date | 2026-09-29 |
| Host | Linux x86_64, 4 cores |
| CPU | `Intel(R) Xeon(R) Processor`, `/proc/cpuinfo` `cpu MHz` 2400.000, `cpu cores` 4 |
| Frequency scaling | `/sys/devices/system/cpu/cpu0/cpufreq` is absent |
| Toolchain | Zig 0.16.0 |
| Build | `cd zig && zig build bench -Doptimize=ReleaseFast` |
| Kernel | `desk/residual_strip_n256`, 50_000 iterations, inner warmup `iters/20` |
| Result | 2510.057 ns per call |

The same binary reprinted the older kernels. Those reprints are a different process from the 0.9.0 six-run table. Do not swap them in.

## 2026-09-28 — `0.9.0-zig-falsify` (microbenchmarks)

These numbers are single-process kernel microbenchmarks. They are not a quote-to-trade latency, not a p99 of a running engine, and not a market-making result.

| Item | Recorded value |
| --- | --- |
| Date | 2026-09-28 |
| Host | Linux x86_64, KVM guest |
| CPU | `Intel(R) Xeon(R) Processor`, family 6, model 207, stepping 2, 4 cores, `/proc/cpuinfo` reports `cpu MHz` 2400.000 |
| Frequency scaling | `/sys/devices/system/cpu/cpu0/cpufreq` is absent, so the governor was not readable. The MHz field above is what the guest exported. |
| Toolchain | Zig 0.16.0 |
| Build | `cd zig && zig build -Doptimize=ReleaseFast` |
| Binary | `zig/zig-out/bin/jev_omm_bench` |
| Clock | `CLOCK_MONOTONIC` inside the bench |
| Inner loop | Each kernel discards `iters/20` calls, then reports the mean nanoseconds of the timed loop. Inputs are volatile so the compiler cannot delete the work. |
| Repeats | 6 process invocations. Run 0 is cold (first execution of that binary in the sequence). Runs 1–5 are warm. |
| Statistics | Median, p95, and p99 are of the five warm per-run means. With five samples, p95 and p99 are linear interpolations of the upper order statistics, close to the slowest warm run. They are not percentiles of individual iterations. |

The bench binary was already built before run 0; "cold" is the first execution (page faults and first touch), not a rebuild. The inner `iters/20` warmup still runs on every invocation, including the cold one.

| kernel | cold ns | warm median | warm p95 | warm p99 |
| --- | --- | --- | --- | --- |
| `bs_greeks/price_and_greeks_atm_call` | 72.122 | 72.190 | 72.263 | 72.269 |
| `bs_greeks/price_and_greeks_batch_64` | 4455.798 | 4457.322 | 4843.677 | 4920.750 |
| `quote_cycle/bs_risk_as_quote` | 90.150 | 89.792 | 89.936 | 89.958 |
| `quote_cycle/quote_cycle_x256` | 22281.997 | 22293.767 | 22317.756 | 22321.002 |
| `surface/sabr_iv_single` | 23.846 | 23.761 | 23.771 | 23.771 |
| `surface/sabr_iv_batch_64` | 1504.419 | 1504.682 | 1505.557 | 1505.688 |
| `gueant/make_quote` | 103.362 | 103.310 | 103.522 | 103.548 |
| `multi_strike/quote_strip_5` | 892.565 | 896.042 | 905.102 | 906.676 |
| `parity/box_spread` | 12.222 | 12.220 | 12.288 | 12.302 |
| `combos/straddle_theo` | 119.212 | 119.297 | 120.695 | 120.931 |
| `hedge/propose_apply_greek_pnl` | 2.596 | 2.595 | 2.649 | 2.659 |
| `scenario/matrix_7x5` | 251.776 | 252.932 | 253.300 | 253.354 |
| `surface/svi_iv_and_density_g` | 7.698 | 7.693 | 7.699 | 7.700 |
| `gueant/ode_offsets_q8_200steps` | 130437.471 | 130679.294 | 131508.513 | 131608.411 |
| `term/aggregate_3_expiries` | 131.937 | 132.001 | 132.125 | 132.128 |
| `option_mm/solve_and_quote_grid21x20` | 11796.203 | 11787.725 | 11799.012 | 11799.601 |
| `hawkes/intensity_8_events` | 127.426 | 127.648 | 128.097 | 128.176 |
| `training/location_arb_pair` | 444.240 | 442.611 | 444.705 | 445.113 |

`bs_greeks/price_and_greeks_batch_64` has a warm p95 far above the median because one of the five warm runs was slower (about 4940 ns versus about 4457 ns). That run is in the table. Do not quote the median alone.

The rows below this section are the 2026-09-26 log from `0.5.0` / `0.7.0`. They are not the 0.9 measurement.

**Measured:** 2026-09-26  
**Host:** Linux x86_64 (cloud agent VM)  
**Toolchain:** Zig 0.16.0 (`$HOME/zig-0.16/zig`, official tarball https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz)  
**Command:** `cd zig && zig build bench -Doptimize=ReleaseFast`  
**Clock:** `CLOCK_MONOTONIC`  
**Note:** Inputs varied each iteration (volatile) to defeat dead-code elimination.  
**Version of the historical table below:** `0.5.0` / `0.7.0` microbenchmarks. The citable 0.9 block is the one above.

## Methodology (read this before quoting a number)

These rows are single-process microbenchmarks of individual kernels. They are not a quote-to-trade latency, not a p99 of a running engine, and not comparable across machines.

A future table that is allowed to be cited as a measurement needs all of the following, written next to the numbers:

- CPU model, core count, and whether frequency scaling was left on
- Zig optimize mode (`ReleaseFast` here) and the exact `zig version`
- warm-up iterations discarded, then the timed loop
- repeated runs (at least 5); report median and p95 of the per-run mean, not one lucky total
- cold vs warm called out when the first invocation pays a page fault or a JIT-less first touch of a big grid
- the bench command and the date

`0.8.0` did not produce a new table. The dated block at the top of this file is the `0.9.0-zig-falsify` measurement, taken with the checklist above. The rows below stay the `0.5.0` / `0.7.0` log. The 0.7 instability gate and the 0.8 risk-book extensions are a few compares; they are not in those historical rows.

**Version stamp on the historical rows:** `0.7.0-zig-state-os` host note, measurements mostly from `0.5.0-zig-oom-citadel-lit` as the next paragraph says.  

The rows below were measured on `0.5.0-zig-oom-citadel-lit`. The `0.6.0` flow-prior, GEX, and COT scalers, and the `0.7.0` instability gate, are unit-tested and are not in this table. The gate is a handful of multiplies on precomputed scalars. It does not reprice the book.

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
