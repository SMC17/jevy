# jevy

[![ci](https://github.com/SMC17/jevy/actions/workflows/ci.yml/badge.svg)](https://github.com/SMC17/jevy/actions/workflows/ci.yml)

Research-oriented **options market-making laboratory** for Sean Collins ([SMC17/jevy](https://github.com/SMC17/jevy)).

Zig hot path (pricing, quoters, risk, fills, surface, hedge, event log) plus a Python research layer. **TypeSafe System One / Jev** is the decision layer only (Choice / Score / Noul) — it never emits orders. Live `TYPESAFE_API_KEY` wiring is optional and not required. This tree does not claim production market-making readiness.

**Simulation / paper only.** No live brokers, exchange SDKs, or venue API keys. No bundled OPRA or NBBO tape. Empirical gaps (no licensed historical tape, no live Jev scoreboard, microbenchmarks are not system latency) are called out in [`docs/ablation_synthetic.md`](docs/ablation_synthetic.md), [`docs/NUMERICAL.md`](docs/NUMERICAL.md), [`docs/JEV_SCORE.md`](docs/JEV_SCORE.md), and [`docs/PERF.md`](docs/PERF.md).

## Architecture (Zig-first hot path)

| Layer | Language | Role |
| --- | --- | --- |
| **Hot path** | **Zig 0.16** (`zig/`) | BS + vanna/volga, A–S, Guéant asymptotic **and ODE**, **constant-vega option MM**, SVI/SSVI, multi-expiry term risk, Poisson **and queue** fills, Hawkes intensity, **flow prior / dealer-gamma / COT scalers**, **instability gate** (identity when off), training kernels, hedge (including spot–vol tilt), event log — C ABI `.so` for Python (`0.8.0-zig-evidence`) |
| **Research glue** | Python (`jev_omm/`) | Config, TypeSafe System One / Jev decisions (Choice/Score/Noul only), latent-state engines, surface/quoter mirrors, training desk, COT / GEX / ETF feature builders, Dupire / rough-vol research, demos, tests |

Classical **A–S reservation price stays pure math**. Jev answers only feed `policy.py` → `QuoteAdjustments`.

The `0.7.0` layer is a latent-state desk: predict the state that makes the next trade obligatory, then gate quotes with `Instability = |F| / L_exec`. See [`docs/STATE_OS.md`](docs/STATE_OS.md). `0.8.0-zig-evidence` does not add another model frontier. It fixes simulator units and hedge accounting, selects the LOB fill path from `run_simulation`, and checks Zig against Python. Zig and Python only.

## Quick start — Zig

```bash
# Zig 0.16+ (this tree targets 0.16). Example local install:
#   curl -fL -o /tmp/zig.tar.xz https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz
#   mkdir -p "$HOME/zig-0.16" && tar -xJf /tmp/zig.tar.xz -C "$HOME/zig-0.16" --strip-components=1
export PATH="$HOME/zig-0.16:$PATH"
cd zig

zig build test                      # unit tests
zig build demo                      # pure-Zig paper sim (+ JSONL event log)
zig build demo -- --multi-strike --gueant   # 5-strike strip + Guéant asymptotics
zig build demo -- --hedge-scenario          # parity/box + banded hedge + greek PnL + scenarios
zig build frontiers                         # SVI, multi-expiry term risk, LOB, Guéant ODE
zig build training                          # Citadel-style paper cases
zig build replay -- jev_omm_events.jsonl   # recompute markout/PnL from log
zig build -Doptimize=ReleaseFast    # libjev_omm.so + demo + bench bins
zig build bench -Doptimize=ReleaseFast

# Artifacts
#   zig-out/lib/libjev_omm.so
#   zig-out/bin/jev_omm_demo
#   zig-out/bin/jev_omm_bench
#   zig-out/bin/jev_omm_training
```

After `zig build -Doptimize=ReleaseFast`, copy/refresh the Python-facing dylib:

```bash
cp -f zig-out/lib/libjev_omm.so ../jev_omm/native/libjev_omm.so
```

`*.so` is gitignored. Python falls back to pure Python when the library is absent.

## Quick start — Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Prefers Zig lib when present (jev_omm/native/libjev_omm.so); else pure Python
python -c "from jev_omm.pricing import ZIG_AVAILABLE, native_version; print(ZIG_AVAILABLE, native_version())"

python -m jev_omm.demo                 # primary fill backend is LOB; intensity is events/second
python -m jev_omm.demo_multistrike --gueant
python -m jev_omm.demo_desk          # Akuna curriculum desk demo
python -m jev_omm.demo_frontiers     # SVI + term book + LOB + Guéant ODE
python -m jev_omm.demo_training      # training cases + option-vega toy
pytest -q                            # golden Zig tests skip unless libjev_omm.so is built
JEV_OMM_REQUIRE_NATIVE=1 pytest -q   # CI mode: missing .so is a failure
```

The paper demo does not pass `fill_intensity_base=5e4`. That number was an events-per-year hack multiplied by a one-minute year-fraction. Intensity is `fill_intensity_per_second`. Poisson remains available as `SimConfig(fill_model="poisson")`.

Override library path: `export JEV_OMM_LIB=/path/to/libjev_omm.so`.

## Reproduce the dependency lock

`requirements.lock` pins the Python packages resolved for this tree (Python 3.12). It is a version lock, not a hash lock.

```bash
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

To regenerate after an intentional upgrade: install `.[dev]` into a clean environment and rewrite the lock from `importlib.metadata` for `numpy`, `scipy`, `pandas`, `pydantic`, `rich`, `pytest`, and their dependencies. CI itself installs from `pyproject.toml` (`pip install -e ".[dev]"`), then builds the Zig library and runs pytest against it. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## TypeSafe / Jev (optional)

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is TypeSafe’s System One model
([docs.typesafe.ai](https://docs.typesafe.ai/) — Choice / Score / Noul, `system_one`, `jev-latest`).

```bash
export TYPESAFE_API_KEY=...   # do not invent keys; omit for offline fallback
```

Client tries SDK → `POST /v1/systemone`; on missing key or failure → `DeterministicFallbackClient`.
DecisionSnapshots in the event log record `source=live|fallback`.
See [`docs/SYSTEM_ONE_JEV.md`](docs/SYSTEM_ONE_JEV.md).

## Event log + replay

```bash
cd zig && zig build demo                 # writes jev_omm_events.jsonl + prints sha256
zig build replay -- jev_omm_events.jsonl # deterministic markout/PnL recompute
# Python notebooks:
#   from jev_omm.obs.event_log import read_jsonl, replay_jsonl
```

## Module map

### Zig (`zig/src/`)

| File | Role |
| --- | --- |
| `black_scholes.zig` | BS price + δ/Γ/ν/θ |
| `parity.zig` | PCP, synthetics/CR, boxes + implied rate |
| `combos.zig` | Vertical / fly / straddle / strangle theos |
| `hedge.zig` | Banded Δ hedge, slippage, greek PnL buckets, spot–vol hedge tilt |
| `toxicity.zig` | Research-grade VPIN-style / imbalance features |
| `scenario.zig` | Spot×IV scenario risk matrix |
| `as_quoter.zig` | A–S reservation + spread + greek penalties; routes `gueant_ode` and `option_vega` |
| `option_mm.zig` | Constant-vega HJB premiums (Baldacci–Bergault–Guéant) + Stoikov–Sağlam Theorem 4 |
| `gueant.zig` / `gueant_ode.zig` | Asymptotic closed form **and** finite-horizon / spectral ODE (arXiv 1105.3115); \((A,k)\) MLE |
| `multi_strike.zig` / `term_book.zig` | Single-expiry strip; multi-expiry book, bucket vega, vanna, volga, term slope |
| `svi.zig` | Raw SVI + SSVI, butterfly/calendar gates, sticky strike/delta, calibration |
| `risk_limits.zig` | Hard inventory / greek / PnL stops |
| `fills.zig` / `lob.zig` | Poisson fills; queue/LOB model (depth, latency, partials, toxic markout, queue value) |
| `hawkes.zig` | Self-exciting intensity, excitation, fill-rate scale |
| `flow_signals.zig` | Lee–Ready, OFI, layered-cancel score, flow prior (identity at zero) |
| `positioning.zig` | GEX / COT / basis scalers, dollar gamma, charm–vanna hedge overlay |
| `varswap.zig` | Variance-strike trapezoid + stylized variance/vol-swap greeks |
| `training.zig` | Shared LCG + paper training cases, including flow, dealer gamma, COT, and the state-OS cases |
| `surface.zig` | Hagan SABR-lite IV (kept; SVI is the primary research surface) |
| `markout.zig` | Spread / markout / inventory attribution |
| `pnl.zig` | Mark-to-model PnL |
| `state_os.zig` | Instability gate and the training-case formulas (LETF, TDF, gen-3 cover). Identity when off |
| `c_abi.zig` | Exported C ABI for Python ctypes (`0.8.0-zig-evidence`), including `jev_omm_state_gate` and `jev_omm_evaluate_risk_ext` |
| `event_log.zig` | Sequenced JSONL (+ LobAdd / LobExecute / LobCancel) + SHA-256 + replay |
| `demo.zig` / `demo_frontiers.zig` / `demo_training.zig` / `replay.zig` / `bench.zig` | Paper demos, training desk, log replay, microbenchmarks |

### Python (`jev_omm/`)

| Package | Role |
| --- | --- |
| `pricing/` | `_native.py` (Zig ctypes) → BS / parity / combos / variance-swap weights |
| `surface/` | **SVI/SSVI** (primary) + SABR-lite + Dupire local vol + rough Bergomi paths |
| `hedge/` | Banded delta hedge + greek PnL + spot–vol tilt (Zig preferred) |
| `flow/` | Research-grade toxicity, Hawkes, Lee–Ready / OFI / spoof score → Decision state |
| `positioning/` | COT, dealer gamma, ETF create/redeem, futures roll, factor overlay |
| `state_os/` | S_t, instability gate, forced-flow engines, research cores, warehouse terms |
| `data/fixtures/` | Synthetic CFTC-shaped CSV. Live Socrata fetch is gated |
| `pnl/` | Mark PnL + markout attribution |
| `quoter/` | A–S / Guéant asymptotic / **Guéant ODE** / **option-vega** / multi-strike |
| `training/` | Citadel-style case engine, scores, JSONL replay |
| `decisions/` | TypeSafe System One schemas, client (SDK→HTTP→fallback), policy. No live key required |
| `obs/event_log.py` | JSONL reader/replay for notebooks |
| `risk/` | Hard limits, scenario matrix, **multi-expiry term risk** |
| `execution/` / `backtest/` | `run_simulation` fill backend: **LOB** (primary) or explicit Poisson. Hedges are booked into cash and the underlier |
| `demo.py` / `demo_desk.py` / `demo_frontiers.py` / `demo_training.py` | Paper demos, including the training cases |
| `research/` | Synthetic walk-forward, ablation metrics, numerical checks. No live vendor |
| `decisions/scoreboard.py` | Paper Brier / log loss / ECE. Jev still does not emit orders |

Docs: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/ablation_synthetic.md`](docs/ablation_synthetic.md) · [`docs/NUMERICAL.md`](docs/NUMERICAL.md) · [`docs/JEV_SCORE.md`](docs/JEV_SCORE.md) · [`docs/PERF.md`](docs/PERF.md) · [`docs/STATE_OS.md`](docs/STATE_OS.md) · [`docs/FRONTIERS.md`](docs/FRONTIERS.md) · [`docs/TRAINING_CASES.md`](docs/TRAINING_CASES.md) · [`docs/LITERATURE_CANON.md`](docs/LITERATURE_CANON.md) · [`docs/MODULES.md`](docs/MODULES.md) · [`docs/AKUNA_AND_DESK_CURRICULUM.md`](docs/AKUNA_AND_DESK_CURRICULUM.md)

## License

Proprietary — Sean Collins. See [`LICENSE`](LICENSE). There is no open-source grant.
