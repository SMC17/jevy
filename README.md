# jevy

Research-grade **options market-making** for Sean Collins ([SMC17/jevy](https://github.com/SMC17/jevy)).

Zig hot path (pricing, quoters, risk, fills, surface, hedge, event log) plus a Python research layer. **TypeSafe System One / Jev** is the decision layer only (Choice / Score / Noul) — it never emits orders. Live `TYPESAFE_API_KEY` wiring is optional and not required.

**Simulation / paper only.** No live brokers, exchange SDKs, or venue API keys.

## Architecture (Zig-first hot path)

| Layer | Language | Role |
| --- | --- | --- |
| **Hot path** | **Zig 0.16** (`zig/`) | Black–Scholes, Avellaneda–Stoikov, risk limits, Poisson fills, mark PnL — C ABI `.so` for Python |
| **Research glue** | Python (`jev_omm/`) | Config, TypeSafe System One / Jev decisions, SABR/parametric surface glue, markout, demo, tests |
| **Abandoned** | `_abandoned_rust/` | Early Rust spike — do not build; Zig is the chosen hot path |

Classical **A–S reservation price stays pure math**. Jev answers only feed `policy.py` → `QuoteAdjustments`.

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
zig build replay -- jev_omm_events.jsonl   # recompute markout/PnL from log
zig build -Doptimize=ReleaseFast    # libjev_omm.so + demo + bench bins
zig build bench -Doptimize=ReleaseFast

# Artifacts
#   zig-out/lib/libjev_omm.so
#   zig-out/bin/jev_omm_demo
#   zig-out/bin/jev_omm_bench
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

python -m jev_omm.demo
python -m jev_omm.demo_multistrike --gueant
python -m jev_omm.demo_desk          # Akuna curriculum desk demo
pytest -q
```

Override library path: `export JEV_OMM_LIB=/path/to/libjev_omm.so`.

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
| `hedge.zig` | Banded Δ hedge, slippage, greek PnL buckets |
| `toxicity.zig` | Research-grade VPIN-style / imbalance features |
| `scenario.zig` | Spot×IV scenario risk matrix |
| `as_quoter.zig` | A–S reservation + spread + greek penalties |
| `gueant.zig` / `multi_strike.zig` | Guéant asymptotics + desk strip |
| `risk_limits.zig` | Hard inventory / greek / PnL stops |
| `fills.zig` | Poisson fill sampler |
| `surface.zig` | Hagan SABR-lite IV |
| `markout.zig` | Spread / markout / inventory attribution |
| `pnl.zig` | Mark-to-model PnL |
| `c_abi.zig` | Exported C ABI for Python ctypes (`0.3.0-zig-akuna-depth`) |
| `event_log.zig` | Sequenced JSONL (+ HedgeFill / GreekPnl) + SHA-256 + replay |
| `demo.zig` / `replay.zig` / `bench.zig` | Paper demo (`--hedge-scenario`), log replay, microbenchmarks |

### Python (`jev_omm/`)

| Package | Role |
| --- | --- |
| `pricing/` | `_native.py` (Zig ctypes) → BS / parity / combos |
| `surface/` | SABR-lite (Zig preferred) + PLACEHOLDER parametric |
| `hedge/` | Banded delta hedge + greek PnL (Zig preferred) |
| `flow/` | Research-grade toxicity features → Decision state |
| `pnl/` | Mark PnL + markout attribution |
| `quoter/` | Pure-Python A–S / Guéant / multi-strike |
| `decisions/` | TypeSafe System One schemas, client (SDK→HTTP→fallback), policy |
| `obs/event_log.py` | JSONL reader/replay for notebooks |
| `risk/` | Hard limits + scenario matrix |
| `execution/` / `backtest/` | Sim glue |
| `demo.py` / `demo_desk.py` | Paper demos |

Docs: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/PERF.md`](docs/PERF.md) · [`docs/MODULES.md`](docs/MODULES.md) · [`docs/AKUNA_AND_DESK_CURRICULUM.md`](docs/AKUNA_AND_DESK_CURRICULUM.md)

## License

Proprietary — Sean Collins.
