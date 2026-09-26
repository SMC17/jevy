# Architecture — Jev Options Market-Making Research System

**Owner:** Sean Collins  
**Code package:** Zig hot path `zig/` + Python glue `jev_omm/` at the **jevy** repo root ([SMC17/jevy](https://github.com/SMC17/jevy)).  
**Mode:** simulation / paper replay only (no live exchange credentials)  
**Companions:** [LITERATURE_AND_DESIGN_BRIEF.md](./LITERATURE_AND_DESIGN_BRIEF.md) · [SYSTEM_ONE_JEV.md](./SYSTEM_ONE_JEV.md) · [PERF.md](./PERF.md)

---

## 0. Language split (Zig hot path / Python System One)

| Concern | Implementation |
| --- | --- |
| BS price + greeks (incl. vanna/volga), A–S / **Guéant asymptotic and ODE** / **option-vega HJB**, multi-strike strip, **multi-expiry term risk**, hard risk, Poisson **and queue** fills, SABR-lite **and SVI/SSVI**, markout, mark PnL, parity/boxes/combos, banded hedge **and spot–vol tilt**, scenario matrix, toxicity **and Hawkes**, variance-swap weights, training-case kernels, sequenced event log + replay | **Zig** (`zig/src/`), shipped as `libjev_omm.so` (C ABI) |
| TypeSafe / Jev decisions, policy, config, surface glue (prefer Zig SABR), paper demo orchestration, JSONL notebooks | **Python** (`jev_omm/`) — research glue only |
| Python default pricing import | `jev_omm.pricing` → ctypes Zig if `.so` present, else pure Python |

Rust crates (if any) live under `_abandoned_rust/` and are **not** part of the build.

---

## 1. Design thesis

1. **Deterministic core** (Zig) prices and hedges: Avellaneda–Stoikov (AS) reservation/spreads, Black–Scholes / parametric surface fair values, analytic greeks, hard risk limits.
2. **System One / Jev Decision Layer** (Python) wraps *brittle heuristic* judgments (toxicity, regime, aggressiveness, hedge urgency, kill/widen) as typed probabilistic smart-ifs — never as free-form orders.
   - Blog: https://typesafe.ai/blog/introducing-system-one-models-and-jev  
   - Docs: https://docs.typesafe.ai/ (Choice / Score / Noul; `TypeSafeClient.system_one`; model `jev-latest`)
3. **Sequenced event log** (exchange-inspired) enables replay, fuzz, and attribution — see Jane Street [How to Build an Exchange](https://www.janestreet.com/tech-talks/building-an-exchange/).
4. **Fail safe, not fail aggressive:** API down, low confidence, or stale MD → widen / pull / flatten via deterministic fallback.

---

## 2. Pipeline (v0)

```
MarketData ──► Surface / FairValue ──► Quoter (AS + greek penalties)
                                              │
                                              ▼
                                    Decision Layer (System One / Jev)
                                       Choice / Score / Noul battery
                                              │
                          QuoteMods / HedgeCmd / RiskMode (code policy)
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
              Risk / Inventory            Hedge                      Execution / Sim
                    │                         │                         │
                    └─────────────────────────┴────────────► PnL / Attribution
                                                                    │
                                                                    ▼
                                                              Observability
```

**Control-flow rule (TypeSafe):** code owns branching; the model only returns typed answers for narrow questions on a structured `state` snapshot. Actuators never accept model text.

---

## 3. Component contracts

### 3.1 MarketData
- **In:** historical OPRA/NBBO-style captures, synthetic LOB generator, or recorded JSONL ticks.
- **Out:** sequenced `BookTop` / `Trade` / `UnderlyingTick` events with monotonic `seq` and exchange timestamps.
- **Invariants:** single consumer clock for sim; gap detection sets `latency.feed_gap` for Decision state.
- **Non-goal (v0):** FPGA / kernel-bypass feeds (documented only as future shape).

### 3.2 Surface / FairValue
- Fit IV surface: **raw SVI / SSVI** is the primary research surface (`svi.zig`); Hagan SABR-lite remains for one-slice work; parametric placeholder retained. Butterfly (density) and calendar checks set `surface_suspect`. Sticky-strike vs sticky-delta is an explicit regime when spot moves. See [FRONTIERS.md](./FRONTIERS.md).
- Emit `FairValue(option_id) → mid, bid_fv, ask_fv, greeks, residual_z, fit_rmse`.
- Freeze / previous-fit fallback when Decision marks `surface_suspect` or fit explodes.

### 3.2b Parity / combos / hedge / scenarios (Akuna depth)
- **Parity:** European PCP, synthetic forward / conversion-reversal, box PV + implied rate — executable sides only.
- **Combos:** vertical / butterfly / straddle / strangle package theos from BS legs.
- **Hedge:** banded Δ hedge with underlier slippage; greek PnL buckets (½ Γ (ΔS)², θ, ν, inventory MTM); `HedgeFill` / `GreekPnl` event-log types.
- **Toxicity:** research-grade VPIN-style / imbalance features into Decision `flow.*` (not production VPIN).
- **Scenario matrix:** spot×IV shock grid with soft/hard loss hooks (Akuna 201 risk analysis).

### 3.3 Quoter (AS / Guéant + greek penalties)
- Deterministic AS **or** Guéant–Lehalle–Fernandez-Tapia (`QuoterConfig.mode`):
  - `as_finite_horizon`: classic A–S reservation/spread with rolling horizon T−t
  - `gueant_asymptotic`: stationary closed form (arXiv 1105.3115) with mid-touch intensity A
  - `gueant_ode`: finite-horizon ODE / principal eigenmode of the linear system, inventory cap Q
  - `option_vega`: Baldacci–Bergault–Guéant constant-vega grid (arXiv 1907.12433). Reservation and premiums are a function of portfolio vega. Cash A–S math is not used in this mode.
  - reservation from inventory; half-spread from (A, k, γ, σ); (A, k) can be fit from a synthetic tape
- **Multi-strike strip** (`multi_strike.zig`): 5 strikes around spot, shared portfolio-Δ tilt
- **Multi-expiry book** (`term_book.zig`): bucket vega, term-structure slope, vanna, volga, per-expiry scenario tilt
- **Greek penalties:** per-contract Γ/ν + `portfolio_delta_penalty`
- **Out:** `Quote{reservation, half_spread, size}` (+ `strike` on event log) *before* Decision scaling.
- **Must not** call System One.

### 3.4 Decision Layer (System One / Jev) — first-class
See [SYSTEM_ONE_JEV.md](./SYSTEM_ONE_JEV.md).

| Item | Spec |
| --- | --- |
| Model | `jev-latest` |
| API | SDK `TypeSafeClient.system_one` then `POST /v1/systemone`; model `jev-latest`; tag `source=live` |
| Primitives | **Choice** → choice+probabilities+confidence; **Score** → score+probabilities+confidence; **Noul** → P(yes)∈[0,1] |
| State | book, flow marks, inventory/greeks, surface residuals, event flags, latency |
| v0 questions | `regime` Choice{calm,trend,event,auction}; `toxicity` Score; nouls: `informed_flow`, `widen_quotes`, `pull_quotes`, `hedge_now`, `surface_suspect`; `size_tier` Choice{tiny,normal,large} |
| Policy (code) | AS math unchanged; multiply size by tier; if pull or low confidence → cancel; if widen → inflate AS spread; if hedge_now → Hedge module |
| Fallback | `DeterministicFallbackClient` when `TYPESAFE_API_KEY` unset or live call fails; DecisionSnapshot `source=fallback` |

**Forbidden:** model-emitted prices, sizes as raw strings, client order IDs, or chat rationales on the hot path.

### 3.5 Risk / Inventory
- Hard limits: net Δ, Γ, ν, notional, per-strike caps, max quotes outstanding.
- Soft limits feed Decision state; hard breaches force `RiskMode=flatten` regardless of nouls.
- Inventory state is delta-normalized where AS uses `q`.

### 3.6 Hedge
- Converts residual delta into underlying hedge tickets **in sim**.
- Optional spot–vol target `qS* = −Δ − ρ ξ V^π / (2 √ν S)` (Baldacci appendix). Banded delta hedge is unchanged.
- Urgency: continuous hedge vs Decision `hedge_now` noul gate.
- Slippage model explicit and attributed.

### 3.7 Execution / Sim
- Sequenced matching stub (price-time or simplified touch fill with queue priority).
- Partial fills, cancels, rejects as first-class events.
- Same event schema for historical replay and Monte Carlo.

### 3.8 PnL / Attribution
- Decompose: spread capture, inventory MTM, adverse selection / markout at 1/5/30-step horizons (Zig `markout.zig` + Python `pnl/markout.py`); hedge PnL / fees later.
- Tag each interval with Decision features (regime, toxicity, widen_mult, size_mult, fallback).

### 3.9 Observability
- Structured logs + expect-test friendly snapshots (Jane Street tooling theme).
- Metrics: quote uptime, tox-gated pull rate, confidence histograms, fill markout, greek utilization.
- Replay from event log without live Jev (stored answers or forced fallback).

### 3.10 Sequenced event log + replay (exchange-inspired)
- **Zig** `event_log.zig`: append-only JSONL with monotonic `seq` + sim `ts`.
- Event types: `BookTop`, `UnderlyingTick`, `Quote`, `Fill`, `Cancel`, `DecisionSnapshot`, `RiskBreach`.
- `DecisionSnapshot` records answers + Choice/Score confidence + `source=live|fallback`.
- Optional `state` object on DecisionSnapshot (portfolio Δ/Γ/ν, `n_strikes`, `quoter_mode`) for multi-strike.
- `Quote` may include `strike` when logging a strip (single-strike logs omit it).
- Demo writes `jev_omm_events.jsonl` and prints SHA-256; **same seed → same log hash**.
- Replay: `zig build replay -- path.jsonl` or `jev_omm_replay` recomputes markout/PnL from BookTop.mid + Fill.
- Python: `jev_omm.obs.event_log` reads JSONL for notebooks; `run_simulation(..., event_log_path=...)` records DecisionSnapshots.
- Compact binary format: deferred (JSONL preferred for research readability).

---

## 4. Decision integration (detail)

```
each quote cycle:
  state = build_mm_state(book, flow, inv, surface, events, latency)
  as0   = quoter.propose(state)                  # deterministic
  try:
      ans = typesafe.system_one(state, MM_QUESTIONS, model="jev-latest")
      mods = policy.apply(ans, as0)              # confidence gates in code
  except / timeout:
      mods = policy.fallback(state, as0)
  if mods.cancel or risk.blocks(mods):
      execution.cancel_all(our_orders)
  else:
      execution.replace_quotes(as0 scaled by mods)
  if mods.hedge_now or risk.delta_breach:
      hedge.rebalance(urgency=mods.hedge_now)
```

Composite example (weights in code, per TypeSafe composite-scoring pattern):

```python
adverse = 0.5 * answers["informed_flow"].noul + 0.5 * (answers["toxicity"].score / 3.0)
if adverse > 0.7 and answers["toxicity"].confidence >= 0.55:
    mods.widen_mult = max(mods.widen_mult, 1.0 + adverse)
```

---

## 5. Module layout

Zig owns the hot path (`zig/src/`). Python modules under `jev_omm/`:

| Module | Responsibility |
| --- | --- |
| `jev_omm/models/types.py` | Events, BookTop, FairValue, AsQuote, QuoteMods |
| `jev_omm/config.py` | γ, A, k, limits, confidence thresholds, feature flags |
| `jev_omm/pricing/black_scholes.py` | BS price + greeks |
| `jev_omm/surface/sabr.py` | Hagan SABR-lite (Zig preferred) |
| `jev_omm/surface/parametric.py` | PLACEHOLDER parametric smile |
| `jev_omm/pnl/markout.py` | Spread / markout / inventory attribution |
| `jev_omm/quoter/avellaneda_stoikov.py` | AS reservation & spread |
| `jev_omm/decision/state.py` | `build_mm_state` |
| `jev_omm/decision/questions.py` | MM System One battery |
| `jev_omm/decision/client.py` | TypeSafe wrapper + timeout |
| `jev_omm/decision/policy.py` | Confidence gates → QuoteMods |
| `jev_omm/risk/limits.py` | Hard/soft greek & inventory limits |
| `jev_omm/hedge/` | Delta hedge sim |
| `jev_omm/execution/` | Sequenced sim matcher |
| `jev_omm/pnl/` | Attribution |
| `jev_omm/obs/` | Logging / metrics + JSONL event log reader/replay |
| `zig/src/event_log.zig` | Sequenced JSONL writer + replay harness |
| `jev_omm/backtest/` | Replay driver |

Optional dep: `typesafe-sdk` (never required for offline fallback tests).

---

## 6. Phasing

| Phase | Deliver |
| --- | --- |
| **v0** | Synthetic/historical L1 replay; BS + AS; soft greeks; Decision battery with fallback; PnL attribution; docs (this) |
| **v1** | Guéant asymptotics + multi-strike strip (landed); full ODE/spectral quotes; markout-based tox labels |
| **v2** | Multi-name portfolio risk; auction regimes; short-term α (CJP); paper gateway **still keyless in-repo** |
| **Later** | Hardware-shaped MD path research; not a v0 blocker |

---

## 7. Non-goals (architecture)

- Live order routing or storing exchange secrets
- LLM string generation on the quote path
- Replacing AS/BS with a neural price
- Claiming colo / FPGA latency in Python (Zig hot path is the research speed path; still not colo)
- Hidden side effects inside the TypeSafe client wrapper (pure decision → mods)

---

## 8. Safety & research ethics

- Default `TRADING_MODE=sim`.
- Refuse to load live API key envs for venues in this research tree.
- System One used only for typed decisions; human/research review for threshold calibration.
