# Architecture — Jev Options Market-Making Research System

**Owner:** Sean Collins  
**Code package:** Zig hot path `zig/` + Python glue `jev_omm/` at the **jevy** repo root ([SMC17/jevy](https://github.com/SMC17/jevy)).  
**Mode:** simulation / paper replay only (no live exchange credentials)  
**Companions:** [LITERATURE_AND_DESIGN_BRIEF.md](./LITERATURE_AND_DESIGN_BRIEF.md) · [SYSTEM_ONE_JEV.md](./SYSTEM_ONE_JEV.md) · [PERF.md](./PERF.md)

---

## 0. Language split (Zig hot path / Python System One)

| Concern | Implementation |
| --- | --- |
| BS price + greeks (incl. vanna/volga), A–S / **Guéant asymptotic and ODE** / **option-vega HJB**, multi-strike strip, **multi-expiry term risk**, hard risk, Poisson **and queue** fills, SABR-lite **and SVI/SSVI**, markout, mark PnL, parity/boxes/combos, banded hedge **and spot–vol tilt**, scenario matrix, toxicity **and Hawkes**, **flow prior / GEX / COT scalers**, variance-swap weights, training-case kernels, sequenced event log + replay | **Zig** (`zig/src/`), shipped as `libjev_omm.so` (C ABI) |
| TypeSafe / Jev decisions, policy, config, surface glue (prefer Zig SABR), COT/GEX/ETF feature builders, paper demo orchestration, JSONL notebooks | **Python** (`jev_omm/`) — research glue only |
| Python default pricing import | `jev_omm.pricing` → ctypes Zig if `.so` present, else pure Python |

Zig and Python only. This tree has no Rust crate.

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
- **Toxicity:** research-grade VPIN-style / imbalance features into Decision `flow.*` (not production VPIN). Lee–Ready, OFI, off-exchange share, and a layered-cancel score join that dict. Zeros leave the fallback rules unchanged.
- **Positioning:** precomputed GEX, COT z-score, pin gap, basis z, put/call ratio, charm/vanna hedge quantities on Decision `positioning.*`. The fallback reads them only when they are set. Zig applies the same scalers with no network I/O.
- **Scenario matrix:** spot×IV shock grid with soft/hard loss hooks (Akuna 201 risk analysis).

### 3.3 Quoter (AS / Guéant + greek penalties)
- Deterministic AS **or** Guéant–Lehalle–Fernandez-Tapia (`QuoterConfig.mode`):
  - `as_finite_horizon`: classic A–S reservation/spread with rolling session time `max(T_horizon − t, dt)`. `T_horizon` is a session length in years, not option expiry. Passing `t_remaining=None` keeps a fixed receding horizon; `run_simulation` does not do that.
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
- Hard limits that are always on: inventory, net Δ (option delta plus underlier shares), Γ, ν, loss.
- Notional, per-strike absolute inventory, and quotes outstanding are checked only when the limit is set **and** the caller passes the object. An unset limit is a no-op.
- Soft limits feed Decision state; hard breaches stop quoting regardless of nouls.
- Inventory state is delta-normalized where AS uses `q`.

### 3.6 Hedge
- Converts residual delta into underlying hedge tickets **in sim**.
- When `hedge_now` fires and `|net delta|` exceeds the band, the fill is booked: underlier position, cash at the slipped price, cumulative slippage, and marked PnL (`cash + option_qty × option_mid + underlier_qty × spot`). A hedge that only lands in `result.hedges` is not accounting.
- Optional spot–vol target `qS* = −Δ − ρ ξ V^π / (2 √ν S)` (Baldacci appendix). Banded delta hedge is unchanged.
- Urgency: Decision `hedge_now` noul gate. The simulator does not hedge continuously.
- Slippage is inside the fill price (so it is inside cash). `hedge_slippage` is the attribution total and is not subtracted again.

### 3.7 Execution / Sim
- `SimConfig.fill_model` selects the backend inside `run_simulation`. `lob` is the primary path (queue depth, cancel-ahead, partials, on the seconds clock). `poisson` is the explicit touch model.
- Clocks: `dt_seconds` for fills, `dt_years = dt_seconds / (252 × 6.5 × 3600)` for GBM and Black–Scholes. `fill_intensity_per_second` is events per second at zero distance from mid. Do not multiply an events-per-year intensity by `dt_years` and call it calibrated.
- Partial fills and cancels are first-class in the LOB stepper. The Poisson path caps size at the posted quote.
- Same event schema for historical replay and Monte Carlo. Live OPRA/NBBO is not wired; see `docs/ablation_synthetic.md`.

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

## 6b. Latent-state gate (`0.7.0-zig-state-os`)

Python builds `S_t`, forced flow, executable liquidity, constraint level-sets, and clocks (`jev_omm/state_os/`). Zig multiplies a precomputed instability scalar into spread, size, and hedge urgency (`state_os.zig`). The multiplier is 1 when the flag is off. The offline fallback maps a high ratio, a binding constraint, a live parent, or a GEX sign disagreement onto Choice / Score / Noul. Full write-up: [STATE_OS.md](./STATE_OS.md).

## 6d. Multi-product sleeve desk (`1.0.0-zig-desk`)

```
products × surfaces × sleeves × residual risk → portfolio + attribution
```

`SurfaceBook` holds per-underlier SVI slices (spot, forward, rate, dividend explicit; a missing spot stays missing). Butterfly and calendar gates are the existing SVI checks. A calendar failure damps the marked total variance up to the previous expiry.

Eight paper sleeves share that book and keep separate PnL. Raw PnL and residual PnL are both on the scoreboard. Residual slopes are demeaned OLS on the index return and on `greekPnlStep`'s gamma and vega buckets. The intercept stays in the residual mean. The allocator is inverse-vol, with `|ρ| > 0.50` shrink, a cut of the worse collinear leg, and a 0.40 name cap.

On seed 11 the skew sleeve and the fly sleeve still print residual Pearson 0.93, and the spread sleeve and the variance sleeve still print 0.63. Those pairs are flagged in [ablation_sleeve_corr.md](./ablation_sleeve_corr.md). The box and the flow sleeve sit apart from that. This is a synthetic fixture (`synthetic_fixture=1`), not a live surface.

Desk Choice / Noul questions (`sleeve_weight`, `kill_sleeve`) are offline-only and are the identity when `desk.enabled` is off. They do not emit orders.

## 6e. Orthogonal multi-product desk (`1.1.0-zig-ortho`)

```
independent smile and convexity factors
        → greek strip (beta, gamma, vega, volga, vanna, variance)
        → pairwise gate (residualize, or merge if |ρ| ≥ √0.5)
        → strip again so the residual stays orthogonal to the factors
        → inverse-vol, shrink at 0.35, cap at 0.35
```

Eight synthetic products (`EQ_INDEX`, `EQ_SINGLE`, `EQ_LOWBETA`, `FX_PAIR`, `FX_EM`, `COMMO_ENERGY`, `RATES_STIR`, `CRYPTO_BETA`). Twenty sleeves: the 1.0 eight, plus wing, sticky regime, dispersion, roll yield, charm, vanna, queue, COT fade, autocall warehouse, funding-versus-box, hub cross-impact, and a rough-vol stress path. New sleeves that would have reloaded beta or vega were rewritten or restricted to one product. See [ORTHOGONALITY.md](./ORTHOGONALITY.md) and [SLEEVES.md](./SLEEVES.md).

On the fifteen-path grid (five seeds, three regimes, 80 steps) no residual pair has mean `|ρ|` above 0.25. The 1.0 failures, 0.93 and 0.63, reprint under `legacy_config()` and fall to about 0.02 and 0.02 as means after the split. Seed 11 can still print `|ρ|` 0.36 on `roll_yield` / `vanna_tilt`, under the gate and over the shrink. The flow sleeve's per-step residual Sharpe on the full book is a smooth synthetic spread. It is not an annualized track record.

Extra offline questions: `merge_sleeve`, `cut_corr_pair`, `product_kill`. Identity when `desk.enabled` is off. They do not emit orders. The k-factor strip is `jev_omm_residual_strip_k`. The three-factor export is unchanged.

## 6f. Honest desk (`1.2.0-zig-honest`)

```
greek strip
        → smoothness penalty (flat leftover → weight 0)
        → roll residual removed from vanna (mean kept), strip again
        → inverse-vol with σ clipped at the panel's 75th percentile
        → min weight 0.03 for a positive-mean sleeve that still has a test-half edge
        → walk-forward sleeve_kill when the test-half mean is ≤ 0
        → soft caps on gross turnover and peak |inventory|
```

`legacy_ortho_config()` reprints the 1.1 book (no penalty, no clip, no floor, no caps, no roll/vanna split). `legacy_config()` still reprints the 1.0 correlations. The raw per-step residual Sharpe is reported and then penalized. It is not annualized and it is not a capacity. See [DESK_HONESTY.md](./DESK_HONESTY.md) and [ablation_desk_honest.md](./ablation_desk_honest.md).

Offline questions add `sleeve_kill`, answered when `desk.edge_fail` is set. `product_edge_fail` raises `product_kill`. Both are flags. Jev does not emit an order.

## 6c. Evidence pass (`0.8.0-zig-evidence`)

No new warehouse or engine. The simulator's primary fill path is the existing LOB stepper; Poisson is `fill_model=poisson`. Finite-horizon A–S uses rolling `T − t`. Hedges update position, cash, slippage, and marked PnL. Optional risk limits (notional, per-strike, quotes outstanding) are no-ops until set. Checked-in synthetic ablation, numerical self-checks, and a Jev paper scoreboard are under `docs/`. They are not a live track record.

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
