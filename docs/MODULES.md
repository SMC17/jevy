# Module reference & upgrade path

Simulation / paper only. Live brokers are out of scope for this scaffold.

## `models/`

Domain types: `OptionContract`, `Greeks`, `Quote`, `Fill`, `Position`, `RiskSnapshot`, `MarketState`.

**Upgrade:** multi-strike books, portfolio Greeks aggregation, fee/rebate models.

## `pricing/`

European Black–Scholes–Merton analytic price + delta/gamma/vega/theta.

| Impl | Role |
| --- | --- |
| `black_scholes.py` / `black_scholes.zig` | BS price + greeks |
| `parity.py` / `parity.zig` | PCP, synthetics/CR, boxes + implied rate (executable sides) |
| `combos.py` / `combos.zig` | Vertical / fly / straddle / strangle package theos |

**Real today**, including vanna (\(\partial^2V/\partial S\partial\sigma\)) and volga (\(\partial^2V/\partial\sigma^2\)). **Upgrade:** American (tree/PDE), local vol, Heston, dividend schedules, discrete cash dividends; rho greek export.

## `desk/` (`1.1.0-zig-ortho`)

Paper multi-product book. `SurfaceBook` in `surface/book.py` fits raw SVI per underlier and expiry, runs the existing butterfly and calendar gates, and damps a calendar break instead of quoting through it. It also reports a front Dupire local variance and a sticky-delta minus sticky-strike gap. Twenty sleeves in `desk/sleeves.py` each expose a target, a risk budget, greeks, and a PnL stream. The first eight are the 1.0 names. `enabled=False` is a zero target. Flow, COT, the autocall warehouse, and the instability gate stay at the identity when `gates_on` is false.

`pnl/residual.py` and `zig/src/desk.zig` strip index beta, spot gamma, vega, and (by default) volga, vanna, and a variance column orthogonal to spot gamma. The residual keeps the intercept. `desk/orthogonal.py` residualizes or merges a pair whose residual `|ρ|` is above 0.40, then the harness strips the factor columns again. `desk/allocator.py` is inverse-vol with a 0.35 correlation shrink, a 0.25 haircut on a negative residual mean, a cut of the worse leg of a collinear pair, a Sharpe tilt of 0.25 on the desk, and a 0.35 concentration cap. The scoreboard is in [ablation_sleeve_corr.md](./ablation_sleeve_corr.md). Write-ups: [ORTHOGONALITY.md](./ORTHOGONALITY.md), [SURFACE_DESK.md](./SURFACE_DESK.md), [SLEEVES.md](./SLEEVES.md), [RESIDUAL_PNL.md](./RESIDUAL_PNL.md).

The fixture `data/fixtures/surfaces_synthetic.csv` is labeled `synthetic_fixture=1`. Eight underliers. It is not an OPRA tape. `CRYPTO_BETA` is `crypto_synthetic`.

Optional desk questions `sleeve_weight`, `kill_sleeve`, `surface_suspect`, `merge_sleeve`, `cut_corr_pair`, and `product_kill` are a separate System One map. The offline fallback is the identity when `desk.enabled` is off. Jev still does not emit an order.

## `surface/`

| Impl | Role |
| --- | --- |
| `svi.py` / `svi.zig` | **Primary.** Raw SVI + power-law SSVI, Nelder–Mead fit, butterfly/calendar gates, sticky-strike vs sticky-delta. Cite https://arxiv.org/abs/1204.0646 |
| `SabrIVSurface` | Hagan SABR-lite (Zig `surface.zig` / `jev_omm_sabr_iv` when `.so` present; Python Hagan fallback otherwise) |
| `ParametricIVSurface` | Explicitly labeled **PLACEHOLDER** toy smile (ATM + skew + smile in log-moneyness) |

ATM limit and edge cases documented in `zig/src/surface.zig` and `jev_omm/surface/sabr.py`. SVI conventions: [`FRONTIERS.md`](./FRONTIERS.md).

## `quoter/`

| Impl | Role |
| --- | --- |
| `avellaneda_stoikov.py` / `as_quoter.zig` | Classic A–S finite-horizon reservation + half-spread |
| `gueant.py` / `gueant.zig` | Guéant–Lehalle–Fernandez-Tapia asymptotics (arXiv 1105.3115) |
| `gueant_ode.py` / `gueant_ode.zig` | Finite-horizon ODE + principal eigenmode; Poisson MLE for \((A,k)\) on a synthetic tape |
| `multi_strike.py` / `multi_strike.zig` | Desk strip (≈5 strikes), shared portfolio-Δ tilt |

| `option_mm.py` / `option_mm.zig` | Constant-vega HJB (Baldacci–Bergault–Guéant). Exponential or logistic intensity. Stoikov–Sağlam Theorem 4 premiums. `iv_alpha` reservation shift |

Toggle via `QuoterConfig.mode` ∈ `{as_finite_horizon, gueant_asymptotic, gueant_ode, option_vega}`. `spread_mult` / `size_mult` from the decisions layer. The model does not emit orders.

## `decisions/` (TypeSafe System One / Jev)

| File | Role |
| --- | --- |
| `schemas.py` | Choice / Score / Noul–shaped questions & answers; `build_mm_state` / `build_mm_questions` |
| `client.py` | `TypeSafeDecisionClient` (`TYPESAFE_API_KEY` → SDK then `POST /v1/systemone`, `source=live`) + `DeterministicFallbackClient` + `ResilientDecisionClient` |
| `policy.py` | Confidence gates, composite toxicity, map → `QuoteAdjustments` |

**Pattern:** atomic questions in parallel; compose in code ([docs.typesafe.ai](https://docs.typesafe.ai)).

**Upgrade:** pin `jev-1.x.x` (not only `jev-latest`), log model id + confidence histograms into DecisionSnapshot, A/B offline vs live, add more speculative Nouls (fan-out pattern).

Live key: `export TYPESAFE_API_KEY=...` (do not invent keys). Without a key, fallback always works.

## `risk/`

Hard inventory / delta / vega / gamma / loss limits → `quoting_allowed=False`.

| Impl | Role |
| --- | --- |
| `limits.py` / `risk_limits.zig` | Hard stops |
| `scenario.py` / `scenario.zig` | Spot×IV shock grid (Taylor or reprice); soft/hard hooks |
| `term.py` / `term_book.zig` | Multi-expiry greeks, bucket vega, term-structure slope, vanna/volga limits, tilt scenarios |

**Upgrade:** VaR/ES, kill-switch webhooks (still paper until wired carefully).

## `hedge/`

Banded delta hedge + underlier slippage + Natenberg greek PnL buckets (`hedge.zig` / `hedge/delta.py`). Flatten-to-zero when |Δ| > band (optional to-edge). WW-style band helper documented.

Spot–vol tilt: `spot_vol_hedge_qty` / `spotVolHedgeQty` (Baldacci appendix).

**Upgrade:** futures/ETF execution sim with queue; borrow; discrete hedge calendar. Charm/color bands.

## `flow/`

Research-grade rolling imbalance / simplified VPIN-style buckets → Decision `flow.*` features (`toxicity.zig` / `flow/toxicity.py`). **Not production VPIN.**

Hawkes excitation (`hawkes.zig` / `flow/hawkes.py`) adds `flow.hawkes_intensity` and `flow.hawkes_excitation`. The fallback toxicity Score reads the excitation. `fill_intensity` can scale a synthetic Poisson rate.

`flow/signals.py` / `flow_signals.zig` add Lee–Ready signs, one-level OFI, a synthetic off-exchange weight, and a layered-cancel score. `flow_prior` maps those plus VPIN into a spread multiplier and a size multiplier. Zeros are the identity.

## `positioning/`

Precomputed, paper-only features. Zig consumes scalars; it does not fetch.

| Piece | Role |
| --- | --- |
| `cot.py` | CFTC legacy / disaggregated / TFF rows, net spec, commercial hedge ratio, trailing z, week-over-week. Fixtures under `jev_omm/data/fixtures/`. Network pull is gated |
| `gex.py` | Dealer dollar gamma, zero-gamma level, max pain, put/call ratios, charm/vanna hedge proxy |
| `adjust.py` / `positioning.zig` | Reservation shift, spread, size, hedge-band scale, hedge urgency. Identity when flags are off |
| `overlays.py` | ETF create/redeem pressure, annualized futures roll, residual beta overlay |

`cot_fade` fades a crowded z-score. `gex_adjust` tightens the hedge band in short gamma and widens it in long gamma. `overlay_hedge_qty` adds charm and vanna; zeros do not change the banded delta hedge.

## `execution/`

Poisson fills (`fills.py`) with intensity decaying in distance-from-mid, plus a synthetic queue model (`lob.py` / `lob.zig`): depth, queue position, cancel latency, partial fills, adverse-selection markout, sequenced LobAdd / LobExecute / LobCancel.

`queue_value` / `depth_ahead` score a resting order and the size in front of it on a small multi-level book.

**Upgrade:** historical LOB replay (still no live market-data session).

## `backtest/`

Discrete-event loop: exogenous GBM spot → BS mid → risk → decisions → A–S quotes → fills.

**Upgrade:** historical tape, multi-series, transaction costs, walk-forward.

## `obs/` (event log)

| Impl | Role |
| --- | --- |
| Zig `event_log.zig` / `jev_omm_demo` / `jev_omm_replay` | Append-only JSONL; SHA-256; replay markout/PnL |
| `jev_omm/obs/event_log.py` | Python writer/reader; notebook-friendly `read_jsonl` / `replay_jsonl` |
| `DecisionSnapshot` | answers + confidence + `source=live\|fallback` |

**Upgrade:** compact binary twin; gap detection; expect-tests on log hashes.

## `pnl/` / logging

Mark-to-model PnL; **markout attribution** (`MarkoutTracker` / Zig `markout.zig`):
spread capture vs adverse selection (1/5/30-step markout) vs inventory MTM.
Rich-friendly run summary + attribution table in demos.

Greek P&L explain landed in `hedge.greekPnlStep` / event-log `GreekPnl` + `HedgeFill`.

`pnl/residual.py` projects sleeve PnL on the index return and on those gamma and vega buckets. Formulas: [RESIDUAL_PNL.md](./RESIDUAL_PNL.md).

**Upgrade:** structured logging, metrics export.

## `state_os/`

Latent-state desk. `vector.py` is S_t and `|F|/L_exec`. `gate.py` / `state_os.zig` scale the quote and are the identity when off. `engines.py` is the accounting and clock layer. `gex_flow.py` separates flow-signed gamma from the open-interest book and refuses an ES conversion of SPY. Research cores (`latent_book`, `metaorder`, `impact`, `constraints`, `cross_impact`, `funding`, `hawkes_tv`) and warehouse terms (`warehouse.py`) are Python. Write-up: [STATE_OS.md](./STATE_OS.md).

## `training/`

Citadel-style paper cases, plus `flow_vpin`, `dealer_gamma`, `cot_fade`, and the seven state-OS cases. See [TRAINING_CASES.md](./TRAINING_CASES.md). Zig twin: `zig/src/training.zig`, `zig build training`. Default grade is the `desk` policy. `mm_inventory` also exposes an ungraded `predatory` research mode. The state-OS cases share formulas with `state_os.zig`. Earlier case scores are unchanged.

## `surface/dupire.py`, `surface/rough_vol.py`, `pricing/varswap.py`

Research surface extensions. Dupire local variance (Python). Rough Bergomi stress paths (Python). Variance- and vol-swap weights (Python + `varswap.zig`). See [LITERATURE_CANON.md](./LITERATURE_CANON.md).

## `config.py`

Pydantic `EngineConfig` tree.

**Upgrade:** YAML/TOML profiles, per-underlying overrides.
