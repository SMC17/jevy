# Executable fills

**Version:** `1.3.0-zig-fills`  
**Code:** `jev_omm/execution/executable.py`, `jev_omm/execution/lob.py`, `jev_omm/research/tape_walk.py`, `jev_omm/desk/harness.py`, `zig/src/lob.zig` (`kappaForTouch`, `queueDecision`).  
**Mode:** paper. `synthetic_fixture=1` on synthetic rows. No live broker, no order. Jev does not emit an order.

The 1.2 scoreboard was a residual stream. Model quoters on the local tape filled nothing, because γ = 0.12 and κ = 1.5 post an indifference half-spread of about 0.64 price units and the books here are a few cents wide. This pass posts the same quoters and the survivor sleeves on the LOB path. The evidence is fill count, join rate, adverse markout, and fill PnL. The 1.2 residual Sharpe is still computed and is not replaced by the fill number. `legacy_honest_config()` turns the fill columns off and turns the full sleeve set back on.

Numbers are in [ablation_fills.md](./ablation_fills.md).

## Why the 0.9 quotes did not trade

On `tape_synthetic.csv` the train-window median touch is 0.0474. No test-window print goes through a 0.64 half-spread. `fill_model=print` still shows that zero for Avellaneda–Stoikov, Guéant, and the option-vega quoter. Fixed-spread is clamped at 0.25 and catches five sweeps. Join-touch, with no queue, catches 28 prints and loses money after the 0.05 research fee.

That print rule is still the regression. It is not the default.

## Touch units

The A–S intensity half-spread is

δ = (1/γ) ln(1 + γ/κ).

`kappa_for_touch(γ, δ, κ_prior)` inverts it. With δ equal to the train touch, γ = 0.12 gives κ ≈ 21.04 on the fixture and κ ≈ 79.94 on the 20-row OPRA preview. The hazard fit is a different object (events per second at a distance) and is not written into `QuoterConfig.A`. On both tapes the train window does not identify k, so the hazard κ stays 1.5. The executable κ is only the unit map. γ ≤ 0 or δ ≤ 0 returns the prior. That is the identity.

Fixed-spread is not rescaled. It stays clamped at 0.25. Join-touch posts the market and does not improve.

## LOB posting

Default `walk_forward` uses `fill_model="lob"`.

- A model bid inside the spread is posted there. Nothing is displayed ahead of it. The next print at the old touch reaches it.
- A model at the touch joins. Displayed `bid_sz` / `ask_sz` is the queue. On this fixture the print is 2 contracts and the smallest displayed size is 5, so a joiner fills 0.
- A model behind the touch stays behind. It does not step up to the touch. A print through that price fills it. That is how fixed-spread still trades.
- `queue_edge=False` is stay: same ahead, no cancel, spread and size unchanged.
- `queue_edge=True` cancels when `toxic_flow ≥ 0.85` or when both staying and improving have negative queue value. Otherwise it improves (ahead → 0, half the capture in the value) when that beats staying. Cancel latency is 10% of the bar. Rates are contracts per second. The horizon is the gap to the next row, in seconds.

The fill is the next row's print. No print is invented. `CRYPTO_BETA` is still `crypto_synthetic`.

## Missing spot

The OPRA preview has no underlying. `spot_provenance=absent`. The quote reference is the CBBO mid, `quote_mid_provenance=cbbo`. The strike is not stored as a spot, and GEX stays off, so `as_flow` and `as_flow_gex` match. sha256 of the cbbo-1m body is `b8ed8988cd064a35252049a6306f39a39040f54c1c1c8ebee1536a8a96882848`, the same preview as [ablation_real_or_fixture.md](./ablation_real_or_fixture.md). Raw bytes stay in gitignored `data/local/`.

## Desk

`DeskConfig` defaults to `fill_model="lob"`, `executable_fills=True`, `queue_edge=True`, `lean_book=True`.

Lean turns off the sleeves the 1.2 baseline walk-forward killed: `parity_box`, `roll_yield`, `cot_fade_sleeve`, `warehouse_autocall`, `box_rate`. They remain in `SLEEVE_IDS`. An explicit `enabled` entry turns one back on. `legacy_honest_config()`, `legacy_ortho_config()`, and `legacy_config()` set `lean_book=False` and `executable_fills=False`.

Each enabled sleeve emits a bid and ask around the research mid 1. Half-spread is `0.02 * spread_mult`. The horizon is 60 seconds, not the daily factor step and not a year. Intensity is 0.05 contracts per second. The adverse jump is 0.04 price units times toxicity.

- `book="touch"` for every sleeve except the two below. Depth is 2 contracts.
- `queue_sniper` uses depth `max(0, 12 − 6 queue)`. That queue does not clear in 60 seconds unless the edge improves.
- `flow_toxicity` keeps its widened `spread_mult`, so it sits behind the touch. The queue edge does not pull it inside. Toxicity still charges the markout.

Fill PnL is posted half-spread times contracts, minus the toxic jump, minus the research fee. It is a column. It is not added to the residual the allocator sees. The smoothness penalty is unchanged.

`no_fill` still records the quote and leaves inventory, residual PnL, and fill PnL at 0.

## What the queue ablation is for

Eight seeds, one resting bid, `simulate` in `lob.py`.

| case | edge | mean fills | mean markout |
| --- | --- | --- | --- |
| toxic (`toxic_flow=1`, ahead 0, intensity 8/s) | off | 1.000 | −0.0300 |
| toxic | on | 0.625 | −0.0188 |
| deep queue (ahead 50, intensity 5/s, no toxicity) | off | 0.000 | 0.000 |
| deep queue | on | 1.000 | +0.0500 |

Toxicity with the edge on fills less and loses less. A dead queue with the edge on improves and fills; off stays at zero. The desk's `queue_sniper` does the same thing: 208 fills with the edge on, 0 with it off, on a 40-step seed-11 path. The touch sleeves fill fewer contracts with the edge on (362 versus 486) and their adverse markout is less negative (−3.79 versus −8.41), because toxic steps cancel.

## What this file is not

An annualized Sharpe. A claim that posting inside a two-cent book is a strategy. A live OPRA backtest. The 20-row preview is still a preview.
