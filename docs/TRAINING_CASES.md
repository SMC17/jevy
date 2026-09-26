# Training cases — Citadel-style paper desk

**Version:** `0.5.0-zig-oom-citadel-lit`  
**Code:** `jev_omm/training/` (research desk, JSONL, Decision hook) and `zig/src/training.zig` (same LCG and constants).  
**Mode:** simulation / paper only. No live venues. The decision model never emits orders.

Lessons are taken from the public write-ups of Citadel's training software, not from a live Citadel system:

- https://www.reddit.com/r/Trading/comments/122y2zq/what_i_learned_from_citadels_training_software/
- https://www.reddit.com/r/options/comments/122pz4e/what_i_learned_from_citadels_training_software/
- https://medium.datadriveninvestor.com/this-is-what-citadels-training-software-taught-me-741c3996a5b5

Chicago desk context (curriculum, not a case): https://www.reddit.com/r/quant/comments/pwzknt/small_prop_trading_firms_in_chicago/hekivya/

## What a case is

Each case has a **role**, an **information set**, **constraints**, and a **score**. Two scripted policies run on the same seed:

| Policy | What it does |
| --- | --- |
| `naive` | Trades the obvious edge and skips the lesson (no hedge, slow, always facilitate, ignore arb gates). |
| `desk` | Applies the lesson in code. |

Score (`score_path` / Zig `scorePath`):

- **absolute PnL** — terminal marked wealth
- **relative score** — absolute PnL minus a peer (the naive run). Competitive hook, not a live leaderboard
- **inventory-path penalty** — `λ_q · mean(q²)`
- **beta penalty** — `λ_β · mean(β²)` for unhedged factor exposure
- **exec penalty** — latency shock or an explicit arb fine
- **risk-adjusted** — absolute − inventory penalty − beta penalty − exec penalty

JSONL records are `CaseMeta`, `CaseScore` (paths included), and case events. `replay_scores` recomputes risk-adjusted PnL from the stored paths.

```bash
python -m jev_omm.demo_training
cd zig && zig build training
```

## 1. `location_arb`

**Role:** commodity location arbitrageur.  
**Information:** both venue mids, the common factor, and the betas (1.0 and 0.55).  
**Lesson:** buying the cheap venue and selling the rich one still leaves residual market beta. Hedge it or the factor move dominates the spread.

The desk sets the factor future so net beta is zero before the next shock. The naive book does not. Tests: `test_location_arb_hedges_beta`, Zig `location arb desk hedges residual beta`.

## 2. `etf_ap_arb`

**Role:** ETF authorized participant.  
**Information:** ETF-vs-basket premium path, create/redeem fee, latency.  
**Lesson:** when the arb is near risk-free (edge large versus residual execution vol), size up and trade before the premium decays. Waiting is the risk.

Desk latency is 1 step and size is 12. Naive latency is 8 and size is 1.

The offline Decision client sees `state["arb"]["near_risk_free"]` and returns Choice `size_tier=large` (`source=fallback`). `apply_policy` may still cut `size_mult` when the toxicity Score confidence is under the floor. **The case engine keeps max size.** That override is code. The model does not emit an order. Tests: `test_etf_sizes_before_the_edge_dies`, `test_near_risk_free_choice_is_large_and_not_an_order`.

## 3. `liability_facilitator`

**Role:** customer-flow facilitator.  
**Information:** customer side, whether the print was informed, recent informed fraction, inventory.  
**Lesson:** uninformed flow pays the half-spread (0.08). Informed flow marks out by 0.28. The desk pulls when the recent informed fraction exceeds 0.55 or `|inventory| ≥ 4`. The naive book always takes the print.

Tests: `test_facilitator_cuts_informed_inventory`.

## 4. `mm_inventory`

**Role:** single-name market maker in a one-sided wave.  
**Information:** queue ahead, trade intensity, cancel latency. Reuses the LOB fluid fill (`expectedFills` / `expected_fills`).  
**Lesson:** a fast cancel (latency 0.20 versus 0.90) fills less of a toxic wave. The desk also stops bidding once inventory is large. Spread earned on the wave does not pay for the adverse jump.

Tests: `test_mm_inventory_beats_the_wave`, Zig `mm desk cancel and skew beat the one-sided wave`.

## 5. `vol_surface_mm`

**Role:** volatility-surface market maker.  
**Information:** a clean raw SVI, a poisoned smile that fails the butterfly / Lee gate, two slices that fail the calendar check, sticky-strike versus sticky-delta after a forward move.  
**Lesson:** do not quote through butterfly or calendar arbitrage. Sticky-strike and sticky-delta are different marks at the same strike (Gatheral–Jacquier, https://arxiv.org/abs/1204.0646). The desk refuses the bad package. The naive book pays an arb penalty and a regime-mismatch penalty.

Tests: `test_vol_surface_refuses_arb`. Surface math itself stays in `svi.zig` / `jev_omm/surface/svi.py`.

## What this is not

- Not a multi-agent replica of other trainees. The relative score is against the naive policy on the same seed.
- Not a live matching engine. Fills are scripted or the existing synthetic LOB.
- Not a path that requires `TYPESAFE_API_KEY`.
