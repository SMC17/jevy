# Training cases — Citadel-style paper desk

**Version:** `0.5.0-zig-oom-citadel-lit`  
**Code:** `jev_omm/training/` (research desk, JSONL, Decision hook) and `zig/src/training.zig` (same LCG and constants).  
**Mode:** simulation / paper only. No live venues. The decision model never emits orders.

Lessons follow the public write-up of a simulated trading class. This tree does not ship that software and does not claim the cases are Citadel's.

Primary article (the `/p/what-i-learned-from-citadels-training-software` path 404s):

- https://www.predictingalpha.com/blogs/what-i-learned-from-citadels-training-software (Sean Ryan, 17 Mar 2023)

Reddit originals of the same essay:

- https://www.reddit.com/r/Trading/comments/122y2zq/what_i_learned_from_citadels_training_software/
- https://www.reddit.com/r/options/comments/122pz4e/what_i_learned_from_citadels_training_software/

Mirror already cited in the design brief: https://medium.datadriveninvestor.com/this-is-what-citadels-training-software-taught-me-741c3996a5b5

The r/quant Chicago thread is career context only. See [AKUNA_AND_DESK_CURRICULUM.md](./AKUNA_AND_DESK_CURRICULUM.md). It is not a case, and this repo does not encode firm names from that comment.

## What a case is

Each case has a **role**, an **information set**, **constraints**, and a **score**. Two scripted policies run on the same seed. `mm_inventory` also has a research policy that is not graded.

| Policy | What it does |
| --- | --- |
| `naive` | Trades the obvious edge and skips the lesson (no hedge, slow hand schedule, always facilitate, ignore arb gates). |
| `desk` | Applies the lesson in code. This is the graded policy. |
| `predatory` | `mm_inventory` only. Joins a one-sided wave and sells into a later peer cover. Printed for comparison. Excluded from `run_pair` and the leaderboard. |

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
**Information:** both venue mids, the common factor, betas 1.0 and 0.55, and the constraint that the sim has no futures.  
**Lesson:** buying the cheap location and selling the rich one is positive expected value and still leaves residual oil beta. In the original simulator you could not hedge: you cannot short spot oil, and futures were not in the sim (author reply on the r/Trading thread). In real life a futures hedge is possible and still carries basis risk.

The naive book is the simulator: it trades the basis and holds the residual beta while the factor has a downward drift. The desk book is an **out-of-sim overlay**. It shorts futures one-for-one against oil beta, but the future's true beta is 0.85, so about 15% of the oil beta remains, and the futures basis is marked. Tests: `test_location_arb_hedges_beta`, Zig `location arb sim cannot hedge; futures overlay leaves basis risk`.

## 2. `pm_fair_value`

**Role:** portfolio manager. Three names, known fair values.  
**Information:** price, fair value, and a unit market beta on each name.  
**Lesson:** long what is cheap, short what is rich. Even when nothing looks rich, short the name closest to fair so the book is market-neutral, then size the cheap names (desk size 2, naive size 1 and no shorts). The score is distance-to-fair PnL minus a residual-beta penalty. With the opposing leg, desk beta is zero and the risk-adjusted score beats the net-long naive book.

Tests: `test_pm_fair_value_is_market_neutral`, Zig `pm fair value desk is market neutral and sizes the edge`.

## 3. `etf_ap_arb`

**Role:** ETF authorized participant.  
**Information:** ETF-vs-basket premium path, create/redeem fee, latency.  
**Lesson:** when the arb is near risk-free (edge large versus residual execution vol), size up and trade before the premium decays. Waiting is the risk. The write-up's failure mode is a slow or wrong create/redeem algo.

Desk latency is 1 step and size is 12. Naive latency is 8 and size is 1.

The offline Decision client sees `state["arb"]["near_risk_free"]` and returns Choice `size_tier=large` (`source=fallback`). `apply_policy` may still cut `size_mult` when the toxicity Score confidence is under the floor. **The case engine keeps max size.** That override is code. The model does not emit an order. Tests: `test_etf_sizes_before_the_edge_dies`, `test_near_risk_free_choice_is_large_and_not_an_order`.

## 4. `liability_facilitator`

**Role:** liability broker.  
**Information:** a forced client block (low toxicity prior 0.15), the child-slice gap, and later discretionary prints (toxicity prior 0.70).  
**Lesson:** the client sells a large block at a discount. You cannot lift the whole position without pushing the market, so you work it down in slices of 2. Risk is adverse drift while inventory is still on. The write-up contrasts a hand schedule of about a minute with an algo that finishes in four or five seconds. Here that is gap 12 versus gap 1. The desk skips discretionary adds while the block is being worked. The naive book waits and takes those prints.

Tests: `test_facilitator_cuts_informed_inventory`, Zig `liability desk slices the block faster than the hand schedule`.

## 5. `mm_inventory`

**Role:** market maker under one-sided flow.  
**Information:** queue ahead, cancel latency, and a tag on each print: `forced` (prior 0.20: earnings hedges, window dressing, producer/consumer hedges, mechanical rebalance) or `discretionary` (prior 0.75).  
**Lesson, graded:** earn the spread, cancel the toxic queue (latency 0.20 versus 0.90), take forced flow inside an inventory band, and skew back when inventory builds. Default `run_pair` grades this book. Spread PnL is positive and inventory stays small.

**Research mode `predatory`:** join the discretionary wave, hold the inventory, then sell into the later cover. Raw PnL is higher. The inventory-path penalty (`λ_q = 0.08`) makes its risk-adjusted score worse than the desk. It is not on the leaderboard. The write-up is explicit that this made money and was not a successful market-making program.

Tests: `test_mm_inventory_beats_the_wave`, Zig `mm default grade is the stable book, not the predatory cover`.

## 6. `vol_surface_mm`

**Role:** volatility-surface market maker.  
**Information:** a clean raw SVI, a poisoned smile that fails the butterfly / Lee gate, two slices that fail the calendar check, sticky-strike versus sticky-delta after a forward move.  
**Lesson:** do not quote through butterfly or calendar arbitrage. Sticky-strike and sticky-delta are different marks at the same strike (Gatheral–Jacquier, https://arxiv.org/abs/1204.0646). The desk refuses the bad package. The naive book pays an arb penalty and a regime-mismatch penalty.

This case is the options-desk extension. The public essay's market-making section is single-stock. Tests: `test_vol_surface_refuses_arb`. Surface math itself stays in `svi.zig` / `jev_omm/surface/svi.py`.

## What this is not

- Not a multi-agent replica of other trainees. The relative score is against the naive policy on the same seed. Peer panic in `mm_inventory` is a scripted research path.
- Not a live matching engine. Fills are scripted or the existing synthetic LOB.
- Not a path that requires `TYPESAFE_API_KEY`.
- Not a directory of Chicago firms. The `hekivya` comment is cited only as desk-culture background.
