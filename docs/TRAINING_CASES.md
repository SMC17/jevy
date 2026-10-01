# Training cases — Citadel-style paper desk

**Version:** `0.7.0-zig-state-os`  
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

## 7. `flow_vpin`

**Role:** market maker on a signed tape.  
**Information:** bucket VPIN, one-level order-flow imbalance, an off-exchange share, and a layered-cancel score.  
**Lesson:** when those line up, `flow_prior` widens the quote and cuts size. The naive book keeps size 1 and spread multiplier 1 and pays the adverse move.

The graded PnL is that scaler, in Python and in Zig, on the same LCG. A Decision snapshot on a toxic state returns Choice `size_tier=tiny` from the fallback client (`source=fallback`). The snapshot is not a ticket. Tests: `test_training_flow_gamma_and_cot`, Zig `flow desk quotes through toxicity and keeps more pnl`.

## 8. `dealer_gamma`

**Role:** options market maker in two dealer-gamma regimes.  
**Information:** normalized GEX, a pin, and a hedge band.  
**Lesson:** while GEX is positive the desk leans with long gamma (tighter quote, larger size, wider hedge band) and the spot mean-reverts. When GEX turns negative the desk widens, cuts size, and hedges inside a tighter band while the spot trends. The naive book never changes the band. Desk absolute PnL is positive; the naive book is negative.

`short_premium` (dealers short both wings) does not produce a flip. The case passes a normalized GEX scalar; it does not pretend the sign was measured. Max pain is not used as a forecast. The short-gamma Decision snapshot sets `hedge_now` and `size_tier=tiny`. Tests: `test_training_flow_gamma_and_cot`, Zig `dealer gamma desk beats a flat band`.

## 9. `cot_fade`

**Role:** overlay on a weekly speculative positioning print.  
**Information:** a z-score of net speculative futures positions.  
**Lesson:** the next return in this case mean-reverts. The desk fades only when `|z| ≥ 1.5`. The naive book takes the sign of z, including mild prints. Extreme COT also widens the fallback quote. `cot_fade` itself moves the reservation against the crowd; that shift is code.

Tests: `test_training_flow_gamma_and_cot`, Zig `cot desk fades the extreme and beats the crowd`.

## 10–16. State-OS cases

Same score as the earlier cases. The desk policy is code. A Decision snapshot, when the case logs one, is Choice / Score / Noul from `DeterministicFallbackClient`. It is not an order. Zig runs the same arithmetic through `state_os.zig`.

| Case | Lesson |
| --- | --- |
| `letf_day` | Cheng–Madhavan demand `AUM (L² − L) r` is a buy on an up day. The desk sells that demand and covers the revert. The naive book buys the close. |
| `instability_spike` | A calm bar is the identity. `|F|/L = 4` pulls the desk quote. The naive book keeps size 1 into the print. |
| `remaining_parent` | Size is cut only while the parent fraction is still positive. After the parent stops, the desk size goes back to 1. |
| `constraint_gate` | Vol above the cap binds the level-set and the desk size is 0. Inside the cap both books quote. |
| `gex_disagree` | Structural gamma is long and flow-signed gamma is short. The desk cuts size. The naive book leans with the pin and the path follows the flow. |
| `tdf_threshold` | A 250 bp drift trades back to 175 bp from target (`−7.5` on 1000 of AUM). A 100 bp drift trades 0. |
| `overwrite_roll` | Gen-3 cover is `0.50 + 2(IV − IV_ref)`. The desk sells rich implied vol. The naive book skips the roll. |

Tests: `test_state_os_training_cases_teach_the_gate`, Zig `state-os cases: the desk policy beats the naive one`.

## 17. `toxic_sleeve` (`1.1.0-zig-ortho`, was `1.0.0-zig-desk`)

**Role:** sleeve allocator across two synthetic names.  
**Information:** residual series for `mm_spread` and a collinear toxic sleeve, correlation cap 0.35, concentration cap 0.35, Sharpe tilt 0.  
**Lesson:** equal weight keeps a negative sleeve that is the same factor as a better one. The desk allocator zeros the worse leg and caps the survivor at 0.35. PnL on the fixed series is 0.0735 versus 0 for the naive book. The 1.0 lesson used cap 0.40 and printed 0.084.

The offline fallback sees `desk.enabled`. Off, `sleeve_weight` is `hold` and `kill_sleeve` is 0, and the policy multiplier stays 1. On, with the toxic flag, kill noul is above 0.70 and code sets the weight. The answer is not an order. Carr and Wu, Review of Financial Studies 2009, [doi:10.1093/rfs/hhn039](https://doi.org/10.1093/rfs/hhn039), is the variance-premium citation for the residual object. The allocator rule itself is the one in [SLEEVES.md](./SLEEVES.md).

Tests: `test_toxic_sleeve_training_case`, Zig `toxic sleeve allocator cuts the loser`.

## 18. `ortho_break` (`1.1.0-zig-ortho`)

**Role:** two sleeves, one smile factor.  
**Information:** a skew residual and a fly series `0.93 * skew − 0.04`, so Pearson is 1. Correlation cap 0.35. Concentration cap is 1 on this lesson so the cut is visible without the 0.35 name cap. Tilt 0.  
**Lesson:** equal weight double-counts the factor. The allocator keeps the skew leg at weight 1 and drops the fly. Desk PnL is 0.21. Naive PnL is 0.08265.

`ortho_break` on the desk state asks `merge_sleeve` and `cut_corr_pair`. Code sets the weight. Gatheral and Jacquier, [arXiv:1204.0646](https://arxiv.org/abs/1204.0646).

Tests: `test_ortho_and_multi_product_training_cases`, Zig `orthogonality break keeps the better smile leg`.

## 19. `toxic_multi` (`1.1.0-zig-ortho`)

**Role:** three synthetic names, one negative clone.  
**Information:** a good series, an unrelated series, and the good series minus 0.07. Names `EQ_INDEX`, `EQ_SINGLE`, `EQ_LOWBETA`. Caps 0.35, tilt 0.  
**Lesson:** a third name does not rescue a negative copy. Desk weights are 0.35, 0.35, and 0 (sum 0.70, the rest is cash). Desk PnL is 0.10535. Naive equal weight is 0.03033.

`product_toxic` asks `product_kill`. Code sets the weight. Carr and Wu, [doi:10.1093/rfs/hhn039](https://doi.org/10.1093/rfs/hhn039).

Tests: `test_ortho_and_multi_product_training_cases`, Zig `orthogonality break keeps the better smile leg` (the same test calls the multi-name case).

## What this is not

- Not a multi-agent replica of other trainees. The relative score is against the naive policy on the same seed. Peer panic in `mm_inventory` is a scripted research path.
- Not a live matching engine. Fills are scripted or the existing synthetic LOB.
- Not a path that requires `TYPESAFE_API_KEY`.
- Not a directory of Chicago firms. The `hekivya` comment is cited only as desk-culture background.
