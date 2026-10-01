# Desk honesty

**Version:** `1.2.0-zig-honest`  
**Code:** `jev_omm/desk/honesty.py`, `jev_omm/desk/allocator.py`, `jev_omm/desk/harness.py`, `jev_omm/desk/falsify.py`, `zig/src/desk.zig` (`smoothnessPenalty`, `allocateInverseVolClipped`).  
**Mode:** paper. `synthetic_fixture=1` on every synthetic row. No live broker, no order. Jev answers Choice / Noul and does not emit an order.

`1.1.0-zig-ortho` left three lies on the scoreboard. This pass measures them and stops paying them. The numbers are in [ablation_desk_honest.md](./ablation_desk_honest.md). `legacy_ortho_config()` reprints the 1.1 residual correlations and the 1.1 inverse-vol weights. `legacy_config()` still reprints the 1.0 pair correlations.

## 1. A leftover Sharpe is not a capacity

After the greek strip, each sleeve reports four diagnostics. Zig and Python use the same formulas.

| Diagnostic | Definition | What a high number means |
| --- | --- | --- |
| AC(1) | lag-1 correlation of the demeaned residual | a slow leftover |
| DC share | `n * mean² / sum(x²)`, the lowest Fourier bin of the uncentered series | the residual is mostly its mean |
| Constant+trend R² | R² of `x ~ a + b t` against a zero baseline | a constant or a drift, not noise |
| Low-freq share | power in Fourier bin k=1 over bins `1 .. n//2`, after demeaning | a slow cycle in the noise |

Demeaning deletes a constant before AC(1) or the low-frequency bin can see it. On seed 11, `flow_toxicity` has AC(1) 0.066 and low-freq share 0.019, and DC share 0.991. The raw series, before the strip, has DC share 0.050. The strip created the constant. That is the 10.65.

Penalty, in order:

1. Sample σ under `1e-5`, or DC share at least 0.95 and at least 0.40 above the raw series: penalty 0, flag `flat`. The weight is multiplied by 0.
2. Otherwise, if the residual is smooth relative to raw PnL (DC gap, AC(1) gap, low-freq gap, or constant+trend gap past the thresholds in `smoothnessPenalty`), the penalty is in `(0, 1)` and the flag is `smooth`.
3. Otherwise the penalty is 1 and the flag is `ok`. That is the identity.

`residual_sharpe_raw` is mean / sample std. `residual_sharpe_penalized` is that number times the penalty. Neither is annualized. Do not multiply by `√252`.

On seed 11 the flow sleeve is `flat`. Raw Sharpe stays 10.65 on the row. Penalized Sharpe is 0. Weight is 0. The 1.1 desk residual Sharpe of 6.57 was that sleeve plus a quiet book. The honest portfolio Sharpe on the same seed, after the penalty, the floor, the kill, and the capacity scale, is 0.42. The pre-penalty allocator (σ already clipped) prints 2.19, because the clip already moves weight off the tiny-σ sleeve. One path.

`mm_spread` still has raw per-step Sharpe 2.37 and DC share 0.850, against a raw-series DC share of 0.668. The gap is under the flag line, so the penalty stays 1. That is a synthetic spread with a real variance, not the flow leftover. It is still not a capacity.

## 2. Inverse-vol was starving the noisy sleeves

`allocate()` is unchanged when `sigma_clip_quantile` is 0. The 1.2 desk passes 0.75.

1. Sharpe tilt still uses the unclipped sample σ. A negative residual mean is still multiplied by 0.25. The correlated-loser cut is unchanged. Shrink stays 0.35. Concentration stays 0.35.
2. Before `1/σ`, σ is capped at the 75th percentile of the enabled panel. A sleeve noisier than that percentile is sized as if its σ were the percentile. Sleeves under the percentile are unchanged.
3. After the smoothness penalty, weights are renormalized and the 0.35 cap is applied again.
4. A sleeve with positive full-sample mean, penalty above 0, and a positive walk-forward test mean is lifted to 0.03 if it sits below that. Donors above 0.03 pay. The sum does not grow.
5. If the second half of the residual has mean ≤ 0, `sleeve_kill` zeros the weight. Cash absorbs it. There is no renormalization after the kill, and none after the capacity scale.

On seed 11, `legacy_ortho_config()` weights versus the honest book:

| Sleeve | 1.1 weight | 1.2 weight | Residual mean | Why |
| --- | --- | --- | --- | --- |
| `sticky_regime` | 0.0002 | 0.0242 | +1.16 | positive mean, floor, then turnover scale 0.81 |
| `vanna_tilt` | 0.0001 | 0.0231 | +1.81 | same, inventory scale 0.77 |
| `queue_sniper` | 0.0001 | 0.0219 | +2.36 | same, turnover scale 0.73 |
| `rough_vol_stress` | 0.0003 | 0.0000 | −0.29 | test-half mean is negative, so the kill fires |

Three of the four clear 0.02. The fourth is a negative residual, and the 0.25 haircut is not what zeros it. The walk-forward gate does. Fifteen-path mean weights (seeds `(11, 23, 42, 7, 99)` × `baseline`, `smile_shock`, `jump`, 80 steps): sticky 0.022, vanna 0.025, queue 0.021, rough 0.012. Rough's mean stays under 0.02 because the kill fires on paths whose test half does not pay.

No sleeve exceeds 0.35 on the seed-11 book. Weight sum is 0.83. The gap is cash: the flat sleeve, the killed sleeves, and the capacity haircut.

## 3. Roll clock versus spot–vol

The pre-trade shocks were already different random numbers. The −0.363 on seed 11 is not a shared Gaussian draw. It is the commodity roll alpha, `previous target × d_roll`, against the commodity vanna alpha, `previous target × d_vanna`. The shocks themselves correlate at about −0.06. Gram–Schmidt on the shocks moves the PnL correlation from −0.363 to −0.339. That is not a split.

The split that changes the book is applied after the greek strip, because that is where the collinearity sits. Vanna's residual is residualized against the roll sleeve's residual. The mean of vanna is kept. Vanna is then stripped again onto its own greek columns. Roll is left alone.

| | Seed 11 ρ | Fifteen-path mean ρ |
| --- | --- | --- |
| `legacy_ortho_config()` | −0.363 | −0.136 |
| `split_roll_vanna` on | −0.004 | −0.0015 |

The new worst fifteen-path mean is `dispersion_index` / `cot_fade_sleeve` at +0.125 (sample std 0.097), which was already the second pair in 1.1. No pair has mean `|ρ|` above 0.25. The gate is still 0.40 hard and 0.35 shrink. Seed 11's new worst pair is `vrp_varswap` / `vanna_tilt` at +0.338, under the shrink. One seed can still shrink. The mean does not.

Smile and convexity means are unchanged: skew/fly −0.0225, mm/vrp +0.0182.

Standardized PCA is the number to quote. Unstandardized PCA is noise-dominated: a few high-σ sleeves own the sum of squares and then receive little inverse-vol weight. The scoreboard says so. It is not a live risk model.

## 4. Capacity

Per sleeve and for the desk: gross turnover (`sum |Δinventory|`), average |inventory|, peak |inventory|, path length of Γ and of ν, and quote revisions per step. Inventory is the sleeve target summed across products. A `no_fill` regime records the quote revision and then leaves inventory and PnL at 0.

Soft caps, on by default: gross turnover 600, peak |inventory| 30. A sleeve over a cap is scaled by `cap / usage`. Both caps off, or a cap of 0, is the identity. The scale does not get renormalized back to 1. Cash holds the difference.

Edge density is penalized residual PnL per unit turnover and per unit peak inventory. On seed 11 the desk prints 0.0033 per unit turnover and 4.60 per unit peak inventory. Those are research units on one synthetic path. They are not an annualized edge and they are not a dollar capacity.

The 3×8 control (EQ_INDEX, EQ_SINGLE, FX_PAIR, the original eight sleeves, same seed) still zeros `flow_toxicity` (raw Sharpe 5.33, flag `flat`) and puts `parity_box` on the 0.35 cap. Desk penalized Sharpe 0.33. Gross turnover 260. The caps do not bind on that smaller book.

## 5. What the falsification does

Adversarial regimes are `smile_shock`, `jump`, `toxic_flow`, `wide_spread`, and `no_fill`, plus `baseline`, five seeds, 40 steps. `toxic_flow` pushes toxicity up and makes the fill draw harder. `wide_spread` multiplies the research fee by 6 and makes fills harder. `no_fill` discards inventory and PnL after the quote is recorded.

A sleeve "keeps a positive penalized mean" when that mean, after the penalty, is above 0 and the mean weight is not zero. That is not a claim of capacity. `flow_toxicity` can print a small positive penalized mean at a penalty near 0.1. The ablation says so.

Walk-forward kill: second-half residual mean ≤ 0 zeros the sleeve. The offline fallback is asked `sleeve_kill` with `desk.edge_fail` set. A product whose second-half raw PnL is ≤ 0 raises `product_kill` via `product_edge_fail`. The product is flagged and left in the synthetic book. Neither answer is an order.

Local tape: the checked-in fixture `tape_synthetic.csv` (`synthetic_fixture=1`, symbol `SYNTH`) and the no-account Databento preview (`sha256 b8ed8988…`, 20 rows, spot missing, not a fixture). Model quoters post wider than those books and receive zero fills. Fixed-spread and join-touch are the ones that trade. Detail is in the ablation. No OPRA print was invented. `CRYPTO_BETA` is still `crypto_synthetic`.

## What this file is not

An annualized Sharpe. A production capacity. A claim that twenty sleeves are twenty risks. A live TypeSafe session.
