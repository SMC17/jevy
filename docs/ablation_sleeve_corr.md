# Sleeve residual correlation

Synthetic paper desk. `synthetic_fixture=1`. Per-step residual Sharpe is mean / sample std after the research fee. It is not annualized. Do not multiply by `√252`.

Current desk: `python -m jev_omm.demo_surface_desk` (`1.1.0-zig-ortho`). The write-up of the factor split and the gate is [ORTHOGONALITY.md](./ORTHOGONALITY.md).

## 1.1.0-zig-ortho — seed 11, 80 steps, eight products, twenty sleeves

`fit_surfaces=False` and `fit_surfaces=True` print the same sleeve PnL on this seed. Sleeves trade the planted factors. The SVI fit is an audit. With fits on (`fit_stride=20`): 4 refits, 25 suspect slices, 25 calendar breaks, EQ_INDEX Dupire front local variance 0.0268, sticky-delta minus sticky-strike ATM gap after a 1% spot move 0.0038. Scenario grid worst −0.3114 at `dS=-2%, dσ=-1%` (research units, end-of-path greeks).

Desk raw PnL 0.8138, residual PnL 4.1431, per-step residual Sharpe 6.567, residual max drawdown 0.0000, weight sum 1.0000. Pre-gate max `|ρ|` 0.3632, post-gate max `|ρ|` 0.3632. The gate does not fire. The 0.35 allocator shrink does, on `roll_yield` / `vanna_tilt` (−0.3632).

No sleeve is at the 0.35 concentration cap. `flow_toxicity` weight 0.212 with per-step residual Sharpe 10.65 and R² 0.636: after the strip, what is left is a smooth synthetic spread. That Sharpe is the sample. It is not a capacity number. `calendar_term` R² is 0.960 because the term sleeve is mostly vega. `vrp_varswap` R² is 0.025. The strip no longer eats the variance premium, and the sleeve's inverse-vol weight is 0.0017.

| sleeve | weight | resid Sharpe | R² | residual PnL |
| --- | --- | --- | --- | --- |
| mm_spread | 0.1164 | 2.368 | 0.587 | 3.210 |
| skew_residual | 0.0160 | 0.405 | 0.359 | 2.769 |
| calendar_term | 0.1422 | 0.406 | 0.960 | 0.312 |
| fly_butterfly | 0.0230 | 0.219 | 0.099 | 0.995 |
| vrp_varswap | 0.0017 | 0.144 | 0.025 | 8.581 |
| flow_toxicity | 0.2123 | 10.650 | 0.636 | 14.922 |
| gex_forced | 0.0206 | −0.242 | 0.469 | −0.291 |
| parity_box | 0.1971 | 0.199 | 0.201 | 0.105 |
| wing_kurtosis | 0.0233 | 0.554 | 0.092 | 2.682 |
| sticky_regime | 0.0002 | 0.181 | 0.063 | 92.792 |
| dispersion_index | 0.0014 | 0.474 | 0.050 | 38.950 |
| roll_yield | 0.0000 | −0.108 | 0.023 | −0.017 |
| charm_bleed | 0.0913 | 1.797 | 0.004 | 2.829 |
| vanna_tilt | 0.0001 | 0.201 | 0.041 | 153.411 |
| queue_sniper | 0.0001 | 0.199 | 0.016 | 188.412 |
| cot_fade_sleeve | 0.0047 | 0.244 | 0.009 | 5.433 |
| warehouse_autocall | 0.0057 | −0.087 | 0.006 | −0.377 |
| box_rate | 0.1418 | 0.041 | 0.014 | 0.029 |
| cross_impact | 0.0018 | 0.057 | 0.062 | 3.235 |
| rough_vol_stress | 0.0003 | −0.232 | 0.054 | −22.806 |

`sticky_regime`, `vanna_tilt`, `queue_sniper`, and `rough_vol_stress` have large residual totals and almost no weight. Inverse-vol sizes by `1/σ`. Those series are noisy. They are not cut. `roll_yield` is a negative-mean series and the 0.25 haircut plus the shrink against `vanna_tilt` leaves it at 0 on this seed.

Headline pairs on this same seed: `skew_residual` / `fly_butterfly` +0.0270, `mm_spread` / `vrp_varswap` +0.1077. The next largest magnitudes after the flagged pair are `vrp_varswap` / `vanna_tilt` +0.331, `wing_kurtosis` / `box_rate` +0.312, `flow_toxicity` / `box_rate` +0.300. All of those sit under the 0.35 shrink.

### Same eight sleeves, three products, before and after the split

Seed 11, 80 steps, `fit_surfaces=False`. Residual streams under `legacy_config()` match the 1.0 table below. Correlations:

| Pair | `legacy_config` | smile + convexity split, gate on, caps 0.35 |
| --- | --- | --- |
| `skew_residual` / `fly_butterfly` | 0.9301 | 0.0237 |
| `mm_spread` / `vrp_varswap` | 0.6295 | −0.0394 |
| max `|ρ|` | 0.9301 | 0.2732 |

After the split the gate does not fire. `parity_box` weight is 0.350. Desk per-step residual Sharpe on that eight-sleeve book is 4.29. That is one path, and most of it is the flow sleeve's leftover spread (per-step residual Sharpe 5.33, R² 0.270).

`legacy_config()` weights are not the 1.0 weights in the historical table. The residual series match. The allocator now multiplies a negative residual mean by 0.25, so `gex_forced` prints 0.049 instead of 0.158 and the positive sleeves absorb the difference. `parity_box` is still 0.400 on that replay because the legacy cap is 0.40.

### Fifteen paths

Seeds `(11, 23, 42, 7, 99)` × regimes `(baseline, smile_shock, jump)`, 80 steps, `fit_surfaces=False`, twenty sleeves.

| | Mean ρ | Sample std |
| --- | --- | --- |
| `skew_residual` / `fly_butterfly` | −0.0225 | (inside the grid; not the worst) |
| `mm_spread` / `vrp_varswap` | +0.0182 | |
| worst mean: `roll_yield` / `vanna_tilt` | −0.1359 | 0.1531 |
| `dispersion_index` / `cot_fade_sleeve` | +0.1252 | 0.0973 |
| `flow_toxicity` / `gex_forced` | +0.0772 | 0.1789 |

Zero pairs have mean `|ρ|` above 0.25. Zero above 0.40. Seed 11 can still print 0.36 on `roll_yield` / `vanna_tilt`. The mean across paths does not.

## 1.0.0-zig-desk — historical seed 11 (kept)

The table below is the published 1.0 run. It is not what `DeskConfig()` prints now.

Reproduced as residual streams by `legacy_config()` (see the note above on weights). Original command was `python -m jev_omm.demo_surface_desk` on `1.0.0-zig-desk`.

`DeskConfig` at 1.0: seed 11, 80 steps, `fit_stride=20`, products `EQ_INDEX`, `EQ_SINGLE`, `FX_PAIR`, all eight sleeves, correlation cap 0.50, concentration cap 0.40, research fee 0.001 per unit of absolute target, vega column on. Four surface refits, 10 of 36 slices flagged `surface_suspect`.

Products: EQ_INDEX, EQ_SINGLE, FX_PAIR.

Desk raw PnL 0.0480, residual PnL 0.1521, residual Sharpe 0.6449, residual max drawdown 0.0067, weight sum 1.0000.

| sleeve | weight | raw | residual | mean resid | resid Sharpe | R² | β | γ | ν | fills |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mm_spread | 0.0804 | -0.2657 | 0.9151 | 0.0114 | 1.0104 | 0.7431 | 0.9479 | 0.9032 | 1.0126 | 91 |
| skew_residual | 0.0301 | 0.7915 | 0.7363 | 0.0092 | 0.4497 | 0.6200 | 0.2821 | 0.6311 | 1.1536 | 237 |
| calendar_term | 0.1818 | 0.1165 | 0.1681 | 0.0021 | 0.3334 | 0.8860 | 0.1687 | -3.6440 | 1.0447 | 237 |
| fly_butterfly | 0.0425 | 0.5882 | 0.5228 | 0.0065 | 0.4515 | 0.2531 | 0.1036 | 0.6141 | 1.4604 | 237 |
| vrp_varswap | 0.0000 | -0.2333 | -0.2041 | -0.0026 | -0.9177 | 0.9267 | 0.3972 | 1.1478 | 0.9916 | 240 |
| flow_toxicity | 0.1073 | 0.0261 | 0.0478 | 0.0006 | 0.0560 | 0.0117 | 0.1238 | 0.8508 | -0.0189 | 99 |
| gex_forced | 0.1579 | -0.0600 | -0.0896 | -0.0011 | -0.1543 | 0.3480 | 0.5341 | -0.2735 | 0.9785 | 108 |
| parity_box | 0.4000 | 0.0150 | 0.0315 | 0.0004 | 0.1606 | 0.2106 | 0.0421 | -5.2792 | 1.3017 | 237 |

## Pairwise residual Pearson

| | mm_spread | skew_residual | calendar_term | fly_butterfly | vrp_varswap | flow_toxicity | gex_forced | parity_box |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mm_spread | 1.0000 | 0.0062 | -0.0113 | -0.0537 | 0.6295 | 0.0594 | 0.0101 | -0.0494 |
| skew_residual | 0.0062 | 1.0000 | 0.1271 | 0.9301 | 0.0717 | 0.1469 | 0.2062 | -0.2171 |
| calendar_term | -0.0113 | 0.1271 | 1.0000 | 0.0902 | 0.2603 | 0.0077 | 0.3425 | -0.0037 |
| fly_butterfly | -0.0537 | 0.9301 | 0.0902 | 1.0000 | -0.0648 | 0.1385 | 0.1856 | -0.2243 |
| vrp_varswap | 0.6295 | 0.0717 | 0.2603 | -0.0648 | 1.0000 | 0.2511 | 0.2362 | -0.0819 |
| flow_toxicity | 0.0594 | 0.1469 | 0.0077 | 0.1385 | 0.2511 | 1.0000 | 0.0939 | -0.0591 |
| gex_forced | 0.0101 | 0.2062 | 0.3425 | 0.1856 | 0.2362 | 0.0939 | 1.0000 | -0.1031 |
| parity_box | -0.0494 | -0.2171 | -0.0037 | -0.2243 | -0.0819 | -0.0591 | -0.1031 | 1.0000 |

Spearman is computed in the same run and stored on `DeskScoreboard.spearman`. The flag uses Pearson.

## What is orthogonal on this sample

These residual pairs sit under 0.50, most of them well under:

- `parity_box` against every other sleeve (largest magnitude −0.22 versus the fly). The box is the rates sleeve. Equity beta and gamma do not describe it, and it does not tag along with the smile.
- `flow_toxicity` against the book (largest is 0.25 versus variance). Toxicity is a size/spread rule. R² 0.012 says the factor strip had almost nothing to remove.
- `mm_spread` against skew, calendar, fly, flow, gex, and the box. After the greek strip, the spread leftover does not track the smile.
- `calendar_term` against skew and the fly (0.13 and 0.09). The term state is its own shock. It still shares 0.34 with `gex_forced`, under the flag line.
- `gex_forced` against `mm_spread` (0.01). The chase loads on beta in the raw series (β̂ 0.53) and that part comes out in the strip.

## What is not orthogonal

Two pairs are flagged. They are not a surprise, and they are not fixed by adding more of the same greek column.

1. **`skew_residual` / `fly_butterfly`, Pearson 0.93.** Both sleeves fade a state driven by one common smile shock (`0.85` and `0.80` of the same draw, plus a private residual). Beta, gamma, and vega do not span that shock. The fly's factor R² is only 0.25, so most of its PnL is the smile, and that smile is the skew sleeve's PnL. Treating these as two risk budgets would double-count one factor. The allocator shrinks both (`0.50/0.93`) and then inverse-vol keeps the weights small (0.030 and 0.043) because the residual vol is large. They are still both in the book.

2. **`mm_spread` / `vrp_varswap`, Pearson 0.63.** Both are short convexity. The strip removes each sleeve's own gamma bucket (R² 0.74 and 0.93, gamma coefficients 0.90 and 1.15) and still leaves a shared leftover. Product returns are `beta × index + idiosyncratic`, and one gamma column per sleeve cannot soak up every name's squared move with a single coefficient. The variance sleeve's residual mean is negative. Rule 4 of the allocator zeros it because `mm_spread` has the higher residual mean at a correlation above 0.50. Weight on `vrp_varswap` is 0 on this seed. That is the cut working, not a claim that variance and spot-gamma have been separated in general.

## What the weights are doing

Inverse-vol plus the 0.40 cap is why `parity_box` prints weight 0.4000 while its residual Sharpe is 0.16, and why `mm_spread` prints weight 0.0804 while its residual Sharpe is 1.01. The scoreboard is the ranking. The weight vector is a concentration limit. Weight sum is 1.0000, so the cap's excess was handed to sleeves that were still under 0.40. Cash was not required on this seed.

`gex_forced` keeps weight 0.16 with a negative residual mean. Nothing else cleared the 0.50 correlation line against it, so the loser-cut did not fire. A negative sleeve that is not collinear is allowed to stay. That is the rule as written. It is a weak result for the forced-flow sleeve on this path.

## What this file is not

A live-tape correlation, an OPRA surface, or a capacity study. The smiles, the spot factor, and the fills are synthetic. `synthetic_fixture=1` on every fixture row and on the scoreboard. The 1.0 table is the historical sample. The 1.1 numbers are the section at the top. `python -m jev_omm.demo_surface_desk` prints the current book.
