# Orthogonality

**Version:** `1.1.0-zig-ortho`, kept as the record of that pass. The current desk is `1.2.0-zig-honest`. See [DESK_HONESTY.md](./DESK_HONESTY.md).  
**Code:** `jev_omm/desk/orthogonal.py`, `jev_omm/desk/harness.py`, `jev_omm/desk/sleeves.py`.  
**Mode:** paper. Every path below is `synthetic_fixture=1`. No live tape, no order.

`legacy_ortho_config()` still reprints the seed-11 correlations in this file, including `roll_yield` / `vanna_tilt` at −0.363. The 1.2 split takes that pair's fifteen-path mean from −0.136 to −0.0015. The gate thresholds did not move.

`1.0.0-zig-desk` left two residual pairs that were the same economic factor counted twice. This layer splits those factors in the simulator, then runs a hard gate on whatever correlation is still there. The gate is a research control on a synthetic book. It is not a covariance model for a live desk.

## What 1.0 actually shared

On seed 11, 80 steps, three names, eight sleeves, the published residual Pearson values were:

| Pair | Pearson | Why |
| --- | --- | --- |
| `skew_residual` / `fly_butterfly` | 0.9301 | Both faded one smile draw (0.85 and 0.80 of the same shock). Beta, gamma, and vega do not span that shock. |
| `mm_spread` / `vrp_varswap` | 0.6295 | Both were short the same-bar squared move. One gamma column per sleeve did not separate inventory gamma from a variance premium. |

`legacy_config()` still plants that economy (`factorize_smile=False`, `separate_convexity=False`, no gate, caps 0.50 / 0.40). Re-running it reprints those two correlations and the same per-sleeve residual totals as [ablation_sleeve_corr.md](./ablation_sleeve_corr.md). Weights on that replay are not the 1.0 weight vector: a negative residual mean is now multiplied by 0.25 before the cap, so `gex_forced` shrinks and the positive sleeves take the slack. `parity_box` still sits on the 0.40 cap in that replay.

## Factor split

Two flags, both on by default.

**Smile.** Skew, butterfly, and wing are separate draws. In the `smile_shock` regime a shared draw is planted and then Gram–Schmidt residualized before a sleeve trades it. Funding is residualized against the box so `box_rate` is not a second copy of `parity_box`.

**Convexity.** With `separate_convexity`, the market-maker inventory gamma is hedged down and the leftover delta leak is `0.06 * beta`. The variance sleeve's alpha is `(iv² − lagged realized variance) * dt` plus its own premium shock. Its spot gamma is 0. The vega it books is the flat-smile variance-swap vega from `pricing/varswap.py` (Carr–Wu, [doi:10.1093/rfs/hhn039](https://doi.org/10.1093/rfs/hhn039)). Turning the flag off restores the 1.0 short-gamma formulas.

Same eight sleeves, same three products, seed 11, 80 steps, surfaces not refit:

| Pair | Legacy ρ | After the split |
| --- | --- | --- |
| `skew_residual` / `fly_butterfly` | 0.9301 | 0.0237 |
| `mm_spread` / `vrp_varswap` | 0.6295 | −0.0394 |

Pre-gate max `|ρ|` on that book is 0.273. The gate does not fire. `parity_box` weight is 0.350, the new concentration cap.

## Hard gate

After the greek strip, `enforce_orthogonality` looks at enabled residual pairs.

1. The worst pair with `|ρ| > 0.40` is residualized: the weaker absolute per-step Sharpe against the stronger. The mean of the series that is residualized stays. A flat leftover has no sample variance and the allocator gives it weight 0.
2. If `|ρ| ≥ √0.5` (more than half the variance is shared), the weaker sleeve is merged: its series is zeroed and it is disabled.
3. If a residualized pair is still above 0.40, that sleeve is cut the same way.

The projection is against another sleeve's residual, which is orthogonal to that sleeve's greeks and not to this sleeve's. The harness therefore runs the factor strip again on each enabled residual, same columns, intercept kept. If that second strip pushes a pair back over 0.40, the gate runs again, up to four passes. The scoreboard residual is the series after that loop. Factor R² on the row is still the first strip of raw PnL.

PCA in the scoreboard note is a research summary on the residual panel. Columns are standardized, so the share is of correlation variance. The unstandardized sum of squares is also printed, because a few noisy sleeves dominate the raw scale. It is not a live risk model.

## Multi-seed, not seed 11

`EVAL_SEEDS = (11, 23, 42, 7, 99)` and `EVAL_REGIMES = (baseline, smile_shock, jump)`. `jump` adds a ±0.04 move on 5% of steps from a second RNG (`seed + 101`) so the 1.0 shock stream is unchanged. `multi_seed_corr` uses `n_steps=80` and `fit_surfaces=False` (sleeve PnL does not read the SVI fit). Fifteen paths, twenty sleeves:

| | Mean residual ρ |
| --- | --- |
| `skew_residual` / `fly_butterfly` | −0.0225 |
| `mm_spread` / `vrp_varswap` | +0.0182 |
| Worst mean (any pair) | −0.1359 (`roll_yield` / `vanna_tilt`, sample std 0.153) |

No pair has mean `|ρ|` above 0.25 on that grid, and none above 0.40. The next means are `dispersion_index` / `cot_fade_sleeve` at +0.125 and `parity_box` / `rough_vol_stress` at −0.107. Sample standard deviations are the same order as the means. That is fifteen synthetic paths, not a confidence interval for a live book.

## What still lines up

Seed 11 alone, full twenty-sleeve book, 80 steps:

- `roll_yield` / `vanna_tilt` prints −0.363. That is under the 0.40 gate, so the gate is a no-op, and over the 0.35 allocator shrink, so both weights are scaled by `0.35 / 0.363`. Across the fifteen-path grid the same pair's mean is −0.136. One seed is not the book.
- `flow_toxicity` / `gex_forced` mean ρ is +0.077 with sample std 0.179. The mean is small. A single path can still be large. The allocator shrinks a path when that path clears 0.35.
- Standardized PCA on the seed-11 residual panel: the first four components explain 0.368 of correlation variance. Unstandardized, the same four explain 0.992, because `queue_sniper`, `vanna_tilt`, and `sticky_regime` dominate the sum of squares and then receive almost no inverse-vol weight. Pairwise `|ρ|` and a raw PCA answer different questions. Quote the standardized one.

Sleeves that are economically adjacent were not cloned into a second name. Dispersion is one AR factor, vega 0, booked only on `EQ_INDEX`. Cross-impact is one hub book on `EQ_INDEX`. Roll yield is commodity and rates only. Charm and COT use different clocks (`t % 5 == 1` and `t % 5 == 4`) so they do not share a spike.

## What this file is not

A claim that twenty sleeves are twenty independent risks in a market. A capacity study. An annualized Sharpe. The per-step residual Sharpe on the full synthetic book is high because the greek strip leaves a smooth spread in `flow_toxicity`. That number is in the ablation. Do not multiply it by `√252`.
