# Residual PnL

**Version:** `1.1.0-zig-ortho` for the strip below. The 1.2 smoothness penalty is in [DESK_HONESTY.md](./DESK_HONESTY.md). The raw per-step Sharpe is still mean / sample std. It is not annualized, and a flat leftover is not a capacity.  
**Code:** `jev_omm/pnl/residual.py` (`strip_residual`, `strip_factors`), `zig/src/desk.zig` (`stripResidual`, `stripResidualFactors`), `jev_omm/hedge/delta.py` (`greek_pnl_step`).

The research object is the sleeve's period PnL after the comovement with spot and with the greek buckets has been removed. Raw PnL is still reported. A strip that does not explain the variance is reported with a low R².

## Factors

Let `r_t` be the sleeve's raw PnL on step `t`, already after the research fee (`0.001` times absolute target in the harness).

| Factor | Definition | Same expression as |
| --- | --- | --- |
| Beta `f_β` | Index simple return `ΔS_index / S_index` | the equity factor, not the sleeve's own delta |
| Gamma `f_Γ` | `greek_pnl_step` gamma bucket `0.5 Γ (ΔS)²` | `hedge.greekPnlStep` |
| Vega `f_ν` | `ν Δσ`, optional | the vega bucket in the same step |
| Volga | `0.5 * volga * (Δσ)²`, when `extended_strip` | the volga term booked by `book_higher_greeks` |
| Vanna | `vanna * product return * Δσ` | the same |
| Variance | quadratic variation of product returns, Gram–Schmidt against that sleeve's spot-gamma column | a convexity column that is not a second copy of gamma |

`strip_residual` is the three-column path (beta, gamma, optional vega). `strip_factors` takes up to six columns, C ABI `jev_omm_residual_strip_k`. The old `jev_omm_residual_strip` export is unchanged. Column order on the desk is beta, spot-gamma, vega (or a zero column if vega is off), volga, vanna, variance.

In the harness, `ΔS` passed into the greek step is the product simple return (research units, S = 1). That is `0.5 Γ S² (ΔS/S)²` at S = 1. Greek factors are summed across products inside the sleeve. Beta is the single index return, shared by every sleeve.

Theta is inside raw PnL and is not a regression column. It stays in the residual.

The pairwise gate runs after this strip. It can put a sleeve's residual back onto another sleeve's leftover, which is not orthogonal to this sleeve's greek columns. The harness strips the same columns a second time, keeps the intercept, and if a pair climbs back over the 0.40 gate it repeats, up to four passes. The published residual is that series. The R² on the scoreboard row is the first strip of raw PnL.

## Slopes and the residual

Slopes come from least squares on **demeaned** factors, with ridge `1e-12` on the diagonal of the centered Gram matrix. That is the slope from a regression that includes an intercept. Zig and Python build the same normal equations (`dot(f_i, f_j) − n μ_i μ_j`).

The stored residual does not subtract the intercept:

```
r_resid = r − β̂ f_β − γ̂ f_Γ − ν̂ f_ν − (volga, vanna, variance terms when the six-column strip is on)
```

So `mean(r_resid)` is the intercept: average PnL after the average factor contribution. A constant premium is not eaten by a through-origin fit. A through-origin fit was tried and rejected here because a nonzero factor mean soaks up the constant and the slopes move. The worked example below is the case that shows it.

R² is `1 − SS(r_resid − mean) / SS(r − mean)`. It is the fraction of variance the slopes explain. It is not a claim that the mean was removed. A flat input has R² 0.

`strip_vega=False` drops the vega column. The default desk run keeps it.

## Worked synthetic example

Six steps, planted as `r = 0.5 f_β + 1.0 f_Γ + 0.25 f_ν + 0.01`:

| t | f_β | f_Γ | f_ν |
| --- | --- | --- | --- |
| 0 | 0.010 | 0.0010 | 0.10 |
| 1 | −0.020 | 0.0040 | −0.20 |
| 2 | 0.015 | 0.0002 | 0.00 |
| 3 | 0.000 | 0.0030 | 0.05 |
| 4 | −0.010 | 0.0015 | −0.04 |
| 5 | 0.008 | 0.0004 | 0.02 |

Recovered slopes: `β̂ ≈ 0.500`, `γ̂ ≈ 1.000`, `ν̂ ≈ 0.250`. Mean residual `≈ 0.010`. R² is 1 at printing precision. The residual series is the constant `0.01` up to `1e-6`. Unit tests in `tests/test_desk.py` and `zig/src/desk.zig` lock that.

Gamma on a one-step check: `Γ = 0.04`, `ΔS = 2` gives `0.5 * 0.04 * 4 = 0.08`, equal to `greek_pnl_step`.

## What the 1.0 desk run did

That config was seed 11, 80 steps, three synthetic names, eight sleeves, vega included, no volga or vanna column. Full table in [ablation_sleeve_corr.md](./ablation_sleeve_corr.md). The readings below are that historical sample.

| Sleeve | R² | Reading on this sample |
| --- | --- | --- |
| `vrp_varswap` | 0.927 | The short-variance leg is the squared move plus vega. The strip explains most of the variance. Residual mean stays negative, and the allocator cuts the sleeve because that mean is worse than `mm_spread` at residual correlation 0.63. |
| `calendar_term` | 0.886 | Raw variance is mostly the vega column (ν̂ ≈ 1.04). The gamma coefficient is unstable (−3.6) because that column is small. Residual Sharpe per step is 0.33, which is the term fade after vega. |
| `mm_spread` | 0.743 | Short gamma and the delta leak are real. β̂ ≈ 0.95, γ̂ ≈ 0.90, ν̂ ≈ 1.01. Residual Sharpe per step is about 1.0 because the spread is left in the mean. Inverse-vol still gives it weight 0.08: the residual is volatile. |
| `skew_residual` | 0.620 | Part vega, part a small gamma column. The smile residual remains. |
| `gex_forced` | 0.348 | Some index beta (β̂ ≈ 0.53). The chase does not pay on this seed (residual Sharpe −0.15). |
| `fly_butterfly` | 0.253 | Most of the curvature PnL is not beta, gamma, or vega. It lines up with the skew sleeve instead. |
| `parity_box` | 0.211 | The box is only weakly related to the equity factors. The gamma coefficient (−5.3) is an unstable small column, not a measured rate exposure. |
| `flow_toxicity` | 0.012 | Beta, gamma, and vega do not explain this stream. Say so. Per-step residual Sharpe is 0.06. |

Per-step Sharpe is `mean / sample std` of the residual. It is not annualized. Multiplying by `√252` would invent a track record. These are 80 synthetic days.

## Headline versus weights (1.0 sample)

Residual Sharpe ranks `mm_spread`, then the fly and the skew, then the calendar. The allocator's largest weight is `parity_box` at the 0.40 cap, because its residual volatility is the lowest and the cap then binds. Both facts are on the scoreboard. The desk residual PnL on this seed is 0.152 against raw 0.048, with per-step residual Sharpe 0.64 and residual max drawdown 0.0067. That is one synthetic path.

## What the 1.1 strip changes on the same seed

Eight products, twenty sleeves, six-column strip, seed 11, 80 steps. The full weight table is in the ablation. Readings that the extra columns and the convexity split actually move:

| Sleeve | R² | Reading on this sample |
| --- | --- | --- |
| `calendar_term` | 0.960 | Still mostly vega. The extra columns do not create a term residual. |
| `flow_toxicity` | 0.636 | The 1.0 R² was 0.012. Higher-greek booking and the hardened flow path put variance into the strip. What remains is a smooth spread, per-step residual Sharpe 10.65. That is the sample mean over a small sample std. It is not an annualized result. |
| `mm_spread` | 0.587 | Inventory gamma is hedged down. Residual Sharpe per step is 2.37. Weight 0.116, not the cap. |
| `vrp_varswap` | 0.025 | The premium is no longer the squared spot move, so the strip explains little. Weight 0.0017. Correlation with `mm_spread` on this seed is +0.108, and the fifteen-path mean is +0.018. |
| `fly_butterfly` | 0.099 | Curvature is its own draw. Correlation with skew on this seed is +0.027. The 1.0 figure was 0.930. |
| `charm_bleed` | 0.004 | Almost none of the weekend bleed is in the greek columns. Weight 0.091, per-step residual Sharpe 1.80. |

R² above 0.70 means most of the variance sits in the factor strip. R² under 0.15 means the strip is not the story. Both sentences are on the scoreboard when they apply. The desk residual PnL on this path is 4.143 against raw 0.814, per-step residual Sharpe 6.57, residual max drawdown 0. The Sharpe is the flow sleeve plus a few quiet positive means. One synthetic path.

## What 1.2 does with that 10.65

The 10.65 is still the raw number on seed 11. `flow_toxicity`'s residual has DC share 0.991 against a raw-series DC share of 0.050: the strip left a near-constant. AC(1) of the demeaned residual is only 0.066, so a lag-1 test alone does not see it. The penalty uses the uncentered DC bin, which is the constant. On seed 11 the flag is `flat`, the penalty is 0, and the weight is 0. The penalized Sharpe is 0. The raw 10.65 stays on the scoreboard so the leftover is visible. It is not a capacity. Across the fifteen-path grid the same sleeve is flat on 5 paths, smooth on 8, and unmarked on 2. Mean penalized per-step Sharpe is 0.68. Mean weight is 0.024.
