# Residual PnL

**Version:** `1.0.0-zig-desk`  
**Code:** `jev_omm/pnl/residual.py`, `zig/src/desk.zig` (`stripResidual`), `jev_omm/hedge/delta.py` (`greek_pnl_step`).

The research object is the sleeve's period PnL after the comovement with spot and with the greek buckets has been removed. Raw PnL is still reported. A strip that does not explain the variance is reported with a low R². The default seed does that for `flow_toxicity` (R² 0.012).

## Factors

Let `r_t` be the sleeve's raw PnL on step `t`, already after the research fee (`0.001` times absolute target in the harness).

| Factor | Definition | Same expression as |
| --- | --- | --- |
| Beta `f_β` | Index simple return `ΔS_index / S_index` | the equity factor, not the sleeve's own delta |
| Gamma `f_Γ` | `greek_pnl_step` gamma bucket `0.5 Γ (ΔS)²` | `hedge.greekPnlStep` |
| Vega `f_ν` | `ν Δσ`, optional | the vega bucket in the same step |

In the harness, `ΔS` passed into the greek step is the product simple return (research units, S = 1). That is `0.5 Γ S² (ΔS/S)²` at S = 1. Gamma and vega factors are summed across products inside the sleeve, because that is the book's greek PnL. Beta is the single index return, shared by every sleeve.

Theta is inside raw PnL and is not a regression column. It stays in the residual.

## Slopes and the residual

Slopes come from least squares on **demeaned** factors, with ridge `1e-12` on the diagonal of the centered Gram matrix. That is the slope from a regression that includes an intercept. Zig and Python build the same normal equations (`dot(f_i, f_j) − n μ_i μ_j`).

The stored residual does not subtract the intercept:

```
r_resid = r − β̂ f_β − γ̂ f_Γ − ν̂ f_ν
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

## What the desk run actually did

Default config: seed 11, 80 steps, three synthetic names, eight sleeves, vega included. Full table in [ablation_sleeve_corr.md](./ablation_sleeve_corr.md).

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

## Headline versus weights

Residual Sharpe ranks `mm_spread`, then the fly and the skew, then the calendar. The allocator's largest weight is `parity_box` at the 0.40 cap, because its residual volatility is the lowest and the cap then binds. Both facts are on the scoreboard. The desk residual PnL on this seed is 0.152 against raw 0.048, with per-step residual Sharpe 0.64 and residual max drawdown 0.0067. That is one synthetic path.
