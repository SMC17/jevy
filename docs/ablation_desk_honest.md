# Desk honesty ablation

Synthetic paper desk unless a row says otherwise. `synthetic_fixture=1` on synthetic paths. Per-step Sharpe is mean / sample std. It is not annualized. Do not multiply by `√252`. The raw number is not a capacity.

Current desk: `python -m jev_omm.demo_surface_desk` (`1.2.0-zig-honest`). Method: [DESK_HONESTY.md](./DESK_HONESTY.md). The 1.1 table this replaces as the live scoreboard is [ablation_sleeve_corr.md](./ablation_sleeve_corr.md). `legacy_ortho_config()` reprints that book's correlations.

## Seed 11, 80 steps, eight products, twenty sleeves

`fit_surfaces=False`. Desk raw PnL 15.22, residual PnL 12.17, raw residual Sharpe 2.19 (σ already clipped, penalty not yet applied), penalized residual Sharpe 0.42, residual max drawdown 0.685, weight sum 0.831. Pre-gate max `|ρ|` 0.338, post-gate max `|ρ|` 0.338. The 0.40 gate does not fire. Gross turnover 3669, peak |inventory| 2.65, penalized residual PnL per unit turnover 0.0033, per unit peak inventory 4.60. Gamma path length 429, vega path length 52.7, quote revisions per step 114.

The 1.1 desk residual Sharpe on this seed was 6.57, with `flow_toxicity` at weight 0.212 and raw Sharpe 10.65. That weight is now 0.

| sleeve | weight | raw Sharpe | penalized | flag | DC share | turnover | peak inv | cap scale | test kill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mm_spread | 0.0937 | 2.368 | 2.368 | ok | 0.850 | 315.00 | 7.00 | 1.000 | |
| skew_residual | 0.0300 | 0.405 | 0.405 | ok | 0.143 | 84.80 | 3.57 | 1.000 | |
| calendar_term | 0.1124 | 0.406 | 0.406 | ok | 0.143 | 29.85 | 0.68 | 1.000 | |
| fly_butterfly | 0.0300 | 0.219 | 0.219 | ok | 0.046 | 58.68 | 3.04 | 1.000 | |
| vrp_varswap | 0.0300 | 0.144 | 0.144 | ok | 0.021 | 8.00 | 8.00 | 1.000 | |
| flow_toxicity | 0.0000 | 10.650 | 0.000 | flat | 0.991 | 37.01 | 7.84 | 1.000 | |
| gex_forced | 0.0000 | −0.242 | −0.242 | ok | 0.056 | 72.45 | 2.52 | 1.000 | yes |
| parity_box | 0.1522 | 0.199 | 0.199 | ok | 0.038 | 75.79 | 5.01 | 1.000 | |
| wing_kurtosis | 0.0300 | 0.554 | 0.554 | ok | 0.237 | 61.44 | 3.07 | 1.000 | |
| sticky_regime | 0.0242 | 0.181 | 0.181 | ok | 0.032 | 742.72 | 22.28 | 0.808 | |
| dispersion_index | 0.0300 | 0.474 | 0.474 | ok | 0.186 | 81.03 | 4.69 | 1.000 | |
| roll_yield | 0.0000 | −0.112 | −0.112 | ok | 0.013 | 0.23 | 0.01 | 1.000 | yes |
| charm_bleed | 0.0755 | 1.797 | 1.797 | ok | 0.766 | 208.00 | 9.60 | 1.000 | |
| vanna_tilt | 0.0231 | 0.200 | 0.200 | ok | 0.039 | 668.50 | 38.99 | 0.769 | |
| queue_sniper | 0.0219 | 0.199 | 0.199 | ok | 0.038 | 820.36 | 36.10 | 0.731 | |
| cot_fade_sleeve | 0.0300 | 0.244 | 0.244 | ok | 0.057 | 29.43 | 3.74 | 1.000 | |
| warehouse_autocall | 0.0060 | −0.087 | −0.087 | ok | 0.008 | 15.31 | 1.15 | 1.000 | |
| box_rate | 0.1121 | 0.041 | 0.041 | ok | 0.002 | 56.04 | 5.65 | 1.000 | |
| cross_impact | 0.0300 | 0.057 | 0.057 | ok | 0.003 | 84.16 | 3.94 | 1.000 | |
| rough_vol_stress | 0.0000 | −0.232 | −0.232 | ok | 0.052 | 220.14 | 13.65 | 1.000 | yes |

`flow_toxicity` AC(1) is 0.066. The flat flag is the DC share, not the lag. `sticky_regime`, `vanna_tilt`, and `queue_sniper` clear 0.02 after the capacity haircut. `rough_vol_stress` does not: the second-half residual mean is negative. `warehouse_autocall` keeps a small weight with a negative full-sample mean because the test half is positive, so the kill does not fire, and the 0.25 haircut leaves it small. That is the rule.

## Fifteen paths

Seeds `(11, 23, 42, 7, 99)` × `(baseline, smile_shock, jump)`, 80 steps, `fit_surfaces=False`.

| | Mean ρ |
| --- | --- |
| `skew_residual` / `fly_butterfly` | −0.0225 |
| `mm_spread` / `vrp_varswap` | +0.0182 |
| `roll_yield` / `vanna_tilt` | −0.0015 |
| worst mean: `dispersion_index` / `cot_fade_sleeve` | +0.1252 (sample std 0.097) |

The 1.1 worst mean was `roll_yield` / `vanna_tilt` at −0.136. No pair has mean `|ρ|` above 0.25. Seed 11 can still print 0.338 on `vrp_varswap` / `vanna_tilt`, under the 0.35 shrink.

Mean weight after penalty, floor, kill, and capacity scale:

| sleeve | mean weight | mean penalized Sharpe | flags (15 paths) |
| --- | --- | --- | --- |
| sticky_regime | 0.0224 | 0.204 | ok 15 |
| vanna_tilt | 0.0250 | 0.352 | ok 15 |
| queue_sniper | 0.0212 | 0.239 | ok 15 |
| rough_vol_stress | 0.0120 | 0.013 | ok 15 |
| flow_toxicity | 0.0235 | 0.681 | flat 5, smooth 8, ok 2 |

Flow is not flat on every path. Where the residual is not a constant, the penalty is partial and a small weight remains. The mean penalized Sharpe 0.68 is the honest reading of a sleeve whose seed-11 raw Sharpe was 10.65. It is still one synthetic grid.

## 3×8 control

Same seed, 80 steps, products `EQ_INDEX`, `EQ_SINGLE`, `FX_PAIR`, sleeves the original eight. Desk penalized Sharpe 0.33, raw residual Sharpe 1.26, weight sum 0.94, gross turnover 260, peak |inventory| 0.75. Caps do not bind.

| sleeve | weight | raw Sharpe | flag |
| --- | --- | --- | --- |
| mm_spread | 0.1928 | 1.400 | ok |
| skew_residual | 0.0578 | 0.251 | ok |
| calendar_term | 0.2188 | 0.303 | ok |
| fly_butterfly | 0.0665 | 0.265 | ok |
| vrp_varswap | 0.0558 | 0.108 | ok |
| flow_toxicity | 0.0000 | 5.331 | flat |
| gex_forced | 0.0000 | −0.186 | ok |
| parity_box | 0.3500 | 0.161 | ok |

The flow leftover is flat on the smaller book too. `parity_box` sits on the 0.35 cap. `gex_forced` is killed on the test half.

## Adversarial regimes

Five seeds, 40 steps, honesty and capacity on. A row is a survivor when the mean of (residual mean × penalty) is positive and the mean weight is not zero. Survivors are not capacities. `flow_toxicity` on the baseline grid has mean penalty 0.10 and mean weight 0.007 with a small positive penalized mean. That is a penalized leftover, not an edge worth sizing.

| regime | what it does | survivors worth naming | dead, including the usual negative means |
| --- | --- | --- | --- |
| baseline | the 1.2 book | mm, skew, calendar, fly, vrp, charm, dispersion, cross, sticky, vanna, queue, wing, gex, rough. Flow's penalized mean is positive and its weight is ~0.007 | parity, roll, cot, warehouse, box |
| smile_shock | shared smile draw, then residualized | same list as baseline | same five |
| jump | ±0.04 on 5% of steps | baseline list, plus cot and box | parity, roll, warehouse |
| toxic_flow | toxicity pushed toward 1, fills harder | skew, calendar, fly, vrp, wing, sticky, dispersion, charm, vanna, queue, cross, gex, rough. **mm and flow die** | mm, flow, parity, roll, cot, warehouse, box |
| wide_spread | research fee ×6, fills harder | mm, skew, fly, vrp, wing, sticky, dispersion, charm, vanna, queue, cross, rough. **flow and calendar die** | calendar, flow, gex, parity, roll, cot, warehouse, box |
| no_fill | quote is recorded, inventory and PnL stay 0 | none | all twenty |

`sticky_regime`, `vanna_tilt`, and `queue_sniper` keep a positive penalized mean on every regime except `no_fill`. `rough_vol_stress` does too on this 40-step grid, with a small weight. On the 80-step seed-11 path it does not. The longer path is the one in the weight table above.

## Local tape

No licensed history. Two files, both replayed with the existing 0.9 quoters. Posting rule: join the touch when the model is tighter, sit behind when it is wider, never cross. Fee 0.05 per contract on the preview comparison. Not a venue card.

### Synthetic fixture

`jev_omm/data/fixtures/tape_synthetic.csv`. `synthetic_fixture=1`. Symbol `SYNTH`. 96 rows. Spot is present because the file invented it.

| quoter | fills |
| --- | --- |
| fixed_spread | 5 |
| avellaneda_stoikov | 0 |
| gueant_asymptotic | 0 |
| option_vega | 0 |
| as_flow | 0 |
| as_flow_gex | 0 |
| as_flow_gex_state | 0 |
| join_touch | 28 |

Six of eight quoters do not trade. Join-touch trades because it posts the market. That overstates fills: there is no queue ahead.

### Databento public preview

`GET https://api.databento.com/v0/dataset/sample`, `OPRA.PILLAR`, `cbbo-1m`. No API key. sha256 `b8ed8988cd064a35252049a6306f39a39040f54c1c1c8ebee1536a8a96882848`, the same body as [ablation_real_or_fixture.md](./ablation_real_or_fixture.md). 20 rows, symbol `TSLA  230901C00250000`, spot missing (left missing). `synthetic_fixture` is false: this is a vendor preview, not a fixture, and it is not a session. Raw bytes stay in gitignored `data/local/`.

| quoter | fills | pnl |
| --- | --- | --- |
| fixed_spread | 3 | 0.315 |
| avellaneda_stoikov | 0 | 0 |
| gueant_asymptotic | 0 | 0 |
| option_vega | 0 | 0 |
| as_flow | 0 | 0 |
| as_flow_gex | 0 | 0 |
| as_flow_gex_state | 0 | 0 |
| join_touch | 9 | −1.235 |

The model quoters are too wide for a one-to-a-few-cent preview and receive zero fills. Join-touch loses money after the research fee. Same result as the 0.9 write-up. Nothing here was filled in from an equity print.

## What still fails

- `mm_spread` raw per-step Sharpe stays above 2 on seed 11. The smoothness rule does not call it flat. It is a synthetic spread. It is not a capacity.
- `charm_bleed` DC share is 0.77 because the weekend clock is a smooth level in the raw PnL as well as in the residual. The penalty compares the residual with the raw series and leaves it alone. The 1.80 raw Sharpe is the clock. Do not annualize it.
- `flow_toxicity` is not flat on every seed. Eight of fifteen paths are `smooth` rather than `flat`, and the mean weight is 0.024. The penalty cuts the Sharpe. It does not delete the sleeve from the grid.
- `rough_vol_stress` does not clear 0.02 on the fifteen-path mean. Paths with a negative test half are killed. That is intentional.
- Seed 11 still has a pair at 0.338 (`vrp_varswap` / `vanna_tilt`). Under the gate, under the shrink. The fifteen-path mean of that pair is +0.090.
- `no_fill` kills the whole book. A paper sleeve that does not need a fill draw was still booking PnL in 1.1. The regime now refuses that. The other regimes are still synthetic.
- The Databento preview is 20 rows with no underlying. Zero model fills is the result. It is not a backtest.
- Unstandardized PCA remains noise-dominated. Quote the standardized share only.
