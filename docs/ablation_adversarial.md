# Adversarial synthetic ablations

Research laboratory output. Every price is invented (`SYNTHETIC_FIXTURE=1`). Five seeds (7, 11, 19, 23, 29). Each path is 64 one-minute bars. The book is a Black–Scholes call on a GBM spot, then a regime distortion. This is not OPRA and not a walk-forward on a licensed tape.

Posting rule matches the tape replay: join the touch when the model is tighter, sit behind when it is wider, never cross. The next bar's trade fills us with no queue ahead, which overstates touch fills. Crossed, locked, and stale rows are not quoted. Fee is a research charge, not a venue card. Baseline fee is 0.02 per contract. The `fees` regime uses the baseline book with fee 0.20 and rebate 0.05.

`mean_gap_vs_fixed` is the paired PnL gap against fixed-spread on the same seed. A row **earns** only when that mean gap is positive and larger than the seed std of the gap. **behind** means the mean gap is negative and larger in magnitude than the seed std. Anything else **does not earn** — including a higher mean that is inside the seed noise.

σ = 0.45 and κ = 1.5 are the 0.8 ablation knobs. Feature pack `off` is the identity. GEX uses the invented spot; it is not dealer gamma.

## baseline

Book `baseline`, fee 0.02, rebate 0.00, seeds 5.

| label | mean_pnl | std_pnl | mean_gap_vs_fixed | std_gap | mean_markout_1 | mean_fills | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_spread | 2.0667 | 0.4704 | 0.0000 | 0.0000 | 0.0125 | 8.6000 | reference |
| avellaneda_stoikov | 0.0000 | 0.0000 | -2.0667 | 0.4704 | 0.0000 | 0.0000 | behind |
| gueant_asymptotic | 0.0000 | 0.0000 | -2.0667 | 0.4704 | 0.0000 | 0.0000 | behind |
| option_vega | 0.0000 | 0.0000 | -2.0667 | 0.4704 | 0.0000 | 0.0000 | behind |
| as_flow | 0.0000 | 0.0000 | -2.0667 | 0.4704 | 0.0000 | 0.0000 | behind |
| as_flow_gex | 0.0000 | 0.0000 | -2.0667 | 0.4704 | 0.0000 | 0.0000 | behind |
| as_flow_gex_state | 0.0000 | 0.0000 | -2.0667 | 0.4704 | 0.0000 | 0.0000 | behind |
| join_touch | 0.0259 | 0.9932 | -2.0408 | 1.2404 | -0.5558 | 32.0000 | behind |

No model earns a gap over fixed-spread outside seed noise.
Behind fixed-spread: avellaneda_stoikov, gueant_asymptotic, option_vega, as_flow, as_flow_gex, as_flow_gex_state, join_touch.

## crossed_locked

Book `crossed_locked`, fee 0.02, rebate 0.00, seeds 5.

| label | mean_pnl | std_pnl | mean_gap_vs_fixed | std_gap | mean_markout_1 | mean_fills | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_spread | 1.6288 | 0.6803 | 0.0000 | 0.0000 | 0.0289 | 6.6000 | reference |
| avellaneda_stoikov | 0.0000 | 0.0000 | -1.6288 | 0.6803 | 0.0000 | 0.0000 | behind |
| gueant_asymptotic | 0.0000 | 0.0000 | -1.6288 | 0.6803 | 0.0000 | 0.0000 | behind |
| option_vega | 0.0000 | 0.0000 | -1.6288 | 0.6803 | 0.0000 | 0.0000 | behind |
| as_flow | 0.0000 | 0.0000 | -1.6288 | 0.6803 | 0.0000 | 0.0000 | behind |
| as_flow_gex | 0.0000 | 0.0000 | -1.6288 | 0.6803 | 0.0000 | 0.0000 | behind |
| as_flow_gex_state | 0.0000 | 0.0000 | -1.6288 | 0.6803 | 0.0000 | 0.0000 | behind |
| join_touch | 0.0357 | 0.8595 | -1.5932 | 1.3025 | -0.3944 | 25.2000 | behind |

No model earns a gap over fixed-spread outside seed noise.
Behind fixed-spread: avellaneda_stoikov, gueant_asymptotic, option_vega, as_flow, as_flow_gex, as_flow_gex_state, join_touch.

## stale

Book `stale`, fee 0.02, rebate 0.00, seeds 5.

| label | mean_pnl | std_pnl | mean_gap_vs_fixed | std_gap | mean_markout_1 | mean_fills | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_spread | 2.3841 | 0.6261 | 0.0000 | 0.0000 | 0.0674 | 11.6000 | reference |
| avellaneda_stoikov | 0.0000 | 0.0000 | -2.3841 | 0.6261 | 0.0000 | 0.0000 | behind |
| gueant_asymptotic | 0.0000 | 0.0000 | -2.3841 | 0.6261 | 0.0000 | 0.0000 | behind |
| option_vega | 0.0000 | 0.0000 | -2.3841 | 0.6261 | 0.0000 | 0.0000 | behind |
| as_flow | 0.0000 | 0.0000 | -2.3841 | 0.6261 | 0.0000 | 0.0000 | behind |
| as_flow_gex | 0.0000 | 0.0000 | -2.3841 | 0.6261 | 0.0000 | 0.0000 | behind |
| as_flow_gex_state | 0.0000 | 0.0000 | -2.3841 | 0.6261 | 0.0000 | 0.0000 | behind |
| join_touch | 0.0249 | 1.1288 | -2.3591 | 0.9518 | -0.4063 | 30.6000 | behind |

No model earns a gap over fixed-spread outside seed noise.
Behind fixed-spread: avellaneda_stoikov, gueant_asymptotic, option_vega, as_flow, as_flow_gex, as_flow_gex_state, join_touch.

## one_sided

Book `one_sided`, fee 0.02, rebate 0.00, seeds 5.

| label | mean_pnl | std_pnl | mean_gap_vs_fixed | std_gap | mean_markout_1 | mean_fills | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_spread | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | reference |
| avellaneda_stoikov | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | does not earn |
| gueant_asymptotic | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | does not earn |
| option_vega | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | does not earn |
| as_flow | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | does not earn |
| as_flow_gex | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | does not earn |
| as_flow_gex_state | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | does not earn |
| join_touch | -0.9335 | 1.2145 | -0.9335 | 1.2145 | -0.1806 | 6.0000 | does not earn |

No model earns a gap over fixed-spread outside seed noise.
Does not earn (gap inside seed noise, or tied): avellaneda_stoikov, gueant_asymptotic, option_vega, as_flow, as_flow_gex, as_flow_gex_state, join_touch.

## fees

Book `baseline`, fee 0.20, rebate 0.05, seeds 5.

| label | mean_pnl | std_pnl | mean_gap_vs_fixed | std_gap | mean_markout_1 | mean_fills | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_spread | 0.9487 | 0.5271 | 0.0000 | 0.0000 | 0.0125 | 8.6000 | reference |
| avellaneda_stoikov | 0.0000 | 0.0000 | -0.9487 | 0.5271 | 0.0000 | 0.0000 | behind |
| gueant_asymptotic | 0.0000 | 0.0000 | -0.9487 | 0.5271 | 0.0000 | 0.0000 | behind |
| option_vega | 0.0000 | 0.0000 | -0.9487 | 0.5271 | 0.0000 | 0.0000 | behind |
| as_flow | 0.0000 | 0.0000 | -0.9487 | 0.5271 | 0.0000 | 0.0000 | behind |
| as_flow_gex | 0.0000 | 0.0000 | -0.9487 | 0.5271 | 0.0000 | 0.0000 | behind |
| as_flow_gex_state | 0.0000 | 0.0000 | -0.9487 | 0.5271 | 0.0000 | 0.0000 | behind |
| join_touch | -4.1341 | 1.0915 | -5.0828 | 1.2273 | -0.5558 | 32.0000 | behind |

No model earns a gap over fixed-spread outside seed noise.
Behind fixed-spread: avellaneda_stoikov, gueant_asymptotic, option_vega, as_flow, as_flow_gex, as_flow_gex_state, join_touch.

## thin

Book `thin`, fee 0.02, rebate 0.00, seeds 5.

| label | mean_pnl | std_pnl | mean_gap_vs_fixed | std_gap | mean_markout_1 | mean_fills | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_spread | 2.4477 | 0.6358 | 0.0000 | 0.0000 | -0.2830 | 13.0000 | reference |
| avellaneda_stoikov | 2.6058 | 0.8938 | 0.1581 | 1.4208 | -0.1340 | 4.8000 | does not earn |
| gueant_asymptotic | 2.2299 | 0.7127 | -0.2179 | 1.0719 | -0.1470 | 4.0000 | does not earn |
| option_vega | 1.3475 | 0.3062 | -1.1003 | 0.6767 | -0.1298 | 2.2000 | behind |
| as_flow | 0.0000 | 0.0000 | -2.4477 | 0.6358 | 0.0000 | 0.0000 | behind |
| as_flow_gex | 0.0000 | 0.0000 | -2.4477 | 0.6358 | 0.0000 | 0.0000 | behind |
| as_flow_gex_state | 0.0000 | 0.0000 | -2.4477 | 0.6358 | 0.0000 | 0.0000 | behind |
| join_touch | 4.9907 | 1.7951 | 2.5429 | 1.2941 | -0.6128 | 32.2000 | earns |

Earns versus fixed-spread: join_touch.
Behind fixed-spread: option_vega, as_flow, as_flow_gex, as_flow_gex_state.
Does not earn (gap inside seed noise, or tied): avellaneda_stoikov, gueant_asymptotic.

## jumps

Book `jumps`, fee 0.02, rebate 0.00, seeds 5.

| label | mean_pnl | std_pnl | mean_gap_vs_fixed | std_gap | mean_markout_1 | mean_fills | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_spread | -0.2855 | 4.2908 | 0.0000 | 0.0000 | -3.4972 | 10.4000 | reference |
| avellaneda_stoikov | -0.4189 | 0.8764 | -0.1334 | 4.0780 | -3.9257 | 2.2000 | does not earn |
| gueant_asymptotic | -0.4215 | 0.8708 | -0.1360 | 4.0788 | -3.9257 | 2.2000 | does not earn |
| option_vega | -0.3698 | 0.8731 | -0.0843 | 4.0785 | -3.9257 | 2.2000 | does not earn |
| as_flow | -0.1293 | 0.9051 | 0.1562 | 4.0646 | -3.9257 | 2.2000 | does not earn |
| as_flow_gex | 0.7143 | 1.0273 | 0.9998 | 4.0449 | -3.5848 | 2.0000 | does not earn |
| as_flow_gex_state | 0.2475 | 0.9676 | 0.5330 | 4.8092 | -0.7818 | 0.4000 | does not earn |
| join_touch | -1.0215 | 7.6007 | -0.7360 | 8.3994 | -2.2900 | 34.0000 | does not earn |

No model earns a gap over fixed-spread outside seed noise.
Does not earn (gap inside seed noise, or tied): avellaneda_stoikov, gueant_asymptotic, option_vega, as_flow, as_flow_gex, as_flow_gex_state, join_touch.

Join-touch is a reference posting rule. A loss there means the invented flow was toxic at the touch after the research fee. Beating it is not live alpha.
