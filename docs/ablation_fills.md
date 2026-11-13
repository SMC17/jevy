# Fill ablation

`1.3.0-zig-fills`. Method: [EXECUTABLE_FILLS.md](./EXECUTABLE_FILLS.md). Per-step residual Sharpe is not in these tables. Fill PnL is cash after the research fee, marked to the last mid, plus the one-row markout column. It is not annualized.

`join_rate` is the share of quoted steps with at least one side at or inside the touch. The print rule does not record it. `mean_half` is the half-spread actually posted.

Fee 0.05 per contract on the tapes. Not a venue card.

## Synthetic fixture

`jev_omm/data/fixtures/tape_synthetic.csv`. `synthetic_fixture=1`. Symbol `SYNTH`. 96 rows, train ends at 48. Spot provenance `underlying_print`. Quote mid provenance `cbbo`. Train touch 0.0474. Print rate 0.0341 contracts/second. Executable κ 21.0390. Hazard κ 1.5000 (`moments_k_held`).

### Print rule (the 1.2 zeros)

| label | pnl | n_fills | contracts | fees | markout_1 | mean_half | quote_uptime |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_spread | 1.0944 | 5 | 5 | 0.2500 | 0.1194 | 0.2500 | 1.0000 |
| avellaneda_stoikov | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.6414 | 1.0000 |
| gueant_asymptotic | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.6477 | 1.0000 |
| option_vega | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.6705 | 1.0000 |
| as_flow | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.7825 | 1.0000 |
| as_flow_gex | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 1.0458 | 1.0000 |
| as_flow_gex_state | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 1.6117 | 0.4043 |
| join_touch | -0.0626 | 28 | 28 | 1.4000 | -0.5620 | 0.0474 | 1.0000 |

Same fills as [ablation_desk_honest.md](./ablation_desk_honest.md). The 28 join-touch fills have no queue in front of them.

### LOB, touch units, queue edge on

| label | pnl | n_fills | contracts | fees | markout_1 | mean_half | join_rate | quote_uptime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_spread | 1.0944 | 5 | 5 | 0.2500 | 0.1194 | 0.2500 | 0.0000 | 1.0000 |
| avellaneda_stoikov | -0.2485 | 26 | 26 | 1.3000 | -0.5545 | 0.0474 | 0.3778 | 0.9574 |
| gueant_asymptotic | -0.1082 | 28 | 28 | 1.4000 | -0.5586 | 0.0499 | 0.7174 | 0.9787 |
| option_vega | -0.5167 | 26 | 26 | 1.3000 | -0.5141 | 0.0527 | 0.6957 | 0.9787 |
| as_flow | 0.1460 | 24 | 24 | 1.2000 | -0.5407 | 0.0578 | 0.0652 | 0.9787 |
| as_flow_gex | 0.5748 | 14 | 14 | 0.7000 | -0.2487 | 0.0774 | 0.0000 | 1.0000 |
| as_flow_gex_state | 0.4034 | 5 | 5 | 0.2500 | 0.0908 | 0.1281 | 0.0000 | 0.4043 |
| join_touch | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.0474 | 1.0000 | 1.0000 |

Avellaneda–Stoikov, Guéant, and option-vega all trade. None beat fixed-spread on PnL. Their markouts are negative. Join-touch joins every quoted step and fills nothing, because the print does not clear displayed size. The same LOB book with executable units off (κ = 1.5, half-spread 0.64) gives Avellaneda–Stoikov 0 fills. The unit map is what creates the fills. Turning it off is the identity on κ.

`as_flow_gex` PnL 0.5748 is still below fixed-spread 1.0944. Spot is present, so GEX is allowed to widen the quote. That is a feature pack, not a new model.

## Databento OPRA.PILLAR cbbo-1m

Public sample, no API key. sha256 `b8ed8988cd064a35252049a6306f39a39040f54c1c1c8ebee1536a8a96882848`. 20 rows, symbol `TSLA  230901C00250000`, train ends at 10. `synthetic_fixture` is false. Spot provenance `absent`. Quote mid provenance `cbbo`. Train touch 0.0125. Print rate 0.0283 contracts/second. Executable κ 79.9400. Hazard κ 1.5000.

### Print rule

| label | pnl | n_fills | markout_1 | mean_half |
| --- | --- | --- | --- | --- |
| fixed_spread | 0.3150 | 3 | -0.6750 | 0.2500 |
| avellaneda_stoikov | 0.0000 | 0 | 0.0000 | 0.6414 |
| gueant_asymptotic | 0.0000 | 0 | 0.0000 | 0.6477 |
| option_vega | 0.0000 | 0 | 0.0000 | 0.6705 |
| as_flow | 0.0000 | 0 | 0.0000 | 0.7839 |
| as_flow_gex | 0.0000 | 0 | 0.0000 | 0.7839 |
| as_flow_gex_state | 0.0000 | 0 | 0.0000 | 0.6414 |
| join_touch | -1.2350 | 9 | -1.1600 | 0.0089 |

### LOB, touch units, queue edge on

| label | pnl | n_fills | contracts | fees | markout_1 | mean_half | join_rate | quote_uptime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_spread | 0.3150 | 3 | 3 | 0.1500 | -0.6750 | 0.2500 | 0.0000 | 1.0000 |
| avellaneda_stoikov | -0.9025 | 7 | 7 | 0.3500 | -0.9650 | 0.0125 | 0.0000 | 0.8889 |
| gueant_asymptotic | -0.1213 | 4 | 4 | 0.2000 | -0.5550 | 0.0160 | 0.0000 | 1.0000 |
| option_vega | -0.0774 | 4 | 4 | 0.2000 | -0.5550 | 0.0288 | 0.0000 | 1.0000 |
| as_flow | -1.1280 | 8 | 8 | 0.4000 | -1.1600 | 0.0153 | 0.0000 | 1.0000 |
| as_flow_gex | -1.1280 | 8 | 8 | 0.4000 | -1.1600 | 0.0153 | 0.0000 | 1.0000 |
| as_flow_gex_state | -0.1425 | 1 | 1 | 0.0500 | -0.3000 | 0.0125 | 0.0000 | 0.1111 |
| join_touch | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.0089 | 1.0000 | 1.0000 |

N is 10 test rows. Every theory row loses to fixed-spread. Fixed-spread's cash PnL is positive and its one-row markout is −0.675; the cash result is not a clean markout win. `as_flow` equals `as_flow_gex` because spot is absent and GEX is the identity. Join-touch again fills 0 once displayed size is in the way. No equity print was joined in to fill the missing spot.

## Queue edge, eight seeds

Stochastic LOB, not the tape. Intensity and horizon are per second.

| case | edge | mean fills | mean markout |
| --- | --- | --- | --- |
| toxic, ahead 0, 8 contracts/s | off | 1.000 | −0.0300 |
| toxic, ahead 0, 8 contracts/s | on | 0.625 | −0.0188 |
| ahead 50, 5 contracts/s, toxic 0 | off | 0.000 | +0.0000 |
| ahead 50, 5 contracts/s, toxic 0 | on | 1.000 | +0.0500 |

## Paper desk, seed 11, 40 steps

`DeskConfig(n_steps=40, seed=11, fit_surfaces=False)`. `synthetic_fixture=1`. Lean book: parity, roll, COT, warehouse, and box are off. Fill PnL is not inside the residual. Residual is the 1.2 object on a shorter path than the honesty table, so the residual column is not a reprint of that table.

| sleeve | weight | lob fills, edge on | lob fills, edge off | adverse, on | adverse, off | fill pnl | join rate | residual pnl | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mm_spread | 0.1903 | 362.13 | 485.91 | −3.793 | −8.413 | 2.607 | 0.675 | 1.495 | 141.0 |
| sticky_regime | 0.0300 | 362.13 | 485.91 | −3.793 | −8.413 | 2.607 | 0.675 | 47.738 | 316.0 |
| vanna_tilt | 0.0300 | 362.13 | 485.91 | −3.793 | −8.413 | 2.607 | 0.675 | 49.157 | 277.3 |
| queue_sniper | 0.0300 | 208.00 | 0.00 | −1.385 | 0.000 | 0.487 | 0.925 | 115.062 | 356.1 |
| charm_bleed | 0.1509 | 362.13 | 485.91 | −3.793 | −8.413 | 2.607 | 0.675 | 1.488 | 105.6 |
| flow_toxicity | 0.0527 | 43.98 | 43.98 | −0.158 | −0.158 | 0.727 | 0.000 | 3.818 | 28.3 |

`mm_spread`, `sticky_regime`, `vanna_tilt`, and `charm_bleed` post the same research touch, so they share a fill count. The queue edge cancels toxic steps: fewer fills, less negative markout. `queue_sniper` fills only when the edge improves a queue that does not clear; off is zero. `flow_toxicity` sits behind its widened spread, so the edge does not change its fills, and it fills less than `mm_spread`. Its markout is still charged.

Weighted desk fill PnL on this path is 2.07. Unweighted adverse markout is −47.06. Join rate is 0.60. None of those numbers is a capacity.

## What still loses

- On both tapes, every theory row's test PnL is below fixed-spread. Getting fills did not produce a markout win.
- Join-touch with a queue fills nothing on these files. The old 28 and 9 were the no-queue assumption.
- Sleeves that post the same touch share a fill. The LOB column does not invent a separate book per sleeve.
- The preview is 20 rows with no underlying. It is not a session.
