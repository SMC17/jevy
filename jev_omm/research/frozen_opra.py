"""Frozen aggregates from the 2026-09-28 OPRA.PILLAR cbbo-1m preview.

Raw rows are not stored here. Regenerate with scripts/fetch_databento_sample.py
and jev_omm.research.tape_walk.render_walk_section if the sha256 changes.
"""

SECTION = """## OPRA.PILLAR public sample, schema cbbo-1m

- Source: `databento-sample:OPRA.PILLAR:cbbo-1m`
- Synthetic fixture: no
- Symbols: TSLA  230901C00250000
- Rows: 20 (train ends at index 10, test rows 10)
- Underlying spot: absent (not filled in)
- Book labels (whole file): two-sided 20, crossed 0, locked 0, stale 0, one-sided 0, missing 0
- Research fee 0.0500 per contract, rebate 0.0000. Not a venue fee card.
- Train σ of the option mid (annualized, reported only): 72.6119 via `train_mids`.
- σ used by the quoters: 0.45 ($/√yr, the 0.8 ablation knob). The train annualization is not substituted.
- Hazard: A=0.005986/s at δ=0, k=1.5000, method `moments_k_held`, MLE A=7.318e-03, MLE k=14.2020, loglik=-14.2501 on 20 touch-sides.
- Kappa on the test window: 1.5000.

Test-window replay. `mean_half` is the half-spread we actually posted (after joining or sitting behind the market). `markout_1` is one tape row after the fill, signed size times the mid change. `join_touch` posts the market itself and is a reference, not a model.

| label | pnl | n_fills | contracts | fees | rebates | markout_1 | mean_half | quote_uptime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_spread | 0.3150 | 3.0000 | 3.0000 | 0.1500 | 0.0000 | -0.6750 | 0.2500 | 1.0000 |
| avellaneda_stoikov | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6414 | 1.0000 |
| gueant_asymptotic | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6477 | 1.0000 |
| option_vega | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6705 | 1.0000 |
| as_flow | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7839 | 1.0000 |
| as_flow_gex | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7839 | 1.0000 |
| as_flow_gex_state | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6414 | 0.1111 |
| join_touch | -1.2350 | 9.0000 | 9.0000 | 0.4500 | 0.0000 | -1.1600 | 0.0089 | 1.0000 |

Uncalibrated A–S (kappa held at 1.5) on the same test rows: PnL 0.0000, fills 0.0000, mean half 0.6414.

Join-touch PnL -1.2350 does not beat fixed-spread 0.3150 after the research fee. The touch is not free edge on this window. Below fixed-spread on this single draw: avellaneda_stoikov, gueant_asymptotic, option_vega, as_flow, as_flow_gex, as_flow_gex_state. No theory row beat fixed-spread on this draw. That is the result. Fitted kappa matched the prior on the test window (the train window did not identify a different k, or it did not move PnL). Fixed-spread terminal PnL is 0.3150 while one-step markout is -0.6750. The cash result is not a clean markout win.

GEX was not applied. Spot is absent, `allow_gex` is false, and that channel is the identity. `as_flow` and `as_flow_gex` match for that reason. State can still pull quotes; a low `quote_uptime` on `as_flow_gex_state` is the gate, not a GEX effect.

With γ = 0.12 and κ = 1.5 the dominant half-spread term is (1/γ) ln(1 + γ/κ) ≈ 0.64 dollars, before the market join/behind clip. A preview NBBO of one to a few cents does not trade against that quote. Fixed-spread is clamped at 0.25 and can catch a sweep the wider quotes miss. That is the formula at these knobs, not a claim that 0.25 is optimal.

Public Databento sample preview (no account). Not a full session. Underlying spot is not in the OPRA preview and is left missing.

"""
