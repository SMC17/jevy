# Tape walk-forward (public preview and synthetic fixture)

Research laboratory output. Two tapes, named for what they are.

## What was loaded

Databento's no-account sample endpoint `GET https://api.databento.com/v0/dataset/sample` returned a JSON list of CSV lines for `dataset=OPRA.PILLAR`. No API key. `schema=cbbo-1s` was an empty list on 2026-09-28. `schema=cbbo-1m` returned a header plus 20 rows: consolidated BBO and last trade for `TSLA  230901C00250000` (OSI call, expiry 2023-09-01, strike 250) from 2023-08-28 13:31Z through 13:50Z. That is a vendor preview, not a session and not a licensed history.

sha256 of the cbbo-1m JSON body: `b8ed8988cd064a35252049a6306f39a39040f54c1c1c8ebee1536a8a96882848`.

The underlying spot is not in the preview. It is left missing. No TSLA stock print was joined in from another sample (those files are a different product and, for the equity previews checked the same day, a different date).

Redistribution of even a short OPRA preview is not clearly granted, so the raw body stays in gitignored `data/local/`. The table is aggregates from a local replay. Re-download with `python scripts/fetch_databento_sample.py` and check the sha256 before quoting the numbers as current.

A second preview, `schema=tcbbo` (trade plus NBBO at the trade), sha256 `c1e45b59cc08a78041e800cec512fa8eb59ff8ab97f7f261c9f462d2e5f4cc52`, has 3 locked rows, 0 crossed, 1 missing bid/ask, and 16 two-sided, out of 20. Those locked prints are real preview rows. They are not mixed into the cbbo-1m replay below; the minute bars in cbbo-1m were two-sided.

Posting rule: on a two-sided book, join the touch when the model is tighter and sit behind when it is wider. Never cross. The next row's trade fills us if it prints at or through our price. There is no queue ahead, so touch fills are overstated. Crossed, locked, stale, and missing books are not quoted. Fee 0.05 per contract, rebate 0, is a research schedule, not an OPRA fee card.

The checked-in fixture `jev_omm/data/fixtures/tape_synthetic.csv` is `synthetic_fixture=1` on every row. The symbol `SYNTH` and the prices are invented. It is the runnable offline path. It is not OPRA.

## OPRA.PILLAR public sample, schema cbbo-1m

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

## Synthetic fixture (invented prices)

- Source: `csv:tape_synthetic.csv`
- Synthetic fixture: yes
- Symbols: SYNTH 260101C00100000
- Rows: 96 (train ends at index 48, test rows 48)
- Underlying spot: present
- Book labels (whole file): two-sided 90, crossed 2, locked 2, stale 2, one-sided 0, missing 0
- Research fee 0.0500 per contract, rebate 0.0000. Not a venue fee card.
- Train σ of the option mid (annualized, reported only): 9.8192 via `train_mids`.
- σ used by the quoters: 0.45 ($/√yr, the 0.8 ablation knob). The train annualization is not substituted.
- Hazard: A=0.004901/s at δ=0, k=1.5000, method `moments_k_held`, MLE A=2.088e-02, MLE k=32.2459, loglik=-52.4724 on 84 touch-sides.
- Kappa on the test window: 1.5000.

Test-window replay. `mean_half` is the half-spread we actually posted (after joining or sitting behind the market). `markout_1` is one tape row after the fill, signed size times the mid change. `join_touch` posts the market itself and is a reference, not a model.

| label | pnl | n_fills | contracts | fees | rebates | markout_1 | mean_half | quote_uptime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_spread | 1.0944 | 5.0000 | 5.0000 | 0.2500 | 0.0000 | 0.1194 | 0.2500 | 1.0000 |
| avellaneda_stoikov | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6414 | 1.0000 |
| gueant_asymptotic | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6477 | 1.0000 |
| option_vega | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6705 | 1.0000 |
| as_flow | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7825 | 1.0000 |
| as_flow_gex | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0458 | 1.0000 |
| as_flow_gex_state | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.6117 | 0.4043 |
| join_touch | -0.0626 | 28.0000 | 28.0000 | 1.4000 | 0.0000 | -0.5620 | 0.0474 | 1.0000 |

Uncalibrated A–S (kappa held at 1.5) on the same test rows: PnL 0.0000, fills 0.0000, mean half 0.6414.

Join-touch PnL -0.0626 does not beat fixed-spread 1.0944 after the research fee. The touch is not free edge on this window. Below fixed-spread on this single draw: avellaneda_stoikov, gueant_asymptotic, option_vega, as_flow, as_flow_gex, as_flow_gex_state. No theory row beat fixed-spread on this draw. That is the result. Fitted kappa matched the prior on the test window (the train window did not identify a different k, or it did not move PnL).

With γ = 0.12 and κ = 1.5 the dominant half-spread term is (1/γ) ln(1 + γ/κ) ≈ 0.64 dollars, before the market join/behind clip. A preview NBBO of one to a few cents does not trade against that quote. Fixed-spread is clamped at 0.25 and can catch a sweep the wider quotes miss. That is the formula at these knobs, not a claim that 0.25 is optimal.

