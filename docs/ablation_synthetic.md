# Synthetic ablation (checked-in)

Research laboratory output. The tape is a seeded GBM plus a Poisson touch fill (`fill_model=poisson`, 48 steps, one trading minute each, seed 11). It is not OPRA, NBBO, or a production market-making result.

Markout columns are 1 / 5 / 30 **simulator steps** (here, minutes), not exchange seconds. `fill_rate` is fill events per step and can exceed 1 when both sides trade. `greek_utilization` is mean absolute net delta divided by the hard delta limit. Feature packs are the existing flow, GEX, and instability scalers; `off` is the identity. `fixed_spread` is Avellaneda–Stoikov with γ = 0 and the half-spread clamped to 0.25.

A schema-compatible synthetic fixture is checked in at `jev_omm/data/fixtures/tape_synthetic.csv` with `synthetic_fixture=1`. Those prices are invented. A licensed CSV or Parquet with columns `time_seconds,spot,bid,ask,bid_sz,ask_sz` (spot optional when bid and ask are present) is passed to `load_tape`. Live vendor URLs are refused. See `docs/DATA.md` and `docs/ablation_real_or_fixture.md`.

| label | pnl | fill_rate | realized_spread | markout_1 | markout_5 | markout_30 | hedge_cost | inventory_variance | max_drawdown | quote_uptime | greek_utilization | n_fills | n_hedges |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_spread | 12.6805 | 1.0417 | 12.5000 | 0.2304 | 0.3738 | -0.1414 | 0.0000 | 1.4931 | 0.1839 | 1.0000 | 0.0081 | 50 | 0 |
| avellaneda_stoikov | 20.3051 | 0.5208 | 21.6462 | 0.0806 | -0.4121 | 0.1766 | 0.0000 | 1.8594 | 0.3469 | 1.0000 | 0.0135 | 25 | 0 |
| gueant_asymptotic | 20.4305 | 0.5208 | 21.7716 | 0.0806 | -0.4121 | 0.1766 | 0.0000 | 1.8594 | 0.3469 | 1.0000 | 0.0135 | 25 | 0 |
| option_vega | 11.7277 | 0.2708 | 11.9576 | -0.0838 | -0.0578 | -0.0671 | 0.0000 | 1.6927 | 0.4366 | 1.0000 | 0.0132 | 13 | 0 |
| as_flow | 22.8456 | 0.5417 | 23.6350 | -0.0445 | -0.1430 | 0.0365 | 0.0000 | 10.3542 | 0.7226 | 1.0000 | 0.0259 | 26 | 0 |
| as_flow_gex | 13.2567 | 0.2292 | 13.1349 | 0.0920 | 0.1554 | -0.1046 | 0.0000 | 1.0933 | 0.1606 | 1.0000 | 0.0073 | 11 | 0 |
| as_flow_gex_state | 14.8217 | 0.2292 | 14.7739 | 0.0719 | 0.1201 | -0.0826 | 0.0000 | 1.0972 | 0.1631 | 1.0000 | 0.0078 | 11 | 0 |

## Walk-forward fill hazard (synthetic)

- Tape source: `synthetic`
- Train fills: 32
- Fitted A (events/second at δ=0): 0.024462
- Fitted k (per price unit): 1.5000
- Fit method: `moments_k_held` (joint MLE A=2.061e-09, k=0.0000; rejected when δ does not move)
- Train log-likelihood of the joint MLE: -508.9810 on 80 side-steps
- Test PnL, default intensity: 24.1161 (28 fills)
- Test PnL, intensity and κ set from the train fit: 21.0982 (24 fills)

A is events/second at δ=0 on this synthetic clock. It is not substituted into QuoterConfig.A (Guéant, arXiv 1105.3115). Fit method: moments_k_held.

If the calibrated test PnL is not better, that is the result. The fit is a hazard on this generator, not a claim that (A, k) transfer to listed options.
