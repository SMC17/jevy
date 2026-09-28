# Jev paper scoreboard (synthetic)

Each round has a visible state `x = P(adverse fill)`. Quoting earns +0.05 if the fill is not adverse and −0.20 if it is. Pulling earns 0. Deterministic policy always quotes. Oracle policy pulls when `x ≥ 0.55` and is given the true probability. Noisy policy uses `clip(x + N(0, 0.25))`. No orders are emitted. No TypeSafe key is used.

- n = 4000
- deterministic PnL / round: -0.0761
- oracle-assisted PnL / round: -0.0119 (gap 0.0643)
- noisy-assisted PnL / round: -0.0171 (gap 0.0590)

| forecast | Brier | log loss | ECE |
| --- | --- | --- | --- |
| oracle (the true x) | 0.1698 | 0.5088 | 0.0136 |
| noisy | 0.2126 | 0.7534 | 0.0860 |
| flat 0.5 | 0.2500 | 0.6931 | 0.0045 |

Noisy ECE by regime: calm 0.1281, toxic 0.1495.

On this draw the noisy assisted policy beats always-quote by 0.0590 per round after the stylized spread and adverse cost. That is still a synthetic Bernoulli tape, not a market.

Oracle Jev is handed P(adverse). A higher oracle PnL is not live alpha. Read noisy_minus_deterministic before claiming the assisted policy helps.

## Sim-loop scoreboard (paper simulator)

Each quote step of `run_simulation` stores the offline fallback answer (Choice / Score / Noul) and a realized outcome from that same run. No TypeSafe key. No order. Feature pack is `off` (identity). Fill model is Poisson, 40 steps, intensity 0.03/s.

Outcomes, not a toy coin:
- `informed_flow`, `toxicity`, `widen_quotes`, `size_tier`: scored only on steps that filled. Outcome 1 when one-step signed markout is negative (the mid moved against the fill).
- `pull_quotes`: 1 if that fill was adverse or the step breached a limit.
- `hedge_now`: 1 if the step recorded an inventory/risk breach.
- `regime`: probability of `stressed` versus breach or a 40 bp spot move.
- `surface_suspect`: 1 if the next mid moved by more than the posted half-spread.

Research fee 0.02 per filled contract, subtracted from simulator PnL. Train seeds [1, 2, 3], test seeds [4, 5, 6].

- Identity test PnL mean ± std: 25.6514 ± 1.8137
- Full fallback test PnL mean ± std: 30.7426 ± 2.8783 (gap vs identity 5.0912; train gap 2.7644)
- Gated policy kept: regime, toxicity, size_tier
- Gated off after train: informed_flow, widen_quotes, pull_quotes, hedge_now, surface_suspect
- Gated policy test PnL: 30.7426 (gap vs identity 5.0912)

| question | train gap | test gap | OOS | n | Brier | log loss | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regime | 1.2040 | 1.4796 | helps OOS | 120 | 0.0034 | 0.0597 | 0.0579 |
| toxicity | 2.6157 | 5.0805 | helps OOS | 83 | 0.2546 | 0.7172 | 0.1542 |
| informed_flow | 0.0000 | 0.0000 | does not help OOS | 83 | 0.2414 | 0.6786 | 0.1080 |
| widen_quotes | 0.0000 | 0.0000 | does not help OOS | 83 | 0.2425 | 0.6833 | 0.1163 |
| pull_quotes | 0.0000 | 0.0000 | does not help OOS | 120 | 0.2164 | 0.6940 | 0.1700 |
| hedge_now | 0.0000 | 0.0000 | does not help OOS | 120 | 0.0504 | 0.2421 | 0.2113 |
| size_tier | 3.2758 | 4.2902 | helps OOS | 83 | 0.3044 | 0.9301 | 0.2707 |
| surface_suspect | 0.0000 | 0.0000 | does not help OOS | 117 | 0.0064 | 0.0834 | 0.0800 |

Informed-flow ECE by the fallback's own regime label: calm 0.1790, trending 0.1300, volatile 0.6948.
Score rows on the test seeds: 809.

The train-gated policy beat identity by 5.0912 per run on the test seeds after the 0.02 research fee. That is one synthetic Poisson tape, not a market, and not a live Jev call. Kept questions with Brier at least 0.24 (no better than a fair coin): toxicity, size_tier. A PnL gap from a coarse size or spread cut is not a calibrated forecast.

A positive gap is not live alpha. The client is the deterministic fallback, not `jev-latest`. Read the OOS column before keeping a question.
