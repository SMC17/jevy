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
