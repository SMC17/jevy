# Sleeves

**Version:** `1.0.0-zig-desk`  
**Code:** `jev_omm/desk/sleeves.py`, `jev_omm/desk/allocator.py`, `zig/src/desk.zig` (`allocateInverseVol`).  
**Mode:** paper targets and paper PnL. `quote_or_target` does not send an order.

A sleeve is a name, a risk budget, a target, a greek vector, and a PnL stream. The eight sleeves share the surface book, the flow prior, and the instability gate. Their PnL streams stay separate so the residual correlation is a number you can print.

`enabled=False` is the identity: target 0, PnL 0, weight 0. `gates_on=False` leaves `flow_prior` and `state_gate` at multiplier 1. Zeros in those functions were already the identity.

## The eight

Research units use S = 1, so the greek step's `ΔS` is the simple return. `greek_pnl_step` is unchanged: gamma PnL is `0.5 Γ (ΔS)²`.

| Sleeve | What it trades | Unhedged risk it keeps | Where the code already lived |
| --- | --- | --- | --- |
| `mm_spread` | A–S half-spread on a short-gamma inventory, with a delta leak proportional to beta | inventory delta, gamma | `quoter/avellaneda_stoikov.py` |
| `skew_residual` | Fade the planted wing state. Small delta, keep vanna | vanna / volga | SVI residual field |
| `calendar_term` | Fade the term state | theta, term vega | multi-expiry total variance |
| `fly_butterfly` | Fade the curvature state. Shares the smile shock with skew | volga | same smile shock, on purpose |
| `vrp_varswap` | Short one variance unit. Premium is `σ² Δt`. Convexity `−r²` is the gamma bucket (`Γ = −2`) | variance, some delta | `pricing/varswap.py` sign; Carr–Wu |
| `flow_toxicity` | `flow_prior` widens and cuts size. Spread if filled, adverse selection if filled into toxicity | adverse selection | `flow/signals.py` |
| `gex_forced` | Chase the previous index return when dealer gamma is short. `state_gate` cuts size | warehouse-style forced flow | `state_os/gate.py` |
| `parity_box` | Fade a box-versus-rate gap. `box_theo` is the package value | rates / borrow | `pricing/parity.py` |

Citations used here, all already public:

- Avellaneda and Stoikov, Quantitative Finance 2008, [doi:10.1080/14697680701381228](https://doi.org/10.1080/14697680701381228)
- Carr and Wu, Variance Risk Premiums, Review of Financial Studies 2009, [doi:10.1093/rfs/hhn039](https://doi.org/10.1093/rfs/hhn039)
- Gatheral and Jacquier, [arXiv:1204.0646](https://arxiv.org/abs/1204.0646)

`flow_prior` and `state_gate` keep the citations already in those modules. This desk does not claim a production VPIN or a live dealer-gamma feed.

Static `RISK_BUDGET` numbers are labels (1.0 down to 0.35). They are not exchange limits.

## Allocator

Same function in Python and Zig. On residual PnL, in this order:

1. Sample standard deviation, divisor `n − 1`, floor `1e-8`. Disabled sleeves stay at 0.
2. Raw weight proportional to `1/σ`.
3. Each enabled pair with `|ρ| > 0.50` multiplies both raw weights by `0.50 / |ρ|`. A name in several pairs is scaled once per pair.
4. A sleeve with a negative residual mean is set to 0 when another enabled sleeve has `|ρ| > 0.50` and a strictly higher mean. The worse leg of a collinear pair is the one that is cut.
5. Renormalize survivors to sum to 1.
6. Cap any weight at 0.40. Excess goes to uncapped positive weights. If every survivor is capped, the rest is cash and the weights sum to less than 1.

Inverse-vol is a risk budget. It is not a sort by residual Sharpe. On the default seed the highest residual Sharpe is `mm_spread` and the largest weight is `parity_box`, because the box residual is the quietest series and then hits the cap. Both numbers are on the scoreboard. See [ablation_sleeve_corr.md](./ablation_sleeve_corr.md).

## Training case

`toxic_sleeve` is two names (`EQ_INDEX`, `EQ_SINGLE`) and two sleeves. The toxic series is the good series minus 0.07, so Pearson is 1 and the toxic mean is negative. Naive weight is 1/2 each and the sum of PnL is 0. The desk allocator puts 0.40 on the good sleeve and 0 on the toxic one (the cap binds, the rest is cash). PnL is 0.084.

The offline fallback is asked `sleeve_weight` and `kill_sleeve`. With `desk.enabled=0` the answer is `hold` and kill-noul 0, and `apply_desk_policy` returns the identity. With the gate on and `sleeve_toxic=1` the kill noul is above 0.70 and the code weight is 0. The answer is a Choice / Noul. It is not an order.

## Desk questions

`build_desk_questions()` is a separate map (`sleeve_weight`, `kill_sleeve`, `surface_suspect`). The MM battery is unchanged when these ids are not in the question dict. `DeterministicFallbackClient` only adds the desk answers when they were asked. No live `TYPESAFE_API_KEY`.
