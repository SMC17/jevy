# Sleeves

**Version:** `1.1.0-zig-ortho`  
**Code:** `jev_omm/desk/sleeves.py`, `jev_omm/desk/allocator.py`, `jev_omm/desk/orthogonal.py`, `zig/src/desk.zig` (`allocateInverseVol`).  
**Mode:** paper targets and paper PnL. `quote_or_target` does not send an order.

A sleeve is a name, a risk budget, a target, a greek vector, and a PnL stream. Twenty sleeves share the surface book. Their PnL streams stay separate so the residual correlation is a number you can print. The factor split and the gate are in [ORTHOGONALITY.md](./ORTHOGONALITY.md). The seed-11 scoreboard is in [ablation_sleeve_corr.md](./ablation_sleeve_corr.md).

`enabled=False` is the identity: target 0, PnL 0, weight 0. `gates_on=False` leaves `flow_prior` and `state_gate` at multiplier 1. Zeros in those functions were already the identity.

## The eight from 1.0, with the split on

Research units use S = 1, so the greek step's `ΔS` is the simple return. `greek_pnl_step` is unchanged: gamma PnL is `0.5 Γ (ΔS)²`. `factorize_smile=False` and `separate_convexity=False` restore the 1.0 formulas (shared smile shock, inventory gamma on both the spread and the variance sleeve). Defaults are the split.

| Sleeve | What it trades | Unhedged risk it keeps | Where the code already lived |
| --- | --- | --- | --- |
| `mm_spread` | A–S half-spread on a short-gamma inventory, with a delta leak proportional to beta | inventory delta, gamma | `quoter/avellaneda_stoikov.py` |
| `skew_residual` | Fade the planted wing state. Small delta, keep vanna | vanna / volga | SVI residual field |
| `calendar_term` | Fade the term state | theta, term vega | multi-expiry total variance |
| `fly_butterfly` | Fade an independent curvature draw. The smile-shock regime plants a shared draw and then residualizes it | volga | own curvature factor |
| `vrp_varswap` | Implied minus lagged realized variance, times `dt`, plus its own premium shock. Spot gamma is 0. Vega is `variance_swap_vega` | variance premium | `pricing/varswap.py`; Carr–Wu |
| `flow_toxicity` | `flow_prior` widens and cuts size. Spread if filled, adverse selection if filled into toxicity | adverse selection | `flow/signals.py` |
| `gex_forced` | Chase the previous index return when dealer gamma is short. `state_gate` cuts size | warehouse-style forced flow | `state_os/gate.py` |
| `parity_box` | Fade a box-versus-rate gap. `box_theo` is the package value | rates / borrow | `pricing/parity.py` |

Citations used here, all already public:

- Avellaneda and Stoikov, Quantitative Finance 2008, [doi:10.1080/14697680701381228](https://doi.org/10.1080/14697680701381228)
- Carr and Wu, Variance Risk Premiums, Review of Financial Studies 2009, [doi:10.1093/rfs/hhn039](https://doi.org/10.1093/rfs/hhn039)
- Gatheral and Jacquier, [arXiv:1204.0646](https://arxiv.org/abs/1204.0646)

`flow_prior` and `state_gate` keep the citations already in those modules. This desk does not claim a production VPIN or a live dealer-gamma feed.

Static `RISK_BUDGET` numbers are labels (1.0 down to 0.35). They are not exchange limits.

## Twelve added in 1.1

Each one is booked only where it has an economic reason to exist. Cloning a sleeve onto every product was rejected when that copy reloaded beta or vega.

| Sleeve | What it trades | Where it is booked |
| --- | --- | --- |
| `wing_kurtosis` | Independent far-wing draw (`k⁴` bump on the quote grid). Smile-shock regime residualizes it against skew | every product |
| `sticky_regime` | Sticky-strike versus sticky-delta state | every product |
| `dispersion_index` | Own AR dispersion factor, vega 0. An IV-spread was collinear with vega (R² 1) and was rejected | `EQ_INDEX` only |
| `roll_yield` | `annualized_roll` on the seasonal commodity and the rates name | `commodity`, `rates` |
| `charm_bleed` | Weekend charm via `charm_tau`, clock `t % 5 == 1` | every product |
| `vanna_tilt` | Vanna mis-mark | every product |
| `queue_sniper` | `expected_fills` as a size, fading the queue state | every product |
| `cot_fade_sleeve` | `cot_fade` when gates are on. Clock `t % 5 == 4`, so it does not share charm's spike. Identity when `gates_on` is false | every product |
| `warehouse_autocall` | `autocall_flow` times `state_gate`. Identity zero when gates are off | every product |
| `box_rate` | Funding residualized against the box. Not a second `parity_box` | every product |
| `cross_impact` | One hub AR factor through `hub_spoke_impact`. Lagged returns reloaded beta and were rejected | `EQ_INDEX` only |
| `rough_vol_stress` | `rough_bergomi_variance` (H = 0.15, η = 0.40, ξ0 = 0.04, seed + 3). Research label, not a calibrated rough-vol book | every product |

`book_higher_greeks` adds `0.5 * volga * (Δσ)²` and `vanna * return * Δσ` into raw PnL. The strip then removes volga, vanna, and a variance column orthogonal to spot gamma. Off, those terms are not booked and the strip is the 1.0 three-factor regression.

## Allocator

Same function in Python and Zig. On residual PnL, in this order. Defaults: correlation shrink 0.35, concentration cap 0.35, Sharpe tilt 0.25 on the desk (0 inside `allocate()` and inside `toxic_sleeve_weights()`, so that lesson stays exact).

1. Sample standard deviation, divisor `n − 1`. A series with σ < 1e-5 gets raw weight 0. Otherwise floor σ at 1e-8. Disabled sleeves stay at 0.
2. Raw weight proportional to `boost / σ`. `boost = min(3, 1 + tilt * max(Sharpe, 0))`. A negative residual mean multiplies `boost` by 0.25. The hard zero is still rule 4.
3. Each enabled pair with `|ρ| > 0.35` multiplies both raw weights by `0.35 / |ρ|`. A name in several pairs is scaled once per pair.
4. A sleeve with a negative residual mean is set to 0 when another enabled sleeve has `|ρ| > 0.35` and a strictly higher mean.
5. Renormalize survivors to sum to 1.
6. Cap any weight at 0.35. Excess goes to uncapped positive weights. If every survivor is capped, the rest is cash and the weights sum to less than 1.

Inverse-vol is a risk budget. It is not a sort by residual Sharpe. On the twenty-sleeve seed-11 path the highest per-step residual Sharpe is `flow_toxicity` (10.65, a smooth synthetic spread after the strip) at weight 0.212, and several research sleeves with large residual totals sit near weight 0 because their sample σ is large. No name is at the 0.35 cap on that path. `parity_box` is at the cap on the eight-sleeve three-product book. See [ablation_sleeve_corr.md](./ablation_sleeve_corr.md).

The 1.0 rule was shrink 0.50 and cap 0.40, with no negative-mean haircut and no Sharpe tilt. `legacy_config()` still passes 0.50 and 0.40. The 0.25 haircut is in the allocator itself, so a legacy replay does not reprint the 1.0 weights. The residual series do reprint.

## Training cases

`toxic_sleeve` is two names (`EQ_INDEX`, `EQ_SINGLE`) and two sleeves. The toxic series is the good series minus 0.07, so Pearson is 1 and the toxic mean is negative. Naive weight is 1/2 each and the sum of PnL is 0. The desk allocator puts 0.35 on the good sleeve and 0 on the toxic one (the cap binds, the rest is cash). PnL is 0.0735.

`ortho_break` is a fly series equal to `0.93 * skew − 0.04`. Pearson is 1. With `max_weight=1`, correlation cap 0.35, and tilt 0, the allocator keeps the skew leg at weight 1 and drops the fly. Desk PnL is 0.21. Naive equal weight is 0.08265.

`toxic_multi` is three series on `EQ_INDEX`, `EQ_SINGLE`, and `EQ_LOWBETA`. The third is the first minus 0.07. Tilt 0. Desk weights are 0.35, 0.35, and 0 (sum 0.70, the rest is cash). Desk PnL is 0.10535. Naive equal weight is 0.03033.

The offline fallback is asked the desk map. With `desk.enabled=0` the policy is the identity. With the gate on, `sleeve_toxic` can kill, `ortho_break` can merge, and `product_toxic` can kill a product. The answer is a Choice / Noul. Code sets the weight. It is not an order.

## Desk questions

`build_desk_questions()` is a separate map (`sleeve_weight`, `kill_sleeve`, `surface_suspect`, `merge_sleeve`, `cut_corr_pair`, `product_kill`). The MM battery is unchanged when these ids are not in the question dict. `DeterministicFallbackClient` only adds the desk answers when they were asked. No live `TYPESAFE_API_KEY`.
