# Latent-state / forced-flow operating system

**Version:** `0.7.0-zig-state-os`  
**Mode:** simulation / paper only. No live venues, brokers, or order path.  
**Languages:** Zig hot path and Python research desk. There is no Rust crate in this tree.

The desk does not forecast ΔP. It estimates the state that makes the next trade obligatory, and it lets price be downstream of that state.

## State

```
S_t = (P, σ, Σ, L, I, C, F, G, τ, B, O, K_stack, D_liab, R)
```

| Slot | Meaning in this tree |
| --- | --- |
| P | Spot used by the one-name research book |
| σ | Realized or implied vol, a scalar on the gate |
| Σ | Surface summary (ATM, skew). The SVI book is unchanged |
| L | **Executable** liquidity, not displayed depth |
| I | Latent inventory by participant class (a dict on `MarketState`) |
| C | Binding constraints. A level-set, used as a gate |
| F | Scheduled and conditional forced flow |
| G | Hub-spoke cross-impact. See the common-flow caveat |
| τ | Settlement and funding clock, in [0, 1] |
| B, O, K_stack | Barrier vector, observation count, coupon stack |
| D_liab | Liability duration |
| R | Capital regime: `economic`, `statutory`, or `rating` |

Code: `jev_omm/state_os/vector.py`.

```
F(S)     = Σ_k w_k(S) f_k(S)
F_net    = F_forced − F_already_positioned − F_cannot_start
L_exec   = L_base + (∂L/∂Q) |F|          floored at 0
Instability(S) = |F_net| / L_exec
```

`L_exec` is evaluated on the path F will take. If that path consumes the book, ∂L/∂Q is negative and the ratio rises. `L_exec ≤ 0` with `|F| > 0` is an air pocket: the ratio is infinite, and the decision scalar stores the gate clip `4`.

`enabled = 0` reports instability 0 and leaves every quote scaler at the identity, even if the raw ratio is large.

## How it hits the desk

Two actuators, both identity when the flag is off.

1. **Zig / Python scaler.** `stateGate` / `state_gate` (`zig/src/state_os.zig`, `jev_omm/state_os/gate.py`). `apply_features` multiplies it in after flow, GEX, and COT. C ABI: `jev_omm_state_gate`, `jev_omm_instability`.

   | Ratio / flag | Spread | Size | Hedge urgency | Pull |
   | --- | --- | --- | --- | --- |
   | off, any inputs | 1 | 1 | 0 | no |
   | on, ratio 0, no parent, no constraint | 1 | 1 | 0 | no |
   | on, ratio `u` | `1 + 0.80 min(u, 2)` | `1 / (1 + 1.10 min(u, 2))`, then cut further by a live parent | `0.45 min(u, 2)` once `u ≥ 1` | when `u ≥ 2` |
   | live parent, fraction `p` | unchanged by `p` alone | extra `1 / (1 + 0.75 p)` | unchanged | no, unless the ratio or a constraint says so |
   | binding constraint | same formula | 0 | at least 0.70 | yes |

   A small reservation shift leans with the sign of `F / L_exec`. It is not a return forecast. Hard pull sets size to 0 after the usual 0.20 floor, so a spike can actually clear the quote.

2. **Decision layer.** `build_mm_state(..., latent=...)` copies the scalars. `DeterministicFallbackClient` maps them onto the existing Choice / Score / Noul battery. `apply_policy` records `instability`, `parent_remaining`, and `constraint_active` on `QuoteAdjustments`. The model still does not emit an order. A missing key or `enabled = 0` leaves the previous fallback rules alone. Live `TYPESAFE_API_KEY` stays optional and unwired in tests.

Classical A–S / Guéant reservation math is not replaced.

## Stand-up engines

All of these take caller inputs. None of them download a calendar or a vendor tape.

| Engine | Formula | Citation |
| --- | --- | --- |
| LETF close | `AUM (L² − L) r` | Cheng & Madhavan, JOIM 2009, [SSRN 1539120](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1539120) |
| Net liquidity | Fed assets − TGA − ON RRP, plus the composition of the change | [H.4.1](https://www.federalreserve.gov/releases/h41/), [TGA](https://fiscaldata.treasury.gov/datasets/daily-treasury-statement/operating-cash-balance), [ON RRP](https://www.newyorkfed.org/markets/desk-operations/reverse-repo). Fixture levels are synthetic |
| Buyback blackout | Issuer window from the caller's quarter-end and earnings dates. Aggregate withheld pace | [17 CFR § 240.10b-18](https://www.ecfr.gov/current/title-17/chapter-II/part-240/section-240.10b-18) is a safe harbor, not a halt. Blackout windows as corporate policy: Bettis, Coles, Lemmon, [JFE 2000](https://doi.org/10.1016/S0304-405X(00)00055-6). Not a live earnings calendar |
| Pension 60/40 | Exact fixed-mix trade and the linear `AUM w(1−w)(Rb−Rs)` | Perold & Sharpe, [FAJ 1988](https://doi.org/10.2469/faj.v44.n1.16) |
| Flow-signed vs structural GEX | OI gamma (existing) versus signed customer volume. Disagreement is a flag. Vanna/charm convert to ES only for SPX | Barbon & Buraschi for the hedge-feedback sign. SPX multiplier 100, ES point value 50. SPY returns null ES fields |
| Securities lending | utilization × fee × DTC × (1 + supply that left) | D'Avolio, [JFE 2002](https://doi.org/10.1016/S0304-405X(02)00206-4). Fixture, not a lending vendor |
| Auction imbalance | `(buy − sell) / (buy + sell)` | Cushing & Madhavan, [JFM 2000](https://doi.org/10.1016/S1386-4181(99)00014-0) |
| Vol-control / CTA / risk parity | Close-to-close vol on 20/60/120/250, `min(cap, σ*/σ)`, sign of price/SMA, inverse-vol weights, dollar step | Moreira & Muir, [JF 2017](https://doi.org/10.1111/jofi.12513); Hurst, Ooi, Pedersen, [JPM 2017](https://doi.org/10.3905/jpm.2017.44.1.015); Asness, Frazzini, Pedersen, [FAJ 2012](https://doi.org/10.2469/faj.v68.n1.1) |
| TDF | Linear glide. Trade only when `\|w−w*\| > 200 bp`, and only back to 175 bp from target | [Vanguard, *The rebalancing edge*](https://corporate.vanguard.com/content/dam/corp/research/pdf/the_rebalancing_edge_optimizing_target_date_fund_rebalancing_through_threshold_based_strategies.pdf) |
| Overwrite roll | Gen-1: twelve monthly third Fridays, full cover. Gen-3: `clip(0.50 + 2(IV−IV_ref))` and a slightly OTM strike when IV is rich | [Cboe BXM](https://www.cboe.com/us/indices/dashboard/bxm/). The Friday list is the expiry rule, not a fund's holdings |
| Clock τ | OpEx (third Friday), quarter-end SLR window, VM hour, AM vs PM settlement gap | SPX facts: [Cboe](https://www.cboe.com/tradable_products/sp_500/spx_options/). Quarter-end balance-sheet pressure: Du, Tepper, Verdelhan, [JF 2018](https://doi.org/10.1111/jofi.12620). The SLR window is a research window, not a bank optimizer |

## Research cores

Tested on synthetic paths. They do not claim a fitted live tape.

| Core | What a passing test shows |
| --- | --- |
| Latent book | `L_exec = L_disp + E[R] − E[C] + E[H]`. Cancel hazard rises in distance and in σ. Analytic ∂E[C]/∂σ matches a finite difference. Icebergs raise L. ∂L/∂σ on the surface is negative |
| Metaorder remaining | Square-root impact is concave (`I(4Q) = 2 I(Q)`). A synthetic parent reverts only part of the way (permanent piece stays). Remaining size tracks the stop rule `dI/dQ = λ` while the child prints, and is 0 when the tape goes flat. A sine path is rejected as noise |
| Impact residual | `ΔP = Y σ √(Q/V) + spread × max(\|Q\|−D, 0)/D + ε`. On a draw from that function plus independent noise, `corr(ε, Q)` is smaller than `corr(ΔP, Q)` |
| Constraint level-sets | Vol-target, price floor, clock, funding, borrow. Normal VaR `σ*(t)` falls as the horizon lengthens. A binding set blocks quoting. A clear set does not |
| Hub-spoke cross-impact | `G_hh`, `G_ss`, `G_hs`. When the only link is a common factor, naive OLS beta is large and the residualized beta is near 0. A real spoke loading survives residualizing on an independent factor |
| Funding capacity | Specials−GC, fails, auction tail, and \|CIP\| dominate a SOFR change. A 50 bp SOFR move with nothing else scores below a 25 bp specials gap |
| Hawkes `n_t` | `n_t = 1 − μ/λ_t` (Filimonov & Sornette). A burst is endogenous. An isolated jump is exogenous. A small return stays quiet. `n_t` is not a crash time |

## Warehouse terms

Same machine: the client is long the income or the protection, the warehouse is short the embedded option, and the hedge target depends on R.

| Term | Geometry the test checks |
| --- | --- |
| Autocall | Digital density and knock-in mass peak on the barrier. Worst-of weight rises when correlation is low. Missed knock-out extends duration. Guillaume, [JOD 2015](https://doi.org/10.3905/jod.2015.22.3.073) |
| RILA / VA | Issuer delta of cap call minus the buffer put spread. A zero buffer and a far cap is flat. `economic` / `statutory` / `rating` return different hedges. Koijen & Yogo, [AER 2015](https://doi.org/10.1257/aer.20121036) |
| MBS cusp | `∂D_eff/∂y` is on when the coupon sits on the yield and **off** for a deep discount. Richard & Roll, [JF 1989](https://doi.org/10.1111/j.1540-6261.1989.tb05062.x). Not an OAS engine |
| LDI | Cash need is repo top-up plus VM, and it is zero when yield velocity is under the floor. The 2022 gilt episode is the historical reference ([BoE FSR, December 2022](https://www.bankofengland.co.uk/financial-stability-report/2022/december-2022)), not a live feed |
| TDF + spend | Threshold trade above, plus a scheduled withdrawal `−rate × balance` |
| Internal net | Tape residual is gross minus what the warehouse nets with itself. Visible GEX is the book times that fraction |
| Mass on the cusp | Fraction of distances inside a bandwidth. No personality parameter |

## Training cases

`letf_day`, `instability_spike`, `remaining_parent`, `constraint_gate`, `gex_disagree`, `tdf_threshold`, `overwrite_roll`. Desk beats naive on absolute and risk-adjusted PnL in Python and in Zig. Decision snapshots, where present, are Choice / Score / Noul from the offline fallback. See [TRAINING_CASES.md](./TRAINING_CASES.md).

## Deferred, on purpose

These are not stubbed as if a live feed existed.

- Full autocall prospectus tape
- Live OPRA, FINRA, or a securities-lending vendor
- 13F parser
- Live MBS coupon stack / OAS
- Live BoE or ESMA LDI stress
- Power ISO, crypto ADL
- Lazy Prices NLP, MNPI, other alt data
- A bank's actual SLR optimization
- Agent-based explosion (the training cases stay scripted)
- Live `TYPESAFE_API_KEY` (the fallback remains the default)

## Tests

```bash
pytest -q
cd zig && zig build test
```
