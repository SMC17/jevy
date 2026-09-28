# Frontiers — surface, term risk, queue fills, Guéant ODE, then training and option-vega MM

**Version:** `0.6.0-zig-flow-positioning` (`jev_omm_version`)  
Frontiers 1–4 below shipped in `0.4.0-zig-frontiers-1-4` and stay as specified. Frontiers 5–8 are the `0.5.0` layer. Frontier 9 is the `0.6.0` flow and positioning layer.  
**Mode:** simulation / paper only. No live exchange SDKs, brokers, or venue keys.  
**Decision layer:** TypeSafe System One / Jev stays in Python (`decisions/`). It returns Choice / Score / Noul answers. It does not emit orders. A live `TYPESAFE_API_KEY` is optional and is **not** required to build, test, or run these demos — missing key uses `DeterministicFallbackClient`.

Hot path is Zig (`zig/src/`). Python modules under `jev_omm/` mirror the same formulas for research glue.

---

## 1. Surface — raw SVI, SSVI, arbitrage, sticky regimes

**Code:** `zig/src/svi.zig`, `jev_omm/surface/svi.py`.  
SABR-lite stays in `surface.zig` / `surface/sabr.py` as a one-slice alternative. SVI is the primary research surface.

**Cite:** Gatheral & Jacquier, *Arbitrage-free SVI volatility surfaces*, https://arxiv.org/abs/1204.0646 (PDF https://arxiv.org/pdf/1204.0646).

### Raw SVI

Total implied variance \(w(k)=\sigma_{\mathrm{BS}}^2(k)\,T\), log-moneyness \(k=\ln(K/F)\):

\[
w(k)=a+b\bigl\{\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}\bigr\}
\]

`a` level, `b≥0` wing slope, `ρ∈(−1,1)` skew, `m` location, `σ>0` curvature (not Black vol).

Closed minimum: \(w_{\min}=a+b\sigma\sqrt{1-\rho^2}\).

### Butterfly (density) and Lee wing

Gatheral–Jacquier \(g(k)\) is sampled on \(k\in[-1.5,1.5]\):

\[
g(k)=\Bigl(1-\frac{k w'(k)}{2w(k)}\Bigr)^2-\frac{w'(k)^2}{4}\Bigl(\frac{1}{w(k)}+\frac{1}{4}\Bigr)+\frac{w''(k)}{2}
\]

No butterfly arbitrage on the grid when \(g(k)\ge 0\), \(w>0\), and the wing slope obeys Roger Lee’s bound as used in that paper:

\[
b(1+|\rho|)\le 2
\]

A failing check, a failing calendar check, or a large fit residual (`rmse>0.02` or max abs residual `>0.05`) sets `surface_suspect`, the same flag the decision layer already consumes.

### SSVI

Power-law surface SVI, ATM total variance \(\theta\):

\[
w(k,\theta)=\frac{\theta}{2}\Bigl(1+\rho\,\varphi(\theta)\,k+\sqrt{(\varphi(\theta)k+\rho)^2+(1-\rho^2)}\Bigr)
\]

\[
\varphi(\theta)=\eta\big/\bigl(\theta^\gamma(1+\theta)^{1-\gamma}\bigr),\quad \gamma\in[0,1]
\]

At \(k=0\), \(w=\theta\). A sufficient no-calendar condition used here is \(\eta(1+|\rho|)\le 2\). Pointwise, two slices pass when total variance is non-decreasing in \(\theta\) on the same \(k\) grid. Raw slices have the same pointwise calendar test (`rawCalendarOk`).

### Calibration

Derivative-free Nelder–Mead on \((a,\ln b,\mathrm{atanh}\,\rho,m,\ln\sigma)\) minimizes squared error in total variance, with a penalty outside the Lee bound and for \(w_{\min}<0\). On a noiseless planted smile the fit RMSE is numerical noise and the butterfly gate stays green.

### Sticky regimes

When the forward moves \(F_0\to F_1\) at a fixed strike \(K\):

| Regime | What is held fixed | Evaluation |
| --- | --- | --- |
| `sticky_strike` | Black IV at strike \(K\) | \(w\) at \(k_0=\ln(K/F_0)\) |
| `sticky_delta` | Black IV at log-moneyness (delta proxy) | \(w\) at \(k_1=\ln(K/F_1)\) |

SVI is parameterized in log-moneyness, so re-evaluating with the new forward **is** sticky-delta. Sticky-strike keeps the pre-move moneyness. With a skewed smile the two IVs at a fixed strike diverge after the spot move; IV at a strike that scales with the forward is unchanged under sticky-delta.

---

## 2. Multi-expiry book and term-structure risk

**Code:** `zig/src/term_book.zig`, `jev_omm/risk/term.py`.  
BS vanna and volga are filled in `black_scholes.zig` / `pricing/black_scholes.py`.

### Greek conventions

\(\sigma\) is decimal Black vol, \(t\) is in years. Vega is \(\partial V/\partial\sigma\), **not** “per vol point” (multiply by 0.01 for a one-vol-point bump).

| Greek | Definition | PnL use |
| --- | --- | --- |
| vega | \(\partial V/\partial\sigma = S e^{-qT} n(d_1)\sqrt{T}\) | parallel: \(\nu\,\mathrm{d}\sigma\) |
| vanna | \(\partial^2 V/\partial S\partial\sigma = -e^{-qT} n(d_1) d_2/\sigma\) | cross: \(\mathrm{vanna}\,\mathrm{d}S\,\mathrm{d}\sigma\) |
| volga | \(\partial^2 V/\partial\sigma^2 = \nu\, d_1 d_2/\sigma\) | second order: \(\tfrac12\mathrm{volga}\,(\mathrm{d}\sigma)^2\) |

Vanna and volga are the same for European calls and puts (they come from the vega factor). Checked by finite difference on delta and vega.

### Term structure

Legs are bucketed by expiry.

- **Parallel vega** — sum of bucket vegas.
- **Bucket vega** — one expiry.
- **Term-structure vega (slope)** — \(\sum_i \nu_i (T_i-T_{\mathrm{front}})\), units vega·years.  
  A tilt \(\mathrm{d}\sigma(T)=\varphi(T-T_{\mathrm{front}})\) has first-order vega PnL `term_vega_slope * φ`, plus the volga square term inside `scenarioPnl`.

Taylor cell for a bucket:

\[
\Pi \approx \Delta\,\mathrm{d}S+\tfrac12\Gamma(\mathrm{d}S)^2+\nu\,\mathrm{d}\sigma+\mathrm{vanna}\,\mathrm{d}S\,\mathrm{d}\sigma+\tfrac12\mathrm{volga}\,(\mathrm{d}\sigma)^2
\]

with \(\mathrm{d}\sigma_i=\mathrm{d}\sigma_{\parallel}+\varphi(T_i-T_{\mathrm{front}})\). `scenarioReprice` reprices each leg with Black–Scholes for a cross-check on small shocks.

### Limits

`TermLimitConfig` / `evaluateLimits` can halt on parallel vega, any bucket vega, |vanna|, |volga|, or |term slope|. The single-name hard limits in `risk_limits.zig` are unchanged.

### Quotes

`quoteExpiries` marks each strike with SSVI, \(\theta(T)=\sigma_{\mathrm{atm}}^2 T\), shared \((\rho,\eta,\gamma)\), then quotes with the configured mode (including `gueant_ode`). The 5-strike SABR strip in `multi_strike.zig` is unchanged.

---

## 3. LOB / queue-aware fills

**Code:** `zig/src/lob.zig`, `jev_omm/execution/lob.py`.  
The Poisson touch model in `fills.zig` / `execution/fills.py` remains. This layer is still synthetic — no live market data.

A resting order sits behind `ahead` contracts. Two flows consume the queue:

- aggressive trades at `trade_intensity` (contracts / year) advance the queue **and** can fill us;
- cancels of size ahead at `cancel_ahead` advance the queue without filling us.

Fluid limit (the comparative statics the tests lock):

\[
t_{\mathrm{clear}}=\frac{\mathrm{ahead}}{\lambda_{\mathrm{trade}}+\lambda_{\mathrm{cancel}}}
\]

\[
\mathrm{fills}=\min\bigl(\mathrm{our\_size},\;\lambda_{\mathrm{trade}}\max(0,\,\mathrm{exposure}-t_{\mathrm{clear}})\bigr)
\]

`exposure` is the horizon, or `cancel_latency` when a cancel requested at \(t=0\) becomes effective. A **late** cancel (latency well below the horizon) therefore cuts off the rest of a toxic window and **reduces adverse fills versus never cancelling**. A longer latency leaves more residual adverse fills than a fast cancel. Deeper `ahead` increases time-to-first-fill and lowers fills inside a fixed window. If traded volume lands strictly inside `our_size`, the fill is partial.

On a fill the mid jumps against the liquidity provider by `adverse_jump * toxic_flow * size`. Markout is signed from the LP side (bid: mid_after − price). Larger `toxic_flow` worsens markout.

The stochastic stepper emits sequenced `LobAdd` / `LobExecute` / `LobCancel` records (add / cancel / execute), including partial executes. `event_log.appendLob` writes them into the JSONL stream; replay ignores them for cash PnL and still hashes the bytes.

---

## 4. Guéant ODE / spectral quotes and \((A,k)\) from tape

**Code:** `zig/src/gueant_ode.zig`, `jev_omm/quoter/gueant_ode.py`.  
Asymptotic closed form stays in `gueant.zig`.  
**Cite:** Guéant, Lehalle, Fernandez-Tapia, https://arxiv.org/abs/1105.3115.

Intensity \(\lambda(\delta)=A e^{-k\delta}\). Ansatz

\[
u=-\exp\bigl(-\gamma(x+qs)+\omega_q(t)\bigr),\qquad \omega_q(T)=0.
\]

In time-to-go \(\tau=T-t\):

\[
\frac{\mathrm{d}\omega_q}{\mathrm{d}\tau}=\frac{\sigma^2\gamma^2 q^2}{2}-\sum_{\mathrm{nb}} C\exp\Bigl(-\frac{k}{\gamma}(\omega_{\mathrm{nb}}-\omega_q)\Bigr)
\]

\[
C=A\cdot\frac{\gamma}{k+\gamma}\cdot\Bigl(\frac{k}{k+\gamma}\Bigr)^{k/\gamma}
\]

The neighbor is missing at the inventory caps \(\pm Q\) (no bid at \(+Q\), no ask at \(-Q\)). RK4 integrates \(\omega\) from 0. The change of variables \(v_q=\exp(-(k/\gamma)\omega_q)\) is the paper’s linear system \(\mathrm{d}v/\mathrm{d}\tau=Mv\). `spectralOffsets` runs that linear system with RK4 and renormalization so the principal eigenmode dominates; quotes use component ratios.

\[
\delta^b(q)=\frac1\gamma\ln\Bigl(1+\frac\gamma k\Bigr)+\frac{\omega_{q+1}-\omega_q}{\gamma}
\]

\[
\delta^a(q)=\frac1\gamma\ln\Bigl(1+\frac\gamma k\Bigr)+\frac{\omega_{q-1}-\omega_q}{\gamma}
\]

Reservation shift from mid is \((\delta^a-\delta^b)/2\). At \(q\) near 0 and long horizon the ODE and the spectral mode match the asymptotic \(\xi\) formula in `gueant.zig` (tests: relative error under 2% at \(q=0\)). At \(q\) next to the cap the ODE bid is materially wider than the unbounded asymptotic, and at \(q=Q\) the bid size is zero.

### Mode

`QuoterConfig.mode`:

| Value | Zig enum | Behavior |
| --- | --- | --- |
| `as_finite_horizon` | `.as_finite_horizon` | classic A–S with \(T-t\) |
| `gueant_asymptotic` | `.gueant_asymptotic` | stationary closed form |
| `gueant_ode` | `.gueant_ode` | ODE above, horizon `t_horizon` / `T_horizon`, cap `inventory_cap`, `ode_steps` RK4 steps |

C ABI `CQuoterConfig.mode`: 0 / 1 / 2. `jev_omm_gueant_ode_offsets` exports the raw distances.

### \((A,k)\) from a synthetic tape

Bins of `(δ, exposure, fills)` with \(\lambda(\delta)=A e^{-k\delta}\). Estimator: weighted least squares on \(\ln(n/E)=\ln A-k\delta\), then Newton MLE for Poisson counts. No live tape. Tests recover a planted \((A,k)\) within 5% on expected counts and within 15% on one seeded Poisson tape.

---

## Demos

```bash
export PATH="$HOME/zig-0.16:$PATH"   # Zig 0.16+; see README for the tarball
cd zig
zig build test
zig build frontiers                  # SVI, multi-expiry, LOB, ODE
zig build -Doptimize=ReleaseFast
zig build bench -Doptimize=ReleaseFast
cp -f zig-out/lib/libjev_omm.so ../jev_omm/native/libjev_omm.so   # optional; *.so is gitignored

cd ..
python -m jev_omm.demo_frontiers
pytest -q
```

---

## 5. Citadel-style training desk

**Code:** `jev_omm/training/`, `zig/src/training.zig`, `zig build training`, `python -m jev_omm.demo_training`.  
**Write-up:** [TRAINING_CASES.md](./TRAINING_CASES.md).

Paper cases (`location_arb`, `pm_fair_value`, `etf_ap_arb`, `liability_facilitator`, `mm_inventory`, `vol_surface_mm`, and from `0.6.0` `flow_vpin`, `dealer_gamma`, `cot_fade`). Each has a role, an information set, constraints, and a score: absolute PnL, relative PnL versus the naive policy, inventory-path penalty, unhedged-beta penalty, execution penalty. JSONL replay recomputes the score from stored paths.

`location_arb` cannot hedge inside the sim (no futures). The desk path is an out-of-sim futures overlay whose true beta is 0.85, so basis risk remains. `pm_fair_value` longs cheap names and forces a market-neutral opposing leg. `liability_facilitator` takes a forced client block and works it in slices (algo gap 1 vs hand gap 12) while skipping discretionary adds. `mm_inventory` tags `forced` vs `discretionary` flow. The graded policy widens and skews. A `predatory` research mode joins the wave and sells into peer covering; it is not the default grade.

`etf_ap_arb` asks the offline Decision client for `size_tier`. A near-risk-free flag returns Choice `large`. Policy may still cut `size_mult` on a low confidence Score. The case engine keeps max size. The model does not emit an order. No `TYPESAFE_API_KEY`.

## 6. Option-inventory market making

**Code:** `zig/src/option_mm.zig`, `jev_omm/quoter/option_mm.py`.  
**Mode:** `QuoterConfig.mode = "option_vega"` (C ABI mode `3`).  
**Cite:** Baldacci, Bergault, Guéant, https://arxiv.org/abs/1907.12433. Stoikov–Sağlam Theorem 4, https://doi.org/10.1007/s11147-009-9036-3. Lucic–Tse IV edge, https://ssrn.com/abstract=4729290.

Constant-vega state \(V^\pi\). Explicit Euler grid for the mean-variance value. Premiums from the Hamiltonian of an exponential or logistic intensity. Sides that would leave \(|V^\pi|\le\overline{\mathcal{V}}\) quote size zero. `vol_edge` is \((a_\mathbb{P}-a_\mathbb{Q})/(2\sqrt{\nu})\). `iv_alpha` shifts the reservation by contract vega times (theo IV − market IV). Spot–vol hedge in `hedge.spotVolHedgeQty`:

\[
q^{S*}=-\Delta-\frac{\rho\xi V^\pi}{2\sqrt{\nu}\,S}.
\]

Cash A–S, Guéant asymptotic, and Guéant ODE are unchanged when `mode` is not `option_vega`.

The unit-test toy is a small grid (`γ=0.5`, `ξ=1`, `A=40`, `k=2`, `V̄=40`, 31×60). It is not the euro-notional Baldacci §4 example.

## 7. Hawkes flow and deeper queue value

**Code:** `zig/src/hawkes.zig`, `jev_omm/flow/hawkes.py`.  
**Cite:** Hawkes 1971, https://doi.org/10.1093/biomet/58.1.83; Bacry–Mastromatteo–Muzy, https://arxiv.org/abs/1502.04592.

\(\lambda(t)=\mu+\sum\alpha e^{-\beta(t-t_i)}\). Excitation \((\lambda-\mu)/\mu\) is copied onto Decision `flow.hawkes_excitation` and raises the fallback toxicity Score. `fillIntensity` scales a baseline Poisson rate. It does not route orders.

`queueValue` in `lob.zig` is fills × (spread capture − adverse cost). `depthAhead` sums size in front of our level on a small multi-level book. The single-order fluid model from frontier 3 is unchanged.

## 8. Local vol, rough paths, variance swaps (research)

**Code:** `jev_omm/surface/dupire.py`, `jev_omm/surface/rough_vol.py`, `jev_omm/pricing/varswap.py`, `zig/src/varswap.zig`.

- Dupire local variance from total variance \(w(k,T)\), and a finite-difference check on flat Black calls. A flat smile returns \(\sigma_{\mathrm{loc}}=\sigma\). Python only.
- Fractional-Brownian covariance and a one-factor rough Bergomi variance path for stress sims ([arXiv:1410.3394](https://arxiv.org/abs/1410.3394)). Not a quoter.
- Variance-strike replication \(K_{\mathrm{var}}=(2/T)\int\mathrm{OTM}/K^2\,dK\) and the sqrt convexity adjustment for a vol swap. Stylized vega \(2\sigma\).

Details and links: [LITERATURE_CANON.md](./LITERATURE_CANON.md).

## 9. Flow and positioning (paper)

**Code:** `jev_omm/flow/signals.py`, `jev_omm/positioning/`, `zig/src/flow_signals.zig`, `zig/src/positioning.zig`.  
**C ABI:** `jev_omm_flow_prior`, `jev_omm_gex_adjust`, `jev_omm_cot_fade`.  
**Version string:** `0.6.0-zig-flow-positioning`.

Features are precomputed. Zig does no network I/O. Every scaler is the identity when its inputs are zero or its flag is off, so existing quote modes are unchanged.

| Edge | What landed | What it changes |
| --- | --- | --- |
| Signed tape | Lee–Ready, one-level OFI, aggressive imbalance, a synthetic off-exchange share, bucket VPIN | `flow_prior` → spread and size |
| Layered book | Cancel-behind-touch score (not a spoofing strategy) | same toxicity blend |
| Dealer gamma | OI × BS gamma × a dealer sign. `short_premium` (default, both signs −1) stays negative and has no flip. `dashboard_flip` (calls +1, puts −1) is the dashboard assumption that can cross zero | reservation shift, spread, size, hedge-band multiplier, hedge urgency |
| Pin | Max pain and the zero-gamma level, blended only if normalized GEX is positive | `pin_gap` into the long-gamma shift |
| Charm / vanna | Central-difference charm and existing vanna, as a hedge quantity | `overlay_hedge_qty`. Zero shocks add nothing |
| COT | Legacy, disaggregated, and TFF column maps. Net spec, commercial hedge ratio, week-over-week, trailing z. Equity index from TFF, commodities from the disaggregated file | `cot_fade` shifts reservation against the crowded side and cuts size |
| ETF / basis / beta | Create vs redeem pressure, annualized roll z, futures overlay `q = −exposure / β` | feature builders. Graded `etf_ap_arb` and `location_arb` scores are unchanged |

Training cases `flow_vpin`, `dealer_gamma`, and `cot_fade` grade the desk policy against a naive one. Python and Zig share the LCG and the scaler constants. Citations: [LITERATURE_CANON.md](./LITERATURE_CANON.md).

The public CFTC JSON endpoint is implemented and **gated** (`JEV_COT_NETWORK=1` or `allow_network=True`). Tests read `jev_omm/data/fixtures/` only. Fixture numbers are synthetic; the column names match the Socrata schema.

## Still later

- Live `TYPESAFE_API_KEY` / pinning `jev-1.x` (hooks exist; fallback is the default).
- Historical OPRA/LOB replay instead of the synthetic queue. Off-exchange volume here is a flag, not a FINRA feed.
- SEC Form 13F. Quarterly, lagged, and a poor fit for a quote-time feature without a holdings parser. Deferred on purpose.
- Full no-arbitrage SVI calibration with cross-expiry joint SSVI (joint MLE across expiries is not).
- The Baldacci §4 euro grid (20 strikes × 4 expiries, \(\overline{\mathcal{V}}=10^7\)) as a production lookup. The HJB here is a small research grid; `solveGrid` accepts several contract vegas, the quoter wrapper prices one name.
- Multi-agent peer market makers (the MM case is a synthetic wave plus cancel latency).
- Full Bergomi forward-variance curve, Heston PDE, American exercise.
- Color hedge bands. Charm and vanna are overlays on a precomputed shock, not a new band schedule. Vanna and volga already live on the term book.
- Two-name dispersion.
- Any live order path. Out of scope.
