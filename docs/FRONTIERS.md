# Frontiers 1–4 — surface, term risk, queue fills, Guéant ODE

**Version:** `0.4.0-zig-frontiers-1-4` (`jev_omm_version`)  
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

## Still later

- Live `TYPESAFE_API_KEY` / pinning `jev-1.x` (hooks exist; fallback is the default).
- Historical OPRA/LOB replay instead of the synthetic queue.
- Full no-arbitrage SVI calibration with cross-expiry joint SSVI (shared \(\rho,\eta,\gamma\), per-expiry \(\theta\) is evaluated; joint MLE across expiries is not).
- American, local vol, Heston.
- Any live order path. Out of scope.
