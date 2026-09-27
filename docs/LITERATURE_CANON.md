# Literature canon

Citations used by this tree. URLs below were checked against the public record. Books are cited by author, title, and publisher when no stable free URL was verified — those lines do not invent a link.

Map: citation → module → test. Simulation / paper only.

## Training desk (practitioner write-ups)

| Source | Module | Test |
| --- | --- | --- |
| [Predicting Alpha — What I learned from Citadel's Training Software](https://www.predictingalpha.com/blogs/what-i-learned-from-citadels-training-software) (Sean Ryan, 17 Mar 2023). The `/p/` path on that host 404s; use `/blogs/`. | `jev_omm/training/`, `zig/src/training.zig` | `tests/test_training.py` |
| [Reddit r/Trading](https://www.reddit.com/r/Trading/comments/122y2zq/what_i_learned_from_citadels_training_software/) | same | same |
| [Reddit r/options](https://www.reddit.com/r/options/comments/122pz4e/what_i_learned_from_citadels_training_software/) | same | same |
| [Medium / DataDrivenInvestor mirror](https://medium.datadriveninvestor.com/this-is-what-citadels-training-software-taught-me-741c3996a5b5) | same | same |
| [r/quant Chicago prop thread](https://www.reddit.com/r/quant/comments/pwzknt/small_prop_trading_firms_in_chicago/hekivya/) | curriculum note only — desk culture / Chicago MM ecosystem. Not a case and not a firm directory. | — |

Case-by-case lesson map: [TRAINING_CASES.md](./TRAINING_CASES.md).

## Cash market making

| Source | Module | Test |
| --- | --- | --- |
| Avellaneda & Stoikov, *High-frequency trading in a limit order book*, Quantitative Finance 2008. [PDF](https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf), [DOI 10.1080/14697680701381228](https://doi.org/10.1080/14697680701381228) | `as_quoter.zig`, `quoter/avellaneda_stoikov.py` | `tests/test_quoter_skew.py` |
| Guéant, Lehalle, Fernandez-Tapia, [arXiv:1105.3115](https://arxiv.org/abs/1105.3115) | `gueant.zig`, `gueant_ode.zig` | `tests/test_gueant.py`, `tests/test_gueant_ode.py` |
| Cartea, Jaimungal, Penalva, *Algorithmic and High-Frequency Trading*, CUP 2015. [Cambridge](https://www.cambridge.org/core/books/algorithmic-and-highfrequency-trading/) | Decision state, fill intensity | `tests/test_decisions_policy.py` |
| Almgren & Chriss, *Optimal execution of portfolio transactions*, Journal of Risk 3, 2001 | cited only; no execution solver in-tree | — |

## Option inventory market making

| Source | Module | Test |
| --- | --- | --- |
| Stoikov & Sağlam, *Option market making under inventory risk*, Review of Derivatives Research 12, 2009. [DOI 10.1007/s11147-009-9036-3](https://doi.org/10.1007/s11147-009-9036-3), [SSRN 1393818](https://ssrn.com/abstract=1393818), [Cornell PDF](https://people.orie.cornell.edu/sfs33/StoikovSaglam.pdf) | `option_mm.zig` `stoikovSaglamPremiums`, `quoter/option_mm.py` | `test_stoikov_saglam_theorem4_toy` |
| Baldacci, Bergault, Guéant, *Algorithmic market making for options*, [arXiv:1907.12433](https://arxiv.org/abs/1907.12433) | `option_mm.zig` HJB grid, `hedge.spotVolHedgeQty` | `tests/test_option_mm.py`, Zig `long portfolio vega widens the bid` |
| El Aoud & Abergel, *A stochastic control approach for options market making*, Market Microstructure and Liquidity 2015. [HAL hal-01061852](https://hal.science/hal-01061852), [DOI 10.1142/S2382626615500069](https://doi.org/10.1142/S2382626615500069) | cited; single-name Δ-hedged control is the ancestor of the vega reduction | — |
| Lucic & Tse, *Optimal option market making and volatility arbitrage*, [SSRN 4729290](https://ssrn.com/abstract=4729290), [DOI 10.2139/ssrn.4729290](https://doi.org/10.2139/ssrn.4729290) | `iv_alpha` reservation shift (theo IV − market IV) | `test_iv_alpha_lifts_reservation` |

Implemented Baldacci objects:

- Portfolio vega \(V^\pi=\sum q_i\mathcal{V}^i\) with \(\mathcal{V}^i\) frozen (constant-vega).
- Objective spread \(+ V^\pi(a_\mathbb{P}-a_\mathbb{Q})/(2\sqrt{\nu}) - (\gamma\xi^2/8)(1-\rho^2)(V^\pi)^2\).
- Intensity exponential \(\Lambda(\delta)=A e^{-k\delta}\), or logistic \(\Lambda(\delta)=\lambda/(1+e^{\alpha+\beta\delta/\mathcal{V}^i})\). Touch probability at \(\delta=0\) is \(1/(1+e^{\alpha})\). The paper's numerical section uses \(\alpha=0.7\) (about 33%).
- Hard set \(|V^\pi|\le\overline{\mathcal{V}}\). A side that would leave the set is size zero.
- Appendix hedge \(q^{S*}=-\Delta^\pi - \rho\xi V^\pi/(2\sqrt{\nu}\,S)\).

The research toy (`researchToy` / `research_toy`) uses a small vega grid so the skew is visible in unit tests. It is not the euro-notional example in Baldacci §4 (\(\overline{\mathcal{V}}=10^7\), \(\gamma=10^{-3}\), \(\xi=0.2\), \(\rho=-0.5\)).

Stoikov–Sağlam Theorem 4 (linear intensity \(\lambda(\varepsilon)=C-D\varepsilon\), one period, delta-hedged):

\[
\varepsilon^{a}=\mathrm{clip}_{[0,C/D]}\!\left(\frac{C}{2D}-\gamma k(q-\tfrac12)\right),\quad
\varepsilon^{b}=\mathrm{clip}_{[0,C/D]}\!\left(\frac{C}{2D}+\gamma k(q+\tfrac12)\right)
\]

with the printed risk scale

\[
k=\left(\tfrac12\sigma^2(T-t_n)+\alpha^2(T_{\mathrm{mat}}-t_n)^2\right)\Gamma^2 S^4\sigma^2(T-t_n).
\]

Figure-style constants used in comments and `stoikov_risk_scale`: \(C=40\), \(D=200\), \(S=100\), \(\sigma=0.01\). Risk-neutral premiums are \(C/(2D)=0.1\).

## Surface, local vol, rough vol

| Source | Module | Test |
| --- | --- | --- |
| Gatheral & Jacquier, *Arbitrage-free SVI volatility surfaces*, [arXiv:1204.0646](https://arxiv.org/abs/1204.0646) | `svi.zig`, `surface/svi.py` | `tests/test_svi.py`, `test_vol_surface_refuses_arb` |
| Hagan et al., SABR, Wilmott 2002. [wilmott.com/managing-smile-risk](https://www.wilmott.com/managing-smile-risk/) | `surface.zig`, `surface/sabr.py` | `tests/test_surface_sabr.py` |
| Dupire, *Pricing with a Smile*, Risk, 1994. No free canonical URL verified | `surface/dupire.py` (Python research) | `test_flat_total_variance_is_flat_local_vol`, `test_dupire_on_flat_black_recovers_iv` |
| Gatheral, Jaisson, Rosenbaum, *Volatility is rough*, [arXiv:1410.3394](https://arxiv.org/abs/1410.3394), [DOI 10.1080/14697688.2017.1393551](https://doi.org/10.1080/14697688.2017.1393551) | `surface/rough_vol.py` | `test_fbm_variance_and_rough_bergomi_positive` |
| Gatheral, *The Volatility Surface*, Wiley 2006 | total-variance Dupire algebra in `dupire.py` | same Dupire tests |
| Bergomi, *Stochastic Volatility Modeling*, Chapman & Hall 2016 | not implemented as a full forward-variance model | — |

## Variance and vol swaps

| Source | Module | Test |
| --- | --- | --- |
| Demeterfi, Derman, Kamal, Zou, *More Than You Ever Wanted to Know About Volatility Swaps*, Goldman Sachs Quantitative Strategies, 1999. [PDF](https://emanuelderman.com/wp-content/uploads/1999/02/gs-volatility_swaps.pdf), [DOI 10.3905/jod.1999.319129](https://doi.org/10.3905/jod.1999.319129) | `pricing/varswap.py`, `varswap.zig` | `test_variance_swap_weights_and_flat_replication` |
| Bossu, Strasser, Guichard, *Just what you need to know about Variance Swaps*. [PDF](http://docs.sbossu.com/bossu-strasser-guichard-varswap.pdf). Related: [Wilmott introduction](https://www.wilmott.com/introduction-to-variance-swaps-wilmott-magazine-article-sebastien-bossu/) | `volSwapFromVariance` convexity | `test "stylized vega and vol convexity"` |

Replication used here: \(K_{\mathrm{var}}=(2/T)\,df\int \mathrm{OTM}(K)/K^2\,dK\). Flat-smile vega of that strike is \(2\sigma\). Vol strike \(\approx\sqrt{K_{\mathrm{var}}}\,(1-\mathrm{Var}/(8 K_{\mathrm{var}}^2))\).

## Microstructure

| Source | Module | Test |
| --- | --- | --- |
| Hawkes, *Spectra of some self-exciting and mutually exciting point processes*, Biometrika 58(1), 1971. [DOI 10.1093/biomet/58.1.83](https://doi.org/10.1093/biomet/58.1.83) | `hawkes.zig`, `flow/hawkes.py` → Decision `flow.hawkes_*` | `test_hawkes_burst_then_decay` |
| Bacry, Mastromatteo, Muzy, *Hawkes processes in finance*, [arXiv:1502.04592](https://arxiv.org/abs/1502.04592), [DOI 10.1142/S2382626615500057](https://doi.org/10.1142/S2382626615500057) | same | `test_hawkes_excitation_raises_fallback_toxicity` |
| Easley, López de Prado, O'Hara, flow toxicity. [DOI 10.1093/rfs/hhs053](https://doi.org/10.1093/rfs/hhs053), [intro note](https://www.quantresearch.org/From%20PIN%20to%20VPIN.pdf) | `toxicity.zig`, `flow/toxicity.py` | `tests/test_toxicity.py` |
| Synthetic queue / cancel latency (desk practice; fluid limit in `lob.zig`) | `lob.zig` `queueValue`, `execution/lob.py` | `tests/test_lob.py`, `test_queue_value_falls_with_depth_when_the_spread_pays` |

## Desk books (no free URL verified)

These are the onboarding list from the r/quant consensus in the design brief. They are not re-implemented beyond the pieces already in the Akuna map (parity, bands, greek PnL, scenarios).

- Natenberg, *Option Volatility and Pricing*. Greek PnL in `hedge.zig` (`test "gamma pnl = 1/2 Gamma (dS)^2"`).
- Hull, *Options, Futures, and Other Derivatives*.
- Taleb, *Dynamic Hedging*.
- Sinclair, *Volatility Trading*; *Positional Option Trading*.
- Bennett, *Trading Volatility*.
- Baird, *Option Market Making*.
- Wilmott (selected chapters); Whalley–Wilmott band helper in `hedge.zig`.
- Bouchaud, Bonart, Donier, Gould, *Trades, Quotes and Prices*, Cambridge, 2018.
- Guéant, *The Financial Mathematics of Market Liquidity*, CRC, 2016.
- Moontower / Kris Abdelmessih, [implying the cost of carry](https://blog.moontower.ai/implying-the-cost-of-carry-in-options/) — parity and boxes.

## Decision layer

TypeSafe System One / Jev: [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [docs](https://docs.typesafe.ai/). Choice / Score / Noul only. `DeterministicFallbackClient` is the no-key path. Tests: `tests/test_decisions_client.py`, `tests/test_decisions_policy.py`.
