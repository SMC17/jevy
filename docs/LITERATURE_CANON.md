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

## Flow, positioning, dealer gamma

Verified links only. Synthetic fixtures stand in for the files. No vendor pull is required.

| Source | Module | Test |
| --- | --- | --- |
| CFTC Commitments of Traders. [Index](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm), [historical zips](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm), [TFF notes](https://www.cftc.gov/sites/default/files/idc/groups/public/@commitmentsoftraders/documents/file/tfmexplanatorynotes.pdf), [TFF field list](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalViewable/cotvariablestfm.html). Socrata ids: legacy `6dca-aqww`, disaggregated `72hh-3qpy`, TFF `gpe5-46if` on `publicreporting.cftc.gov`. | `positioning/cot.py`, fixtures in `jev_omm/data/fixtures/` | `test_cot_fixtures_map_and_extreme_z_and_network_gate`, `test_training_flow_gamma_and_cot` |
| Lee & Ready, *Inferring Trade Direction from Intraday Data*, Journal of Finance 46 (1991). [DOI 10.1111/j.1540-6261.1991.tb02683.x](https://doi.org/10.1111/j.1540-6261.1991.tb02683.x) | `flow/signals.py` `LeeReady`, `flow_signals.zig` | `test_lee_ready_ofi_and_spoof`, Zig `lee-ready quote test then tick test` |
| Cont, Kukanov, Stoikov, *The Price Impact of Order Book Events*, Journal of Financial Econometrics 12 (2014). [DOI 10.1093/jjfinec/nbt003](https://doi.org/10.1093/jjfinec/nbt003), [arXiv:1011.6402](https://arxiv.org/abs/1011.6402) | `ofi_increment` | `test_lee_ready_ofi_and_spoof` |
| Easley, López de Prado, O'Hara VPIN, already cited above | `flow_prior` weights the existing bucket VPIN | `test_training_flow_gamma_and_cot` |
| Cartea, Jaimungal, Wang, *Spoofing and Price Manipulation in Order-Driven Markets*, Applied Mathematical Finance 27 (2020). [DOI 10.1080/1350486X.2020.1726783](https://doi.org/10.1080/1350486X.2020.1726783) | `spoof_score` — layered cancel fraction, not their control problem | `test_lee_ready_ofi_and_spoof` |
| Barbon & Buraschi, *Gamma Fragility*. [SSRN 3725454](https://doi.org/10.2139/ssrn.3725454), [PDF](https://www.abarbon.com/assets/Barbon_Buraschi_2021_Gamma_Fragility.pdf) | `gex_adjust` / `gexAdjust`: short gamma widens and tightens the hedge band; long gamma leans to a pin and widens the band | `test_apply_features_identity_and_golden_numbers`, Zig `long gamma leans to the pin` |
| Garleanu, Pedersen, Poteshman, *Demand-Based Option Pricing*, RFS 22 (2009) 4259–4299. [NBER w11843](https://doi.org/10.3386/w11843) | `short_premium` posture (dealers short the wings end users buy, especially index puts). Not a measured inventory | `test_gex_postures_flip_and_max_pain_limit` |
| Ni, Pearson, Poteshman, *Stock Price Clustering on Option Expiration Dates*, JFE 78 (2005). [DOI 10.1016/j.jfineco.2004.08.005](https://doi.org/10.1016/j.jfineco.2004.08.005) | `pin_level` only when dealer gamma is positive | same |
| Avellaneda & Lipkin, *A market-induced mechanism for stock pinning*, Quantitative Finance 3 (2003). [DOI 10.1088/1469-7688/3/6/301](https://doi.org/10.1088/1469-7688/3/6/301) | same pin candidate | same |
| Max pain (holder-intrinsic minimizer) | `max_pain`. Documented limit: not a forecast. Ignored when GEX is negative | `test_gex_postures_flip_and_max_pain_limit` |

Charm is a central difference of BS delta (`charm_tau` / `charmTau`, step `1e-4`). Vanna is the greek already on the BS pricer. Both enter `charm_vanna_hedge` as a hedge overlay. Zero shocks leave the hedge quantity unchanged.

Put/call OI and a 25-delta risk-reversal stress are spread multipliers in `apply_features`. They are 1 when the inputs are 0. ETF create/redeem pressure and the futures-overlay quantity deepen the existing training cases; they do not change those scores.

## Decision layer

TypeSafe System One / Jev: [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [docs](https://docs.typesafe.ai/). Choice / Score / Noul only. `DeterministicFallbackClient` is the no-key path. Tests: `tests/test_decisions_client.py`, `tests/test_decisions_policy.py`.

## Latent state and forced flow (`0.7.0`)

Map and formulas: [STATE_OS.md](./STATE_OS.md). Tests: `tests/test_state_os.py` and the Zig `state_os` / training tests. Fixture files under `jev_omm/data/fixtures/state_os_*.json` are synthetic.

| Source | Module | Test |
| --- | --- | --- |
| Cheng & Madhavan, *The Dynamics of Leveraged and Inverse Exchange-Traded Funds*, JOIM Q4 2009. [SSRN 1539120](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1539120) | `letf_rebalance` | `test_standup_engines_and_fixtures`, Zig `letf, tdf, coverage` |
| Federal Reserve H.4.1, Treasury DTS, NY Fed ON RRP (links in STATE_OS.md) | `net_liquidity` | same. Levels in the fixture are not a release |
| 17 CFR § 240.10b-18. [eCFR](https://www.ecfr.gov/current/title-17/chapter-II/part-240/section-240.10b-18). Bettis, Coles, Lemmon, JFE 2000. [DOI 10.1016/S0304-405X(00)00055-6](https://doi.org/10.1016/S0304-405X(00)00055-6) | `in_issuer_blackout` | caller dates only |
| Perold & Sharpe, FAJ 1988. [DOI 10.2469/faj.v44.n1.16](https://doi.org/10.2469/faj.v44.n1.16) | `pension_equity_trade` | exact and linear signs |
| Moreira & Muir, JF 2017. [DOI 10.1111/jofi.12513](https://doi.org/10.1111/jofi.12513) | `vol_target_weight` | boundary step |
| Hurst, Ooi, Pedersen, JPM 2017. [DOI 10.3905/jpm.2017.44.1.015](https://doi.org/10.3905/jpm.2017.44.1.015). Hamill, Rattray, Van Hemert. [SSRN 2831926](https://doi.org/10.2139/ssrn.2831926) | `cta_weight` | uptrend sign |
| Asness, Frazzini, Pedersen, FAJ 2012. [DOI 10.2469/faj.v68.n1.1](https://doi.org/10.2469/faj.v68.n1.1) | `risk_parity_weights` | inverse-vol |
| Vanguard, *The rebalancing edge*. [PDF](https://corporate.vanguard.com/content/dam/corp/research/pdf/the_rebalancing_edge_optimizing_target_date_fund_rebalancing_through_threshold_based_strategies.pdf) | `tdf_trade` 200/175 | `tdf_threshold` |
| Cboe BXM. [Dashboard](https://www.cboe.com/us/indices/dashboard/bxm/) | `gen1_roll_dates`, `gen3_coverage` | twelve Fridays; rich IV raises cover |
| Cboe SPX contract facts. [Page](https://www.cboe.com/tradable_products/sp_500/spx_options/). CME E-mini point value 50 | `spx_vanna_charm_es` | SPY conversion is null |
| Cushing & Madhavan, JFM 2000. [DOI 10.1016/S1386-4181(99)00014-0](https://doi.org/10.1016/S1386-4181(99)00014-0) | `auction_imbalance` | 80/20 → 0.6 |
| D'Avolio, JFE 2002. [DOI 10.1016/S0304-405X(02)00206-4](https://doi.org/10.1016/S0304-405X(02)00206-4). Duffie, Gârleanu, Pedersen, JFE 2002. [DOI 10.1016/S0304-405X(02)00226-X](https://doi.org/10.1016/S0304-405X(02)00226-X) | `borrow_pressure` | fixture, not a vendor tape |
| Du, Tepper, Verdelhan, JF 2018. [DOI 10.1111/jofi.12620](https://doi.org/10.1111/jofi.12620) | `in_slr_window`, CIP term in `scarcity_rent` | window and SOFR-versus-specials |
| Duffie, *Special Repo Rates*, JF 1996. [DOI 10.1111/j.1540-6261.1996.tb02692.x](https://doi.org/10.1111/j.1540-6261.1996.tb02692.x) | `scarcity_rent` | specials outweigh a SOFR change |
| Eisler, Bouchaud, Kockelkoren, QF 2012. [DOI 10.1080/14697688.2010.528444](https://doi.org/10.1080/14697688.2010.528444) | `latent_book` | ∂E[C]/∂σ |
| Tóth et al., Phys. Rev. X 2011. [DOI 10.1103/PhysRevX.1.021006](https://doi.org/10.1103/PhysRevX.1.021006). Bacry, Iuga, Lasnier, Lehalle, MML 2015. [DOI 10.1142/S2382626615500094](https://doi.org/10.1142/S2382626615500094). Almgren, Thum, Hauptmann, Li, Risk, July 2005 | `metaorder` | concave impact, partial reversion, noise rejected |
| Benzaquen, Mastromatteo, Eisler, Bouchaud, J. Stat. Mech. 2017. [DOI 10.1088/1742-5468/aa53f7](https://doi.org/10.1088/1742-5468/aa53f7) | `cross_impact` | common-flow trap |
| Filimonov & Sornette, Phys. Rev. E 2012. [DOI 10.1103/PhysRevE.85.056108](https://doi.org/10.1103/PhysRevE.85.056108). Hardiman, Bercot, Bouchaud, EPJB 2013. [DOI 10.1140/epjb/e2013-40107-3](https://doi.org/10.1140/epjb/e2013-40107-3) | `hawkes_tv` | endogenous versus exogenous |
| Guillaume, JOD 2015. [DOI 10.3905/jod.2015.22.3.073](https://doi.org/10.3905/jod.2015.22.3.073) | `autocall_flow` | mass on the barrier |
| Koijen & Yogo, AER 2015. [DOI 10.1257/aer.20121036](https://doi.org/10.1257/aer.20121036) | `hedge_by_regime` | economic ≠ statutory |
| Richard & Roll, JF 1989. [DOI 10.1111/j.1540-6261.1989.tb05062.x](https://doi.org/10.1111/j.1540-6261.1989.tb05062.x) | `mbs_hedge` | deep discount is off |
| Bank of England FSR, December 2022. [Report](https://www.bankofengland.co.uk/financial-stability-report/2022/december-2022) | `ldi_cash_need` | velocity gate, not a live feed |
