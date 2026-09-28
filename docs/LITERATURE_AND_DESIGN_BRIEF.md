# Literature and Design Brief — Equity Options Market-Making Research System

**Owner:** Sean Collins  
**Repo root:** jevy ([SMC17/jevy](https://github.com/SMC17/jevy))  
**Scope:** Simulation / paper research only — no live exchange keys, no production routing  
**Companion docs:** [ARCHITECTURE.md](./ARCHITECTURE.md), [SYSTEM_ONE_JEV.md](./SYSTEM_ONE_JEV.md), [LITERATURE_CANON.md](./LITERATURE_CANON.md) (citations mapped to modules), [TRAINING_CASES.md](./TRAINING_CASES.md), [FRONTIERS.md](./FRONTIERS.md)

`0.5.0-zig-oom-citadel-lit` adds quoter mode `option_vega` (constant-vega inventory MM) and the training desk. `0.6.0-zig-flow-positioning` adds paper flow and positioning features (COT, signed tape, dealer gamma) that scale quotes and the hedge band when the caller turns them on. `0.7.0-zig-state-os` adds the latent-state gate: instability `|F|/L_exec` and the forced-flow engines in [STATE_OS.md](./STATE_OS.md). The canon file is the citation index; this brief stays the design narrative.

This brief consolidates **fetched** industry and academic sources into design principles for a professional equity-options MM research stack. Citations prefer pages actually retrieved; do not invent papers or URLs.

---

## 1. Design principles from industry systems (Jane Street & peers)

### 1.1 Tools for traders / options desk UX
Source: [Building Tools for Traders — Signals & Threads](https://signalsandthreads.com/building-tools-for-traders/) (Ian Henry).

- Options MM is a **low-dimensional opinion** (vol surface / moments) expressed through **hundreds of strikes × expiries** on CLObs — software must bridge that impedance mismatch.
- Desks need two tool classes: (1) **exploration / aggregation** across und×expiry; (2) **attention management** (alerts with configurable fatigue controls).
- Expert UIs: extreme information density, keyboard-first, fast feedback loops with traders sitting next to engineers.
- Full-stack typed protocols (shared types client↔server) beat ad-hoc JSON APIs for internal tooling velocity.
- Expect-test / snapshot workflows make complex trading state changes reviewable.

**Implication for jev-omm:** treat observability and attribution as first-class modules from v0; design quote/risk views around surface residuals and greek inventory, not single-symbol tickers.

### 1.2 Multicast, sequencers, mechanical sympathy
Source: [Multicast and the Markets — Signals & Threads](https://signalsandthreads.com/multicast-and-the-markets/) (Brian Nigito).

- Exchange fabric: **TCP order entry** + **UDP multicast market data**; fairness = identical anonymized stream to all.
- Core abstraction: a **sequencer** (often single-threaded / single core) imposing total order; consumers are **state-machine replicas**.
- Specialized “reliable multicast” that **never slows the publisher** for stragglers; gap-fill via retransmitters; domain-aware snapshots beat byte-replay.
- **Mechanical sympathy:** poll NIC, reduce copies, deterministic service times; Layer-1 crosspoints for ns-scale fanout; FPGA / programmable NICs for content-aware filter when aggregate exchange bandwidth exceeds host NICs.
- Speed/determinism → better prices under **fragmentation** (simulate “buy anywhere but not everywhere” by fast cancel).

**Implication:** even in a Python research sim, preserve **sequenced event logs**, deterministic replay, and clear separation of private fills vs public book — architecture of [How to Build an Exchange](https://www.janestreet.com/tech-talks/building-an-exchange/).

### 1.3 How to build an exchange (JX)
Source: [How to Build an Exchange — Jane Street tech talk](https://www.janestreet.com/tech-talks/building-an-exchange/) (Brian Nigito).

- LOB = price-time priority; messages ≈ add / cancel / execute.
- Requirements: scale (≈ millions msg/s class), fairness, durability, robustness to bad clients.
- Architecture: matching engine + ports + drop copies + market data + “Cancel Fairy”-style side logic — **multicast** fans state to all; retransmitters for UDP loss; **passive ME** follows primary output for failover.
- **State-machine replication** → weeks of replay + fuzz before rollout; timers must be sequenced events.
- Takeaway: performance buys **architectural simplicity**, not vanity latency.

**Implication:** Execution/Sim module should be a sequenced matching stub; Risk/Kill should be sidecar consumers of the same event stream.

### 1.4 Programmable hardware
Source: [Programmable Hardware — Signals & Threads](https://signalsandthreads.com/programmable-hardware/) (Andy Ray / Hardcaml; page also mirrored as signalsandthreads Programmable Hardware).

- FPGAs for line-rate packet parse/filter when software cannot absorb multi-venue 10–25Gb feeds; Hardcaml = OCaml DSL for hardware with software-grade testing.
- Research system **non-goal:** custom FPGA. **Goal:** design data paths that *could* move to hardware later (flat schemas, minimal copies, clear wire formats).

### 1.5 Practitioner engineering culture (Optiver / IMC / HRT — public pages)
Real public URLs (career/engineering narrative, not strategy whitepapers):

- [HRT — Engineering and interviewing](https://www.hudsonrivertrading.com/hrtbeat/engineering-and-interviewing-at-hrt/) — Trading Tech (latency, FPGA) vs R&D (scale); C++/Python split; options accel teams on live path.
- [IMC — Software engineering](https://www.imc.com/us/what-we-do/technology/software-engineering) — in-house stack; traders ↔ quants ↔ FPGA engineers.
- [Optiver engineering (Pragmatic Engineer interview)](https://newsletter.pragmaticengineer.com/p/optiver) — latency historically core; increasing weight on models/research infra.

**Implication:** separate **research/sim** (Python, this repo) from hypothetical **live path**; do not pretend v0 is colo FPGA.

---

## 2. Market-making theory stack

### 2.1 Avellaneda–Stoikov (2008)
- Paper: *High-frequency trading in a limit order book*, Quantitative Finance. Author PDF: https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf  
- DOI: https://doi.org/10.1080/14697680701381228  
- Core: reservation (indifference) price from inventory + utility; optimal bid/ask distance from mid via intensity λ(δ)=A e^{-kδ}; inventory reduces variance of PnL vs naive symmetric quotes.

**v0 use:** deterministic quoter core (`jev_omm/quoter/avellaneda_stoikov.py`).

### 2.2 Guéant–Lehalle–Fernandez-Tapia (arXiv 1105.3115)
- https://arxiv.org/abs/1105.3115 · PDF https://arxiv.org/pdf/1105.3115  
- Inventory-limited AS; HJB → linear ODE system; asymptotic quotes via spectral / closed-form approximations; extensions: drift, market impact / adverse selection.

**Landed:** asymptotic closed form (`gueant.zig`) and the finite-horizon linear ODE / principal eigenmode (`gueant_ode.zig`), toggled by `QuoterConfig.mode` (`as_finite_horizon` | `gueant_asymptotic` | `gueant_ode` | `option_vega`). \((A,k)\) MLE from a synthetic tape. Market-impact extension of the intensity is still future.

### 2.5 Option inventory (constant vega)

**Landed in `0.5.0`:** `QuoterConfig.mode = option_vega` routes reservation and half-spread through `option_mm.zig` / `jev_omm/quoter/option_mm.py`. Portfolio state is frozen vega \(V^\pi\). Premiums come from a small explicit-Euler HJB (exponential or logistic intensity) plus the Stoikov–Sağlam linear-intensity closed form. Spot–vol hedge tilt lives in `hedge.spotVolHedgeQty`. Citations and the toy parameter set: [LITERATURE_CANON.md](./LITERATURE_CANON.md). Cash A–S and Guéant stay the modes they were.

### 2.3 Cartea–Jaimungal–Penalva
- Book: *Algorithmic and High-Frequency Trading* (CUP, 2015) — https://www.cambridge.org/core/books/algorithmic-and-highfrequency-trading/  
- Ch. 10 market making: inventory limits, at-the-touch, volume choice, **adverse selection / short-term alpha**.  
- Companion code: https://sebastian.statistics.utoronto.ca/books/algo-and-hf-trading/code/

**v0+ use:** short-term alpha state in intensity; Decision Layer nouls for informed flow complement (not replace) α̂.

### 2.4 Classical precursors (cite via AS / GLFT discussions)
- Ho–Stoll inventory dealer models (referenced throughout AS / GLFT).  
- Use as historical framing only unless PDFs are fetched later.

---

## 3. Options-specific extensions

| Topic | Role in stack | Anchor sources |
| --- | --- | --- |
| **Greek-aware quoting** | Penalize quotes by projected Δ/Γ/ν inventory after fill; AS reservation on **delta-normalized** inventory | Desk practice + AS inventory term |
| **Delta hedge** | Separate Hedge module; urgency from inventory + System One `hedge_now` noul | Standard MM decomposition |
| **IV surface / SVI** | Raw SVI and SSVI total variance, butterfly \(g(k)\), calendar monotonicity, sticky-strike vs sticky-delta | Gatheral & Jacquier, https://arxiv.org/abs/1204.0646 |
| **IV surface / SABR** | One-slice alternative; Hagan et al. SABR (“Managing Smile Risk”, Wilmott 2002) — https://www.wilmott.com/managing-smile-risk/ | `surface.zig` |
| **Adverse selection** | VPIN / flow toxicity (Easley, López de Prado, O’Hara — *Flow Toxicity and Liquidity…*, RFS) DOI https://doi.org/10.1093/rfs/hhs053; intro note https://www.quantresearch.org/From%20PIN%20to%20VPIN.pdf; Kyle λ as price-impact per flow | Score/Noul battery in Decision Layer |
| **Surface residual gates** | `surface_suspect` noul + RMSE/z-score rules | Jane Street options tooling themes (surface as traded object) |

---

## 4. Decision Layer — TypeSafe System One / Jev (first-class)

**Not a chat bot.** Per https://typesafe.ai/blog/introducing-system-one-models-and-jev and https://docs.typesafe.ai/:

- Unstructured/structured **program state in** → **typed probabilistic decisions out**
- Primitives: **Choice** (choice+probs+confidence), **Score** (score+probs+confidence), **Noul** (P(yes)∈[0,1])
- Model: **`jev-latest`** via `TypeSafeClient.system_one(state=..., questions=...)` / `POST /v1/systemone`
- ~70–500 ms; parallel questions; RLCD-calibrated confidence
- Pattern: many independent questions; **branch on probabilities & confidence in code**
- **Never** emit orders from the model — actuators only

Full schemas, gates, and API-down fallback: **[SYSTEM_ONE_JEV.md](./SYSTEM_ONE_JEV.md)**.

v0 battery (summary): `regime` Choice · `toxicity` Score · `informed_flow` / `widen_quotes` / `pull_quotes` / `hedge_now` / `surface_suspect` Nouls · `size_tier` Choice → scale AS outputs / cancel / hedge.

---

## 5. Module / data-flow recommendations

See [ARCHITECTURE.md](./ARCHITECTURE.md). Pipeline:

`MarketData → Surface/FairValue → Quoter (AS + greek penalties) → Decision (System One) → Risk/Inventory → Hedge → Execution/Sim → PnL/Attribution → Observability`

Event log is append-only and replayable (exchange-inspired sequencer discipline).

---

## 6. v0 vs later phases

### v0 (this foundation + thin Python)
- Single name / few options; synthetic or historical L1/L2 replay
- BS or simple parametric surface; AS quoter with inventory skew
- Soft greek limits; naive delta hedge in sim
- System One client **optional** (fallback rules if no API key)
- PnL attribution: spread capture vs inventory vs hedge slippage
- **No** live keys, **no** smart-order router to real venues

### Later
- Full Guéant ODE / spectral quotes; Cartea short-term α
- SABR/SSVI calibration; multi-expiry surface arb checks
- Proper VPIN / Kyle λ estimators on tick data
- Multi-name, portfolio risk, auction modes
- FPGA-shaped market-data gateway (research spike only)
- Paper trading adapters still without production secrets in-repo

---

## 7. Curated reading list (real URLs + one-line why)

1. https://signalsandthreads.com/building-tools-for-traders/ — Options desk tooling, surface impedance, alerts.
2. https://signalsandthreads.com/multicast-and-the-markets/ — Multicast MD, sequencers, mechanical sympathy.
3. https://signalsandthreads.com/programmable-hardware/ — FPGA / Hardcaml context for feed handling.
4. https://www.janestreet.com/tech-talks/building-an-exchange/ — JX: SMR, multicast, ports, replay testing.
5. https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf — Canonical AS MM model.
6. https://arxiv.org/abs/1105.3115 — Inventory-limited MM + closed-form asymptotics (Guéant–Lehalle–Fernandez-Tapia).
7. https://www.cambridge.org/core/books/algorithmic-and-highfrequency-trading/ — CJP book; Ch.10 MM + adverse selection.
8. https://sebastian.statistics.utoronto.ca/books/algo-and-hf-trading/code/ — Notebooks for CJP MM chapter.
9. https://doi.org/10.1093/rfs/hhs053 — VPIN / flow toxicity (adverse selection measure).
10. https://www.wilmott.com/managing-smile-risk/ — SABR smile dynamics for surface FV.
11. https://typesafe.ai/blog/introducing-system-one-models-and-jev — System One / Jev product thesis.
12. https://docs.typesafe.ai/ — Choice / Score / Noul, confidence routing, Python SDK.
13. https://www.hudsonrivertrading.com/hrtbeat/engineering-and-interviewing-at-hrt/ — Prop-shop eng topology.
14. https://www.imc.com/us/what-we-do/technology/software-engineering — MM firm eng culture (public).

## 8. Explicit non-goals

- Live trading, exchange API keys, or secrets in this repository  
- Claiming production latency / FPGA parity in Python v0  
- Replacing deterministic pricing/greeks with LLM string generation  
- Unverified citations or fake “Optiver secret sauce” blogs  
- CSAM, market manipulation how-tos, or bypassing venue rules  

**Operating mode:** research simulation and paper-style replay only.


---

## 8. Akuna Options 101/201 + desk curriculum (landed v0.3)

Public anchors (fetched / cited only):
- https://akunacapital.com/work-with-us/options-101/
- https://blog.moontower.ai/implying-the-cost-of-carry-in-options/
- optionseducation.org put-call parity; Wikipedia box spread
- Natenberg dynamic hedge framing (γ-PnL ≈ ½ Γ (ΔS)² vs θ)
- VPIN / flow toxicity framing: DOI https://doi.org/10.1093/rfs/hhs053 ; intro https://www.quantresearch.org/From%20PIN%20to%20VPIN.pdf

**Landed:** European PCP + synthetics/CR + boxes (`parity.zig`); combo theos (`combos.zig`); banded delta hedge + slippage + greek PnL buckets (`hedge.zig`); research-grade toxicity features → Decision state (`toxicity.zig`); scenario spot×IV matrix (`scenario.zig`).

Full topic→file map: [AKUNA_AND_DESK_CURRICULUM.md](./AKUNA_AND_DESK_CURRICULUM.md).
