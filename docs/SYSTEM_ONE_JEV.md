# System One / Jev Decision Layer

**Audience:** Sean Collins — options MM research system (`jev-omm`)  
**Status:** Design foundation (sim / paper only)  
**Primary citations:**
- Product announcement: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- API docs: https://docs.typesafe.ai/
- Primitives: https://docs.typesafe.ai/primitives.md
- Confidence: https://docs.typesafe.ai/confidence.md
- Confidence-gated routing: https://docs.typesafe.ai/patterns/confidence-routing.md
- How to build: https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md
- Python SDK: https://docs.typesafe.ai/sdk/python.md

---

## 1. What Jev is (and is not)

**Jev** is TypeSafe’s flagship **System One** model: a frontier model class for **automation**, not chat.

| Property | Chat LLMs | System One / Jev |
| --- | --- | --- |
| Training objective | RLHF / preference / verifiable text | **RLCD** — Reinforcement Learning for Calibrated Decisions |
| Input emphasis | Sequential messages | **Structured program state** |
| Output | Strings (must parse/validate) | **Typed** Choice / Score / Noul + probabilities |
| Sampling | Autoregressive tokens | **Parallel** evaluation of all questions |
| Latency (typical) | Seconds | ~70–500 ms end-to-end |
| Role in this system | Not used on the quote path | **Smart if-statements** inside deterministic workflows |

Jev is **not** this chat assistant. It never emits free-form text on the trading path. It never emits orders. Code owns control flow; Jev only answers narrow typed questions.

Live `TYPESAFE_API_KEY` wiring is optional. This tree does not require a key: `DeterministicFallbackClient` answers the same Choice / Score / Noul questions offline. Do not invent a key.

From TypeSafe’s design rules ([how-to-build](https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md)):

1. **Code owns control flow** — deterministic AS / Black–Scholes / greeks math stays in Python.
2. **Decompose** into many atomic parallel questions on one state.
3. **Confidence-gated routing** — act / widen / escalate / flatten based on confidence + nouls.
4. **Composite scoring** — combine nouls/scores with weights **in code**.
5. **Speculative fan-out** — ask extra questions in the same call; ignore unused answers.
6. **Never let the model emit orders** — only typed decisions that **actuators** consume.

---

## 2. API surface (v0)

### Endpoint / SDK

- HTTP: `POST /v1/systemone` (see https://docs.typesafe.ai/api.md)
- Model alias: **`jev-latest`**
- Python SDK (`typesafe-sdk`):

```python
from typesafe_sdk import TypeSafeClient, Choice, Score, Noul

with TypeSafeClient() as client:  # TYPESAFE_API_KEY in env
    response = client.system_one(
        state=mm_state,           # structured program state
        questions=mm_questions,   # map of Choice | Score | Noul
        # model defaults / selects jev-latest per SDK docs
    )
```

Async variant: `AsyncTypeSafeClient` + `await client.system_one(...)`.

### Primitives (exact return shapes)

| Primitive | Question shape | Returns | Use in MM |
| --- | --- | --- | --- |
| **Choice** | Select one of a closed set | `choice`, `probabilities`, `confidence` | Regime, size tier |
| **Score** | Ordered rubric levels | `score`, `probabilities`, `confidence` | Toxicity levels |
| **Noul** | P(yes) ∈ [0, 1] | `noul` only (no separate confidence) | Informed flow, widen, pull, hedge, surface suspect |

All questions in one request share the same `state`, evaluate **in parallel and in isolation**, and return under the keys you chose ([primitives](https://docs.typesafe.ai/primitives.md)).

---

## 3. Contract: state → questions → actuators

```
┌─────────────────────┐
│  Deterministic core │  MarketData, Surface, Greeks, AS reservation/spread
│  builds mm_state    │
└──────────┬──────────┘
           │ state snapshot (JSON)
           ▼
┌─────────────────────┐
│  System One / Jev   │  Choice / Score / Noul battery (parallel)
│  jev-latest         │  ~70–500 ms, typed probs + confidence
└──────────┬──────────┘
           │ calibrated answers
           ▼
┌─────────────────────┐
│  Decision policy    │  Pure Python: thresholds, weights, gates
│  (code, not model)  │  → QuoteMods | HedgeCmd | RiskMode
└──────────┬──────────┘
           │ constrained actuators only
           ▼
┌─────────────────────┐
│  Quoter / Hedge /   │  Never call System One for price math or order IDs
│  Execution / Sim    │
└─────────────────────┘
```

**Invariant:** classical Avellaneda–Stoikov reservation price, Black–Scholes / SABR fair value, and greek limits are **deterministic**. Jev wraps **brittle heuristic decisions** (toxicity, regime, aggressiveness, urgency, kill/widen).

---

## 4. v0 MM decision battery

### 4.1 State schema (program state, not a chat prompt)

Build a dense, structured snapshot each quote cycle. Prefer numbers and enums over prose.

```python
mm_state = {
    "ts": "...",
    "underlying": "NVDA",
    "book": {
        "bid": 118.40, "ask": 118.42, "bid_sz": 400, "ask_sz": 350,
        "microprice": 118.409, "spread_ticks": 2,
        "imbalance": 0.12,  # (bid_sz - ask_sz) / (bid_sz + ask_sz)
    },
    "flow": {
        "last_trades": [...],          # side, size, price, aggressor
        "signed_volume_30s": -1200,
        "trade_intensity_z": 1.8,
        "vpin_proxy": 0.62,            # research proxy, not live VPIN license claim
    },
    "inventory": {
        "q_underlying_equiv": 1.2,     # delta-normalized
        "q_option_contracts": {"...": 12},
        "net_delta": 85.0,
        "net_vega": -40.0,
        "net_gamma": 12.0,
    },
    "surface": {
        "atm_iv": 0.34,
        "fit_rmse": 0.004,
        "max_residual_z": 2.1,
        "skew_1m": -0.08,
    },
    "events": {
        "earnings_window": False,
        "fomc_window": False,
        "halt_or_luld": False,
        "news_flag": "none",           # none | soft | hard
    },
    "latency": {
        "md_age_ms": 3.2,
        "loop_lag_ms": 1.1,
        "feed_gap": False,
    },
}
```

### 4.2 Questions (one `system_one` call)

```python
from typesafe_sdk import Choice, Score, Noul

MM_QUESTIONS = {
    "regime": Choice(
        instructions=(
            "Given book, flow, inventory, surface residuals, and event flags, "
            "which market regime best describes the next quoting horizon?"
        ),
        criteria={
            "calm": "Stable book, normal trade intensity, no event window",
            "trend": "Directional signed volume / microprice drift dominating",
            "event": "News, earnings, FOMC, halt/LULD, or discontinuous flow",
            "auction": "Open/close auction or crossed/locked special session",
        },
    ),
    "toxicity": Score(
        instructions=(
            "Score adverse-selection / flow toxicity for a passive options MM "
            "posting at current touch distances."
        ),
        criteria=[
            "Benign retail-like flow; little informed pressure",
            "Mild toxicity; slight edge erosion if size is large",
            "Elevated informed flow; widen and cut size",
            "Severe toxicity; prefer pull / widen-only",
        ],
    ),
    "informed_flow": Noul(
        instructions="Is current aggressor flow likely informed against our quotes?",
    ),
    "widen_quotes": Noul(
        instructions="Should we inflate AS half-spreads vs the deterministic baseline?",
    ),
    "pull_quotes": Noul(
        instructions="Should we cancel outstanding quotes (go widen-only / flat quote)?",
    ),
    "hedge_now": Noul(
        instructions="Should the hedge module act immediately on residual delta?",
    ),
    "size_tier": Choice(
        instructions="What quote size tier is appropriate given toxicity and inventory?",
        criteria={
            "tiny": "Minimum size or stub quotes only",
            "normal": "Baseline AS size",
            "large": "Size up; book is calm and inventory room exists",
        },
    ),
    "surface_suspect": Noul(
        instructions="Are IV surface residuals / fit quality unreliable for fair value?",
    ),
}
```

### 4.3 Decision policy (code) — confidence gates

Choice/Score carry `confidence` ∈ [0, 1] derived from the probability distribution ([confidence](https://docs.typesafe.ai/confidence.md)). Noul is already a probability; treat mid-range nouls as uncertain.

Recommended v0 gates (tune on sim; stakes-aware):

| Signal | Gate | Actuator effect |
| --- | --- | --- |
| Any Choice `confidence < 0.55` | **escalate / flatten** | Cancel → widen-only; do not size up |
| `pull_quotes.noul ≥ 0.65` **or** toxicity score ≥ 2.5 | **pull** | Cancel resting quotes |
| `widen_quotes.noul ≥ 0.55` or regime ∈ {event, auction} | **widen** | Multiply AS half-spread by `1 + κ·noul` |
| `size_tier` | **size** | `tiny→0.25×`, `normal→1×`, `large→1.5×` baseline size |
| `hedge_now.noul ≥ 0.6` and conf OK | **hedge** | Invoke hedge module with urgency flag |
| `surface_suspect.noul ≥ 0.6` | **degrade FV** | Widen; freeze surface update; fall back to previous fit |
| API error / timeout | **fallback** | Deterministic rules only (below) |

Pseudo-policy:

```python
def apply_system_one(answers, as_quote, limits) -> QuoteMods:
    regime = answers["regime"]
    tox = answers["toxicity"]
    size = answers["size_tier"]

    # Confidence floor: answer tells WHAT; confidence tells WHETHER ([confidence-routing](https://docs.typesafe.ai/patterns/confidence-routing.md))
    if min(regime.confidence, tox.confidence, size.confidence) < 0.55:
        return QuoteMods.cancel=True, widen_mult=1.5, size_mult=0.0, mode="flatten")

    if answers["pull_quotes"].noul >= 0.65 or tox.score >= 2.5:
        return QuoteMods(cancel=True, mode="pull")

    widen_mult = 1.0 + 0.75 * answers["widen_quotes"].noul
    if regime.choice in ("event", "auction"):
        widen_mult = max(widen_mult, 1.4)

    size_mult = {"tiny": 0.25, "normal": 1.0, "large": 1.5}[size.choice]
    if answers["surface_suspect"].noul >= 0.6:
        widen_mult *= 1.25
        size_mult = min(size_mult, 0.5)

    hedge = answers["hedge_now"].noul >= 0.6
    # AS reservation/spread already computed deterministically — only scale outputs
    return QuoteMods(
        cancel=False,
        reservation=as_quote.reservation,          # untouched math
        half_spread=as_quote.half_spread * widen_mult,
        size=as_quote.size * size_mult,
        hedge_now=hedge,
        regime=regime.choice,
        toxicity=tox.score,
        mode="quote",
    )
```

**Hard rule:** `QuoteMods` feeds the quoter/hedger. No path constructs an exchange order from model text.

---

## 5. Fallback when API unavailable

If `TYPESAFE_API_KEY` is unset, the client errors, or latency > budget (e.g. 400 ms):

1. Log `system_one_fallback=true` with reason.
2. Use **deterministic rules** only:
   - `vpin_proxy > 0.7` or `trade_intensity_z > 2.5` → widen 1.5×, size 0.5×
   - `events.news_flag == "hard"` or `halt_or_luld` → pull
   - `|net_delta| > soft_limit` → `hedge_now=True`
   - `surface.max_residual_z > 3` → widen 1.25×, freeze surface
   - `md_age_ms > 50` or `feed_gap` → pull
3. Never block the quote loop on retries beyond one short attempt; fail open to safe mode (widen/pull), not to aggressive quoting.

---

## 6. Observability

Log every cycle (sim-safe):

- Full `mm_state` hash / summary
- Raw answers: choice/score/noul, probabilities, confidence
- Gated `QuoteMods`
- Fallback flag and rule id
- Attribution: PnL tagged with `regime`, `toxicity`, `widen_mult`, `size_mult`

Replay must be possible **without** live Jev by replaying stored answers or forcing fallback rules.

---

## 7. Non-goals for this layer

- No free-form “explain the market” chat on the hot path
- No model-generated order prices, sizes, or client order IDs
- No replacing AS / BS / SABR with an LLM price
- No live trading keys; sim and paper only
- No hidden coupling between questions (answers are independent; compose in code)

---

## 8. Module placement

Proposed package path (aligns with existing `jev_omm/`):

```
jev_omm/decision/
  __init__.py
  state.py          # build mm_state from book/flow/inv/surface
  questions.py      # MM_QUESTIONS battery
  client.py         # TypeSafeClient wrapper + timeout/fallback
  policy.py         # confidence gates → QuoteMods
  types.py          # QuoteMods, HedgeCmd, RiskMode
```

Wire: `Surface/FairValue → Quoter(AS) → Decision(System One) → Risk → Hedge → Execution/Sim`.
