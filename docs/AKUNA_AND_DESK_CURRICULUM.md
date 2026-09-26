# Akuna Options 101/201 + desk practice → code map

**Repo:** [SMC17/jevy](https://github.com/SMC17/jevy)  
**Mode:** simulation / paper only  
**Version:** `0.5.0-zig-oom-citadel-lit`  
**Public curriculum anchors (real URLs only):**
- https://akunacapital.com/work-with-us/options-101/
- Teachable Options 101 outline themes: terminology, how MM profits, futures/options, payoffs, time premium, **put-call parity**, theos & combination spreads, limits/boundaries, spreads/flies, theo & P&L, Greeks (delta hedge, gamma, theta, vol, vega, **rho & boxes**)
- Options 201 themes: price options, analyze risk, automate workflow
- Put-call parity / synthetics / boxes: optionseducation.org put-call parity; Wikipedia box spread; Moontower imply cost of carry — https://blog.moontower.ai/implying-the-cost-of-carry-in-options/
- Natenberg / dynamic hedge: gamma PnL ≈ ½ Γ (ΔS)² vs theta; realized vs implied
- Desk practice: delta bands (not continuous hedge), scenario matrix, package edges on **executable sides** (not mids)

Status legend: **done** (pre-existing) · **landed this PR** · **future**

---

## Options 101 map

| Topic | Status | Module / file |
| --- | --- | --- |
| Terminology / payoffs / time premium | done (docs) + BS | `docs/LITERATURE_AND_DESIGN_BRIEF.md`, `black_scholes.zig` / `pricing/black_scholes.py` |
| How MM profits (spread capture vs adverse) | done | `markout.zig`, `pnl/markout.py`, event log replay |
| Futures / options underlier | partial | Underlier in hedge + synthetic tape; listed futures routing **future** |
| **Put-call parity** (C−P = DF·(F−K)) | **landed this PR** | `zig/src/parity.zig`, `jev_omm/pricing/parity.py` |
| Theos & combination spreads | **landed this PR** | `zig/src/combos.zig`, `jev_omm/pricing/combos.py` |
| Limits / boundaries | done | `risk_limits.zig`, `risk/limits.py` |
| Spreads / flies (verticals, butterflies) | **landed this PR** | `combos.zig` vertical + butterfly; straddle/strangle |
| Theo & P&L | done + **extended** | mark PnL + markout; **greek buckets** in `hedge.zig` / `hedge/delta.py` |
| Greeks: delta / gamma / theta / vega | done | `black_scholes.zig` analytic greeks |
| **Delta hedge** (banded, not continuous) | **landed this PR** | `zig/src/hedge.zig`, `jev_omm/hedge/delta.py` (was stub) |
| Gamma / theta attribution (Natenberg) | **landed this PR** | `greekPnlStep` — ½ Γ (ΔS)², θ·dt, ν·dσ |
| Vol / vega P&L | **landed this PR** | `vega_pnl` bucket; scenario IV shocks |
| **Rho & boxes** | **landed this PR** | box theo PV, package edge, implied rate from box |
| Synthetics / conversion-reversal | **landed this PR** | `syntheticForwardEdge` on executable sides |
| Package edges on executable sides | **landed this PR** | parity + combos + box — never mids for arb |

## Options 201 map

| Topic | Status | Module / file |
| --- | --- | --- |
| Price options | done | BS + SABR-lite surface |
| **Analyze risk** (scenario matrix) | **landed this PR** | `zig/src/scenario.zig`, `jev_omm/risk/scenario.py` |
| Automate workflow | partial | Zig demo + Python demos + event log/replay; fuller desk UI **future** |
| Soft vs hard scenario limits | **landed this PR** | `soft_breach` / `hard_breach` hooks on matrix |

## Desk practice / systems

| Topic | Status | Module / file |
| --- | --- | --- |
| Delta bands + slippage | **landed this PR** | `HedgeConfig.delta_band`, half-spread / bps slip |
| Whalley–Wilmott-style band helper | **landed this PR** | `whalleyWilmottBand` (documented practical form) |
| Flow toxicity → Decision state | **landed this PR** | `toxicity.zig` / `flow/toxicity.py` → `build_mm_state` `flow.*` |
| Production VPIN / BVC | **future** | Research-grade only today |
| TypeSafe / Jev decision layer | done | `decisions/` — Choice/Score/Noul; never emits orders |
| Multi-strike strip quoting | done | `multi_strike.zig` |
| Guéant asymptotics | done | `gueant.zig` |
| Guéant ODE / spectral + (A, k) tape | **landed** | `gueant_ode.zig` — see [FRONTIERS.md](./FRONTIERS.md) |
| SVI / SSVI + arb + sticky regimes | **landed** | `svi.zig` |
| Multi-expiry vega / vanna / volga | **landed** | `term_book.zig` |
| Queue-aware LOB fills | **landed** | `lob.zig` (synthetic) |
| Queue value / depth ahead | **landed** | `queueValue`, `depthAhead` in `lob.zig`; `execution/lob.py` |
| Citadel-style training cases | **landed** | `jev_omm/training/`, `zig/src/training.zig`, [TRAINING_CASES.md](./TRAINING_CASES.md) |
| Option-vega inventory MM | **landed** | `option_mm.zig`, `quoter/option_mm.py`, mode `option_vega` |
| Spot–vol hedge tilt | **landed** | `hedge.spotVolHedgeQty` / `hedge/delta.py` |
| Hawkes toxic-burst intensity | **landed** | `hawkes.zig`, `flow/hawkes.py` → Decision `flow.hawkes_*` |
| Dupire local vol (research) | **landed** | `surface/dupire.py` (Python) |
| Rough Bergomi stress paths | **landed** | `surface/rough_vol.py` (not a quoter) |
| Variance / vol swap weights | **landed** | `varswap.zig`, `pricing/varswap.py` |
| American / Heston PDE | **future** | See MODULES upgrade path; Dupire local vol is the slice that landed |
| Live venue / broker SDKs | **out of scope** | Sim/paper only |
| Live `TYPESAFE_API_KEY` | **not required** | Hooks exist; fallback is the default |

## Demos

```bash
cd zig && zig build demo -- --hedge-scenario
cd zig && zig build frontiers
cd zig && zig build training
cd .. && source .venv/bin/activate && python -m jev_omm.demo_desk
python -m jev_omm.demo_frontiers
python -m jev_omm.demo_training
```

## Version

C ABI `jev_omm_version` → `0.5.0-zig-oom-citadel-lit`.
