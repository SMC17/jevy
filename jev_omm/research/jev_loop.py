"""Score the offline decision client inside the paper simulator.

On each quote step the fallback client (no TypeSafe key) emits Choice /
Score / Noul. Those probabilities are stored next to what the simulator
then did: a fill, an adverse one-step markout, an inventory-limit breach.
Brier, log loss, and ECE use those outcomes. The Bernoulli toy in
``scoreboard.synthetic_jev_ablation`` is a different experiment and stays.

Deterministic means a neutral client whose answers sit below every policy
threshold, so quote adjustments are the identity. Jev-assisted means
``DeterministicFallbackClient``. Neither emits an order. Economics subtract
a research fee of 0.02 per filled contract from simulator PnL. Questions
whose train-seed gap over the identity is not positive are gated off.
The headline is the test-seed gap of that gated policy. If it is not
positive, the assisted policy does not help out of sample after costs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from jev_omm.backtest.simulator import SimResult, run_simulation
from jev_omm.config import EngineConfig, MarketConfig, QuoterConfig, RiskConfig, SimConfig
from jev_omm.decisions.client import DecisionClient, DeterministicFallbackClient
from jev_omm.decisions.schemas import (
    REGIME_CRITERIA,
    SIZE_TIER_CRITERIA,
    TOXICITY_LEVELS,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResult,
)
from jev_omm.decisions.scoreboard import ScoreTuple, brier, ece, ece_by_regime, log_loss
from jev_omm.models.types import Side

FEE_PER_CONTRACT = 0.02
TRAIN_SEEDS = (1, 2, 3)
TEST_SEEDS = (4, 5, 6)
QUESTIONS = (
    "regime",
    "toxicity",
    "informed_flow",
    "widen_quotes",
    "pull_quotes",
    "hedge_now",
    "size_tier",
    "surface_suspect",
)


def _neutral_result() -> SystemOneResult:
    regime_probs = {k: 0.05 for k in REGIME_CRITERIA}
    regime_probs["calm"] = 0.85
    s = sum(regime_probs.values())
    regime_probs = {k: v / s for k, v in regime_probs.items()}
    size_probs = {k: 0.08 for k in SIZE_TIER_CRITERIA}
    size_probs["normal"] = 0.84
    ss = sum(size_probs.values())
    size_probs = {k: v / ss for k, v in size_probs.items()}
    legend = {str(i): lvl for i, lvl in enumerate(TOXICITY_LEVELS)}
    tox_probs = {"0": 0.7, "1": 0.1, "2": 0.1, "3": 0.1}
    answers = {
        "regime": ChoiceAnswer(choice="calm", probabilities=regime_probs, confidence=0.9),
        "toxicity": ScoreAnswer(score=0.0, legend=legend, probabilities=tox_probs, confidence=0.9),
        "informed_flow": NoulAnswer(noul=0.05),
        "widen_quotes": NoulAnswer(noul=0.05),
        "pull_quotes": NoulAnswer(noul=0.05),
        "hedge_now": NoulAnswer(noul=0.05),
        "size_tier": ChoiceAnswer(choice="normal", probabilities=size_probs, confidence=0.9),
        "surface_suspect": NoulAnswer(noul=0.05),
    }
    return SystemOneResult(model="neutral-identity", answers=answers, source="fallback")


class NeutralClient(DecisionClient):
    """Answers that do not cross a policy threshold. Adjustments stay at 1."""

    def system_one(self, state, questions=None, *, model="neutral-identity"):
        _ = state, questions, model
        return _neutral_result()


class MaskedFallbackClient(DecisionClient):
    """Fallback answers, with every question outside ``keep`` reset to neutral."""

    def __init__(self, keep: frozenset[str]):
        self.keep = keep
        self.inner = DeterministicFallbackClient()

    def system_one(self, state, questions=None, *, model="jev-latest"):
        live = self.inner.system_one(state, questions, model=model)
        base = _neutral_result().answers
        merged = {}
        for key, ans in live.answers.items():
            merged[key] = ans if key in self.keep else base[key]
        return SystemOneResult(model="masked-fallback", answers=merged, source="fallback")


def _engine(seed: int, n_steps: int = 40) -> EngineConfig:
    return EngineConfig(
        market=MarketConfig(),
        quoter=QuoterConfig(
            mode="as_finite_horizon",
            gamma=0.12,
            kappa=1.5,
            sigma=0.45,
            quote_size=1,
            min_half_spread=0.05,
            max_half_spread=2.0,
        ),
        risk=RiskConfig(max_abs_inventory=10, max_abs_delta=80.0, max_loss=5000.0),
        sim=SimConfig(
            n_steps=n_steps,
            seed=seed,
            fill_model="poisson",
            fill_intensity_per_second=0.03,
            feature_pack="off",
            hedge_band=8.0,
        ),
    )


def _net_pnl(result: SimResult) -> float:
    contracts = sum(f.size for f in result.fills)
    return float(result.final_pnl) - FEE_PER_CONTRACT * contracts


def _run(seed: int, client: DecisionClient, n_steps: int = 40) -> SimResult:
    return run_simulation(_engine(seed, n_steps), client=client)


def _prob(result_answers, key: str) -> float | None:
    ans = result_answers.get(key)
    if isinstance(ans, NoulAnswer):
        return float(ans.noul)
    if isinstance(ans, ScoreAnswer):
        return float(ans.score) / 3.0
    if isinstance(ans, ChoiceAnswer):
        if key == "regime":
            return float(ans.probabilities.get("stressed", 0.0))
        if key == "size_tier":
            return float(ans.probabilities.get("tiny", 0.0))
        return float(ans.confidence)
    return None


def _confidence(result_answers, key: str, prob: float) -> float:
    ans = result_answers.get(key)
    if isinstance(ans, (ChoiceAnswer, ScoreAnswer)):
        return float(ans.confidence)
    return float(max(prob, 1.0 - prob))


def rows_from_sim(result: SimResult) -> list[ScoreTuple]:
    """One tuple per scored question on steps where the outcome is defined."""
    fills_by_time: dict[float, list] = {}
    for fill in result.fills:
        fills_by_time.setdefault(float(fill.time), []).append(fill)
    out: list[ScoreTuple] = []
    n = len(result.states)
    for i, state in enumerate(result.states):
        adj = result.adjustments[i] if i < len(result.adjustments) else None
        if adj is None or adj.result is None:
            continue
        answers = adj.result.answers
        regime_ans = answers.get("regime")
        regime = regime_ans.choice if isinstance(regime_ans, ChoiceAnswer) else "unknown"
        fills = fills_by_time.get(float(state.time), [])
        filled = 1.0 if fills else 0.0
        nxt = result.states[i + 1].option_mid if i + 1 < n else None
        adverse = None
        if fills and nxt is not None:
            mark = 0.0
            for fill in fills:
                signed = fill.size if fill.side == Side.BID else -fill.size
                mark += signed * (nxt - fill.mid_at_fill)
            adverse = 1.0 if mark < 0.0 else 0.0
        breached = 1.0 if (i < len(result.step_breach) and result.step_breach[i]) else 0.0
        ret = 0.0
        if i > 0 and result.states[i - 1].spot > 0.0:
            ret = abs(state.spot - result.states[i - 1].spot) / result.states[i - 1].spot
        stressed_y = 1.0 if breached >= 1.0 or ret > 0.004 else 0.0
        surface_y = 0.0
        if nxt is not None and state.quote is not None and state.quote.half_spread > 0.0:
            surface_y = 1.0 if abs(nxt - state.option_mid) > state.quote.half_spread else 0.0
        outcomes = {
            "informed_flow": adverse,
            "toxicity": adverse,
            "pull_quotes": (1.0 if (adverse == 1.0 or breached >= 1.0) else 0.0),
            "widen_quotes": adverse,
            "hedge_now": breached,
            "surface_suspect": surface_y if nxt is not None else None,
            "regime": stressed_y,
            "size_tier": adverse,
        }
        state_blob = json.dumps(
            {
                "i": i,
                "inv": int(result.inventory_path[i]) if i < len(result.inventory_path) else 0,
                "spot": round(float(state.spot), 4),
                "mid": round(float(state.option_mid), 4),
            },
            sort_keys=True,
        )
        for key, y in outcomes.items():
            if y is None:
                continue
            p = _prob(answers, key)
            if p is None:
                continue
            out.append(
                ScoreTuple(
                    state=state_blob,
                    question=key,
                    answer=float(min(1.0, max(0.0, p))),
                    confidence=_confidence(answers, key, float(p)),
                    outcome=float(y),
                    regime=str(regime),
                )
            )
    return out


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return float(np.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def sim_loop_study(
    *,
    train_seeds: tuple[int, ...] = TRAIN_SEEDS,
    test_seeds: tuple[int, ...] = TEST_SEEDS,
    n_steps: int = 40,
) -> dict[str, object]:
    """Train gates on ``train_seeds``. Score economics on ``test_seeds``."""
    identity = NeutralClient()
    full = DeterministicFallbackClient()

    def pnls(seeds: tuple[int, ...], client: DecisionClient) -> list[float]:
        return [_net_pnl(_run(s, client, n_steps)) for s in seeds]

    id_train = pnls(train_seeds, identity)
    full_train = pnls(train_seeds, full)
    question_train: dict[str, float] = {}
    question_test: dict[str, float] = {}
    kept: list[str] = []
    id_test = pnls(test_seeds, identity)
    for key in QUESTIONS:
        client = MaskedFallbackClient(frozenset({key}))
        gap_train = _mean(pnls(train_seeds, client)) - _mean(id_train)
        gap_test = _mean(pnls(test_seeds, client)) - _mean(id_test)
        question_train[key] = gap_train
        question_test[key] = gap_test
        if gap_train > 0.0:
            kept.append(key)
    gated_client = MaskedFallbackClient(frozenset(kept)) if kept else identity
    gated_test = pnls(test_seeds, gated_client)
    full_test = pnls(test_seeds, full)
    # Calibration rows come from the full fallback on the test seeds,
    # because that is the client whose probabilities are being scored.
    scored: list[ScoreTuple] = []
    for seed in test_seeds:
        scored.extend(rows_from_sim(_run(seed, full, n_steps)))
    by_q: dict[str, dict[str, float]] = {}
    for key in QUESTIONS:
        subset = [r for r in scored if r.question == key]
        by_q[key] = {
            "n": float(len(subset)),
            "brier": brier(subset),
            "log_loss": log_loss(subset),
            "ece": ece(subset),
        }
    return {
        "fee_per_contract": FEE_PER_CONTRACT,
        "n_steps": float(n_steps),
        "train_seeds": list(train_seeds),
        "test_seeds": list(test_seeds),
        "identity_test_mean": _mean(id_test),
        "identity_test_std": _std(id_test),
        "full_test_mean": _mean(full_test),
        "full_test_std": _std(full_test),
        "full_minus_identity_test": _mean(full_test) - _mean(id_test),
        "full_minus_identity_train": _mean(full_train) - _mean(id_train),
        "gated_questions": kept,
        "killed_questions": [k for k in QUESTIONS if k not in kept],
        "gated_test_mean": _mean(gated_test),
        "gated_minus_identity_test": _mean(gated_test) - _mean(id_test),
        "question_train_gap": question_train,
        "question_test_gap": question_test,
        "scores": by_q,
        "ece_by_regime": ece_by_regime([r for r in scored if r.question == "informed_flow"]),
        "n_score_rows": float(len(scored)),
    }


def render_sim_loop_markdown(study: dict[str, object] | None = None) -> str:
    s = study if study is not None else sim_loop_study()
    scores: dict[str, dict[str, float]] = s["scores"]  # type: ignore[assignment]
    q_train: dict[str, float] = s["question_train_gap"]  # type: ignore[assignment]
    q_test: dict[str, float] = s["question_test_gap"]  # type: ignore[assignment]
    kept: list[str] = s["gated_questions"]  # type: ignore[assignment]
    killed: list[str] = s["killed_questions"]  # type: ignore[assignment]
    gap = float(s["gated_minus_identity_test"])
    if gap > 0.0:
        verdict = (
            f"The train-gated policy beat identity by {gap:.4f} per run on the "
            "test seeds after the 0.02 research fee. That is one synthetic "
            "Poisson tape, not a market, and not a live Jev call."
        )
    else:
        verdict = (
            f"The train-gated policy does not beat identity on the test seeds "
            f"(gap {gap:.4f} per run after the 0.02 research fee). Questions "
            "that failed the train gap are gated off. The assisted policy does "
            "not help out-of-sample economics after costs on this draw."
        )
    lines = [
        "## Sim-loop scoreboard (paper simulator)",
        "",
        "Each quote step of `run_simulation` stores the offline fallback "
        "answer (Choice / Score / Noul) and a realized outcome from that "
        "same run. No TypeSafe key. No order. Feature pack is `off` "
        "(identity). Fill model is Poisson, 40 steps, intensity 0.03/s.",
        "",
        "Outcomes, not a toy coin:",
        "- `informed_flow`, `toxicity`, `widen_quotes`, `size_tier`: scored "
        "only on steps that filled. Outcome 1 when one-step signed markout "
        "is negative (the mid moved against the fill).",
        "- `pull_quotes`: 1 if that fill was adverse or the step breached a limit.",
        "- `hedge_now`: 1 if the step recorded an inventory/risk breach.",
        "- `regime`: probability of `stressed` versus breach or a 40 bp spot move.",
        "- `surface_suspect`: 1 if the next mid moved by more than the posted half-spread.",
        "",
        f"Research fee {float(s['fee_per_contract']):.2f} per filled contract, "
        "subtracted from simulator PnL. Train seeds "
        f"{s['train_seeds']}, test seeds {s['test_seeds']}.",
        "",
        f"- Identity test PnL mean ± std: {float(s['identity_test_mean']):.4f} "
        f"± {float(s['identity_test_std']):.4f}",
        f"- Full fallback test PnL mean ± std: {float(s['full_test_mean']):.4f} "
        f"± {float(s['full_test_std']):.4f} "
        f"(gap vs identity {float(s['full_minus_identity_test']):.4f}; "
        f"train gap {float(s['full_minus_identity_train']):.4f})",
        f"- Gated policy kept: {', '.join(kept) if kept else '(none)'}",
        f"- Gated off after train: {', '.join(killed) if killed else '(none)'}",
        f"- Gated policy test PnL: {float(s['gated_test_mean']):.4f} "
        f"(gap vs identity {gap:.4f})",
        "",
        "| question | train gap | test gap | OOS | n | Brier | log loss | ECE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in QUESTIONS:
        oos = "helps OOS" if q_test[key] > 0.0 else "does not help OOS"
        sc = scores[key]
        lines.append(
            f"| {key} | {q_train[key]:.4f} | {q_test[key]:.4f} | {oos} | "
            f"{int(sc['n'])} | {sc['brier']:.4f} | {sc['log_loss']:.4f} | {sc['ece']:.4f} |"
        )
    ece_reg: dict[str, float] = s["ece_by_regime"]  # type: ignore[assignment]
    reg = ", ".join(f"{k} {v:.4f}" for k, v in sorted(ece_reg.items())) or "(no fill rows)"
    poor = [k for k in kept if scores[k]["brier"] >= 0.24]
    cal_note = ""
    if poor:
        cal_note = (
            " Kept questions with Brier at least 0.24 (no better than a fair coin): "
            + ", ".join(poor)
            + ". A PnL gap from a coarse size or spread cut is not a calibrated forecast."
        )
    lines += [
        "",
        f"Informed-flow ECE by the fallback's own regime label: {reg}.",
        f"Score rows on the test seeds: {int(float(s['n_score_rows']))}.",
        "",
        verdict + cal_note,
        "",
        "A positive gap is not live alpha. The client is the deterministic "
        "fallback, not `jev-latest`. Read the OOS column before keeping a question.",
        "",
    ]
    return "\n".join(lines)
