"""Case score: absolute PnL, competitive relative hook, risk-adjusted, path penalties.

The model never emits orders. Scores are research metrics on a paper case.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def lcg_next(state: int) -> tuple[int, float]:
    """Same 32-bit LCG as ``zig/src/training.zig``."""
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return state, (state / 4294967296.0) * 2.0 - 1.0


@dataclass
class CaseScore:
    absolute_pnl: float
    inventory_path_penalty: float
    beta_penalty: float
    exec_penalty: float
    risk_adjusted: float
    relative_score: float
    mean_abs_beta: float
    mean_abs_inventory: float
    peer_pnl: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "absolute_pnl": self.absolute_pnl,
            "inventory_path_penalty": self.inventory_path_penalty,
            "beta_penalty": self.beta_penalty,
            "exec_penalty": self.exec_penalty,
            "risk_adjusted": self.risk_adjusted,
            "relative_score": self.relative_score,
            "mean_abs_beta": self.mean_abs_beta,
            "mean_abs_inventory": self.mean_abs_inventory,
            "peer_pnl": self.peer_pnl,
        }


def score_path(
    pnl: float,
    inventory: list[float],
    beta: list[float],
    *,
    inv_lambda: float,
    beta_lambda: float,
    exec_penalty: float = 0.0,
    peer_pnl: float = 0.0,
) -> CaseScore:
    n_i = max(len(inventory), 1)
    n_b = max(len(beta), 1)
    inv_pen = inv_lambda * sum(q * q for q in inventory) / n_i
    beta_pen = beta_lambda * sum(b * b for b in beta) / n_b
    return CaseScore(
        absolute_pnl=pnl,
        inventory_path_penalty=inv_pen,
        beta_penalty=beta_pen,
        exec_penalty=exec_penalty,
        risk_adjusted=pnl - inv_pen - beta_pen - exec_penalty,
        relative_score=pnl - peer_pnl,
        mean_abs_beta=sum(abs(b) for b in beta) / n_b,
        mean_abs_inventory=sum(abs(q) for q in inventory) / n_i,
        peer_pnl=peer_pnl,
    )


@dataclass
class CaseSpec:
    name: str
    role: str
    information_set: list[str]
    constraints: list[str]
    lesson: str
    citation: str


SPECS: dict[str, CaseSpec] = {
    "location_arb": CaseSpec(
        name="location_arb",
        role="commodity location arbitrageur",
        information_set=["venue_a mid", "venue_b mid", "common factor", "betas"],
        constraints=["paper only", "hedge the factor future", "no live venue"],
        lesson="Buy cheap venue, sell rich, and hedge residual market beta or the factor move dominates the edge.",
        citation="https://www.reddit.com/r/Trading/comments/122y2zq/what_i_learned_from_citadels_training_software/",
    ),
    "etf_ap_arb": CaseSpec(
        name="etf_ap_arb",
        role="ETF authorized participant",
        information_set=["ETF mid", "basket mid", "create/redeem fee", "latency"],
        constraints=["edge decays", "execution shock grows with latency", "size in code from Decision size_tier"],
        lesson="When the create/redeem is near risk-free, size aggressively and immediately. Latency is the risk.",
        citation="https://medium.datadriveninvestor.com/this-is-what-citadels-training-software-taught-me-741c3996a5b5",
    ),
    "liability_facilitator": CaseSpec(
        name="liability_facilitator",
        role="customer-flow facilitator",
        information_set=["customer side", "recent informed fraction", "inventory"],
        constraints=["hard inventory cap", "pull when recent flow is toxic"],
        lesson="Absorb mixed informed and uninformed flow. Inventory and markout, not the spread alone, are the score.",
        citation="https://www.reddit.com/r/options/comments/122pz4e/what_i_learned_from_citadels_training_software/",
    ),
    "mm_inventory": CaseSpec(
        name="mm_inventory",
        role="single-name market maker",
        information_set=["one-sided wave", "queue ahead", "cancel latency"],
        constraints=["reuse LOB fluid fills", "skew or stop the heavy side"],
        lesson="Earn the spread under a one-sided wave, but cancel late and you wear the adverse selection.",
        citation="https://www.reddit.com/r/Trading/comments/122y2zq/what_i_learned_from_citadels_training_software/",
    ),
    "vol_surface_mm": CaseSpec(
        name="vol_surface_mm",
        role="volatility-surface market maker",
        information_set=["raw SVI", "butterfly gate", "calendar gate", "sticky regime"],
        constraints=["no butterfly arb", "no calendar arb", "mark the declared sticky regime"],
        lesson="Quote the strip only inside the SVI no-arb set. Sticky-strike and sticky-delta are different marks.",
        citation="https://arxiv.org/abs/1204.0646",
    ),
}


@dataclass
class CaseRun:
    spec: CaseSpec
    strategy: str
    score: CaseScore
    inventory_path: list[float] = field(default_factory=list)
    beta_path: list[float] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    decision_source: str = "fallback"
