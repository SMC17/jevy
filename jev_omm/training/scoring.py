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
        information_set=["venue_a mid", "venue_b mid", "common factor", "venue betas", "no futures inside the sim"],
        constraints=[
            "paper only",
            "original sim has no futures, so the oil book cannot be hedged there",
            "desk overlay is an out-of-sim futures hedge with basis risk",
        ],
        lesson="The location spread is positive expected value and still leaves residual oil beta. The sim cannot hedge it. A real futures hedge would remain, with basis risk.",
        citation="https://www.predictingalpha.com/blogs/what-i-learned-from-citadels-training-software",
    ),
    "pm_fair_value": CaseSpec(
        name="pm_fair_value",
        role="portfolio manager with known fair values",
        information_set=["three names", "fair value of each", "unit market betas"],
        constraints=["paper only", "opposing leg forces net beta to zero", "size the edge only once neutral"],
        lesson="Long what is cheap and short what is rich. If nothing is rich, short the closest name anyway so a market move cannot wipe the fair-value edge.",
        citation="https://www.predictingalpha.com/blogs/what-i-learned-from-citadels-training-software",
    ),
    "etf_ap_arb": CaseSpec(
        name="etf_ap_arb",
        role="ETF authorized participant",
        information_set=["ETF mid", "basket mid", "create/redeem fee", "latency"],
        constraints=["edge decays", "execution shock grows with latency", "size in code from Decision size_tier"],
        lesson="When the create/redeem is near risk-free, size aggressively and immediately. Latency is the risk.",
        citation="https://www.reddit.com/r/Trading/comments/122y2zq/what_i_learned_from_citadels_training_software/",
    ),
    "liability_facilitator": CaseSpec(
        name="liability_facilitator",
        role="liability broker working a client block",
        information_set=["forced block premium", "child-slice schedule", "discretionary prints", "toxicity prior by flow class"],
        constraints=[
            "work the block in slices",
            "algo gap is 1 step, hand gap is 12 steps",
            "do not add discretionary inventory while working",
        ],
        lesson="Take the client's forced block at a discount, then slice it out. A slow hand schedule leaves you in the drift. Discretionary prints are the toxic ones.",
        citation="https://www.reddit.com/r/options/comments/122pz4e/what_i_learned_from_citadels_training_software/",
    ),
    "mm_inventory": CaseSpec(
        name="mm_inventory",
        role="single-name market maker",
        information_set=["forced vs discretionary flow", "queue ahead", "cancel latency", "peer inventory pressure"],
        constraints=[
            "default grade is the disciplined skew",
            "predatory peer-cover is a research mode and is not the grade",
            "reuse LOB fluid fills",
        ],
        lesson="Earn the spread and keep inventory stable. Joining a wave to sell into peer panic can print more raw PnL and is not the graded policy.",
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
    "flow_vpin": CaseSpec(
        name="flow_vpin",
        role="market maker reading a signed tape",
        information_set=["VPIN", "order-flow imbalance", "off-exchange share", "layered-cancel score"],
        constraints=["paper tape", "flow_prior is code", "Decision snapshot is not an order"],
        lesson="Quote wider and smaller when volume-synchronized toxicity, OFI, and layered cancels line up. A tight quote collects the adverse move.",
        citation="https://doi.org/10.1093/rfs/hhs053",
    ),
    "dealer_gamma": CaseSpec(
        name="dealer_gamma",
        role="options market maker in a dealer-gamma regime",
        information_set=["normalized dealer GEX", "pin gap", "hedge band"],
        constraints=["short-premium and dashboard signs are assumptions", "max pain is not a forecast", "no live OI feed"],
        lesson="Long dealer gamma: lean toward the pin and let the hedge band widen. Short dealer gamma: cut size, widen, and hedge sooner.",
        citation="https://doi.org/10.2139/ssrn.3725454",
    ),
    "cot_fade": CaseSpec(
        name="cot_fade",
        role="futures-positioning overlay",
        information_set=["speculative net z-score", "week-over-week change is not required to fade"],
        constraints=["weekly CFTC print", "mean-reverting synthetic return", "no live order"],
        lesson="Do not join an extreme speculative COT print. Fade it. A mild z-score is not an extreme.",
        citation="https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
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
