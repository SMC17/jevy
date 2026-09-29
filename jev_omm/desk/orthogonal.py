"""Smile and sleeve orthogonalization.

Two mechanisms, both paper-only:

1. Factor split. Skew, butterfly, and wing innovations are separate draws.
   In the smile-shock regime a shared draw is planted and then removed with
   Gram–Schmidt before a sleeve trades it. Funding is residualized against
   the box so the rate sleeve is not a second copy of parity.

2. Hard gate. After the greek strip, any enabled pair with |ρ| above the
   threshold is residualized (the weaker Sharpe leg against the stronger).
   A pair that shares more than half its variance (|ρ| > √0.5) is merged:
   the weaker leg is set to zero and logged. A pair that is still above the
   threshold after that pass is cut the same way.

The projection keeps the intercept of the series it residualizes, matching
the strip. A flat leftover has no sample variance; the allocator then
assigns it weight 0.

PCA on the residual panel is a research summary. It is not a live risk model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from jev_omm.desk.scoreboard import per_step_sharpe
from jev_omm.pnl.residual import pearson

# |ρ| above this means more than half the variance is shared (ρ² > 0.5).
MERGE_RHO = math.sqrt(0.5)


def residualize_keep_mean(y: np.ndarray | list[float], x: np.ndarray | list[float]) -> np.ndarray:
    """Remove the centered projection of ``y`` on ``x``. The mean of ``y`` stays."""
    yy = np.asarray(y, dtype=float).reshape(-1).copy()
    xx = np.asarray(x, dtype=float).reshape(-1)
    if yy.size != xx.size or yy.size < 2:
        return yy
    xc = xx - float(np.mean(xx))
    yc = yy - float(np.mean(yy))
    var = float(np.dot(xc, xc))
    if var <= 1e-18:
        return yy
    beta = float(np.dot(xc, yc) / var)
    return yy - beta * xc


def residualize_against_many(y: np.ndarray, cols: list[np.ndarray]) -> np.ndarray:
    out = np.asarray(y, dtype=float).reshape(-1).copy()
    for col in cols:
        out = residualize_keep_mean(out, col)
    return out


@dataclass
class GateLog:
    action: str
    drop: str
    keep: str
    rho_before: float
    rho_after: float
    reason: str


@dataclass
class GateResult:
    series: dict[str, np.ndarray]
    enabled: dict[str, bool]
    logs: list[GateLog] = field(default_factory=list)

    @property
    def notes(self) -> list[str]:
        return [
            f"{row.action} `{row.drop}` against `{row.keep}` "
            f"(|ρ| {row.rho_before:.3f} → {row.rho_after:.3f}): {row.reason}"
            for row in self.logs
        ]


def enforce_orthogonality(
    series: dict[str, np.ndarray],
    enabled: dict[str, bool] | None = None,
    *,
    threshold: float = 0.40,
) -> GateResult:
    """Force pairwise |ρ| at or under ``threshold``.

    Preference is residualize, then merge when the pair shares more than
    half its variance, then cut if a pass still leaves |ρ| above the line.
    """
    out = {k: np.asarray(v, dtype=float).reshape(-1).copy() for k, v in series.items()}
    on = {k: True for k in out} if enabled is None else {k: bool(enabled.get(k, False)) for k in out}
    logs: list[GateLog] = []
    ids = [k for k, flag in on.items() if flag]
    # One pass per pair is enough when each pass zeros the worst offender.
    # A few extra passes catch pairs the projection pushes over the line.
    for _ in range(len(ids) * 2 + 2):
        worst: tuple[str, str, float] | None = None
        worst_abs = threshold
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                rho = pearson(out[a], out[b])
                if abs(rho) > worst_abs + 1e-15:
                    worst_abs = abs(rho)
                    worst = (a, b, rho)
        if worst is None:
            break
        a, b, rho = worst
        sa = abs(per_step_sharpe(out[a]))
        sb = abs(per_step_sharpe(out[b]))
        # Tie-break on the name so the gate is deterministic.
        if sa > sb or (sa == sb and a < b):
            keep, drop = a, b
        else:
            keep, drop = b, a
        before = float(rho)
        shared = abs(before) >= MERGE_RHO - 1e-15
        if shared:
            out[drop] = np.zeros_like(out[drop])
            on[drop] = False
            ids = [k for k in ids if k != drop]
            logs.append(
                GateLog(
                    "merge",
                    drop,
                    keep,
                    before,
                    0.0,
                    "shared variance above one half; weaker sleeve weight forced to 0",
                )
            )
            continue
        out[drop] = residualize_keep_mean(out[drop], out[keep])
        after = pearson(out[keep], out[drop])
        if abs(after) > threshold:
            out[drop] = np.zeros_like(out[drop])
            on[drop] = False
            ids = [k for k in ids if k != drop]
            logs.append(
                GateLog(
                    "cut",
                    drop,
                    keep,
                    before,
                    float(after),
                    "still above the gate after residualizing; weight forced to 0",
                )
            )
        else:
            logs.append(
                GateLog(
                    "residualize",
                    drop,
                    keep,
                    before,
                    float(after),
                    "Gram–Schmidt on the residual stream",
                )
            )
    return GateResult(out, on, logs)


def research_pca(series: dict[str, np.ndarray], ids: list[str], k: int = 4) -> str:
    """Share of residual variance in the first K principal components.

    Research label only. This is not a covariance model for a live book.
    """
    if len(ids) < 2:
        return "Research PCA skipped: fewer than two sleeves."
    cols = []
    raw = []
    kept = []
    for name in ids:
        col = np.asarray(series[name], dtype=float).reshape(-1)
        std = float(np.std(col))
        if std <= 1e-12:
            continue
        centered = col - float(np.mean(col))
        raw.append(centered)
        cols.append(centered / std)
        kept.append(name)
    if len(cols) < 2:
        return "Research PCA skipped: residual panel has no variance."
    try:
        _u, s, _vt = np.linalg.svd(np.column_stack(cols), full_matrices=False)
        _ur, sr, _vtr = np.linalg.svd(np.column_stack(raw), full_matrices=False)
    except np.linalg.LinAlgError:
        return "Research PCA failed to factor the residual panel."
    ev = s * s
    total = float(ev.sum())
    raw_ev = sr * sr
    raw_total = float(raw_ev.sum())
    if total <= 0.0 or raw_total <= 0.0:
        return "Research PCA skipped: residual panel has no variance."
    take = min(k, int(ev.size))
    share = float(ev[:take].sum() / total)
    raw_share = float(raw_ev[:take].sum() / raw_total)
    return (
        f"Research PCA on {len(kept)} sleeves, columns standardized: first {take} "
        f"components explain {share:.3f} of residual correlation variance. "
        f"Unstandardized sum of squares is {raw_share:.3f} in the same {take}, "
        f"because a few noisy sleeves dominate the scale. Not a live risk model."
    )
