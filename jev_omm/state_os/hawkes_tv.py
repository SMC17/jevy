"""Time-varying Hawkes branching ratio as a fragility overlay.

Stationary n = α/β. Locally, Filimonov & Sornette write n_t = 1 − μ/λ_t.
Near 1 the book is reflexive. That is a state label, not a crash time.

An isolated jump with λ near μ is exogenous. A jump that arrives with n_t
already high is endogenous. Quiet returns stay quiet regardless of n_t.

- Hawkes, Biometrika 1971, https://doi.org/10.1093/biomet/58.1.83
- Filimonov & Sornette, Phys. Rev. E 2012, https://doi.org/10.1103/PhysRevE.85.056108
- Hardiman, Bercot, Bouchaud, Eur. Phys. J. B 2013,
  https://doi.org/10.1140/epjb/e2013-40107-3
"""

from __future__ import annotations

from jev_omm.flow.hawkes import HawkesParams, intensity


def local_branching(mu: float, lam: float) -> float:
    """n_t = 1 − μ/λ. Floored at 0 when the intensity has not left the baseline."""
    if lam <= 0.0:
        return 0.0
    return max(0.0, 1.0 - mu / lam)


def n_t(params: HawkesParams, t: float, events: list[float]) -> float:
    lam = intensity(params, t, events)
    return local_branching(max(params.mu, 0.0), lam)


def classify_move(
    ret: float,
    branching: float,
    ret_min: float = 0.01,
    n_star: float = 0.75,
) -> str:
    """``quiet``, ``exogenous``, or ``endogenous``. No timestamp of a crash."""
    if abs(ret) < ret_min:
        return "quiet"
    if branching >= n_star:
        return "endogenous"
    return "exogenous"
