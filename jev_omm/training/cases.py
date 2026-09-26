"""Five paper cases. Same constants and LCG as ``zig/src/training.zig``.

Strategies are code. A Decision snapshot (Choice / Score / Noul from the
offline fallback) may scale size. It does not contain an order.
"""

from __future__ import annotations

from jev_omm.decisions.client import DeterministicFallbackClient
from jev_omm.decisions.policy import apply_policy
from jev_omm.decisions.schemas import build_mm_questions, build_mm_state
from jev_omm.execution.lob import expected_fills
from jev_omm.surface.svi import (
    SviParams,
    butterfly_check,
    implied_vol_after_move,
    raw_calendar_ok,
)
from jev_omm.training.scoring import SPECS, CaseRun, lcg_next, score_path

CASES = (
    "location_arb",
    "etf_ap_arb",
    "liability_facilitator",
    "mm_inventory",
    "vol_surface_mm",
)


def _location(strategy: str, peer_pnl: float) -> tuple[CaseRun, list[float], list[float]]:
    n = 40
    state = 7
    f = 100.0
    basis = 0.80
    qa = qb = qf = cash = 0.0
    inv: list[float] = []
    beta: list[float] = []
    beta_a, beta_b, qty = 1.0, 0.55, 2.0
    events: list[dict] = []
    for _t in range(n):
        a = f
        b = f + basis
        edge = b - a
        traded = 0.0
        if abs(edge) > 0.20:
            traded = qty if edge > 0.0 else -qty
            if edge > 0.0:
                cash -= qty * a
                cash += qty * b
                qa += qty
                qb -= qty
            else:
                cash += qty * a
                cash -= qty * b
                qa -= qty
                qb += qty
        net = beta_a * qa + beta_b * qb
        hedged = 0.0
        if strategy == "desk":
            target = -net
            dq = target - qf
            cash -= dq * f
            qf = target
            hedged = dq
            net = beta_a * qa + beta_b * qb + qf
        inv.append(abs(qa) + abs(qb))
        beta.append(net)
        events.append({"type": "Step", "factor": f, "basis": basis, "traded": traded, "hedge_dq": hedged, "beta": net})
        state, z = lcg_next(state)
        f += 1.5 * z
        state, z2 = lcg_next(state)
        basis = 0.92 * basis + 0.06 + 0.02 * z2
    pnl = cash + qa * f + qb * (f + basis) + qf * f
    sc = score_path(pnl, inv, beta, inv_lambda=0.01, beta_lambda=2.0, peer_pnl=peer_pnl)
    run = CaseRun(SPECS["location_arb"], strategy, sc, inv, beta, events)
    return run, inv, beta


def _etf_path(seed: int, n: int = 17) -> list[float]:
    state = seed
    premium = 0.50
    out = [premium]
    for _ in range(1, n):
        state, z = lcg_next(state)
        premium = premium * 0.72 + 0.015 * z
        out.append(premium)
    return out


def _etf(strategy: str, peer_pnl: float) -> CaseRun:
    path = _etf_path(11)
    fee = 0.02
    latency = 1 if strategy == "desk" else 8
    resid = 0.02 * (latency**0.5)
    edge0 = path[0] - fee
    near_risk_free = edge0 > 4.0 * resid
    size = 12.0 if strategy == "desk" and near_risk_free else 1.0
    state = build_mm_state(
        time=0.0,
        spot=100.0,
        option_mid=path[0],
        iv=0.2,
        inventory=0,
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        cash_pnl=0.0,
        half_spread=edge0,
        quoting_allowed=True,
    )
    state["arb"] = {"near_risk_free": bool(strategy == "desk" and near_risk_free), "edge": edge0, "residual": resid}
    result = DeterministicFallbackClient().system_one(state, build_mm_questions())
    mods = apply_policy(result)
    # Code owns the ticket. The fallback Choice only confirms urgency.
    if strategy == "desk" and mods.size_mult >= 1.5 and near_risk_free:
        size = 12.0
    shock_state = (11 + latency * 17) & 0xFFFFFFFF
    if shock_state == 0:
        shock_state = 1
    _, z = lcg_next(shock_state)
    shock = resid * z
    edge = path[latency] - fee - shock
    impact = 0.0004 * size * size
    pnl = size * edge - impact
    exec_pen = abs(shock) * size
    inv = [size] * latency + [0.0]
    beta = [0.0] * len(inv)
    # Match Zig, which scores only the latency window.
    inv_s = [size] * max(latency, 1)
    beta_s = [0.0] * max(latency, 1)
    sc = score_path(pnl, inv_s, beta_s, inv_lambda=0.0, beta_lambda=0.0, exec_penalty=exec_pen, peer_pnl=peer_pnl)
    events = [
        {"type": "PremiumPath", "premium": path, "latency": latency, "size": size},
        {
            "type": "DecisionSnapshot",
            "source": result.source,
            "model": result.model,
            "size_tier": result.answers["size_tier"].choice if hasattr(result.answers["size_tier"], "choice") else "",
            "size_mult": mods.size_mult,
            "engine_size": size,
            "near_risk_free": bool(strategy == "desk" and near_risk_free),
            "note": (
                "Choice/Score/Noul from the fallback client. "
                "A low-confidence gate may cut size_mult. "
                "The case engine, not the model, keeps max size when the arb is near risk-free."
            ),
        }
    ]
    run = CaseRun(SPECS["etf_ap_arb"], strategy, sc, inv, beta, events, decision_source=result.source)
    return run


def _facilitator(strategy: str, peer_pnl: float) -> CaseRun:
    n = 50
    state = 21
    half, jump = 0.08, 0.28
    inv_pos = 0.0
    pnl = 0.0
    recent: list[float] = []
    inv: list[float] = []
    beta = [0.0] * n
    events: list[dict] = []
    for _t in range(n):
        state, u = lcg_next(state)
        informed = (u + 1.0) / 2.0 < 0.40
        state, side_draw = lcg_next(state)
        cust_buy = side_draw > 0.0
        size = 1.0
        if strategy == "desk":
            window = recent[-6:]
            tox = sum(window) / len(window) if window else 0.0
            if abs(inv_pos) >= 4.0 or tox > 0.55:
                size = 0.0
        if size > 0.0:
            inv_pos += -size if cust_buy else size
            pnl += half * size
            if informed:
                pnl -= jump * size
            recent.append(1.0 if informed else 0.0)
        else:
            recent.append(0.0)
        inv.append(inv_pos)
        events.append({"type": "Flow", "informed": informed, "size": size, "inventory": inv_pos})
    sc = score_path(pnl, inv, beta, inv_lambda=0.05, beta_lambda=0.0, peer_pnl=peer_pnl)
    return CaseRun(SPECS["liability_facilitator"], strategy, sc, inv, beta, events)


def _mm(strategy: str, peer_pnl: float) -> CaseRun:
    ahead, intensity, horizon = 4.0, 25.0, 1.0
    lat = 0.20 if strategy == "desk" else 0.90
    fills = expected_fills(ahead, 6.0, intensity, 0.0, horizon, lat)
    spread, adverse = 0.05, 0.15
    pnl = fills * spread - fills * adverse
    n = 40
    inv_pos = 0.0
    inv: list[float] = []
    beta = [0.0] * n
    for t in range(n):
        hit = True
        if strategy == "desk" and inv_pos >= 5.0:
            hit = False
        if strategy == "desk" and inv_pos >= 3.0 and t % 2 == 0:
            hit = False
        if hit:
            inv_pos += 1.0
            pnl += 0.04
            if t > n // 3:
                pnl -= 0.09
        inv.append(inv_pos)
    sc = score_path(pnl, inv, beta, inv_lambda=0.02, beta_lambda=0.0, peer_pnl=peer_pnl)
    events = [{"type": "LobRace", "cancel_latency": lat, "expected_fills": fills, "queue_pnl": fills * (spread - adverse)}]
    return CaseRun(SPECS["mm_inventory"], strategy, sc, inv, beta, events)


def _vol(strategy: str, peer_pnl: float) -> CaseRun:
    clean = SviParams(0.04, 0.1, -0.4, 0.0, 0.2)
    poisoned = SviParams(0.01, 1.5, -0.9, 0.0, 0.05)
    clean_ok = butterfly_check(clean).ok
    poison_ok = butterfly_check(poisoned).ok
    earlier = SviParams(0.08, 0.1, -0.3, 0.0, 0.2)
    later = SviParams(0.02, 0.1, -0.3, 0.0, 0.2)
    calendar_ok = raw_calendar_ok(earlier, later)
    iv_strike = implied_vol_after_move(clean, 100.0, 110.0, 100.0, 0.25, "sticky_strike")
    iv_delta = implied_vol_after_move(clean, 100.0, 110.0, 100.0, 0.25, "sticky_delta")
    pnl = 1.2 if clean_ok else 0.0
    arb_pen = 0.0
    regime_pen = 0.0
    if strategy == "desk":
        if poison_ok:
            pnl -= 5.0
        if not calendar_ok:
            arb_pen += 0.0
    else:
        if not poison_ok:
            arb_pen += 25.0
        if not calendar_ok:
            arb_pen += 10.0
        regime_pen = abs(iv_strike - iv_delta) * 40.0
    inv = [0.0]
    beta = [0.0]
    sc = score_path(pnl - regime_pen, inv, beta, inv_lambda=0.0, beta_lambda=0.0, exec_penalty=arb_pen, peer_pnl=peer_pnl)
    events = [
        {
            "type": "SurfaceGate",
            "butterfly_clean": clean_ok,
            "butterfly_poison": poison_ok,
            "calendar_ok": calendar_ok,
            "iv_sticky_strike": iv_strike,
            "iv_sticky_delta": iv_delta,
        }
    ]
    return CaseRun(SPECS["vol_surface_mm"], strategy, sc, inv, beta, events)


def run_case(name: str, strategy: str = "desk", peer_pnl: float = 0.0) -> CaseRun:
    if name == "location_arb":
        run, _, _ = _location(strategy, peer_pnl)
        return run
    if name == "etf_ap_arb":
        return _etf(strategy, peer_pnl)
    if name == "liability_facilitator":
        return _facilitator(strategy, peer_pnl)
    if name == "mm_inventory":
        return _mm(strategy, peer_pnl)
    if name == "vol_surface_mm":
        return _vol(strategy, peer_pnl)
    raise KeyError(name)


def run_pair(name: str) -> tuple[CaseRun, CaseRun]:
    naive = run_case(name, "naive", 0.0)
    desk = run_case(name, "desk", naive.score.absolute_pnl)
    return naive, desk


def leaderboard(runs: list[CaseRun]) -> list[CaseRun]:
    """Competitive hook: rank by risk-adjusted PnL, high to low."""
    return sorted(runs, key=lambda r: r.score.risk_adjusted, reverse=True)
