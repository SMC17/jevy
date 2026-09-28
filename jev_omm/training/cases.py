"""Paper cases. Same constants and LCG as ``zig/src/training.zig``.

Strategies are code. A Decision snapshot (Choice / Score / Noul from the
offline fallback) may scale size. It does not contain an order.
"""

from __future__ import annotations

from jev_omm.decisions.client import DeterministicFallbackClient
from jev_omm.decisions.policy import apply_policy
from jev_omm.decisions.schemas import build_mm_questions, build_mm_state
from jev_omm.flow.signals import flow_prior
from jev_omm.positioning.adjust import cot_fade, gex_adjust
from jev_omm.state_os.engines import gen3_coverage, letf_rebalance, tdf_trade
from jev_omm.state_os.gate import state_gate
from jev_omm.state_os.gex_flow import signs_disagree
from jev_omm.state_os.vector import instability as instability_ratio
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
    "pm_fair_value",
    "etf_ap_arb",
    "liability_facilitator",
    "mm_inventory",
    "vol_surface_mm",
    "flow_vpin",
    "dealer_gamma",
    "cot_fade",
    "letf_day",
    "instability_spike",
    "remaining_parent",
    "constraint_gate",
    "gex_disagree",
    "tdf_threshold",
    "overwrite_roll",
)


def _location(strategy: str, peer_pnl: float) -> tuple[CaseRun, list[float], list[float]]:
    """Location spread. The original sim has no futures.

    Venue A moves one-for-one with the factor. Venue B moves with beta 0.55
    plus the transport basis, so a matched long/short is not factor-flat.
    The desk policy is the author's real-life comment, not a sim instrument:
    hedge with futures as if their beta were 1, while the future's true beta
    is 0.85. The gap is basis risk. A negative factor drift is the oil-down
    scenario in the write-up.
    """
    n = 40
    state = 7
    f = 100.0
    basis = 0.80
    fut_basis = 0.12
    qa = qb = qf = cash = 0.0
    inv: list[float] = []
    beta: list[float] = []
    fut_beta = 0.85
    for _t in range(n):
        pa = f
        pb = 100.0 + 0.55 * (f - 100.0) + basis
        pf = 100.0 + fut_beta * (f - 100.0) + fut_basis
        if abs(basis) > 0.20:
            qty = 2.0
            if basis > 0.0:
                cash -= qty * pa
                cash += qty * pb
                qa += qty
                qb -= qty
            else:
                cash += qty * pa
                cash -= qty * pb
                qa -= qty
                qb += qty
        oil = qa + 0.55 * qb
        if strategy == "desk":
            target = -oil
            dq = target - qf
            cash -= dq * pf
            qf = target
            net = oil + fut_beta * qf
        else:
            net = oil
        inv.append(abs(qa) + abs(qb))
        beta.append(net)
        state, z = lcg_next(state)
        f += -0.25 + 1.5 * z
        state, z2 = lcg_next(state)
        basis = 0.92 * basis + 0.06 + 0.02 * z2
        state, z3 = lcg_next(state)
        fut_basis = 0.80 * fut_basis + 0.05 * z3
    pa = f
    pb = 100.0 + 0.55 * (f - 100.0) + basis
    pf = 100.0 + fut_beta * (f - 100.0) + fut_basis
    pnl = cash + qa * pa + qb * pb + qf * pf
    sc = score_path(pnl, inv, beta, inv_lambda=0.01, beta_lambda=2.0, peer_pnl=peer_pnl)
    events = [
        {
            "type": "HedgeConstraint",
            "futures_in_sim": False,
            "hedge": "none" if strategy != "desk" else "irl_futures_basis",
            "futures_beta": fut_beta,
            "residual_beta": beta[-1] if beta else 0.0,
            "note": (
                "Original simulator: cannot short spot oil, and futures were not in the sim. "
                "The desk book is an out-of-sim futures overlay. True futures beta is 0.85, "
                "so a one-for-one hedge leaves residual basis risk."
            ),
        }
    ]
    run = CaseRun(SPECS["location_arb"], strategy, sc, inv, beta, events)
    return run, inv, beta


def _pm(strategy: str, peer_pnl: float) -> CaseRun:
    """Three names with known fair values. Desk forces net beta to zero."""
    n = 36
    state = 31
    fair = [100.0, 100.0, 100.0]
    price = [96.5, 99.4, 102.8]
    q = [0.0, 0.0, 0.0]
    cash = 0.0
    inv: list[float] = []
    beta: list[float] = []
    fair_edge: list[float] = []
    for _t in range(n):
        gap = [fair[i] - price[i] for i in range(3)]
        target = [0.0, 0.0, 0.0]
        if strategy != "desk":
            for i in range(3):
                if gap[i] > 0.40:
                    target[i] = 1.0
        else:
            for i in range(3):
                if gap[i] > 0.40:
                    target[i] = 2.0
                elif gap[i] < -0.40:
                    target[i] = -2.0
            net = sum(target)
            j = min(range(3), key=lambda i: (abs(gap[i]), i))
            target[j] -= net
        for i in range(3):
            dq = target[i] - q[i]
            cash -= dq * price[i]
            q[i] = target[i]
        inv.append(sum(abs(x) for x in q))
        beta.append(sum(q))
        fair_edge.append(sum(q[i] * gap[i] for i in range(3)))
        state, z = lcg_next(state)
        factor = 1.2 * z
        for i in range(3):
            state, zi = lcg_next(state)
            price[i] = price[i] + 0.30 * (fair[i] - price[i]) + factor + 0.04 * zi
    pnl = cash + sum(q[i] * price[i] for i in range(3))
    sc = score_path(pnl, inv, beta, inv_lambda=0.02, beta_lambda=1.5, peer_pnl=peer_pnl)
    events = [
        {
            "type": "FairValueBook",
            "names": 3,
            "terminal_fair_edge": fair_edge[-1] if fair_edge else 0.0,
            "mean_fair_edge": sum(fair_edge) / len(fair_edge),
            "mean_abs_beta": sc.mean_abs_beta,
            "note": "Opposing leg zeros net beta. Fair-value distance is the edge; the factor is the penalty.",
        }
    ]
    return CaseRun(SPECS["pm_fair_value"], strategy, sc, inv, beta, events)


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
    """Liability block, then a TWAP-like working schedule.

    Gap 1 vs gap 12 stylizes the write-up: an algo clears in a few seconds,
    a hand schedule takes on the order of a minute and sits in the drift.
    The client block is forced (low toxicity prior). Later prints are
    discretionary (high toxicity prior); the desk does not add them.
    """
    n = 48
    state = 21
    block = 12.0
    premium = 0.18
    child = 2.0
    gap = 1 if strategy == "desk" else 12
    impact_k = 0.008
    mid = 100.0
    inv_pos = block
    cash = -block * (mid - premium)
    next_slice = gap
    disc_loss = 0.0
    inv: list[float] = []
    beta = [0.0] * n
    events: list[dict] = [
        {
            "type": "Block",
            "flow_class": "forced",
            "tox_prior": 0.15,
            "size": block,
            "premium": premium,
            "slice_gap": gap,
            "note": "Forced client block. Child slices work it down. The model does not send a ticket.",
        }
    ]
    for t in range(n):
        worked = 0.0
        if inv_pos > 1e-9 and t >= next_slice:
            sl = child if inv_pos > child else inv_pos
            cash += sl * mid - impact_k * sl * sl
            inv_pos -= sl
            next_slice = t + gap
            worked = sl
        state, u = lcg_next(state)
        discretionary = (u + 1.0) / 2.0 < 0.30
        state, _side = lcg_next(state)
        took = 0.0
        if discretionary and strategy != "desk":
            cash -= mid
            inv_pos += 1.0
            disc_loss += 0.25
            took = 1.0
        state, z = lcg_next(state)
        mid += -0.012 + 0.02 * z
        inv.append(inv_pos)
        if worked > 0.0 or took > 0.0:
            events.append(
                {
                    "type": "Work",
                    "t": t,
                    "flow_class": "discretionary" if took > 0.0 else "forced",
                    "tox_prior": 0.70 if took > 0.0 else 0.15,
                    "slice": worked,
                    "discretionary_take": took,
                    "inventory": inv_pos,
                }
            )
    pnl = cash + inv_pos * mid - disc_loss
    sc = score_path(pnl, inv, beta, inv_lambda=0.05, beta_lambda=0.0, peer_pnl=peer_pnl)
    return CaseRun(SPECS["liability_facilitator"], strategy, sc, inv, beta, events)


def _mm(strategy: str, peer_pnl: float) -> CaseRun:
    """Disciplined inventory is the grade. Predatory peer-cover is research only.

    Forced flow (must-trade hedges, rebalance) has a low toxicity prior and
    pays the spread. Discretionary flow has a high prior. The research mode
    joins the discretionary wave and sells into the later cover; raw PnL can
    be higher, and the inventory-path penalty keeps it off the default grade.
    """
    if strategy not in ("naive", "desk", "predatory"):
        strategy = "naive"
    ahead, intensity, horizon = 4.0, 25.0, 1.0
    lat = 0.90 if strategy == "naive" else 0.20
    fills = expected_fills(ahead, 6.0, intensity, 0.0, horizon, lat)
    queue_pnl = 0.0 if strategy == "predatory" else fills * (0.05 - 0.15)
    n = 40
    state = 44
    inv_pos = 0.0
    spread = 0.0
    adverse = 0.0
    inv: list[float] = []
    beta = [0.0] * n
    for t in range(n):
        state, u = lcg_next(state)
        forced = (u + 1.0) / 2.0 < 0.45
        wave = (not forced) and (t >= 8)
        if strategy == "naive":
            inv_pos += 1.0
            spread += 0.04
            if wave:
                adverse += 0.09
        elif strategy == "desk":
            if forced and abs(inv_pos) < 4.0:
                state, side = lcg_next(state)
                inv_pos += 1.0 if side > 0.0 else -1.0
                spread += 0.07
            elif abs(inv_pos) >= 4.0:
                inv_pos += -1.0 if inv_pos > 0.0 else 1.0
                spread += 0.03
        else:
            if t < 30 and not forced:
                inv_pos += 1.0
            elif t >= 30 and inv_pos > 0.0:
                sold = min(inv_pos, 4.0)
                inv_pos -= sold
                spread += sold * 0.35
        inv.append(inv_pos)
    pnl = queue_pnl + spread - adverse
    sc = score_path(pnl, inv, beta, inv_lambda=0.08, beta_lambda=0.0, peer_pnl=peer_pnl)
    events = [
        {
            "type": "LobRace",
            "cancel_latency": lat,
            "expected_fills": fills,
            "queue_pnl": queue_pnl,
            "spread_pnl": spread,
            "flow_classes": ["forced", "discretionary"],
            "tox_prior_forced": 0.20,
            "tox_prior_discretionary": 0.75,
            "policy": strategy,
            "graded": strategy == "desk",
            "research_mode": "predatory_peer_cover",
        }
    ]
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


def _flow_vpin(strategy: str, peer_pnl: float) -> CaseRun:
    """Toxic VPIN / OFI tape. Desk quotes wider and smaller. Naive does not.

    The graded PnL uses ``flow_prior`` in code. A Decision snapshot is logged
    from the fallback client and is not an order.
    """
    n = 24
    state = 11
    pnl = 0.0
    inv: list[float] = []
    beta: list[float] = []
    last_toxic = True
    for _t in range(n):
        state, z = lcg_next(state)
        toxic = abs(z) > 0.45
        last_toxic = toxic
        vpin = 0.82 if toxic else 0.12
        ofi = (0.75 if z > 0.0 else -0.75) if toxic else 0.05 * z
        spoof = 0.70 if toxic else 0.05
        off = 0.55 if toxic else 0.10
        aggr = 0.80 if toxic else 0.0
        if strategy == "desk":
            _tox, spread_mult, size_mult = flow_prior(vpin, ofi, aggr, off, spoof)
        else:
            spread_mult, size_mult = 1.0, 1.0
        fill_prob = 0.85 / spread_mult
        adverse = 0.35 if toxic else 0.02
        pnl += fill_prob * size_mult * (0.08 - adverse)
        inv.append(size_mult if toxic else 0.0)
        beta.append(0.0)
    sc = score_path(pnl, inv, beta, inv_lambda=0.02, beta_lambda=0.0, peer_pnl=peer_pnl)
    toxic_state = build_mm_state(
        time=0.0, spot=100.0, option_mid=2.0, iv=0.2, inventory=0,
        delta=0.2, gamma=0.01, vega=8.0, cash_pnl=0.0, half_spread=0.15, quoting_allowed=True,
        toxicity_features={
            "vpin": 0.82,
            "ofi": 0.75,
            "aggr_imbalance": 0.80,
            "off_exchange_share": 0.55,
            "spoof": 0.70,
        },
    )
    result = DeterministicFallbackClient().system_one(toxic_state, build_mm_questions())
    mods = apply_policy(result)
    events = [
        {
            "type": "DecisionSnapshot",
            "source": result.source,
            "model": result.model,
            "size_tier": result.answers["size_tier"].choice,
            "toxicity": float(result.answers["toxicity"].score),
            "size_mult": mods.size_mult,
            "spread_mult": mods.spread_mult,
            "last_step_toxic": last_toxic,
            "note": (
                "Choice/Score/Noul from the fallback client on a toxic tape. "
                "The case engine applies flow_prior. The model returns no ticket."
            ),
        }
    ]
    return CaseRun(SPECS["flow_vpin"], strategy, sc, inv, beta, events, decision_source=result.source)


def _dealer_gamma(strategy: str, peer_pnl: float) -> CaseRun:
    """Long-gamma pin, then a short-gamma trend.

    Desk uses ``gex_adjust``: wider hedge band and larger size while dealers
    are long gamma; wider quotes, smaller size, tighter band when they are short.
    Naive keeps band multiplier 1 and size 1.
    """
    n = 20
    state = 19
    spot = 100.0
    pin = 100.0
    hedge_spot = 100.0
    pnl = 0.0
    inv: list[float] = []
    beta: list[float] = []
    for t in range(n):
        state, z = lcg_next(state)
        if t < 10:
            gex = 0.80
            spot += -0.55 * (spot - pin) + 0.40 * z
            adverse = 0.0
        else:
            gex = -0.80
            spot += 0.55 + 0.15 * z
            adverse = 0.12
        if strategy == "desk":
            _shift, spread, size, band, _urg = gex_adjust(True, gex, (pin - spot) / spot, 0.0, spot)
        else:
            spread, size, band = 1.0, 1.0, 1.0
        gap = spot - hedge_spot
        slip = 0.0
        if abs(gap) > 0.80 * band:
            slip = 0.045 * abs(gap)
            hedge_spot = spot
        pnl += (0.06 * size) / spread - slip - adverse * size
        inv.append(abs(spot - hedge_spot))
        beta.append(0.0)
    sc = score_path(pnl, inv, beta, inv_lambda=0.001, beta_lambda=0.0, peer_pnl=peer_pnl)
    short = build_mm_state(
        time=0.0, spot=spot, option_mid=2.0, iv=0.22, inventory=0,
        delta=0.4, gamma=-0.02, vega=6.0, cash_pnl=0.0, half_spread=0.2, quoting_allowed=True,
        positioning={"gex_enabled": 1.0, "gex_norm": -0.80, "pin_gap": 0.0},
    )
    result = DeterministicFallbackClient().system_one(short, build_mm_questions())
    mods = apply_policy(result)
    events = [
        {
            "type": "DecisionSnapshot",
            "source": result.source,
            "model": result.model,
            "regime": result.answers["regime"].choice,
            "size_tier": result.answers["size_tier"].choice,
            "hedge_now": float(result.answers["hedge_now"].noul),
            "hedge_flag": mods.hedge_now,
            "spread_mult": mods.spread_mult,
            "size_mult": mods.size_mult,
            "note": (
                "Short-gamma state. Choice/Score/Noul only. "
                "Hedge urgency is a flag on the snapshot."
            ),
        }
    ]
    return CaseRun(SPECS["dealer_gamma"], strategy, sc, inv, beta, events, decision_source=result.source)


_COT_Z = (0.3, 1.1, 1.8, 2.6, 3.1, 2.2, 0.6, -0.4, -1.6, -2.5, -3.0, -1.4, 0.2, 0.5)


def _cot_fade(strategy: str, peer_pnl: float) -> CaseRun:
    """Weekly speculative z-score. The next return mean-reverts.

    Desk fades |z| >= 1.5. Naive takes the sign of z. Quote size from
    ``cot_fade`` is logged; the position itself is the fade.
    """
    state = 23
    pnl = 0.0
    inv: list[float] = []
    beta: list[float] = []
    for z in _COT_Z:
        state, noise = lcg_next(state)
        ret = -0.18 * z + 0.01 * noise
        fade = max(-1.0, min(1.0, z / 4.0))
        if strategy == "desk":
            pos = -fade if abs(z) >= 1.5 else 0.0
        else:
            pos = fade
        pnl += pos * ret * 10.0
        inv.append(abs(pos))
        beta.append(0.0)
    sc = score_path(pnl, inv, beta, inv_lambda=0.02, beta_lambda=0.0, peer_pnl=peer_pnl)
    _shift, size_mult = cot_fade(True, 3.1, 100.0)
    crowded = build_mm_state(
        time=0.0, spot=100.0, option_mid=2.0, iv=0.2, inventory=0,
        delta=0.1, gamma=0.01, vega=4.0, cash_pnl=0.0, half_spread=0.12, quoting_allowed=True,
        positioning={"cot_z": 3.1},
    )
    result = DeterministicFallbackClient().system_one(crowded, build_mm_questions())
    mods = apply_policy(result)
    events = [
        {
            "type": "DecisionSnapshot",
            "source": result.source,
            "model": result.model,
            "size_tier": result.answers["size_tier"].choice,
            "spread_mult": mods.spread_mult,
            "policy_size_mult": mods.size_mult,
            "cot_size_mult": size_mult,
            "cot_shift": _shift,
            "note": (
                "Extreme COT z-score. The fallback widens. "
                "cot_fade shifts the reservation in code."
            ),
        }
    ]
    return CaseRun(SPECS["cot_fade"], strategy, sc, inv, beta, events, decision_source=result.source)


def _snapshot(state: dict, engine_size: float) -> tuple[dict, str]:
    result = DeterministicFallbackClient().system_one(state, build_mm_questions())
    mods = apply_policy(result, state=state)
    event = {
        "type": "DecisionSnapshot",
        "source": result.source,
        "model": result.model,
        "size_tier": result.answers["size_tier"].choice,
        "size_mult": mods.size_mult,
        "spread_mult": mods.spread_mult,
        "engine_size": engine_size,
        "pull": mods.pull,
        "hedge_now": mods.hedge_now,
        "instability": mods.instability,
        "note": "Choice/Score/Noul only. The case engine applies the state gate in code.",
    }
    return event, result.source


def _letf_day(strategy: str, peer_pnl: float) -> CaseRun:
    """Cheng–Madhavan close. Desk supplies the buy program and covers the revert."""
    demand = letf_rebalance(100.0, 3.0, 0.02)
    impact = demand / 40.0 * 0.50
    reversion = 0.65 * impact
    if strategy == "desk":
        pnl = demand * reversion
        inv = [demand, 0.0]
    else:
        pnl = -demand * reversion
        inv = [demand, demand]
    beta = [0.0, 0.0]
    sc = score_path(pnl, inv, beta, inv_lambda=0.0, beta_lambda=0.0, peer_pnl=peer_pnl)
    events = [
        {
            "type": "LetfRebalance",
            "demand": demand,
            "reversion": reversion,
            "note": "Q = AUM (L^2 - L) r. Positive demand is a buy of the underlying.",
        }
    ]
    return CaseRun(SPECS["letf_day"], strategy, sc, inv, beta, events)


def _instability_spike(strategy: str, peer_pnl: float) -> CaseRun:
    """Calm bar, then |F|/L = 4. Desk pulls. Naive keeps size 1 into the print."""
    steps = ((0.0, 50.0), (80.0, 20.0))
    pnl = 0.0
    inv: list[float] = []
    last_size = 1.0
    for forced, liquidity in steps:
        ratio = instability_ratio(forced, liquidity)
        if strategy == "desk":
            gate = state_gate(True, ratio, False, 0.0, -forced, liquidity, 100.0)
            size = gate.size_mult
        else:
            size = 1.0
        last_size = size
        adverse = 1.25 if forced > 0.0 else 0.0
        pnl += 0.05 * size - adverse * size
        inv.append(size)
    beta = [0.0, 0.0]
    sc = score_path(pnl, inv, beta, inv_lambda=0.0, beta_lambda=0.0, peer_pnl=peer_pnl)
    hot = build_mm_state(
        time=0.0, spot=100.0, option_mid=2.0, iv=0.2, inventory=0,
        delta=0.2, gamma=0.01, vega=4.0, cash_pnl=0.0, half_spread=0.12, quoting_allowed=True,
        latent={"enabled": 1.0, "instability": 4.0, "f_net": -80.0, "l_exec": 20.0},
    )
    snap, source = _snapshot(hot, last_size if strategy == "desk" else 1.0)
    return CaseRun(SPECS["instability_spike"], strategy, sc, inv, beta, [snap], decision_source=source)


def _remaining_parent(strategy: str, peer_pnl: float) -> CaseRun:
    """Eight printing bars, then four quiet bars. Desk is smaller only while the parent lives."""
    pnl = 0.0
    inv: list[float] = []
    for i in range(12):
        if i < 8:
            remaining = 1.0 - (i + 1) / 8.0
            adverse = 0.20
        else:
            remaining = 0.0
            adverse = 0.0
        if strategy == "desk":
            gate = state_gate(True, 0.0, False, remaining, 0.0, 1.0, 100.0)
            size = gate.size_mult
        else:
            size = 1.0
        pnl += 0.04 * size - adverse * size
        inv.append(size)
    beta = [0.0] * 12
    sc = score_path(pnl, inv, beta, inv_lambda=0.0, beta_lambda=0.0, peer_pnl=peer_pnl)
    live = build_mm_state(
        time=0.0, spot=100.0, option_mid=2.0, iv=0.2, inventory=0,
        delta=0.1, gamma=0.01, vega=4.0, cash_pnl=0.0, half_spread=0.1, quoting_allowed=True,
        latent={"enabled": 1.0, "parent_remaining": 0.75, "l_exec": 1.0},
    )
    snap, source = _snapshot(live, inv[-1] if strategy == "desk" else 1.0)
    return CaseRun(SPECS["remaining_parent"], strategy, sc, inv, beta, [snap], decision_source=source)


def _constraint_gate(strategy: str, peer_pnl: float) -> CaseRun:
    """Vol-target cap at 0.20. Two bars bind. Desk size is 0 on those bars."""
    cap = 0.20
    pnl = 0.0
    inv: list[float] = []
    for sigma in (0.12, 0.16, 0.22, 0.28, 0.18):
        active = sigma > cap
        if strategy == "desk":
            gate = state_gate(True, 0.0, active, 0.0, 0.0, 1.0, 100.0)
            size = gate.size_mult
        else:
            size = 1.0
        shock = 0.80 if active else 0.0
        pnl += 0.05 * size - shock * size
        inv.append(size)
    beta = [0.0] * 5
    sc = score_path(pnl, inv, beta, inv_lambda=0.0, beta_lambda=0.0, peer_pnl=peer_pnl)
    bound = build_mm_state(
        time=0.0, spot=100.0, option_mid=2.0, iv=0.28, inventory=0,
        delta=0.2, gamma=0.02, vega=6.0, cash_pnl=0.0, half_spread=0.12, quoting_allowed=True,
        latent={"enabled": 1.0, "constraint_active": 1.0, "instability": 0.0},
    )
    snap, source = _snapshot(bound, 0.0 if strategy == "desk" else 1.0)
    return CaseRun(SPECS["constraint_gate"], strategy, sc, inv, beta, [snap], decision_source=source)


def _gex_disagree(strategy: str, peer_pnl: float) -> CaseRun:
    """Structural gamma is long. Today's signed flow is short. The path follows the flow."""
    structural = 0.60
    flow = -0.80
    disagree = signs_disagree(structural, flow)
    if strategy == "desk":
        size = 0.40 if disagree else 1.0
        pnl = -0.05 * size
    else:
        size = 1.15
        pnl = -1.0 * 0.25 * size
    inv = [size]
    beta = [0.0]
    sc = score_path(pnl, inv, beta, inv_lambda=0.0, beta_lambda=0.0, peer_pnl=peer_pnl)
    state = build_mm_state(
        time=0.0, spot=100.0, option_mid=2.0, iv=0.2, inventory=0,
        delta=0.3, gamma=0.01, vega=5.0, cash_pnl=0.0, half_spread=0.12, quoting_allowed=True,
        latent={"enabled": 1.0, "gex_disagree": 1.0 if disagree else 0.0},
    )
    snap, source = _snapshot(state, size)
    snap["structural"] = structural
    snap["flow"] = flow
    return CaseRun(SPECS["gex_disagree"], strategy, sc, inv, beta, [snap], decision_source=source)


def _tdf_threshold(strategy: str, peer_pnl: float) -> CaseRun:
    """250 bp of drift trades back to 175 bp. The next equity leg mean-reverts."""
    weight = 0.625
    target = 0.60
    trade = tdf_trade(weight, target, 1000.0) if strategy == "desk" else 0.0
    rx = -0.02
    pnl = trade * rx
    inv = [abs(trade)]
    beta = [0.0]
    sc = score_path(pnl, inv, beta, inv_lambda=0.0, beta_lambda=0.0, peer_pnl=peer_pnl)
    events = [
        {
            "type": "TdfThreshold",
            "trade": trade,
            "inside_band_trade": tdf_trade(0.61, 0.60, 1000.0),
            "note": "200 bp trigger, 175 bp destination. A 100 bp drift is inside the band.",
        }
    ]
    return CaseRun(SPECS["tdf_threshold"], strategy, sc, inv, beta, events)


def _overwrite_roll(strategy: str, peer_pnl: float) -> CaseRun:
    """Rich IV. Desk sells gen-3 cover. Naive skips the roll."""
    if strategy == "desk":
        coverage = gen3_coverage(0.28, 0.18)
        premium = 1.50 * coverage
        pnl = premium - 0.25 * premium
    else:
        coverage = 0.0
        pnl = 0.0
    inv = [coverage]
    beta = [0.0]
    sc = score_path(pnl, inv, beta, inv_lambda=0.0, beta_lambda=0.0, peer_pnl=peer_pnl)
    events = [
        {
            "type": "OverwriteRoll",
            "coverage": coverage,
            "note": "Gen-3 cover = clip(0.50 + 2(IV - IV_ref)). Gen-1 would be full ATM notional.",
        }
    ]
    return CaseRun(SPECS["overwrite_roll"], strategy, sc, inv, beta, events)


def run_case(name: str, strategy: str = "desk", peer_pnl: float = 0.0) -> CaseRun:
    if name == "location_arb":
        run, _, _ = _location(strategy, peer_pnl)
        return run
    if name == "pm_fair_value":
        return _pm(strategy, peer_pnl)
    if name == "etf_ap_arb":
        return _etf(strategy, peer_pnl)
    if name == "liability_facilitator":
        return _facilitator(strategy, peer_pnl)
    if name == "mm_inventory":
        return _mm(strategy, peer_pnl)
    if name == "vol_surface_mm":
        return _vol(strategy, peer_pnl)
    if name == "flow_vpin":
        return _flow_vpin(strategy, peer_pnl)
    if name == "dealer_gamma":
        return _dealer_gamma(strategy, peer_pnl)
    if name == "cot_fade":
        return _cot_fade(strategy, peer_pnl)
    if name == "letf_day":
        return _letf_day(strategy, peer_pnl)
    if name == "instability_spike":
        return _instability_spike(strategy, peer_pnl)
    if name == "remaining_parent":
        return _remaining_parent(strategy, peer_pnl)
    if name == "constraint_gate":
        return _constraint_gate(strategy, peer_pnl)
    if name == "gex_disagree":
        return _gex_disagree(strategy, peer_pnl)
    if name == "tdf_threshold":
        return _tdf_threshold(strategy, peer_pnl)
    if name == "overwrite_roll":
        return _overwrite_roll(strategy, peer_pnl)
    raise KeyError(name)


def run_pair(name: str) -> tuple[CaseRun, CaseRun]:
    naive = run_case(name, "naive", 0.0)
    desk = run_case(name, "desk", naive.score.absolute_pnl)
    return naive, desk


def leaderboard(runs: list[CaseRun]) -> list[CaseRun]:
    """Competitive hook: rank by risk-adjusted PnL, high to low."""
    return sorted(runs, key=lambda r: r.score.risk_adjusted, reverse=True)
