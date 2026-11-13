"""Walk-forward on a local tape. Quotes are the existing models.

Fills are an assumption, not a queue: we post without crossing a two-sided
book (join the touch when the model is tighter, sit behind when it is
wider). The next row's trade fills us if it prints at or through our
price, up to our size. There is no queue ahead. That overstates fills at
the touch. Crossed, locked, stale, and missing books are not quoted.

``QuoterConfig.A`` is not overwritten with the fitted events/second
intensity. That A is Guéant's closed form (arXiv 1105.3115), a different
object. Fitted k replaces kappa only when the train window identifies it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from jev_omm.config import TRADING_SECONDS_PER_YEAR, QuoterConfig
from jev_omm.execution.executable import kappa_for_touch, queue_decision
from jev_omm.quoter.avellaneda_stoikov import make_quote
from jev_omm.quoter.gueant_ode import IntensityFit, IntensityObs, estimate_intensity
from jev_omm.research.features import pack_adjustment
from jev_omm.research.tape import Tape, TapeFeeSchedule, count_states
from jev_omm.research.walkforward import ablation_specs, hazard_from_observations

_TICK = 0.0001


@dataclass
class ReplayRow:
    label: str
    quoter_mode: str
    feature_pack: str
    pnl: float
    n_fills: float
    contracts: float
    fees: float
    rebates: float
    markout_1: float
    mean_half: float
    quote_uptime: float
    n_quoted: int
    join_rate: float = 0.0
    adverse_markout: float = 0.0
    kappa_quote: float = 0.0


def _finite(x: float) -> bool:
    return math.isfinite(x)


def _mid(bid: float, ask: float) -> float:
    return 0.5 * (bid + ask)


def _sigma_from_mids(mids: list[float], dt_seconds: float) -> tuple[float, str]:
    clean = [m for m in mids if _finite(m) and m > 0.0]
    if len(clean) < 3:
        return 0.45, "default_lt_3_mids"
    diffs = [clean[i] - clean[i - 1] for i in range(1, len(clean))]
    std = float(np.std(diffs))
    if std <= 0.0:
        return 0.45, "default_zero_std"
    dt_years = max(dt_seconds, 1.0) / TRADING_SECONDS_PER_YEAR
    return std / math.sqrt(dt_years), "train_mids"


def _quoter(mode: str, *, kappa: float, sigma: float) -> QuoterConfig:
    if mode == "fixed":
        return QuoterConfig(
            mode="as_finite_horizon",
            gamma=0.0,
            kappa=kappa,
            sigma=sigma,
            quote_size=1,
            min_half_spread=0.25,
            max_half_spread=0.25,
            T_horizon=1.0 / 252.0,
        )
    if mode == "join_touch":
        return QuoterConfig(
            mode="as_finite_horizon",
            gamma=0.0,
            kappa=kappa,
            sigma=sigma,
            quote_size=1,
            min_half_spread=0.0001,
            max_half_spread=0.0001,
            T_horizon=1.0 / 252.0,
        )
    return QuoterConfig(
        mode=mode,
        gamma=0.12,
        kappa=kappa,
        sigma=sigma,
        A=140.0,
        quote_size=1,
        min_half_spread=0.01,
        max_half_spread=2.0,
        option_grid_n=11,
        option_grid_steps=8,
        contract_vega=8.0,
        vega_limit=40.0,
        inventory_cap=6,
        ode_steps=80,
    )


def _arrays(tape: Tape) -> dict[str, np.ndarray]:
    n = len(tape)
    nan = np.full(n, np.nan)
    return {
        "bid": tape.bid if tape.bid is not None else nan,
        "ask": tape.ask if tape.ask is not None else nan,
        "trade": tape.trade_px if tape.trade_px is not None else nan,
        "tsz": tape.trade_sz if tape.trade_sz is not None else np.ones(n),
        "spot": tape.spot if tape.spot is not None else nan,
        "strike": tape.strike if tape.strike is not None else np.full(n, np.nan),
    }


def fit_hazard(tape: Tape, *, kappa_prior: float, end: int) -> dict[str, object]:
    """Touch hazard on the train window.

    δ is the market half-spread. A fill observation is 1 when the next
    row's trade prints at or through that touch. Exposure is the gap to
    the next row, in seconds. k is held at the prior when δ does not move.
    """
    arr = _arrays(tape)
    labels = tape.book_state or ["two_sided"] * len(tape)
    obs: list[IntensityObs] = []
    last = min(end, len(tape) - 1)
    for i in range(last):
        if labels[i] != "two_sided":
            continue
        bid, ask = float(arr["bid"][i]), float(arr["ask"][i])
        if not (_finite(bid) and _finite(ask) and ask > bid):
            continue
        half = 0.5 * (ask - bid)
        dt = float(tape.time_seconds[i + 1] - tape.time_seconds[i])
        if dt <= 0.0:
            dt = 60.0
        trade = float(arr["trade"][i + 1])
        bid_hit = 1.0 if _finite(trade) and trade <= bid + 1e-9 else 0.0
        ask_hit = 1.0 if _finite(trade) and trade >= ask - 1e-9 else 0.0
        obs.append(IntensityObs(delta=half, exposure=dt, fills=bid_hit))
        obs.append(IntensityObs(delta=half, exposure=dt, fills=ask_hit))
    fit: IntensityFit = estimate_intensity(obs)
    A, k, method, raw = hazard_from_observations(obs, kappa_prior=kappa_prior)
    return {
        "A_per_second": float(A),
        "k_per_price": float(k),
        "fit_method": method,
        "mle_A": float(raw.A),
        "mle_k": float(raw.k),
        "loglik": float(raw.loglik),
        "n_obs": len(obs),
        "identified": method == "mle",
        "joint_fit_A": float(fit.A),
    }


def _train_touch(tape: Tape, end: int) -> float:
    """Median two-sided half-spread on the train window. Price units, not ticks."""
    arr = _arrays(tape)
    labels = tape.book_state or ["two_sided"] * len(tape)
    halves: list[float] = []
    last = min(end, len(tape))
    for i in range(last):
        if labels[i] != "two_sided":
            continue
        bid, ask = float(arr["bid"][i]), float(arr["ask"][i])
        if _finite(bid) and _finite(ask) and ask > bid:
            halves.append(0.5 * (ask - bid))
    if not halves:
        return 0.05
    return float(np.median(np.asarray(halves, dtype=float)))


def _train_print_rate(tape: Tape, end: int) -> float:
    """Contracts per second of the next-row print, train window only."""
    arr = _arrays(tape)
    labels = tape.book_state or ["two_sided"] * len(tape)
    contracts = 0.0
    seconds = 0.0
    last = min(end, len(tape) - 1)
    for i in range(max(last, 0)):
        if labels[i] != "two_sided":
            continue
        dt = float(tape.time_seconds[i + 1] - tape.time_seconds[i])
        if dt <= 0.0:
            dt = 60.0
        tsz = float(arr["tsz"][i + 1])
        contracts += tsz if _finite(tsz) and tsz > 0.0 else 1.0
        seconds += dt
    if seconds <= 0.0:
        return 2.0 / 60.0
    return float(contracts / seconds)


def _post_side(
    *,
    model_px: float,
    touch_px: float,
    inside: float,
    depth: float,
    toxic: float,
    queue_edge: bool,
    our_size: int,
    dt: float,
    spread_capture: float,
    print_rate: float,
) -> tuple[float | None, float, str]:
    """Post one side. ``inside`` > 0 means the model is through the touch.

    A model that is already inside the spread is posted there (ahead 0).
    A model at the touch joins, and ``queue_edge`` may step one tick inside
    or cancel. A model behind the touch stays behind. It does not improve
    up to the touch. A returned price of None is a pull.
    """
    if inside > _TICK:
        ahead0 = 0.0
        at_touch = False
        already_inside = True
    elif inside >= -_TICK:
        ahead0 = max(depth, 0.0)
        at_touch = True
        already_inside = False
    else:
        return model_px, 0.0, "behind"
    dec = queue_decision(
        ahead0,
        float(max(our_size, 1)),
        print_rate,
        0.0,
        dt,
        spread_capture,
        0.0,
        toxic if queue_edge else 0.0,
        queue_edge,
    )
    if dec.action == 2:
        return None, ahead0, "cancel"
    if dec.action == 1 and at_touch and not already_inside:
        stepped = touch_px + _TICK if model_px <= touch_px + _TICK else model_px
        # Bid steps up; ask steps down. ``inside``'s sign is handled by the caller
        # via ``touch_px`` and the direction encoded in ``step_sign`` — see below.
        return stepped, 0.0, "improve"
    if already_inside:
        return model_px, 0.0, "inside"
    return touch_px, ahead0, "join"


def _replay_lob(
    tape: Tape,
    *,
    label: str,
    mode: str,
    pack: str,
    fees: TapeFeeSchedule,
    kappa: float,
    sigma: float,
    start: int,
    end: int,
    touch: float,
    print_rate: float,
    executable_units: bool,
    queue_edge: bool,
    spreads_in_touch: float,
) -> ReplayRow:
    """LOB posting on a local tape.

    The fill is still the next row's print. Joining a displayed size the
    print does not clear fills nothing. A quote inside the spread has no
    queue ahead of it, so a print at the old touch reaches it. Sitting
    behind fills only when the print trades through that price. No print
    is invented. Missing spot does not become the strike.
    """
    arr = _arrays(tape)
    labels = tape.book_state or ["two_sided"] * len(tape)
    gamma = 0.12
    quote_kappa = float(kappa)
    if executable_units and mode not in ("fixed", "join_touch"):
        quote_kappa = kappa_for_touch(gamma, max(touch, 1e-6) * spreads_in_touch, kappa)
    cfg = _quoter(mode, kappa=quote_kappa, sigma=sigma)
    inv = 0
    cash = 0.0
    fills = 0
    contracts = 0.0
    fee_paid = 0.0
    rebate_paid = 0.0
    markout = 0.0
    halves: list[float] = []
    quoted = 0
    eligible = 0
    joins = 0
    prev_mid: float | None = None
    spot_known = bool(np.isfinite(arr["spot"]).any())
    bsz = arr["bid"] * 0.0
    asz = arr["bid"] * 0.0
    if tape.bid_sz is not None:
        bsz = tape.bid_sz
    if tape.ask_sz is not None:
        asz = tape.ask_sz
    for i in range(start, end - 1):
        state = labels[i]
        bid, ask = float(arr["bid"][i]), float(arr["ask"][i])
        if state != "two_sided" or not (_finite(bid) and _finite(ask) and ask > bid):
            continue
        eligible += 1
        mid = _mid(bid, ask)
        ret_bps = 0.0
        if prev_mid is not None and prev_mid > 0.0:
            ret_bps = 1e4 * (mid - prev_mid) / prev_mid
        move = 0.0 if prev_mid is None else abs(mid - prev_mid)
        toxic = min(1.0, move / max(touch, 1e-6))
        prev_mid = mid
        dt = float(tape.time_seconds[i + 1] - tape.time_seconds[i])
        if dt <= 0.0:
            dt = 60.0
        strike = float(arr["strike"][i]) if _finite(float(arr["strike"][i])) else 100.0
        spot_i = float(arr["spot"][i])
        spot_ok = _finite(spot_i)
        depth_b = float(bsz[i]) if _finite(float(bsz[i])) else 0.0
        depth_a = float(asz[i]) if _finite(float(asz[i])) else 0.0
        if mode == "join_touch":
            post_bid: float | None = bid
            post_ask: float | None = ask
            half = 0.5 * (ask - bid)
            size = 1
            bid_ahead = max(depth_b, 0.0)
            ask_ahead = max(depth_a, 0.0)
            bid_how, ask_how = "join", "join"
            bid_live = True
            ask_live = True
        else:
            adj = pack_adjustment(
                pack,
                ret_bps=ret_bps,
                spot=spot_i if spot_ok else 0.0,
                strike=strike,
                mid=mid,
                allow_gex=spot_known and spot_ok,
            )
            spread_mult = 1.0 if adj is None else adj.spread_mult
            size_mult = 1.0 if adj is None else adj.size_mult
            if adj is not None and adj.size_mult == 0.0:
                continue
            quote = make_quote(mid, inv, cfg, spread_mult=spread_mult, size_mult=size_mult)
            model_bid = min(quote.bid, ask - _TICK)
            model_ask = max(quote.ask, bid + _TICK)
            if model_ask <= model_bid + _TICK:
                continue
            size = max(quote.bid_size, 1)
            capture = max(0.0, mid - model_bid)
            post_bid, bid_ahead, bid_how = _post_side(
                model_px=model_bid,
                touch_px=bid,
                inside=model_bid - bid,
                depth=depth_b,
                toxic=toxic,
                queue_edge=queue_edge,
                our_size=size,
                dt=dt,
                spread_capture=capture,
                print_rate=print_rate,
            )
            if bid_how == "improve":
                post_bid = min(bid + _TICK, ask - _TICK)
                bid_ahead = 0.0
            capture_a = max(0.0, model_ask - mid)
            post_ask, ask_ahead, ask_how = _post_side(
                model_px=model_ask,
                touch_px=ask,
                inside=ask - model_ask,
                depth=depth_a,
                toxic=toxic,
                queue_edge=queue_edge,
                our_size=size,
                dt=dt,
                spread_capture=capture_a,
                print_rate=print_rate,
            )
            if ask_how == "improve":
                post_ask = max(ask - _TICK, bid + _TICK)
                ask_ahead = 0.0
            bid_live = post_bid is not None
            ask_live = post_ask is not None
            if not bid_live and not ask_live:
                continue
            if not bid_live:
                post_bid = mid
                bid_ahead = 0.0
                bid_how = "cancel"
            if not ask_live:
                post_ask = mid
                ask_ahead = 0.0
                ask_how = "cancel"
            if bid_live and ask_live and post_ask <= post_bid + _TICK:
                continue
            if bid_live and ask_live:
                half = 0.5 * (float(post_ask) - float(post_bid))
            elif bid_live:
                half = max(0.0, mid - float(post_bid))
            else:
                half = max(0.0, float(post_ask) - mid)
        halves.append(half)
        quoted += 1
        if bid_how in ("join", "inside", "improve") or ask_how in ("join", "inside", "improve"):
            joins += 1
        trade = float(arr["trade"][i + 1])
        if not _finite(trade):
            continue
        tsz = float(arr["tsz"][i + 1])
        lot = tsz if _finite(tsz) and tsz > 0.0 else 1.0
        side = 0
        px = 0.0
        take = 0
        if bid_live and trade <= float(post_bid) + 1e-9 and inv < 6:
            reachable = lot - bid_ahead
            if reachable >= 1.0 or (bid_ahead <= 0.0 and lot > 0.0):
                take = min(size, max(1, int(lot))) if bid_ahead <= 0.0 else min(size, int(reachable))
                if take > 0:
                    side = 1
                    px = float(post_bid)
        elif ask_live and trade >= float(post_ask) - 1e-9 and inv > -6:
            reachable = lot - ask_ahead
            if reachable >= 1.0 or (ask_ahead <= 0.0 and lot > 0.0):
                take = min(size, max(1, int(lot))) if ask_ahead <= 0.0 else min(size, int(reachable))
                if take > 0:
                    side = -1
                    px = float(post_ask)
        if side == 0 or take <= 0:
            continue
        cash -= side * px * take
        fee_paid += fees.fee_per_contract * take
        rebate_paid += fees.rebate_per_contract * take
        cash -= fees.net_per_contract * take
        inv += side * take
        fills += 1
        contracts += take
        nxt = i + 2
        if nxt < len(tape) and labels[nxt] == "two_sided":
            nb, na = float(arr["bid"][nxt]), float(arr["ask"][nxt])
            if _finite(nb) and _finite(na) and na > nb:
                step_mo = side * take * (_mid(nb, na) - mid)
                markout += step_mo
    last_mid = prev_mid if prev_mid is not None else 0.0
    for j in range(end - 1, start - 1, -1):
        b, a = float(arr["bid"][j]), float(arr["ask"][j])
        if _finite(b) and _finite(a) and a > b:
            last_mid = _mid(b, a)
            break
    pnl = cash + inv * last_mid
    return ReplayRow(
        label=label,
        quoter_mode=mode,
        feature_pack=pack,
        pnl=float(pnl),
        n_fills=float(fills),
        contracts=float(contracts),
        fees=float(fee_paid),
        rebates=float(rebate_paid),
        markout_1=float(markout),
        mean_half=float(sum(halves) / len(halves)) if halves else 0.0,
        quote_uptime=float(quoted / max(eligible, 1)),
        n_quoted=quoted,
        join_rate=float(joins / quoted) if quoted else 0.0,
        adverse_markout=float(markout),
        kappa_quote=float(quote_kappa),
    )


def replay(
    tape: Tape,
    *,
    label: str,
    mode: str,
    pack: str,
    fees: TapeFeeSchedule,
    kappa: float,
    sigma: float,
    start: int = 0,
    end: int | None = None,
    fill_model: str = "print",
    touch: float = 0.05,
    print_rate: float = 2.0 / 60.0,
    executable_units: bool = False,
    queue_edge: bool = False,
    spreads_in_touch: float = 1.0,
) -> ReplayRow:
    """Replay one quoter on ``tape[start:end]``. Inventory is marked to the last mid.

    ``fill_model="print"`` is the 0.9 rule: join when tighter, sit behind
    when wider, fill if the next print trades through, no queue ahead.
    ``fill_model="lob"`` keeps the print as the fill but charges displayed
    size against a joiner and posts an inside quote when the model is tighter.
    """
    stop = len(tape) if end is None else min(end, len(tape))
    if fill_model == "lob":
        return _replay_lob(
            tape,
            label=label,
            mode=mode,
            pack=pack,
            fees=fees,
            kappa=kappa,
            sigma=sigma,
            start=start,
            end=stop,
            touch=touch,
            print_rate=print_rate,
            executable_units=executable_units,
            queue_edge=queue_edge,
            spreads_in_touch=spreads_in_touch,
        )
    if fill_model != "print":
        raise ValueError(f"fill_model must be 'print' or 'lob', got {fill_model!r}")
    arr = _arrays(tape)
    labels = tape.book_state or ["two_sided"] * len(tape)
    stop = len(tape) if end is None else min(end, len(tape))
    cfg = _quoter(mode, kappa=kappa, sigma=sigma)
    inv = 0
    cash = 0.0
    fills = 0
    contracts = 0.0
    fee_paid = 0.0
    rebate_paid = 0.0
    markout = 0.0
    halves: list[float] = []
    quoted = 0
    eligible = 0
    prev_mid: float | None = None
    spot_known = bool(np.isfinite(arr["spot"]).any())
    for i in range(start, stop - 1):
        state = labels[i]
        bid, ask = float(arr["bid"][i]), float(arr["ask"][i])
        if state != "two_sided" or not (_finite(bid) and _finite(ask) and ask > bid):
            continue
        eligible += 1
        mid = _mid(bid, ask)
        ret_bps = 0.0
        if prev_mid is not None and prev_mid > 0.0:
            ret_bps = 1e4 * (mid - prev_mid) / prev_mid
        prev_mid = mid
        strike = float(arr["strike"][i]) if _finite(float(arr["strike"][i])) else 100.0
        spot = float(arr["spot"][i]) if _finite(float(arr["spot"][i])) else strike
        if mode == "join_touch":
            post_bid, post_ask = bid, ask
            half = 0.5 * (ask - bid)
            size = 1
        else:
            adj = pack_adjustment(
                pack,
                ret_bps=ret_bps,
                spot=spot,
                strike=strike,
                mid=mid,
                allow_gex=spot_known,
            )
            spread_mult = 1.0 if adj is None else adj.spread_mult
            size_mult = 1.0 if adj is None else adj.size_mult
            if adj is not None and adj.size_mult == 0.0:
                continue
            quote = make_quote(mid, inv, cfg, spread_mult=spread_mult, size_mult=size_mult)
            # Join when tighter than the market; sit behind when wider. Never cross.
            post_bid = min(quote.bid, bid)
            post_ask = max(quote.ask, ask)
            if post_ask <= post_bid + _TICK:
                continue
            half = 0.5 * (post_ask - post_bid)
            size = max(quote.bid_size, 1)
        halves.append(half)
        quoted += 1
        trade = float(arr["trade"][i + 1])
        if not _finite(trade):
            continue
        tsz = float(arr["tsz"][i + 1])
        take = max(1, int(tsz)) if _finite(tsz) and tsz > 0.0 else 1
        take = min(take, size)
        side = 0
        px = 0.0
        if trade <= post_bid + 1e-9 and inv < 6:
            side = 1
            px = post_bid
        elif trade >= post_ask - 1e-9 and inv > -6:
            side = -1
            px = post_ask
        if side == 0:
            continue
        cash -= side * px * take
        fee_paid += fees.fee_per_contract * take
        rebate_paid += fees.rebate_per_contract * take
        cash -= fees.net_per_contract * take
        inv += side * take
        fills += 1
        contracts += take
        nxt = i + 2
        if nxt < len(tape) and labels[nxt] == "two_sided":
            nb, na = float(arr["bid"][nxt]), float(arr["ask"][nxt])
            if _finite(nb) and _finite(na) and na > nb:
                markout += side * take * (_mid(nb, na) - mid)
    last_mid = prev_mid if prev_mid is not None else 0.0
    for j in range(stop - 1, start - 1, -1):
        b, a = float(arr["bid"][j]), float(arr["ask"][j])
        if _finite(b) and _finite(a) and a > b:
            last_mid = _mid(b, a)
            break
    pnl = cash + inv * last_mid
    return ReplayRow(
        label=label,
        quoter_mode=mode,
        feature_pack=pack,
        pnl=float(pnl),
        n_fills=float(fills),
        contracts=float(contracts),
        fees=float(fee_paid),
        rebates=float(rebate_paid),
        markout_1=float(markout),
        mean_half=float(sum(halves) / len(halves)) if halves else 0.0,
        quote_uptime=float(quoted / max(eligible, 1)),
        n_quoted=quoted,
    )


def walk_forward(
    tape: Tape,
    *,
    fees: TapeFeeSchedule | None = None,
    train_frac: float = 0.5,
    kappa_prior: float = 1.5,
    fill_model: str = "lob",
    executable_units: bool = True,
    queue_edge: bool = True,
    spreads_in_touch: float = 1.0,
) -> dict[str, object]:
    """Fit the touch hazard on the first fraction; score models on the rest.

    The default fill model is ``lob``: displayed size is charged against a
    joiner, and an executable-unit κ puts the classical half-spread on the
    train-window touch. ``fill_model="print"`` is the 0.9 rule (no queue,
    κ held at the prior unless the hazard identifies it) and still shows
    zero model fills on the checked-in fixture.
    """
    fees = fees or TapeFeeSchedule(fee_per_contract=0.05, rebate_per_contract=0.0)
    n = len(tape)
    if n < 4:
        raise ValueError(f"tape {tape.source} has {n} rows; need at least 4")
    cut = max(2, min(n - 2, int(n * train_frac)))
    arr = _arrays(tape)
    train_mids: list[float] = []
    labels = tape.book_state
    for i in range(cut):
        if labels and labels[i] == "two_sided":
            b, a = float(arr["bid"][i]), float(arr["ask"][i])
            if _finite(b) and _finite(a) and a > b:
                train_mids.append(_mid(b, a))
    dt = 60.0
    if n >= 2:
        gaps = np.diff(tape.time_seconds[: cut + 1])
        pos = gaps[gaps > 0]
        if len(pos):
            dt = float(np.median(pos))
    sigma, sigma_how = _sigma_from_mids(train_mids, dt)
    # The laboratory quoter's dollar vol is a config knob. The sample
    # annualization is reported beside it and is not silently substituted:
    # a 20-minute preview annualizes to a nonsense σ and would only widen
    # every quote. Models keep σ = 0.45, the 0.8 ablation value.
    sigma_model = 0.45
    hazard = fit_hazard(tape, kappa_prior=kappa_prior, end=cut)
    kappa = float(hazard["k_per_price"]) if hazard["identified"] else kappa_prior
    touch = _train_touch(tape, cut)
    print_rate = _train_print_rate(tape, cut)
    use_units = bool(executable_units) and fill_model == "lob"
    use_queue = bool(queue_edge) and fill_model == "lob"
    kappa_exec = kappa_for_touch(0.12, max(touch, 1e-6) * spreads_in_touch, kappa_prior)
    rows: list[ReplayRow] = []
    specs = list(ablation_specs()) + [("join_touch", "join_touch", "off")]
    replay_kw = dict(
        fees=fees,
        kappa=kappa,
        sigma=sigma_model,
        start=cut,
        end=n,
        fill_model=fill_model,
        touch=touch,
        print_rate=print_rate,
        executable_units=use_units,
        queue_edge=use_queue,
        spreads_in_touch=spreads_in_touch,
    )
    for label, mode, pack in specs:
        rows.append(replay(tape, label=label, mode=mode, pack=pack, **replay_kw))
    uncal = replay(
        tape,
        label="as_uncalibrated_kappa",
        mode="as_finite_horizon",
        pack="off",
        fees=fees,
        kappa=kappa_prior,
        sigma=sigma_model,
        start=cut,
        end=n,
        fill_model=fill_model,
        touch=touch,
        print_rate=print_rate,
        executable_units=use_units,
        queue_edge=use_queue,
        spreads_in_touch=spreads_in_touch,
    )
    counts = count_states(tape)
    return {
        "source": tape.source,
        "synthetic_fixture": tape.synthetic_fixture,
        "n": n,
        "train_end": cut,
        "test_rows": n - cut,
        "sigma_train": sigma,
        "sigma_how": sigma_how,
        "sigma_model": sigma_model,
        "spot_known": bool(np.isfinite(arr["spot"]).any()),
        "spot_provenance": "underlying_print" if bool(np.isfinite(arr["spot"]).any()) else "absent",
        "quote_mid_provenance": "cbbo",
        "hazard": hazard,
        "kappa_used": kappa,
        "fill_model": fill_model,
        "executable_units": use_units,
        "queue_edge": use_queue,
        "train_touch": touch,
        "print_rate_per_second": print_rate,
        "kappa_executable": kappa_exec,
        "spreads_in_touch": spreads_in_touch,
        "fees": fees,
        "counts": counts,
        "rows": rows,
        "uncalibrated": uncal,
        "symbols": sorted({s for s in tape.symbol if s}),
        "notes": tape.notes,
    }


def _fmt(val: float) -> str:
    if abs(val) < 5e-7:
        val = 0.0
    return f"{val:.4f}"


def _table(rows: list[ReplayRow]) -> str:
    cols = (
        "label",
        "pnl",
        "n_fills",
        "contracts",
        "fees",
        "rebates",
        "markout_1",
        "mean_half",
        "quote_uptime",
    )
    head = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join("---" for _ in cols) + " |\n"
    body = ""
    for row in rows:
        cells = [
            row.label,
            _fmt(row.pnl),
            _fmt(row.n_fills),
            _fmt(row.contracts),
            _fmt(row.fees),
            _fmt(row.rebates),
            _fmt(row.markout_1),
            _fmt(row.mean_half),
            _fmt(row.quote_uptime),
        ]
        body += "| " + " | ".join(cells) + " |\n"
    return head + sep + body


def _verdict(rows: list[ReplayRow], uncal: ReplayRow) -> str:
    by = {r.label: r for r in rows}
    fixed = by.get("fixed_spread")
    touch = by.get("join_touch")
    lines = []
    if fixed is not None and touch is not None:
        if touch.n_fills <= 0 and all(r.n_fills <= 0 for r in rows):
            lines.append(
                "No row was filled on the test window. A zero is a zero: "
                "the models did not earn a fill under this posting rule."
            )
        elif touch.pnl > fixed.pnl:
            lines.append(
                f"Join-touch PnL {_fmt(touch.pnl)} is above fixed-spread "
                f"{_fmt(fixed.pnl)} on this window. Joining is a reference "
                "posting rule, not a new model, and the fill assumption has no queue."
            )
        else:
            lines.append(
                f"Join-touch PnL {_fmt(touch.pnl)} does not beat fixed-spread "
                f"{_fmt(fixed.pnl)} after the research fee. The touch is not "
                "free edge on this window."
            )
    theory = [r for r in rows if r.label not in {"join_touch", "fixed_spread"}]
    if fixed is not None and theory:
        winners = [r.label for r in theory if r.pnl > fixed.pnl + 1e-9]
        losers = [r.label for r in theory if r.pnl < fixed.pnl - 1e-9]
        tied = [r.label for r in theory if abs(r.pnl - fixed.pnl) <= 1e-9]
        if winners:
            lines.append("Above fixed-spread on this single draw: " + ", ".join(winners) + ".")
        if losers:
            lines.append("Below fixed-spread on this single draw: " + ", ".join(losers) + ".")
        if tied:
            lines.append("Tied with fixed-spread: " + ", ".join(tied) + ".")
        if not winners:
            lines.append(
                "No theory row beat fixed-spread on this draw. That is the result."
            )
    gap = uncal.pnl
    cal = by.get("avellaneda_stoikov")
    if cal is not None:
        if cal.pnl > gap + 1e-9:
            lines.append(
                f"Train-fitted kappa lifted A–S test PnL from {_fmt(gap)} to {_fmt(cal.pnl)}."
            )
        elif abs(cal.pnl - gap) <= 1e-9:
            lines.append(
                "Fitted kappa matched the prior on the test window "
                "(the train window did not identify a different k, or it did not move PnL)."
            )
        else:
            lines.append(
                f"Train-fitted kappa did not help A–S "
                f"({_fmt(cal.pnl)} vs uncalibrated {_fmt(gap)})."
            )
    if fixed is not None and fixed.pnl > 0.0 and fixed.markout_1 < 0.0:
        lines.append(
            f"Fixed-spread terminal PnL is {_fmt(fixed.pnl)} while one-step markout "
            f"is {_fmt(fixed.markout_1)}. The cash result is not a clean markout win."
        )
    return " ".join(lines)


def render_walk_section(result: dict[str, object], *, title: str) -> str:
    hazard = result["hazard"]
    fees: TapeFeeSchedule = result["fees"]  # type: ignore[assignment]
    counts: dict[str, int] = result["counts"]  # type: ignore[assignment]
    rows: list[ReplayRow] = result["rows"]  # type: ignore[assignment]
    uncal: ReplayRow = result["uncalibrated"]  # type: ignore[assignment]
    spot = "present" if result["spot_known"] else "absent (not filled in)"
    synth = "yes" if result["synthetic_fixture"] else "no"
    symbols = ", ".join(result["symbols"]) or "(none)"
    text = (
        f"## {title}\n\n"
        f"- Source: `{result['source']}`\n"
        f"- Synthetic fixture: {synth}\n"
        f"- Symbols: {symbols}\n"
        f"- Rows: {result['n']} (train ends at index {result['train_end']}, "
        f"test rows {result['test_rows']})\n"
        f"- Underlying spot: {spot}\n"
        f"- Book labels (whole file): two-sided {counts['two_sided']}, "
        f"crossed {counts['crossed']}, locked {counts['locked']}, "
        f"stale {counts['stale']}, one-sided {counts['one_sided']}, "
        f"missing {counts['missing']}\n"
        f"- Research fee {fees.fee_per_contract:.4f} per contract, "
        f"rebate {fees.rebate_per_contract:.4f}. Not a venue fee card.\n"
        f"- Train σ of the option mid (annualized, reported only): "
        f"{float(result['sigma_train']):.4f} via `{result['sigma_how']}`.\n"
        f"- σ used by the quoters: {float(result['sigma_model']):.2f} "
        f"($/√yr, the 0.8 ablation knob). The train annualization is not substituted.\n"
        f"- Hazard: A={float(hazard['A_per_second']):.6f}/s at δ=0, "
        f"k={float(hazard['k_per_price']):.4f}, method `{hazard['fit_method']}`, "
        f"MLE A={float(hazard['mle_A']):.3e}, MLE k={float(hazard['mle_k']):.4f}, "
        f"loglik={float(hazard['loglik']):.4f} on {hazard['n_obs']} touch-sides.\n"
        f"- Kappa on the test window: {float(result['kappa_used']):.4f}.\n\n"
        "Test-window replay. `mean_half` is the half-spread we actually posted "
        "(after joining or sitting behind the market). `markout_1` is one tape "
        "row after the fill, signed size times the mid change. `join_touch` posts "
        "the market itself and is a reference, not a model.\n\n"
        + _table(rows)
        + "\nUncalibrated A–S (kappa held at 1.5) on the same test rows: "
        f"PnL {_fmt(uncal.pnl)}, fills {_fmt(uncal.n_fills)}, "
        f"mean half {_fmt(uncal.mean_half)}.\n\n"
        + _verdict(rows, uncal)
        + "\n\n"
    )
    if not result["spot_known"]:
        text += (
            "GEX was not applied. Spot is absent, `allow_gex` is false, and that "
            "channel is the identity. `as_flow` and `as_flow_gex` match for that "
            "reason. State can still pull quotes; a low `quote_uptime` on "
            "`as_flow_gex_state` is the gate, not a GEX effect.\n\n"
        )
    if result.get("fill_model") == "lob" and result.get("executable_units"):
        text += (
            f"Fill model `lob`. Train touch {float(result.get('train_touch', 0.0)):.4f}. "
            f"Executable κ {float(result.get('kappa_executable', 0.0)):.4f} sets the "
            "A–S intensity half-spread equal to that touch "
            f"(`spreads_in_touch` {float(result.get('spreads_in_touch', 1.0)):.2f}). "
            "A joiner is behind displayed size; a quote inside the spread is not. "
            "The next print is the only fill. No print was invented. "
            f"Quote mid provenance `{result.get('quote_mid_provenance')}`. "
            f"Spot provenance `{result.get('spot_provenance')}`.\n\n"
        )
    else:
        text += (
            "With γ = 0.12 and κ = 1.5 the dominant half-spread term is "
            "(1/γ) ln(1 + γ/κ) ≈ 0.64 dollars, before the market join/behind clip. "
            "A preview NBBO of one to a few cents does not trade against that quote. "
            "Fixed-spread is clamped at 0.25 and can catch a sweep the wider quotes miss. "
            "That is the formula at these knobs, not a claim that 0.25 is optimal.\n\n"
        )
    if result["notes"]:
        text += f"{result['notes']}\n\n"
    return text


def _fill_table(rows: list[ReplayRow]) -> str:
    cols = (
        "label",
        "pnl",
        "n_fills",
        "contracts",
        "fees",
        "markout_1",
        "mean_half",
        "join_rate",
        "quote_uptime",
    )
    head = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join("---" for _ in cols) + " |\n"
    body = ""
    for row in rows:
        cells = [
            row.label,
            _fmt(row.pnl),
            _fmt(row.n_fills),
            _fmt(row.contracts),
            _fmt(row.fees),
            _fmt(row.markout_1),
            _fmt(row.mean_half),
            _fmt(row.join_rate),
            _fmt(row.quote_uptime),
        ]
        body += "| " + " | ".join(cells) + " |\n"
    return head + sep + body


def render_fill_comparison(tape: Tape, *, title: str) -> str:
    """Print-rule table beside the executable LOB table. Same tape, no new prints."""
    legacy = walk_forward(tape, fill_model="print", executable_units=False, queue_edge=False)
    live = walk_forward(tape)
    spot = live["spot_provenance"]
    text = (
        f"## {title}\n\n"
        f"- Source: `{live['source']}`\n"
        f"- Synthetic fixture: {'yes' if live['synthetic_fixture'] else 'no'}\n"
        f"- Rows: {live['n']} (train ends at {live['train_end']}, test rows {live['test_rows']})\n"
        f"- Spot provenance: `{spot}`. Quote mid provenance: `{live['quote_mid_provenance']}`.\n"
        f"- Train touch: {float(live['train_touch']):.4f} price units. "
        f"Print rate: {float(live['print_rate_per_second']):.6f} contracts/second.\n"
        f"- Executable κ: {float(live['kappa_executable']):.4f} "
        f"(A–S half-spread equals `spreads_in_touch` × train touch). "
        f"Hazard κ stays {float(live['kappa_used']):.4f} and is not the quote κ.\n\n"
        "### Legacy print rule (`fill_model=print`)\n\n"
        "Join when the model is tighter, sit behind when it is wider, no queue. "
        "This is the 0.9 / 1.2 table. Model half-spreads near 0.64 do not trade. "
        "The print rule does not record a join rate.\n\n"
        + _table(legacy["rows"])  # type: ignore[arg-type]
        + "\n### Executable LOB (`fill_model=lob`, touch units, queue edge on)\n\n"
        "The classical half-spread is mapped onto the train touch. A quote inside "
        "the spread has no queue ahead. A joiner sits behind displayed size, and a "
        "2-lot print does not clear it. Fixed-spread stays clamped at 0.25 and only "
        "fills on a sweep. `join_touch` is the touch reference and does not improve.\n\n"
        + _fill_table(live["rows"])  # type: ignore[arg-type]
        + "\n"
        + _verdict(live["rows"], live["uncalibrated"])  # type: ignore[arg-type]
        + "\n\n"
    )
    if spot != "underlying_print":
        text += (
            "Spot is absent. The quoter's reference mid is the CBBO mid "
            "(`quote_mid_provenance=cbbo`). No underlying print was joined in, "
            "and the strike was not written down as a spot.\n\n"
        )
    return text
