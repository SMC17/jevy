//! Citadel-style training kernels. Same LCG and constants as `jev_omm/training/`.
//!
//!   1. location_arb — sim has no futures; residual oil beta. Desk overlay hedges
//!      with a futures beta of 0.85 (basis risk).
//!   2. pm_fair_value — three names, opposing leg forces market neutrality.
//!   3. etf_ap_arb — size the create/redeem before latency kills it.
//!   4. liability_facilitator — work a forced block in slices; skip discretionary adds.
//!   5. mm_inventory — graded policy widens/skews. `.predatory` joins the wave
//!      and is not the default grade.
//!   6. vol_surface_mm — refuse butterfly / calendar violations.
//!   7. flow_vpin — widen and cut size into toxic VPIN / OFI.
//!   8. dealer_gamma — lean with long gamma; defend in short gamma.
//!   9. cot_fade — fade an extreme speculative z-score.
//!  10. letf_day — supply the Cheng–Madhavan close buy, cover the revert.
//!  11. instability_spike — pull when |F|/L reaches 4.
//!  12. remaining_parent — smaller while the parent is still printing.
//!  13. constraint_gate — size 0 once vol crosses the cap.
//!  14. gex_disagree — do not pin when flow-signed gamma disagrees.
//!  15. tdf_threshold — 200 bp trigger, 175 bp destination.
//!  16. overwrite_roll — sell gen-3 cover when IV is rich.
//!
//! Article (the `/p/` path 404s; this is the live URL):
//!   https://www.predictingalpha.com/blogs/what-i-learned-from-citadels-training-software
//! Reddit originals:
//!   https://www.reddit.com/r/Trading/comments/122y2zq/what_i_learned_from_citadels_training_software/
//!   https://www.reddit.com/r/options/comments/122pz4e/what_i_learned_from_citadels_training_software/

const std = @import("std");
const svi = @import("svi.zig");
const lob = @import("lob.zig");
const flow = @import("flow_signals.zig");
const positioning = @import("positioning.zig");
const state_os = @import("state_os.zig");

pub const Strategy = enum(u8) { naive = 0, desk = 1, predatory = 2 };

pub const CaseScore = struct {
    absolute_pnl: f64 = 0.0,
    inventory_path_penalty: f64 = 0.0,
    beta_penalty: f64 = 0.0,
    exec_penalty: f64 = 0.0,
    risk_adjusted: f64 = 0.0,
    relative_score: f64 = 0.0,
    mean_abs_beta: f64 = 0.0,
    mean_abs_inventory: f64 = 0.0,
};

pub fn lcgNext(state: *u32) f64 {
    state.* = state.* *% 1664525 +% 1013904223;
    return @as(f64, @floatFromInt(state.*)) / 4294967296.0 * 2.0 - 1.0;
}

pub fn scorePath(
    pnl: f64,
    inventory: []const f64,
    beta: []const f64,
    inv_lambda: f64,
    beta_lambda: f64,
    exec_penalty: f64,
    peer_pnl: f64,
) CaseScore {
    var inv_ss: f64 = 0.0;
    var inv_abs: f64 = 0.0;
    for (inventory) |q| {
        inv_ss += q * q;
        inv_abs += @abs(q);
    }
    var beta_ss: f64 = 0.0;
    var beta_abs: f64 = 0.0;
    for (beta) |b| {
        beta_ss += b * b;
        beta_abs += @abs(b);
    }
    const n_i: f64 = @floatFromInt(@max(inventory.len, 1));
    const n_b: f64 = @floatFromInt(@max(beta.len, 1));
    const inv_pen = inv_lambda * inv_ss / n_i;
    const beta_pen = beta_lambda * beta_ss / n_b;
    return .{
        .absolute_pnl = pnl,
        .inventory_path_penalty = inv_pen,
        .beta_penalty = beta_pen,
        .exec_penalty = exec_penalty,
        .risk_adjusted = pnl - inv_pen - beta_pen - exec_penalty,
        .relative_score = pnl - peer_pnl,
        .mean_abs_beta = beta_abs / n_b,
        .mean_abs_inventory = inv_abs / n_i,
    };
}

fn location(strategy: Strategy, peer_pnl: f64) CaseScore {
    // Original sim: no futures. Venue B beta is 0.55, so a matched spread
    // is not factor-flat. Desk overlay shorts futures one-for-one against
    // oil beta; the future's true beta is 0.85 (basis risk remains).
    const n: usize = 40;
    var state: u32 = 7;
    var f: f64 = 100.0;
    var basis: f64 = 0.80;
    var fut_basis: f64 = 0.12;
    var qa: f64 = 0.0;
    var qb: f64 = 0.0;
    var qf: f64 = 0.0;
    var cash: f64 = 0.0;
    var inv: [40]f64 = undefined;
    var beta_path: [40]f64 = undefined;
    const fut_beta: f64 = 0.85;
    const qty: f64 = 2.0;
    var t: usize = 0;
    while (t < n) : (t += 1) {
        const pa = f;
        const pb = 100.0 + 0.55 * (f - 100.0) + basis;
        const pf = 100.0 + fut_beta * (f - 100.0) + fut_basis;
        if (@abs(basis) > 0.20) {
            if (basis > 0.0) {
                cash -= qty * pa;
                cash += qty * pb;
                qa += qty;
                qb -= qty;
            } else {
                cash += qty * pa;
                cash -= qty * pb;
                qa -= qty;
                qb += qty;
            }
        }
        const oil = qa + 0.55 * qb;
        var net = oil;
        if (strategy == .desk) {
            const target = -oil;
            const dq = target - qf;
            cash -= dq * pf;
            qf = target;
            net = oil + fut_beta * qf;
        }
        inv[t] = @abs(qa) + @abs(qb);
        beta_path[t] = net;
        const z = lcgNext(&state);
        f += -0.25 + 1.5 * z;
        const z2 = lcgNext(&state);
        basis = 0.92 * basis + 0.06 + 0.02 * z2;
        const z3 = lcgNext(&state);
        fut_basis = 0.80 * fut_basis + 0.05 * z3;
    }
    const pa = f;
    const pb = 100.0 + 0.55 * (f - 100.0) + basis;
    const pf = 100.0 + fut_beta * (f - 100.0) + fut_basis;
    const pnl = cash + qa * pa + qb * pb + qf * pf;
    return scorePath(pnl, &inv, &beta_path, 0.01, 2.0, 0.0, peer_pnl);
}

fn pmFair(strategy: Strategy, peer_pnl: f64) CaseScore {
    const n: usize = 36;
    var state: u32 = 31;
    const fair = [3]f64{ 100.0, 100.0, 100.0 };
    var price = [3]f64{ 96.5, 99.4, 102.8 };
    var q = [3]f64{ 0.0, 0.0, 0.0 };
    var cash: f64 = 0.0;
    var inv: [36]f64 = undefined;
    var beta_path: [36]f64 = undefined;
    var t: usize = 0;
    while (t < n) : (t += 1) {
        const gap = [3]f64{ fair[0] - price[0], fair[1] - price[1], fair[2] - price[2] };
        var target = [3]f64{ 0.0, 0.0, 0.0 };
        if (strategy != .desk) {
            var i: usize = 0;
            while (i < 3) : (i += 1) {
                if (gap[i] > 0.40) target[i] = 1.0;
            }
        } else {
            var i: usize = 0;
            while (i < 3) : (i += 1) {
                if (gap[i] > 0.40) target[i] = 2.0 else if (gap[i] < -0.40) target[i] = -2.0;
            }
            const net = target[0] + target[1] + target[2];
            var j: usize = 0;
            var best = @abs(gap[0]);
            var k: usize = 1;
            while (k < 3) : (k += 1) {
                const a = @abs(gap[k]);
                if (a < best) {
                    best = a;
                    j = k;
                }
            }
            target[j] -= net;
        }
        var i: usize = 0;
        while (i < 3) : (i += 1) {
            const dq = target[i] - q[i];
            cash -= dq * price[i];
            q[i] = target[i];
        }
        inv[t] = @abs(q[0]) + @abs(q[1]) + @abs(q[2]);
        beta_path[t] = q[0] + q[1] + q[2];
        const z = lcgNext(&state);
        const factor = 1.2 * z;
        i = 0;
        while (i < 3) : (i += 1) {
            const zi = lcgNext(&state);
            price[i] = price[i] + 0.30 * (fair[i] - price[i]) + factor + 0.04 * zi;
        }
    }
    const pnl = cash + q[0] * price[0] + q[1] * price[1] + q[2] * price[2];
    return scorePath(pnl, &inv, &beta_path, 0.02, 1.5, 0.0, peer_pnl);
}

fn etfPath(seed: u32, out: []f64) void {
    var state = seed;
    var premium: f64 = 0.50;
    out[0] = premium;
    var t: usize = 1;
    while (t < out.len) : (t += 1) {
        const z = lcgNext(&state);
        premium = premium * 0.72 + 0.015 * z;
        out[t] = premium;
    }
}

fn etf(strategy: Strategy, peer_pnl: f64) CaseScore {
    var path: [17]f64 = undefined;
    etfPath(11, &path);
    const fee: f64 = 0.02;
    const latency: usize = if (strategy == .desk) 1 else 8;
    const resid = 0.02 * @sqrt(@as(f64, @floatFromInt(latency)));
    const edge0 = path[0] - fee;
    const size: f64 = if (strategy == .desk and edge0 > 4.0 * resid) 12.0 else 1.0;
    var shock_state: u32 = (11 +% @as(u32, @intCast(latency * 17)));
    if (shock_state == 0) shock_state = 1;
    const z = lcgNext(&shock_state);
    const shock = resid * z;
    const edge = path[latency] - fee - shock;
    const impact = 0.0004 * size * size;
    const pnl = size * edge - impact;
    const exec_pen = @abs(shock) * size;
    var inv: [16]f64 = [_]f64{0} ** 16;
    var beta: [16]f64 = [_]f64{0} ** 16;
    var i: usize = 0;
    while (i < latency and i < inv.len) : (i += 1) inv[i] = size;
    return scorePath(pnl, inv[0..@max(latency, 1)], beta[0..@max(latency, 1)], 0.0, 0.0, exec_pen, peer_pnl);
}

fn facilitator(strategy: Strategy, peer_pnl: f64) CaseScore {
    // Forced client block, then child slices. Desk gap 1 vs hand gap 12
    // (~4–5s vs ~1 minute). Discretionary prints carry the high toxicity prior.
    const n: usize = 48;
    var state: u32 = 21;
    const block: f64 = 12.0;
    const premium: f64 = 0.18;
    const child: f64 = 2.0;
    const gap: usize = if (strategy == .desk) 1 else 12;
    const impact_k: f64 = 0.008;
    var mid: f64 = 100.0;
    var inv_pos: f64 = block;
    var cash: f64 = -block * (mid - premium);
    var next_slice: usize = gap;
    var disc_loss: f64 = 0.0;
    var inv: [48]f64 = undefined;
    var beta: [48]f64 = [_]f64{0} ** 48;
    var t: usize = 0;
    while (t < n) : (t += 1) {
        if (inv_pos > 1e-9 and t >= next_slice) {
            const sl = if (inv_pos > child) child else inv_pos;
            cash += sl * mid - impact_k * sl * sl;
            inv_pos -= sl;
            next_slice = t + gap;
        }
        const u = lcgNext(&state);
        const discretionary = (u + 1.0) / 2.0 < 0.30;
        _ = lcgNext(&state);
        if (discretionary and strategy != .desk) {
            cash -= mid;
            inv_pos += 1.0;
            disc_loss += 0.25;
        }
        const z = lcgNext(&state);
        mid += -0.012 + 0.02 * z;
        inv[t] = inv_pos;
    }
    const pnl = cash + inv_pos * mid - disc_loss;
    return scorePath(pnl, &inv, &beta, 0.05, 0.0, 0.0, peer_pnl);
}

fn mmInventory(strategy: Strategy, peer_pnl: f64) CaseScore {
    const ahead: f64 = 4.0;
    const intensity: f64 = 25.0;
    const horizon: f64 = 1.0;
    const lat: f64 = if (strategy == .naive) 0.90 else 0.20;
    const fills = lob.expectedFills(ahead, 6.0, intensity, 0.0, horizon, lat);
    var queue_pnl: f64 = fills * (0.05 - 0.15);
    if (strategy == .predatory) queue_pnl = 0.0;
    const n: usize = 40;
    var state: u32 = 44;
    var inv_pos: f64 = 0.0;
    var spread: f64 = 0.0;
    var adverse: f64 = 0.0;
    var inv: [40]f64 = undefined;
    var beta: [40]f64 = [_]f64{0} ** 40;
    var t: usize = 0;
    while (t < n) : (t += 1) {
        const u = lcgNext(&state);
        const forced = (u + 1.0) / 2.0 < 0.45;
        const wave = !forced and t >= 8;
        if (strategy == .naive) {
            inv_pos += 1.0;
            spread += 0.04;
            if (wave) adverse += 0.09;
        } else if (strategy == .desk) {
            if (forced and @abs(inv_pos) < 4.0) {
                const side = lcgNext(&state);
                if (side > 0.0) inv_pos += 1.0 else inv_pos -= 1.0;
                spread += 0.07;
            } else if (@abs(inv_pos) >= 4.0) {
                if (inv_pos > 0.0) inv_pos -= 1.0 else inv_pos += 1.0;
                spread += 0.03;
            }
        } else {
            if (t < 30 and !forced) {
                inv_pos += 1.0;
            } else if (t >= 30 and inv_pos > 0.0) {
                const sold = @min(inv_pos, 4.0);
                inv_pos -= sold;
                spread += sold * 0.35;
            }
        }
        inv[t] = inv_pos;
    }
    const pnl = queue_pnl + spread - adverse;
    return scorePath(pnl, &inv, &beta, 0.08, 0.0, 0.0, peer_pnl);
}

fn volSurface(strategy: Strategy, peer_pnl: f64) CaseScore {
    const clean = svi.SviParams{ .a = 0.04, .b = 0.1, .rho = -0.4, .m = 0.0, .sigma = 0.2 };
    const poisoned = svi.SviParams{ .a = 0.01, .b = 1.5, .rho = -0.9, .m = 0.0, .sigma = 0.05 };
    const clean_ok = svi.butterflyCheck(clean).ok;
    const poison_ok = svi.butterflyCheck(poisoned).ok;
    const earlier = svi.SviParams{ .a = 0.08, .b = 0.1, .rho = -0.3, .m = 0.0, .sigma = 0.2 };
    const later = svi.SviParams{ .a = 0.02, .b = 0.1, .rho = -0.3, .m = 0.0, .sigma = 0.2 };
    const calendar_ok = svi.rawCalendarOk(earlier, later);
    const iv_strike = svi.impliedVolAfterMove(clean, 100.0, 110.0, 100.0, 0.25, .sticky_strike);
    const iv_delta = svi.impliedVolAfterMove(clean, 100.0, 110.0, 100.0, 0.25, .sticky_delta);
    var pnl: f64 = if (clean_ok) 1.2 else 0.0;
    var arb_pen: f64 = 0.0;
    var regime_pen: f64 = 0.0;
    if (strategy == .desk) {
        // Refuse the poisoned smile and the inverted calendar. Mark sticky-delta,
        // which is the SVI coordinate, so there is no regime mismatch penalty.
        if (poison_ok) pnl -= 5.0;
        if (!calendar_ok) arb_pen += 0.0;
    } else {
        if (!poison_ok) arb_pen += 25.0;
        if (!calendar_ok) arb_pen += 10.0;
        regime_pen = @abs(iv_strike - iv_delta) * 40.0;
    }
    const inv = [_]f64{0.0};
    const beta = [_]f64{0.0};
    return scorePath(pnl - regime_pen, &inv, &beta, 0.0, 0.0, arb_pen, peer_pnl);
}

fn flowVpin(strategy: Strategy, peer_pnl: f64) CaseScore {
    const n: usize = 24;
    var state: u32 = 11;
    var pnl: f64 = 0.0;
    var inv: [24]f64 = undefined;
    var beta: [24]f64 = [_]f64{0.0} ** 24;
    var t: usize = 0;
    while (t < n) : (t += 1) {
        const z = lcgNext(&state);
        const toxic = @abs(z) > 0.45;
        const vpin: f64 = if (toxic) 0.82 else 0.12;
        const ofi: f64 = if (toxic) (if (z > 0.0) 0.75 else -0.75) else 0.05 * z;
        const spoof: f64 = if (toxic) 0.70 else 0.05;
        const off: f64 = if (toxic) 0.55 else 0.10;
        const aggr: f64 = if (toxic) 0.80 else 0.0;
        var spread_mult: f64 = 1.0;
        var size_mult: f64 = 1.0;
        if (strategy == .desk) {
            const prior = flow.flowPrior(vpin, ofi, aggr, off, spoof);
            spread_mult = prior.spread_mult;
            size_mult = prior.size_mult;
        }
        const fill_prob = 0.85 / spread_mult;
        const adverse: f64 = if (toxic) 0.35 else 0.02;
        pnl += fill_prob * size_mult * (0.08 - adverse);
        inv[t] = if (toxic) size_mult else 0.0;
    }
    return scorePath(pnl, &inv, &beta, 0.02, 0.0, 0.0, peer_pnl);
}

fn dealerGamma(strategy: Strategy, peer_pnl: f64) CaseScore {
    const n: usize = 20;
    var state: u32 = 19;
    var spot: f64 = 100.0;
    const pin: f64 = 100.0;
    var hedge_spot: f64 = 100.0;
    var pnl: f64 = 0.0;
    var inv: [20]f64 = undefined;
    var beta: [20]f64 = [_]f64{0.0} ** 20;
    var t: usize = 0;
    while (t < n) : (t += 1) {
        const z = lcgNext(&state);
        const gex: f64 = if (t < 10) 0.80 else -0.80;
        var adverse: f64 = 0.0;
        if (t < 10) {
            spot += -0.55 * (spot - pin) + 0.40 * z;
        } else {
            spot += 0.55 + 0.15 * z;
            adverse = 0.12;
        }
        var spread: f64 = 1.0;
        var size: f64 = 1.0;
        var band: f64 = 1.0;
        if (strategy == .desk) {
            const g = positioning.gexAdjust(true, gex, (pin - spot) / spot, 0.0, spot);
            spread = g.spread_mult;
            size = g.size_mult;
            band = g.hedge_band_mult;
        }
        const gap = spot - hedge_spot;
        var slip: f64 = 0.0;
        if (@abs(gap) > 0.80 * band) {
            slip = 0.045 * @abs(gap);
            hedge_spot = spot;
        }
        pnl += (0.06 * size) / spread - slip - adverse * size;
        inv[t] = @abs(spot - hedge_spot);
    }
    return scorePath(pnl, &inv, &beta, 0.001, 0.0, 0.0, peer_pnl);
}

const cot_z = [_]f64{ 0.3, 1.1, 1.8, 2.6, 3.1, 2.2, 0.6, -0.4, -1.6, -2.5, -3.0, -1.4, 0.2, 0.5 };

fn cotFadeCase(strategy: Strategy, peer_pnl: f64) CaseScore {
    var state: u32 = 23;
    var pnl: f64 = 0.0;
    var inv: [14]f64 = undefined;
    var beta: [14]f64 = [_]f64{0.0} ** 14;
    for (cot_z, 0..) |z, i| {
        const noise = lcgNext(&state);
        const ret = -0.18 * z + 0.01 * noise;
        const fade = @min(@max(z / 4.0, -1.0), 1.0);
        const pos: f64 = if (strategy == .desk) (if (@abs(z) >= 1.5) -fade else 0.0) else fade;
        pnl += pos * ret * 10.0;
        inv[i] = @abs(pos);
    }
    return scorePath(pnl, &inv, &beta, 0.02, 0.0, 0.0, peer_pnl);
}

fn letfDay(strategy: Strategy, peer_pnl: f64) CaseScore {
    const demand = state_os.letfRebalance(100.0, 3.0, 0.02);
    const impact = demand / 40.0 * 0.50;
    const reversion = 0.65 * impact;
    const pnl: f64 = if (strategy == .desk) demand * reversion else -demand * reversion;
    const inv = if (strategy == .desk) [_]f64{ demand, 0.0 } else [_]f64{ demand, demand };
    const beta = [_]f64{ 0.0, 0.0 };
    return scorePath(pnl, &inv, &beta, 0.0, 0.0, 0.0, peer_pnl);
}

fn instabilitySpike(strategy: Strategy, peer_pnl: f64) CaseScore {
    const forced = [_]f64{ 0.0, 80.0 };
    const liquidity = [_]f64{ 50.0, 20.0 };
    var pnl: f64 = 0.0;
    var inv: [2]f64 = undefined;
    for (forced, liquidity, 0..) |f, l, i| {
        const ratio = state_os.instability(f, l);
        const size: f64 = if (strategy == .desk)
            state_os.stateGate(true, ratio, false, 0.0, -f, l, 100.0).size_mult
        else
            1.0;
        const adverse: f64 = if (f > 0.0) 1.25 else 0.0;
        pnl += 0.05 * size - adverse * size;
        inv[i] = size;
    }
    const beta = [_]f64{ 0.0, 0.0 };
    return scorePath(pnl, &inv, &beta, 0.0, 0.0, 0.0, peer_pnl);
}

fn remainingParent(strategy: Strategy, peer_pnl: f64) CaseScore {
    var pnl: f64 = 0.0;
    var inv: [12]f64 = undefined;
    var beta = [_]f64{0.0} ** 12;
    for (0..12) |i| {
        const remaining: f64 = if (i < 8) 1.0 - @as(f64, @floatFromInt(i + 1)) / 8.0 else 0.0;
        const adverse: f64 = if (i < 8) 0.20 else 0.0;
        const size: f64 = if (strategy == .desk)
            state_os.stateGate(true, 0.0, false, remaining, 0.0, 1.0, 100.0).size_mult
        else
            1.0;
        pnl += 0.04 * size - adverse * size;
        inv[i] = size;
    }
    return scorePath(pnl, &inv, &beta, 0.0, 0.0, 0.0, peer_pnl);
}

fn constraintGate(strategy: Strategy, peer_pnl: f64) CaseScore {
    const sigmas = [_]f64{ 0.12, 0.16, 0.22, 0.28, 0.18 };
    var pnl: f64 = 0.0;
    var inv: [5]f64 = undefined;
    var beta = [_]f64{0.0} ** 5;
    for (sigmas, 0..) |sigma, i| {
        const active = sigma > 0.20;
        const size: f64 = if (strategy == .desk)
            state_os.stateGate(true, 0.0, active, 0.0, 0.0, 1.0, 100.0).size_mult
        else
            1.0;
        const shock: f64 = if (active) 0.80 else 0.0;
        pnl += 0.05 * size - shock * size;
        inv[i] = size;
    }
    return scorePath(pnl, &inv, &beta, 0.0, 0.0, 0.0, peer_pnl);
}

fn gexDisagree(strategy: Strategy, peer_pnl: f64) CaseScore {
    const disagree = state_os.signsDisagree(0.60, -0.80);
    const size: f64 = if (strategy == .desk) (if (disagree) 0.40 else 1.0) else 1.15;
    const pnl: f64 = if (strategy == .desk) -0.05 * size else -1.0 * 0.25 * size;
    const inv = [_]f64{size};
    const beta = [_]f64{0.0};
    return scorePath(pnl, &inv, &beta, 0.0, 0.0, 0.0, peer_pnl);
}

fn tdfThreshold(strategy: Strategy, peer_pnl: f64) CaseScore {
    const trade: f64 = if (strategy == .desk) state_os.tdfTrade(0.625, 0.60, 1000.0) else 0.0;
    const pnl = trade * -0.02;
    const inv = [_]f64{@abs(trade)};
    const beta = [_]f64{0.0};
    return scorePath(pnl, &inv, &beta, 0.0, 0.0, 0.0, peer_pnl);
}

fn overwriteRoll(strategy: Strategy, peer_pnl: f64) CaseScore {
    const coverage: f64 = if (strategy == .desk) state_os.gen3Coverage(0.28, 0.18) else 0.0;
    const premium = 1.50 * coverage;
    const pnl = premium - 0.25 * premium;
    const inv = [_]f64{coverage};
    const beta = [_]f64{0.0};
    return scorePath(pnl, &inv, &beta, 0.0, 0.0, 0.0, peer_pnl);
}

pub fn runCase(name: []const u8, strategy: Strategy, peer_pnl: f64) CaseScore {
    if (std.mem.eql(u8, name, "location_arb")) return location(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "pm_fair_value")) return pmFair(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "etf_ap_arb")) return etf(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "liability_facilitator")) return facilitator(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "mm_inventory")) return mmInventory(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "vol_surface_mm")) return volSurface(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "flow_vpin")) return flowVpin(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "dealer_gamma")) return dealerGamma(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "cot_fade")) return cotFadeCase(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "letf_day")) return letfDay(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "instability_spike")) return instabilitySpike(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "remaining_parent")) return remainingParent(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "constraint_gate")) return constraintGate(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "gex_disagree")) return gexDisagree(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "tdf_threshold")) return tdfThreshold(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "overwrite_roll")) return overwriteRoll(strategy, peer_pnl);
    return .{};
}

pub const CASES = [_][]const u8{
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
};

test "location arb sim cannot hedge; futures overlay leaves basis risk" {
    const naive = runCase("location_arb", .naive, 0.0);
    const desk = runCase("location_arb", .desk, naive.absolute_pnl);
    const ratio = desk.mean_abs_beta / naive.mean_abs_beta;
    try std.testing.expect(naive.mean_abs_beta > 10.0);
    try std.testing.expect(ratio > 0.10 and ratio < 0.20);
    try std.testing.expect(desk.beta_penalty < naive.beta_penalty);
    try std.testing.expect(desk.absolute_pnl > naive.absolute_pnl);
    try std.testing.expect(desk.risk_adjusted > naive.risk_adjusted);
    try std.testing.expect(desk.relative_score == desk.absolute_pnl - naive.absolute_pnl);
}

test "pm fair value desk is market neutral and sizes the edge" {
    const naive = runCase("pm_fair_value", .naive, 0.0);
    const desk = runCase("pm_fair_value", .desk, naive.absolute_pnl);
    try std.testing.expect(desk.mean_abs_beta < 1e-9);
    try std.testing.expect(naive.mean_abs_beta > 1.0);
    try std.testing.expect(desk.risk_adjusted > naive.risk_adjusted);
}

test "etf desk sizes the near-risk-free create before it decays" {
    const naive = runCase("etf_ap_arb", .naive, 0.0);
    const desk = runCase("etf_ap_arb", .desk, naive.absolute_pnl);
    try std.testing.expect(desk.absolute_pnl > naive.absolute_pnl);
    try std.testing.expect(desk.risk_adjusted > naive.risk_adjusted);
}

test "liability desk slices the block faster than the hand schedule" {
    const naive = runCase("liability_facilitator", .naive, 0.0);
    const desk = runCase("liability_facilitator", .desk, naive.absolute_pnl);
    try std.testing.expect(desk.mean_abs_inventory < naive.mean_abs_inventory);
    try std.testing.expect(desk.absolute_pnl > 0.0);
    try std.testing.expect(desk.risk_adjusted > naive.risk_adjusted);
}

test "mm default grade is the stable book, not the predatory cover" {
    const fast = lob.expectedFills(4.0, 6.0, 25.0, 0.0, 1.0, 0.20);
    const slow = lob.expectedFills(4.0, 6.0, 25.0, 0.0, 1.0, 0.90);
    try std.testing.expect(fast < slow);
    const naive = runCase("mm_inventory", .naive, 0.0);
    const desk = runCase("mm_inventory", .desk, naive.absolute_pnl);
    const pred = runCase("mm_inventory", .predatory, 0.0);
    try std.testing.expect(desk.absolute_pnl > 0.0);
    try std.testing.expect(desk.mean_abs_inventory < naive.mean_abs_inventory);
    try std.testing.expect(desk.risk_adjusted > naive.risk_adjusted);
    try std.testing.expect(pred.absolute_pnl > desk.absolute_pnl);
    try std.testing.expect(desk.risk_adjusted > pred.risk_adjusted);
    try std.testing.expect(desk.mean_abs_inventory < pred.mean_abs_inventory);
}

test "vol surface desk refuses butterfly and calendar arb" {
    const poisoned = svi.SviParams{ .a = 0.01, .b = 1.5, .rho = -0.9, .m = 0.0, .sigma = 0.05 };
    try std.testing.expect(!svi.butterflyCheck(poisoned).ok);
    const earlier = svi.SviParams{ .a = 0.08, .b = 0.1, .rho = -0.3, .m = 0.0, .sigma = 0.2 };
    const later = svi.SviParams{ .a = 0.02, .b = 0.1, .rho = -0.3, .m = 0.0, .sigma = 0.2 };
    try std.testing.expect(!svi.rawCalendarOk(earlier, later));
    const naive = runCase("vol_surface_mm", .naive, 0.0);
    const desk = runCase("vol_surface_mm", .desk, naive.absolute_pnl);
    try std.testing.expect(desk.risk_adjusted > naive.risk_adjusted);
    try std.testing.expect(desk.exec_penalty < naive.exec_penalty);
}

test "flow desk quotes through toxicity and keeps more pnl" {
    const naive = runCase("flow_vpin", .naive, 0.0);
    const desk = runCase("flow_vpin", .desk, naive.absolute_pnl);
    try std.testing.expect(desk.absolute_pnl > naive.absolute_pnl);
    try std.testing.expect(desk.risk_adjusted > naive.risk_adjusted);
    try std.testing.expect(desk.mean_abs_inventory < naive.mean_abs_inventory);
}

test "dealer gamma desk beats a flat band" {
    const naive = runCase("dealer_gamma", .naive, 0.0);
    const desk = runCase("dealer_gamma", .desk, naive.absolute_pnl);
    try std.testing.expect(desk.absolute_pnl > 0.0);
    try std.testing.expect(naive.absolute_pnl < 0.0);
    try std.testing.expect(desk.absolute_pnl > naive.absolute_pnl);
    try std.testing.expect(desk.risk_adjusted > naive.risk_adjusted);
}

test "cot desk fades the extreme and beats the crowd" {
    const naive = runCase("cot_fade", .naive, 0.0);
    const desk = runCase("cot_fade", .desk, naive.absolute_pnl);
    try std.testing.expect(desk.absolute_pnl > 0.0);
    try std.testing.expect(naive.absolute_pnl < 0.0);
    try std.testing.expect(desk.risk_adjusted > naive.risk_adjusted);
}

fn deskBeatsNaive(name: []const u8) !void {
    const naive = runCase(name, .naive, 0.0);
    const desk = runCase(name, .desk, naive.absolute_pnl);
    try std.testing.expect(desk.risk_adjusted > naive.risk_adjusted);
    try std.testing.expect(desk.absolute_pnl > naive.absolute_pnl);
}

test "state-os cases: the desk policy beats the naive one" {
    try deskBeatsNaive("letf_day");
    try deskBeatsNaive("instability_spike");
    try deskBeatsNaive("remaining_parent");
    try deskBeatsNaive("constraint_gate");
    try deskBeatsNaive("gex_disagree");
    try deskBeatsNaive("tdf_threshold");
    try deskBeatsNaive("overwrite_roll");
    try std.testing.expectApproxEqAbs(runCase("letf_day", .desk, 0.0).absolute_pnl, 1.17, 1e-12);
    try std.testing.expectApproxEqAbs(runCase("tdf_threshold", .desk, 0.0).absolute_pnl, 0.15, 1e-12);
}

test "lcg is deterministic" {
    var a: u32 = 7;
    var b: u32 = 7;
    try std.testing.expectApproxEqAbs(lcgNext(&a), lcgNext(&b), 0.0);
    try std.testing.expect(a == b);
}
