//! Citadel-style training kernels: case scores and five scripted paper cases.
//!
//! Lessons (role sims graded on absolute PnL, a competitive relative hook,
//! risk-adjusted PnL, and inventory-path / factor penalties):
//!   1. location_arb — hedge residual market beta
//!   2. etf_ap_arb — size the create/redeem before latency kills it
//!   3. liability_facilitator — shade when flow is informed
//!   4. mm_inventory — cancel/skew through a one-sided wave
//!   5. vol_surface_mm — refuse butterfly / calendar violations
//!
//! The Python research desk (`jev_omm/training/`) is the JSONL + Decision
//! owner. These kernels are the hot, deterministic twin used by
//! `zig build training` and the Zig tests. Same LCG, same constants.
//!
//! Posts (lessons, not a live Citadel system):
//!   https://www.reddit.com/r/Trading/comments/122y2zq/what_i_learned_from_citadels_training_software/
//!   https://medium.datadriveninvestor.com/this-is-what-citadels-training-software-taught-me-741c3996a5b5

const std = @import("std");
const svi = @import("svi.zig");
const lob = @import("lob.zig");

pub const Strategy = enum(u8) { naive = 0, desk = 1 };

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
    const n: usize = 40;
    var state: u32 = 7;
    var f: f64 = 100.0;
    var basis: f64 = 0.80;
    var qa: f64 = 0.0;
    var qb: f64 = 0.0;
    var qf: f64 = 0.0;
    var cash: f64 = 0.0;
    var inv: [40]f64 = undefined;
    var beta: [40]f64 = undefined;
    const beta_a: f64 = 1.0;
    const beta_b: f64 = 0.55;
    const qty: f64 = 2.0;
    var t: usize = 0;
    while (t < n) : (t += 1) {
        const a = f;
        const b = f + basis;
        const edge = b - a;
        if (@abs(edge) > 0.20) {
            if (edge > 0.0) {
                cash -= qty * a;
                cash += qty * b;
                qa += qty;
                qb -= qty;
            } else {
                cash += qty * a;
                cash -= qty * b;
                qa -= qty;
                qb += qty;
            }
        }
        var net = beta_a * qa + beta_b * qb;
        if (strategy == .desk) {
            const target = -net;
            const dq = target - qf;
            cash -= dq * f;
            qf = target;
            net = beta_a * qa + beta_b * qb + qf;
        }
        inv[t] = @abs(qa) + @abs(qb);
        beta[t] = net;
        const z = lcgNext(&state);
        f += 1.5 * z;
        const z2 = lcgNext(&state);
        basis = 0.92 * basis + 0.06 + 0.02 * z2;
    }
    const a = f;
    const b = f + basis;
    const pnl = cash + qa * a + qb * b + qf * f;
    return scorePath(pnl, &inv, &beta, 0.01, 2.0, 0.0, peer_pnl);
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
    const n: usize = 50;
    var state: u32 = 21;
    const half: f64 = 0.08;
    const jump: f64 = 0.28;
    var inv_pos: f64 = 0.0;
    var pnl: f64 = 0.0;
    var recent: [50]f64 = [_]f64{0} ** 50;
    var n_recent: usize = 0;
    var inv: [50]f64 = undefined;
    var beta: [50]f64 = [_]f64{0} ** 50;
    var t: usize = 0;
    while (t < n) : (t += 1) {
        const u = lcgNext(&state);
        const informed = (u + 1.0) / 2.0 < 0.40;
        const side_draw = lcgNext(&state);
        const cust_buy = side_draw > 0.0;
        var size: f64 = 1.0;
        if (strategy == .desk) {
            var tox: f64 = 0.0;
            var c: usize = 0;
            const start = if (n_recent > 6) n_recent - 6 else 0;
            var k = start;
            while (k < n_recent) : (k += 1) {
                tox += recent[k];
                c += 1;
            }
            if (c > 0) tox /= @as(f64, @floatFromInt(c));
            if (@abs(inv_pos) >= 4.0 or tox > 0.55) size = 0.0;
        }
        if (size > 0.0) {
            if (cust_buy) inv_pos -= size else inv_pos += size;
            pnl += half * size;
            if (informed) pnl -= jump * size;
            recent[n_recent] = if (informed) 1.0 else 0.0;
            n_recent += 1;
        } else {
            recent[n_recent] = 0.0;
            n_recent += 1;
        }
        inv[t] = inv_pos;
    }
    return scorePath(pnl, &inv, &beta, 0.05, 0.0, 0.0, peer_pnl);
}

fn mmInventory(strategy: Strategy, peer_pnl: f64) CaseScore {
    const ahead: f64 = 4.0;
    const intensity: f64 = 25.0;
    const horizon: f64 = 1.0;
    const lat: f64 = if (strategy == .desk) 0.20 else 0.90;
    const fills = lob.expectedFills(ahead, 6.0, intensity, 0.0, horizon, lat);
    const spread: f64 = 0.05;
    const adverse: f64 = 0.15;
    var pnl = fills * spread - fills * adverse;
    const n: usize = 40;
    var inv_pos: f64 = 0.0;
    var inv: [40]f64 = undefined;
    var beta: [40]f64 = [_]f64{0} ** 40;
    var t: usize = 0;
    while (t < n) : (t += 1) {
        var hit = true;
        if (strategy == .desk and inv_pos >= 5.0) hit = false;
        if (strategy == .desk and inv_pos >= 3.0 and t % 2 == 0) hit = false;
        if (hit) {
            inv_pos += 1.0;
            pnl += 0.04;
            if (t > n / 3) pnl -= 0.09;
        }
        inv[t] = inv_pos;
    }
    return scorePath(pnl, &inv, &beta, 0.02, 0.0, 0.0, peer_pnl);
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

pub fn runCase(name: []const u8, strategy: Strategy, peer_pnl: f64) CaseScore {
    if (std.mem.eql(u8, name, "location_arb")) return location(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "etf_ap_arb")) return etf(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "liability_facilitator")) return facilitator(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "mm_inventory")) return mmInventory(strategy, peer_pnl);
    if (std.mem.eql(u8, name, "vol_surface_mm")) return volSurface(strategy, peer_pnl);
    return .{};
}

pub const CASES = [_][]const u8{
    "location_arb",
    "etf_ap_arb",
    "liability_facilitator",
    "mm_inventory",
    "vol_surface_mm",
};

test "location arb desk hedges residual beta and scores higher" {
    const naive = runCase("location_arb", .naive, 0.0);
    const desk = runCase("location_arb", .desk, naive.absolute_pnl);
    try std.testing.expect(desk.mean_abs_beta < naive.mean_abs_beta * 0.05 + 1e-9);
    try std.testing.expect(desk.beta_penalty < naive.beta_penalty);
    try std.testing.expect(desk.risk_adjusted > naive.risk_adjusted);
    try std.testing.expect(desk.relative_score == desk.absolute_pnl - naive.absolute_pnl);
}

test "etf desk sizes the near-risk-free create before it decays" {
    const naive = runCase("etf_ap_arb", .naive, 0.0);
    const desk = runCase("etf_ap_arb", .desk, naive.absolute_pnl);
    try std.testing.expect(desk.absolute_pnl > naive.absolute_pnl);
    try std.testing.expect(desk.risk_adjusted > naive.risk_adjusted);
}

test "facilitator desk avoids informed inventory" {
    const naive = runCase("liability_facilitator", .naive, 0.0);
    const desk = runCase("liability_facilitator", .desk, naive.absolute_pnl);
    try std.testing.expect(desk.mean_abs_inventory < naive.mean_abs_inventory);
    try std.testing.expect(desk.risk_adjusted > naive.risk_adjusted);
}

test "mm desk cancel and skew beat the one-sided wave" {
    const fast = lob.expectedFills(4.0, 6.0, 25.0, 0.0, 1.0, 0.20);
    const slow = lob.expectedFills(4.0, 6.0, 25.0, 0.0, 1.0, 0.90);
    try std.testing.expect(fast < slow);
    const naive = runCase("mm_inventory", .naive, 0.0);
    const desk = runCase("mm_inventory", .desk, naive.absolute_pnl);
    try std.testing.expect(desk.mean_abs_inventory < naive.mean_abs_inventory);
    try std.testing.expect(desk.risk_adjusted > naive.risk_adjusted);
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

test "lcg is deterministic" {
    var a: u32 = 7;
    var b: u32 = 7;
    try std.testing.expectApproxEqAbs(lcgNext(&a), lcgNext(&b), 0.0);
    try std.testing.expect(a == b);
}
