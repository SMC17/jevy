//! Latent-state gate. Python builds S_t. This file consumes the scalars.
//!
//! Instability = |F| / L_exec. The quote multipliers are the identity when
//! `enabled` is false. Constants match `jev_omm/state_os/gate.py` and the
//! stand-up formulas used by the training cases.

const std = @import("std");

fn clamp(x: f64, lo: f64, hi: f64) f64 {
    return @min(@max(x, lo), hi);
}

pub const StateGate = struct {
    reservation_shift: f64 = 0.0,
    spread_mult: f64 = 1.0,
    size_mult: f64 = 1.0,
    hedge_urgency: f64 = 0.0,
    pull: bool = false,
    instability: f64 = 0.0,
};

pub fn instability(f_value: f64, l_exec: f64) f64 {
    const forced = @abs(f_value);
    if (l_exec > 0.0) return forced / l_exec;
    if (forced == 0.0) return 0.0;
    return std.math.inf(f64);
}

pub fn stateGate(
    enabled: bool,
    instability_value: f64,
    constraint_active: bool,
    parent_remaining: f64,
    f_signed: f64,
    l_exec: f64,
    mid: f64,
) StateGate {
    if (!enabled) return .{};
    const u_clip: f64 = if (std.math.isInf(instability_value)) 4.0 else @max(instability_value, 0.0);
    const pull = constraint_active or u_clip >= 2.0;
    const mild = @min(u_clip, 2.0);
    var size = 1.0 / (1.0 + 1.10 * mild);
    const parent = clamp(parent_remaining, 0.0, 1.0);
    if (parent > 0.0) size /= 1.0 + 0.75 * parent;
    var urgency: f64 = if (u_clip >= 1.0) 0.45 * mild else 0.0;
    if (constraint_active) urgency = @max(urgency, 0.70);
    var shift: f64 = 0.0;
    if (mid > 0.0 and l_exec > 0.0 and u_clip > 0.0) {
        const pressure = clamp(f_signed / l_exec, -1.0, 1.0);
        shift = 0.0015 * pressure * mid * @min(u_clip, 1.0);
    }
    if (pull) size = 0.0;
    return .{
        .reservation_shift = shift,
        .spread_mult = 1.0 + 0.80 * mild,
        .size_mult = size,
        .hedge_urgency = urgency,
        .pull = pull,
        .instability = u_clip,
    };
}

/// Cheng–Madhavan: AUM * (L^2 - L) * r. https://ssrn.com/abstract=1539120
pub fn letfRebalance(aum: f64, leverage: f64, day_return: f64) f64 {
    return aum * (leverage * leverage - leverage) * day_return;
}

/// 200 bp trigger, 175 bp destination. Inside the band the trade is 0.
pub fn tdfTrade(weight: f64, target: f64, aum: f64) f64 {
    const trigger = 0.0200;
    const destination = 0.0175;
    const gap = weight - target;
    if (@abs(gap) <= trigger) return 0.0;
    const direction: f64 = if (gap > 0.0) 1.0 else -1.0;
    const dest = target + direction * destination;
    return (dest - weight) * aum;
}

pub fn gen3Coverage(iv: f64, iv_ref: f64) f64 {
    const raw = 0.50 + 2.0 * (iv - iv_ref);
    return clamp(raw, 0.0, 1.0);
}

pub fn signsDisagree(structural: f64, flow: f64) bool {
    if (structural == 0.0 or flow == 0.0) return false;
    return (structural > 0.0) != (flow > 0.0);
}

pub fn sqrtImpact(quantity: f64, sigma: f64, volume: f64, y: f64) f64 {
    if (volume <= 0.0 or quantity == 0.0 or sigma < 0.0 or y == 0.0) return 0.0;
    const sign: f64 = if (quantity > 0.0) 1.0 else -1.0;
    return y * sigma * @sqrt(@abs(quantity) / volume) * sign;
}

pub fn netLiquidity(fed_assets: f64, tga: f64, rrp: f64) f64 {
    return fed_assets - tga - rrp;
}

pub fn auctionImbalance(buy_qty: f64, sell_qty: f64) f64 {
    const denom = buy_qty + sell_qty;
    if (denom <= 0.0) return 0.0;
    return (buy_qty - sell_qty) / denom;
}

pub fn pensionLinear(aum: f64, weight: f64, equity_return: f64, bond_return: f64) f64 {
    return aum * weight * (1.0 - weight) * (bond_return - equity_return);
}

test "state gate is the identity when off and when the ratio is zero" {
    const off = stateGate(false, 5.0, true, 0.8, -10.0, 1.0, 100.0);
    try std.testing.expectApproxEqAbs(off.spread_mult, 1.0, 1e-15);
    try std.testing.expectApproxEqAbs(off.size_mult, 1.0, 1e-15);
    try std.testing.expectApproxEqAbs(off.reservation_shift, 0.0, 1e-15);
    try std.testing.expect(!off.pull);
    const calm = stateGate(true, 0.0, false, 0.0, 0.0, 10.0, 100.0);
    try std.testing.expectApproxEqAbs(calm.spread_mult, 1.0, 1e-15);
    try std.testing.expectApproxEqAbs(calm.size_mult, 1.0, 1e-15);
    try std.testing.expect(!calm.pull);
}

test "a ratio of 4 pulls and a live parent only cuts size" {
    try std.testing.expectApproxEqAbs(instability(80.0, 20.0), 4.0, 1e-15);
    const hot = stateGate(true, 4.0, false, 0.0, -80.0, 20.0, 100.0);
    try std.testing.expect(hot.pull);
    try std.testing.expectApproxEqAbs(hot.size_mult, 0.0, 1e-15);
    try std.testing.expectApproxEqAbs(hot.spread_mult, 1.0 + 0.80 * 2.0, 1e-12);
    const parent = stateGate(true, 0.0, false, 0.5, 0.0, 1.0, 100.0);
    try std.testing.expect(!parent.pull);
    try std.testing.expectApproxEqAbs(parent.size_mult, 1.0 / (1.0 + 0.75 * 0.5), 1e-12);
    try std.testing.expect(stateGate(true, 0.0, true, 0.0, 0.0, 1.0, 100.0).pull);
}

test "letf, tdf, coverage, and square-root concavity" {
    try std.testing.expectApproxEqAbs(letfRebalance(100.0, 3.0, 0.02), 12.0, 1e-12);
    try std.testing.expectApproxEqAbs(letfRebalance(100.0, 1.0, 0.02), 0.0, 1e-15);
    try std.testing.expectApproxEqAbs(tdfTrade(0.625, 0.60, 1000.0), -7.5, 1e-12);
    try std.testing.expectApproxEqAbs(tdfTrade(0.61, 0.60, 1000.0), 0.0, 1e-15);
    try std.testing.expectApproxEqAbs(gen3Coverage(0.28, 0.18), 0.70, 1e-12);
    const impact_q = sqrtImpact(100.0, 0.2, 10000.0, 1.0);
    const impact_4q = sqrtImpact(400.0, 0.2, 10000.0, 1.0);
    try std.testing.expectApproxEqAbs(impact_4q, 2.0 * impact_q, 1e-12);
    try std.testing.expect(impact_4q < 4.0 * impact_q);
    try std.testing.expect(signsDisagree(0.6, -0.8));
    try std.testing.expect(!signsDisagree(0.6, 0.0));
    try std.testing.expectApproxEqAbs(netLiquidity(100.0, 20.0, 15.0), 65.0, 1e-12);
    try std.testing.expectApproxEqAbs(auctionImbalance(80.0, 20.0), 0.6, 1e-12);
    try std.testing.expect(pensionLinear(1000.0, 0.6, 0.10, 0.0) < 0.0);
}
