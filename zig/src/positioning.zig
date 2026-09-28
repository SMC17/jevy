//! Precomputed positioning → reservation shift, spread, size, hedge band.
//! No network. Mirrors ``jev_omm/positioning/adjust.py`` and the GEX aggregator
//! in ``jev_omm/positioning/gex.py``.
//!
//! Defaults (flags off, zeros) are the identity. Classical quoter math is untouched.

const std = @import("std");
const bs = @import("black_scholes.zig");
const flow = @import("flow_signals.zig");
const state_os = @import("state_os.zig");

fn clamp(x: f64, lo: f64, hi: f64) f64 {
    return @min(@max(x, lo), hi);
}

pub const GexAdjust = struct {
    reservation_shift: f64 = 0.0,
    spread_mult: f64 = 1.0,
    size_mult: f64 = 1.0,
    hedge_band_mult: f64 = 1.0,
    hedge_urgency: f64 = 0.0,
};

pub fn gexAdjust(enabled: bool, gex_norm: f64, pin_gap: f64, spot_return: f64, mid: f64) GexAdjust {
    if (!enabled) return .{};
    const g = clamp(gex_norm, -1.0, 1.0);
    var spread: f64 = 1.0;
    var size: f64 = 1.0;
    var band: f64 = 1.0;
    var shift: f64 = 0.0;
    if (g >= 0.0) {
        spread = 1.0 - 0.20 * g;
        size = 1.0 + 0.30 * g;
        band = 1.0 + 0.50 * g;
        shift = 0.15 * g * pin_gap * mid;
    } else {
        const a = -g;
        spread = 1.0 + 0.70 * a;
        size = 1.0 / (1.0 + 0.80 * a);
        band = 1.0 / (1.0 + 1.0 * a);
        shift = 0.10 * a * spot_return * mid;
    }
    if (@abs(g) < 0.25) {
        const near = (0.25 - @abs(g)) / 0.25;
        spread *= 1.0 + 0.35 * near;
    }
    const urgency: f64 = if (g < -0.35) 0.40 * (-g) else 0.0;
    return .{
        .reservation_shift = shift,
        .spread_mult = spread,
        .size_mult = size,
        .hedge_band_mult = band,
        .hedge_urgency = urgency,
    };
}

pub fn cotFade(enabled: bool, cot_z: f64, mid: f64) struct { shift: f64, size_mult: f64 } {
    if (!enabled) return .{ .shift = 0.0, .size_mult = 1.0 };
    const fade = clamp(cot_z / 4.0, -1.0, 1.0);
    return .{
        .shift = -fade * 0.0015 * mid,
        .size_mult = 1.0 / (1.0 + 0.50 * @abs(fade)),
    };
}

pub fn basisSpreadMult(basis_z: f64) f64 {
    return 1.0 + 0.20 * clamp(@abs(basis_z) / 2.0, 0.0, 1.0);
}

pub fn rrSpreadMult(rr_stress_value: f64) f64 {
    return 1.0 + 0.15 * clamp(rr_stress_value, 0.0, 1.0);
}

pub fn pcrSpreadMult(put_call_oi: f64) f64 {
    if (put_call_oi <= 0.0) return 1.0;
    return 1.0 + 0.10 * clamp(put_call_oi - 1.0, 0.0, 1.0);
}

pub const PositioningAdjust = struct {
    reservation_shift: f64 = 0.0,
    spread_mult: f64 = 1.0,
    size_mult: f64 = 1.0,
    hedge_band_mult: f64 = 1.0,
    hedge_urgency: f64 = 0.0,
};

pub const FeatureInputs = struct {
    vpin: f64 = 0.0,
    ofi_norm: f64 = 0.0,
    aggr_imbalance: f64 = 0.0,
    off_exchange_share: f64 = 0.0,
    spoof: f64 = 0.0,
    gex_enabled: bool = false,
    gex_norm: f64 = 0.0,
    pin_gap: f64 = 0.0,
    spot_return: f64 = 0.0,
    mid: f64 = 0.0,
    cot_enabled: bool = false,
    cot_z: f64 = 0.0,
    basis_z: f64 = 0.0,
    rr_stress_value: f64 = 0.0,
    put_call_oi: f64 = 0.0,
    state_enabled: bool = false,
    instability: f64 = 0.0,
    constraint_active: bool = false,
    parent_remaining: f64 = 0.0,
    f_signed: f64 = 0.0,
    l_exec: f64 = 0.0,
};

pub fn applyFeatures(inp: FeatureInputs) PositioningAdjust {
    const prior = flow.flowPrior(inp.vpin, inp.ofi_norm, inp.aggr_imbalance, inp.off_exchange_share, inp.spoof);
    const g = gexAdjust(inp.gex_enabled, inp.gex_norm, inp.pin_gap, inp.spot_return, inp.mid);
    const c = cotFade(inp.cot_enabled, inp.cot_z, inp.mid);
    const gate = state_os.stateGate(
        inp.state_enabled,
        inp.instability,
        inp.constraint_active,
        inp.parent_remaining,
        inp.f_signed,
        inp.l_exec,
        inp.mid,
    );
    var spread = prior.spread_mult * g.spread_mult * basisSpreadMult(inp.basis_z) * rrSpreadMult(inp.rr_stress_value) * pcrSpreadMult(inp.put_call_oi) * gate.spread_mult;
    var size = prior.size_mult * g.size_mult * c.size_mult * gate.size_mult;
    spread = clamp(spread, 0.70, 3.50);
    size = clamp(size, 0.20, 1.80);
    if (gate.pull) size = 0.0;
    var urgency = g.hedge_urgency;
    if (prior.toxicity > 0.60) urgency = @max(urgency, 0.30);
    urgency = @max(urgency, gate.hedge_urgency);
    return .{
        .reservation_shift = g.reservation_shift + c.shift + gate.reservation_shift,
        .spread_mult = spread,
        .size_mult = size,
        .hedge_band_mult = g.hedge_band_mult,
        .hedge_urgency = urgency,
    };
}

pub const Posture = enum { short_premium, dashboard_flip };

pub const OiLeg = struct {
    strike: f64,
    call_oi: f64,
    put_oi: f64,
    iv: f64,
    t: f64,
};

fn signs(posture: Posture) struct { call: f64, put: f64 } {
    return switch (posture) {
        .short_premium => .{ .call = -1.0, .put = -1.0 },
        .dashboard_flip => .{ .call = 1.0, .put = -1.0 },
    };
}

pub fn dollarGex1pct(spot: f64, legs: []const OiLeg, posture: Posture, rate: f64, div_yield: f64, multiplier: f64) f64 {
    const s = signs(posture);
    var total: f64 = 0.0;
    for (legs) |leg| {
        const gamma = bs.greeks(spot, leg.strike, leg.t, rate, div_yield, leg.iv, true).gamma;
        const scale = gamma * multiplier * spot * spot * 0.01;
        total += s.call * leg.call_oi * scale;
        total += s.put * leg.put_oi * scale;
    }
    return total;
}

pub fn zeroGammaLevel(
    legs: []const OiLeg,
    posture: Posture,
    spot_min: f64,
    spot_max: f64,
    n: usize,
    rate: f64,
    div_yield: f64,
    multiplier: f64,
) ?f64 {
    if (n < 2 or spot_max <= spot_min) return null;
    var i: usize = 0;
    var prev_s = spot_min;
    var prev_v = dollarGex1pct(spot_min, legs, posture, rate, div_yield, multiplier);
    while (i + 1 < n) : (i += 1) {
        const spot = spot_min + (spot_max - spot_min) * @as(f64, @floatFromInt(i + 1)) / @as(f64, @floatFromInt(n - 1));
        const v = dollarGex1pct(spot, legs, posture, rate, div_yield, multiplier);
        if (prev_v == 0.0) return prev_s;
        if (prev_v * v < 0.0) {
            const w = @abs(prev_v) / (@abs(prev_v) + @abs(v));
            return prev_s + w * (spot - prev_s);
        }
        prev_s = spot;
        prev_v = v;
    }
    return null;
}

pub fn maxPain(legs: []const OiLeg) ?f64 {
    if (legs.len == 0) return null;
    var best_k = legs[0].strike;
    var best_pay = std.math.inf(f64);
    for (legs) |candidate| {
        var pay: f64 = 0.0;
        for (legs) |leg| {
            pay += leg.call_oi * @max(candidate.strike - leg.strike, 0.0);
            pay += leg.put_oi * @max(leg.strike - candidate.strike, 0.0);
        }
        if (pay < best_pay) {
            best_pay = pay;
            best_k = candidate.strike;
        }
    }
    return best_k;
}

pub fn pinLevel(gex_norm: f64, max_pain_strike: ?f64, flip: ?f64, wall: ?f64) ?f64 {
    if (gex_norm <= 0.0) return null;
    if (flip != null and max_pain_strike != null) return 0.5 * flip.? + 0.5 * max_pain_strike.?;
    if (flip != null) return flip;
    if (max_pain_strike != null) return max_pain_strike;
    return wall;
}

pub fn charmVannaHedge(
    spot: f64,
    legs: []const OiLeg,
    posture: Posture,
    rate: f64,
    div_yield: f64,
    dt: f64,
    d_sigma: f64,
    multiplier: f64,
) struct { charm: f64, vanna: f64 } {
    const s = signs(posture);
    var charm_h: f64 = 0.0;
    var vanna_h: f64 = 0.0;
    for (legs) |leg| {
        const sides = [_]struct { call: bool, sign: f64, oi: f64 }{
            .{ .call = true, .sign = s.call, .oi = leg.call_oi },
            .{ .call = false, .sign = s.put, .oi = leg.put_oi },
        };
        for (sides) |side| {
            if (side.oi == 0.0) continue;
            const qty = side.sign * side.oi * multiplier;
            const ch = bs.charmTau(spot, leg.strike, leg.t, rate, div_yield, leg.iv, side.call);
            const d_delta = qty * (-ch) * dt;
            charm_h += -d_delta;
            const vanna = bs.greeks(spot, leg.strike, leg.t, rate, div_yield, leg.iv, side.call).vanna;
            vanna_h += -qty * vanna * d_sigma;
        }
    }
    return .{ .charm = charm_h, .vanna = vanna_h };
}

/// Hedge-band scale and an additive charm/vanna overlay. Identity at mult=1, overlays 0.
pub fn scaledBand(delta_band: f64, band_mult: f64) f64 {
    return delta_band * band_mult;
}

pub fn overlayHedgeQty(base_qty: f64, charm_hedge: f64, vanna_hedge: f64) f64 {
    return base_qty + charm_hedge + vanna_hedge;
}

test "positioning defaults are the identity" {
    const adj = applyFeatures(.{});
    try std.testing.expectApproxEqAbs(adj.reservation_shift, 0.0, 1e-15);
    try std.testing.expectApproxEqAbs(adj.spread_mult, 1.0, 1e-15);
    try std.testing.expectApproxEqAbs(adj.size_mult, 1.0, 1e-15);
    try std.testing.expectApproxEqAbs(adj.hedge_band_mult, 1.0, 1e-15);
    try std.testing.expectApproxEqAbs(adj.hedge_urgency, 0.0, 1e-15);
}

test "long gamma leans to the pin and widens the hedge band" {
    const g = gexAdjust(true, 0.5, 0.01, 0.0, 100.0);
    try std.testing.expectApproxEqAbs(g.reservation_shift, 0.075, 1e-12);
    try std.testing.expectApproxEqAbs(g.spread_mult, 0.9, 1e-12);
    try std.testing.expectApproxEqAbs(g.size_mult, 1.15, 1e-12);
    try std.testing.expectApproxEqAbs(g.hedge_band_mult, 1.25, 1e-12);
    try std.testing.expectApproxEqAbs(g.hedge_urgency, 0.0, 1e-15);
    const off = gexAdjust(false, 0.5, 0.01, 0.0, 100.0);
    try std.testing.expectApproxEqAbs(off.spread_mult, 1.0, 1e-15);
    try std.testing.expectApproxEqAbs(off.hedge_band_mult, 1.0, 1e-15);
}

test "short gamma widens, cuts size, and tightens the hedge band" {
    const g = gexAdjust(true, -0.5, 0.0, 0.01, 100.0);
    try std.testing.expectApproxEqAbs(g.spread_mult, 1.0 + 0.70 * 0.5, 1e-12);
    try std.testing.expectApproxEqAbs(g.size_mult, 1.0 / (1.0 + 0.80 * 0.5), 1e-12);
    try std.testing.expectApproxEqAbs(g.hedge_band_mult, 1.0 / 1.5, 1e-12);
    try std.testing.expect(g.hedge_urgency > 0.0);
    try std.testing.expect(g.reservation_shift > 0.0);
}

test "cot fade leans against a crowded long and is idle at z=0" {
    const hot = cotFade(true, 4.0, 100.0);
    try std.testing.expectApproxEqAbs(hot.shift, -0.15, 1e-12);
    try std.testing.expectApproxEqAbs(hot.size_mult, 1.0 / 1.5, 1e-12);
    const flat = cotFade(true, 0.0, 100.0);
    try std.testing.expectApproxEqAbs(flat.shift, 0.0, 1e-15);
    try std.testing.expectApproxEqAbs(flat.size_mult, 1.0, 1e-15);
    const off = cotFade(false, 4.0, 100.0);
    try std.testing.expectApproxEqAbs(off.size_mult, 1.0, 1e-15);
}

test "short premium gamma stays negative; dashboard book has a flip" {
    const legs = [_]OiLeg{
        .{ .strike = 90.0, .call_oi = 10.0, .put_oi = 400.0, .iv = 0.25, .t = 30.0 / 365.0 },
        .{ .strike = 110.0, .call_oi = 400.0, .put_oi = 10.0, .iv = 0.25, .t = 30.0 / 365.0 },
    };
    const short = dollarGex1pct(100.0, &legs, .short_premium, 0.01, 0.0, 100.0);
    try std.testing.expect(short < 0.0);
    try std.testing.expect(zeroGammaLevel(&legs, .short_premium, 80.0, 120.0, 17, 0.01, 0.0, 100.0) == null);
    const flip = zeroGammaLevel(&legs, .dashboard_flip, 80.0, 120.0, 21, 0.01, 0.0, 100.0);
    try std.testing.expect(flip != null);
    try std.testing.expect(flip.? > 90.0 and flip.? < 110.0);
    try std.testing.expect(pinLevel(-0.4, 100.0, flip, 110.0) == null);
    try std.testing.expect(pinLevel(0.4, 100.0, flip, 110.0) != null);
}

test "charm and vanna overlays are zero when the shocks are zero" {
    const legs = [_]OiLeg{
        .{ .strike = 100.0, .call_oi = 10.0, .put_oi = 10.0, .iv = 0.2, .t = 0.25 },
    };
    const z = charmVannaHedge(100.0, &legs, .short_premium, 0.01, 0.0, 0.0, 0.0, 100.0);
    try std.testing.expectApproxEqAbs(z.charm, 0.0, 1e-12);
    try std.testing.expectApproxEqAbs(z.vanna, 0.0, 1e-12);
    const live = charmVannaHedge(100.0, &legs, .short_premium, 0.01, 0.0, 1.0 / 252.0, -0.01, 100.0);
    try std.testing.expect(@abs(live.charm) + @abs(live.vanna) > 0.0);
    try std.testing.expectApproxEqAbs(overlayHedgeQty(-2.0, 0.0, 0.0), -2.0, 1e-15);
    try std.testing.expectApproxEqAbs(scaledBand(5.0, 1.0), 5.0, 1e-15);
}
