//! European Black–Scholes–Merton price + analytic Greeks.
//! Port of jev_omm/pricing/black_scholes.py

const std = @import("std");
const types = @import("types.zig");
const Greeks = types.Greeks;

const sqrt_2pi: f64 = 2.5066282746310002;
const sqrt_2: f64 = 1.4142135623730951;

/// Abramowitz & Stegun 7.1.26 erf approximation (max error ~1.5e-7).
fn erfApprox(x: f64) f64 {
    const sign: f64 = if (x < 0.0) -1.0 else 1.0;
    const ax = @abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * ax);
    const poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    const y = 1.0 - poly * @exp(-ax * ax);
    return sign * y;
}

inline fn normPdf(x: f64) f64 {
    return @exp(-0.5 * x * x) / sqrt_2pi;
}

inline fn normCdf(x: f64) f64 {
    return 0.5 * (1.0 + erfApprox(x / sqrt_2));
}

fn d1d2(spot: f64, strike: f64, t: f64, rate: f64, div_yield: f64, iv: f64) struct { d1: f64, d2: f64 } {
    if (t <= 0.0 or iv <= 0.0 or spot <= 0.0 or strike <= 0.0) {
        return .{ .d1 = 0.0, .d2 = 0.0 };
    }
    const vol_sqrt_t = iv * @sqrt(t);
    const forward_log = @log(spot / strike) + (rate - div_yield + 0.5 * iv * iv) * t;
    const d1 = forward_log / vol_sqrt_t;
    return .{ .d1 = d1, .d2 = d1 - vol_sqrt_t };
}

pub fn price(spot: f64, strike: f64, t: f64, rate: f64, div_yield: f64, iv: f64, is_call: bool) f64 {
    if (t <= 0.0) {
        if (is_call) return @max(spot - strike, 0.0);
        return @max(strike - spot, 0.0);
    }
    if (iv <= 0.0) {
        const forward = spot * @exp((rate - div_yield) * t);
        const df = @exp(-rate * t);
        if (is_call) return df * @max(forward - strike, 0.0);
        return df * @max(strike - forward, 0.0);
    }
    const d = d1d2(spot, strike, t, rate, div_yield, iv);
    const df = @exp(-rate * t);
    const dq = @exp(-div_yield * t);
    if (is_call) {
        return spot * dq * normCdf(d.d1) - strike * df * normCdf(d.d2);
    }
    return strike * df * normCdf(-d.d2) - spot * dq * normCdf(-d.d1);
}

pub fn greeks(spot: f64, strike: f64, t: f64, rate: f64, div_yield: f64, iv: f64, is_call: bool) Greeks {
    if (t <= 0.0 or spot <= 0.0) {
        const delta: f64 = if (is_call) blk: {
            if (spot > strike) break :blk 1.0;
            if (spot == strike) break :blk 0.5;
            break :blk 0.0;
        } else blk: {
            if (spot < strike) break :blk -1.0;
            if (spot == strike) break :blk -0.5;
            break :blk 0.0;
        };
        return .{ .delta = delta, .gamma = 0.0, .vega = 0.0, .theta = 0.0 };
    }
    if (iv <= 0.0) {
        const forward = spot * @exp((rate - div_yield) * t);
        const dq = @exp(-div_yield * t);
        const delta: f64 = if (is_call) blk: {
            break :blk if (forward > strike) dq else 0.0;
        } else blk: {
            break :blk if (forward < strike) -dq else 0.0;
        };
        return .{ .delta = delta, .gamma = 0.0, .vega = 0.0, .theta = 0.0 };
    }

    const d = d1d2(spot, strike, t, rate, div_yield, iv);
    const dq = @exp(-div_yield * t);
    const df = @exp(-rate * t);
    const pdf_d1 = normPdf(d.d1);
    const sqrt_t = @sqrt(t);
    const gamma = dq * pdf_d1 / (spot * iv * sqrt_t);
    const vega = spot * dq * pdf_d1 * sqrt_t;
    // vanna = ∂Δ/∂σ = −e^{−qT} n(d1) d2 / σ
    // volga = ∂ν/∂σ = ν · d1 · d2 / σ
    // Identical for calls and puts (vega itself does not depend on the right).
    const vanna = -dq * pdf_d1 * d.d2 / iv;
    const volga = vega * d.d1 * d.d2 / iv;

    if (is_call) {
        const delta = dq * normCdf(d.d1);
        const theta = -spot * dq * pdf_d1 * iv / (2.0 * sqrt_t) - rate * strike * df * normCdf(d.d2) + div_yield * spot * dq * normCdf(d.d1);
        return .{ .delta = delta, .gamma = gamma, .vega = vega, .theta = theta, .vanna = vanna, .volga = volga };
    } else {
        const delta = -dq * normCdf(-d.d1);
        const theta = -spot * dq * pdf_d1 * iv / (2.0 * sqrt_t) + rate * strike * df * normCdf(-d.d2) - div_yield * spot * dq * normCdf(-d.d1);
        return .{ .delta = delta, .gamma = gamma, .vega = vega, .theta = theta, .vanna = vanna, .volga = volga };
    }
}

pub fn priceAndGreeks(spot: f64, strike: f64, t: f64, rate: f64, div_yield: f64, iv: f64, is_call: bool) struct { price: f64, greeks: Greeks } {
    return .{
        .price = price(spot, strike, t, rate, div_yield, iv, is_call),
        .greeks = greeks(spot, strike, t, rate, div_yield, iv, is_call),
    };
}

test "call put parity atm" {
    const s: f64 = 100.0;
    const k: f64 = 100.0;
    const t: f64 = 0.25;
    const r: f64 = 0.05;
    const q: f64 = 0.0;
    const iv: f64 = 0.2;
    const c = price(s, k, t, r, q, iv, true);
    const p = price(s, k, t, r, q, iv, false);
    const lhs = c - p;
    const rhs = s * @exp(-q * t) - k * @exp(-r * t);
    try std.testing.expect(@abs(lhs - rhs) < 1e-6);
}

test "call delta between 0 and 1" {
    const g = greeks(100.0, 100.0, 0.25, 0.05, 0.0, 0.2, true);
    try std.testing.expect(g.delta > 0.0 and g.delta < 1.0);
    try std.testing.expect(g.gamma > 0.0);
    try std.testing.expect(g.vega > 0.0);
}

test "put delta between -1 and 0" {
    const g = greeks(100.0, 100.0, 0.25, 0.05, 0.0, 0.2, false);
    try std.testing.expect(g.delta > -1.0 and g.delta < 0.0);
}

test "higher spot higher call price" {
    const p1 = price(100.0, 100.0, 0.25, 0.05, 0.0, 0.2, true);
    const p2 = price(105.0, 100.0, 0.25, 0.05, 0.0, 0.2, true);
    try std.testing.expect(p2 > p1);
}

test "expiry intrinsic" {
    try std.testing.expect(@abs(price(110.0, 100.0, 0.0, 0.05, 0.0, 0.2, true) - 10.0) < 1e-12);
    try std.testing.expect(@abs(price(90.0, 100.0, 0.0, 0.05, 0.0, 0.2, false) - 10.0) < 1e-12);
}

test "vanna matches delta finite difference" {
    const s: f64 = 100.0;
    const k: f64 = 100.0;
    const t: f64 = 0.5;
    const r: f64 = 0.03;
    const q: f64 = 0.01;
    const iv: f64 = 0.25;
    const g = greeks(s, k, t, r, q, iv, true);
    const eps: f64 = 1e-4;
    const d_up = greeks(s, k, t, r, q, iv + eps, true).delta;
    const d_dn = greeks(s, k, t, r, q, iv - eps, true).delta;
    const fd = (d_up - d_dn) / (2.0 * eps);
    try std.testing.expect(@abs(g.vanna - fd) < 1e-4);
    const g_put = greeks(s, k, t, r, q, iv, false);
    try std.testing.expect(@abs(g.vanna - g_put.vanna) < 1e-12);
}

test "volga matches vega finite difference" {
    const s: f64 = 105.0;
    const k: f64 = 100.0;
    const t: f64 = 0.4;
    const r: f64 = 0.02;
    const q: f64 = 0.0;
    const iv: f64 = 0.3;
    const g = greeks(s, k, t, r, q, iv, false);
    const eps: f64 = 1e-4;
    const v_up = greeks(s, k, t, r, q, iv + eps, false).vega;
    const v_dn = greeks(s, k, t, r, q, iv - eps, false).vega;
    const fd = (v_up - v_dn) / (2.0 * eps);
    try std.testing.expect(@abs(g.volga - fd) < 1e-3);
}
