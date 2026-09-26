//! Guéant–Lehalle–Fernandez-Tapia inventory asymptotics (arXiv 1105.3115).
//!
//! Closed-form stationary / asymptotic reservation & spread approximations
//! for the inventory-limited market-making HJB (spectral / Gaussian approx
//! of the principal eigenvector — Prop. 3 / §4 of the paper).
//!
//! Cite: https://arxiv.org/abs/1105.3115
//!       Guéant, Lehalle, Fernandez-Tapia (2013), "Dealing with the Inventory Risk"
//!
//! Formulas (intensity λ(δ)=A e^{-k δ}, risk aversion γ, mid-vol σ):
//!
//!   ξ = √( (σ² γ)/(2 k A) · (1 + γ/k)^{1 + k/γ} )
//!
//!   δ^b*_∞(q) ≈ (1/γ) ln(1+γ/k) + ((2q+1)/2) ξ
//!   δ^a*_∞(q) ≈ (1/γ) ln(1+γ/k) − ((2q−1)/2) ξ
//!   ψ*_∞      ≈ (2/γ) ln(1+γ/k) + ξ          (inventory-independent at this order)
//!
//! Equivalent centered form used here:
//!   r(q)          = mid − q · ξ
//!   half_spread   = (1/γ) ln(1+γ/k) + ξ/2
//!   bid = r − half,  ask = r + half
//!
//! Compare to classic Avellaneda–Stoikov *finite-horizon* formulas in
//! `as_quoter.zig` (toggle via `QuoterConfig.mode`).
//!
//! Notes for options MM research:
//!   - `sigma` is absolute $/√yr vol of the *option mid* (same convention as AS).
//!   - `A` is mid-touch arrival intensity (1/year in our sim clock).
//!   - `kappa` maps to paper's k (intensity decay).
//!   - Inventory hard-cap Q is enforced by risk limits; asymptotics skew continuously.

const std = @import("std");
const types = @import("types.zig");
const QuoterConfig = types.QuoterConfig;
const Greeks = types.Greeks;
const Quote = types.Quote;

/// Inventory-risk scale ξ from the Guéant asymptotic closed form.
pub fn inventoryScale(cfg: *const QuoterConfig) f64 {
    const gamma = cfg.gamma;
    const k = cfg.kappa;
    const A = cfg.A;
    const sigma = cfg.sigma;
    if (gamma <= 0.0 or k <= 0.0 or A <= 0.0 or sigma < 0.0) {
        return 0.0;
    }
    const ratio = 1.0 + gamma / k;
    const exp = 1.0 + k / gamma;
    const inside = (sigma * sigma * gamma) / (2.0 * k * A) * std.math.pow(f64, ratio, exp);
    return @sqrt(@max(inside, 0.0));
}

/// Intensity / adverse-selection base half-spread term (1/γ) ln(1+γ/k).
pub fn intensityHalf(cfg: *const QuoterConfig) f64 {
    if (cfg.gamma <= 0.0 or cfg.kappa <= 0.0) return cfg.min_half_spread;
    return (1.0 / cfg.gamma) * @log(1.0 + cfg.gamma / cfg.kappa);
}

/// Asymptotic reservation price: mid − q ξ (+ optional greek penalties).
pub fn reservationPrice(
    mid: f64,
    inventory: i32,
    cfg: *const QuoterConfig,
    greeks_opt: ?*const Greeks,
) f64 {
    const xi = inventoryScale(cfg);
    var r = mid - @as(f64, @floatFromInt(inventory)) * xi;
    if (greeks_opt) |g| {
        if (inventory != 0) {
            const inv = @as(f64, @floatFromInt(inventory));
            r -= inv * cfg.gamma_penalty * @abs(g.gamma);
            r -= inv * cfg.vega_penalty * @abs(g.vega) * 0.01;
        }
    }
    return r;
}

/// Asymptotic optimal half-spread: (1/γ)ln(1+γ/k) + ξ/2, clamped.
pub fn optimalHalfSpread(cfg: *const QuoterConfig) f64 {
    const half = intensityHalf(cfg) + 0.5 * inventoryScale(cfg);
    return std.math.clamp(half, cfg.min_half_spread, cfg.max_half_spread);
}

/// Bid/ask offsets δ^b*, δ^a* as in the paper's closed-form approx (pre-clamp).
pub fn optimalOffsets(cfg: *const QuoterConfig, inventory: i32) struct { delta_b: f64, delta_a: f64 } {
    const psi = intensityHalf(cfg);
    const xi = inventoryScale(cfg);
    const q = @as(f64, @floatFromInt(inventory));
    const delta_b = psi + 0.5 * (2.0 * q + 1.0) * xi;
    const delta_a = psi - 0.5 * (2.0 * q - 1.0) * xi;
    return .{ .delta_b = delta_b, .delta_a = delta_a };
}

pub fn makeQuote(
    mid_in: f64,
    inventory: i32,
    cfg: *const QuoterConfig,
    greeks_opt: ?*const Greeks,
    spread_mult: f64,
    size_mult: f64,
) Quote {
    var mid = mid_in;
    if (mid <= 0.0) mid = @max(mid, 0.01);

    const r = reservationPrice(mid, inventory, cfg, greeks_opt);
    var half = optimalHalfSpread(cfg) * @max(spread_mult, 0.25);
    half = std.math.clamp(half, cfg.min_half_spread, cfg.max_half_spread);
    const bid = @max(0.01, r - half);
    const ask = @max(bid + 0.01, r + half);
    var size: i32 = @intFromFloat(@round(@as(f64, @floatFromInt(cfg.quote_size)) * @max(size_mult, 0.0)));
    if (size < 1) size = 1;

    return .{
        .bid = bid,
        .ask = ask,
        .bid_size = size,
        .ask_size = size,
        .reservation = r,
        .half_spread = half,
    };
}

test "gueant inventory scale positive" {
    const cfg = QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.5,
        .A = 140.0,
        .mode = .gueant_asymptotic,
    };
    const xi = inventoryScale(&cfg);
    try std.testing.expect(xi > 0.0);
}

test "gueant reservation decreases with long inventory" {
    const cfg = QuoterConfig{
        .gamma = 0.2,
        .kappa = 1.5,
        .sigma = 0.5,
        .A = 100.0,
        .mode = .gueant_asymptotic,
    };
    const mid: f64 = 5.0;
    const r_short = reservationPrice(mid, -5, &cfg, null);
    const r_flat = reservationPrice(mid, 0, &cfg, null);
    const r_long = reservationPrice(mid, 5, &cfg, null);
    try std.testing.expect(r_long < r_flat and r_flat < r_short);
}

test "gueant quotes skew monotone in inventory" {
    const cfg = QuoterConfig{
        .gamma = 0.15,
        .kappa = 1.5,
        .sigma = 0.4,
        .A = 120.0,
        .quote_size = 1,
        .mode = .gueant_asymptotic,
    };
    const mid: f64 = 4.0;
    const q_long = makeQuote(mid, 10, &cfg, null, 1.0, 1.0);
    const q_short = makeQuote(mid, -10, &cfg, null, 1.0, 1.0);
    try std.testing.expect(q_long.reservation < q_short.reservation);
    try std.testing.expect(q_long.bid < q_short.bid);
    try std.testing.expect(q_long.ask < q_short.ask);
    try std.testing.expect(q_long.ask > q_long.bid);
}

test "gueant offsets: long lowers ask offset relative to flat" {
    // Paper: δ^a*(q) decreases in q → sell more aggressively when long.
    const cfg = QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.5,
        .A = 140.0,
    };
    const o_flat = optimalOffsets(&cfg, 0);
    const o_long = optimalOffsets(&cfg, 5);
    const o_short = optimalOffsets(&cfg, -5);
    try std.testing.expect(o_long.delta_a < o_flat.delta_a);
    try std.testing.expect(o_long.delta_b > o_flat.delta_b);
    try std.testing.expect(o_short.delta_a > o_flat.delta_a);
    try std.testing.expect(o_short.delta_b < o_flat.delta_b);
}

test "gueant half spread independent of inventory at this order" {
    const cfg = QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.5,
        .A = 140.0,
        .min_half_spread = 0.01,
        .max_half_spread = 50.0,
    };
    const h0 = optimalHalfSpread(&cfg);
    // half = ψ + ξ/2 does not depend on q (asymmetry is in the offsets)
    try std.testing.expect(h0 > cfg.min_half_spread);
    const xi = inventoryScale(&cfg);
    const psi = intensityHalf(&cfg);
    try std.testing.expect(@abs(h0 - (psi + 0.5 * xi)) < 1e-12);
}

test "gueant limit: larger A shrinks xi (less inventory risk)" {
    var cfg_lo = QuoterConfig{ .gamma = 0.1, .kappa = 1.5, .sigma = 0.5, .A = 50.0 };
    var cfg_hi = QuoterConfig{ .gamma = 0.1, .kappa = 1.5, .sigma = 0.5, .A = 500.0 };
    try std.testing.expect(inventoryScale(&cfg_hi) < inventoryScale(&cfg_lo));
    _ = &cfg_lo;
    _ = &cfg_hi;
}

test "gueant limit: larger sigma grows xi" {
    var cfg_lo = QuoterConfig{ .gamma = 0.1, .kappa = 1.5, .sigma = 0.2, .A = 140.0 };
    var cfg_hi = QuoterConfig{ .gamma = 0.1, .kappa = 1.5, .sigma = 0.8, .A = 140.0 };
    try std.testing.expect(inventoryScale(&cfg_hi) > inventoryScale(&cfg_lo));
    _ = &cfg_lo;
    _ = &cfg_hi;
}

test "gueant greek penalty lowers reservation when long" {
    const cfg = QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.5,
        .A = 140.0,
        .gamma_penalty = 1.0,
        .vega_penalty = 1.0,
    };
    const g = Greeks{ .delta = 0.5, .gamma = 0.02, .vega = 10.0, .theta = -1.0 };
    const r0 = reservationPrice(5.0, 5, &cfg, null);
    const r1 = reservationPrice(5.0, 5, &cfg, &g);
    try std.testing.expect(r1 < r0);
}
