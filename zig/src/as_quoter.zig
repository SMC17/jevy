//! Avellaneda–Stoikov reservation + optimal spread (Python semantics).
//! Port of jev_omm/quoter/avellaneda_stoikov.py
//!
//! When `cfg.mode == .gueant_asymptotic`, delegates to Guéant–Lehalle–
//! Fernandez-Tapia stationary approximations (`gueant.zig`, arXiv 1105.3115).
//! `gueant_ode` solves the finite-horizon / spectral system in `gueant_ode.zig`.

const std = @import("std");
const types = @import("types.zig");
const gueant = @import("gueant.zig");
const gueant_ode = @import("gueant_ode.zig");
const QuoterConfig = types.QuoterConfig;
const Greeks = types.Greeks;
const Quote = types.Quote;

fn horizon(cfg: *const QuoterConfig, t_remaining: ?f64) f64 {
    if (t_remaining) |t| return @max(t, 0.0);
    return cfg.t_horizon;
}

pub fn reservationPrice(
    mid: f64,
    inventory: i32,
    cfg: *const QuoterConfig,
    t_remaining: ?f64,
    greeks_opt: ?*const Greeks,
) f64 {
    if (cfg.mode == .gueant_ode) {
        return gueant_ode.reservationPrice(mid, inventory, cfg, greeks_opt);
    }
    if (cfg.mode == .gueant_asymptotic) {
        return gueant.reservationPrice(mid, inventory, cfg, greeks_opt);
    }
    const t = horizon(cfg, t_remaining);
    var r = mid - @as(f64, @floatFromInt(inventory)) * cfg.gamma * (cfg.sigma * cfg.sigma) * t;
    if (greeks_opt) |g| {
        if (inventory != 0) {
            const inv = @as(f64, @floatFromInt(inventory));
            r -= inv * cfg.gamma_penalty * @abs(g.gamma);
            r -= inv * cfg.vega_penalty * @abs(g.vega) * 0.01;
        }
    }
    return r;
}

pub fn optimalHalfSpread(cfg: *const QuoterConfig, t_remaining: ?f64) f64 {
    if (cfg.mode == .gueant_ode) {
        return gueant_ode.optimalHalfSpread(cfg, 0);
    }
    if (cfg.mode == .gueant_asymptotic) {
        return gueant.optimalHalfSpread(cfg);
    }
    const t = horizon(cfg, t_remaining);
    const risk_term = 0.5 * cfg.gamma * (cfg.sigma * cfg.sigma) * t;
    const intensity_term: f64 = if (cfg.gamma <= 0.0 or cfg.kappa <= 0.0)
        cfg.min_half_spread
    else
        (1.0 / cfg.gamma) * @log(1.0 + cfg.gamma / cfg.kappa);
    const half = risk_term + intensity_term;
    return std.math.clamp(half, cfg.min_half_spread, cfg.max_half_spread);
}

pub fn makeQuote(
    mid_in: f64,
    inventory: i32,
    cfg: *const QuoterConfig,
    t_remaining: ?f64,
    greeks_opt: ?*const Greeks,
    spread_mult: f64,
    size_mult: f64,
) Quote {
    if (cfg.mode == .gueant_ode) {
        return gueant_ode.makeQuote(mid_in, inventory, cfg, greeks_opt, spread_mult, size_mult);
    }
    if (cfg.mode == .gueant_asymptotic) {
        return gueant.makeQuote(mid_in, inventory, cfg, greeks_opt, spread_mult, size_mult);
    }
    var mid = mid_in;
    if (mid <= 0.0) mid = @max(mid, 0.01);

    const r = reservationPrice(mid, inventory, cfg, t_remaining, greeks_opt);
    var half = optimalHalfSpread(cfg, t_remaining) * @max(spread_mult, 0.25);
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

/// Apply an extra reservation tilt from shared portfolio delta (desk-shaped).
/// Positive portfolio_delta (long underlying exposure) lowers reservation → attract sells.
pub fn applyPortfolioDeltaTilt(q: Quote, portfolio_delta: f64, cfg: *const QuoterConfig) Quote {
    const tilt = portfolio_delta * cfg.portfolio_delta_penalty;
    var out = q;
    out.reservation -= tilt;
    out.bid = @max(0.01, out.reservation - out.half_spread);
    out.ask = @max(out.bid + 0.01, out.reservation + out.half_spread);
    return out;
}

test "reservation decreases with long inventory" {
    const cfg = QuoterConfig{
        .gamma = 0.2,
        .sigma = 0.5,
        .t_horizon = 1.0 / 252.0,
    };
    const mid: f64 = 5.0;
    const r_short = reservationPrice(mid, -5, &cfg, null, null);
    const r_flat = reservationPrice(mid, 0, &cfg, null, null);
    const r_long = reservationPrice(mid, 5, &cfg, null, null);
    try std.testing.expect(r_long < r_flat and r_flat < r_short);
}

test "quotes skew monotone in inventory" {
    const cfg = QuoterConfig{
        .gamma = 0.15,
        .kappa = 1.5,
        .sigma = 0.4,
        .quote_size = 1,
    };
    const mid: f64 = 4.0;
    const q_long = makeQuote(mid, 10, &cfg, null, null, 1.0, 1.0);
    const q_short = makeQuote(mid, -10, &cfg, null, null, 1.0, 1.0);
    try std.testing.expect(q_long.reservation < q_short.reservation);
    try std.testing.expect(q_long.bid < q_short.bid);
    try std.testing.expect(q_long.ask < q_short.ask);
    try std.testing.expect(q_long.ask > q_long.bid);
}

test "greek penalty lowers reservation when long" {
    const cfg = QuoterConfig{
        .gamma = 0.1,
        .sigma = 0.5,
        .t_horizon = 1.0 / 252.0,
        .gamma_penalty = 1.0,
        .vega_penalty = 1.0,
    };
    const g = Greeks{ .delta = 0.5, .gamma = 0.02, .vega = 10.0, .theta = -1.0 };
    const r0 = reservationPrice(5.0, 5, &cfg, null, null);
    const r1 = reservationPrice(5.0, 5, &cfg, null, &g);
    try std.testing.expect(r1 < r0);
}

test "half spread clamped" {
    const cfg = QuoterConfig{
        .gamma = 0.001,
        .kappa = 100.0,
        .sigma = 0.01,
        .t_horizon = 1e-9,
        .min_half_spread = 0.05,
        .max_half_spread = 5.0,
    };
    const h = optimalHalfSpread(&cfg, null);
    try std.testing.expect(@abs(h - 0.05) < 1e-12);
}

test "mode toggle routes to gueant ode" {
    const cfg = QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.4,
        .A = 120.0,
        .t_horizon = 0.5,
        .inventory_cap = 6,
        .ode_steps = 400,
        .mode = .gueant_ode,
        .min_half_spread = 0.01,
        .max_half_spread = 20.0,
        .quote_size = 1,
    };
    const q = makeQuote(4.0, 3, &cfg, null, null, 1.0, 1.0);
    try std.testing.expect(q.ask > q.bid);
    try std.testing.expect(q.reservation < 4.0);
}

test "mode toggle routes to gueant" {
    var cfg_as = QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.5,
        .t_horizon = 1.0 / 252.0,
        .A = 140.0,
        .mode = .as_finite_horizon,
    };
    var cfg_g = cfg_as;
    cfg_g.mode = .gueant_asymptotic;
    const mid: f64 = 5.0;
    const r_as = reservationPrice(mid, 8, &cfg_as, null, null);
    const r_g = reservationPrice(mid, 8, &cfg_g, null, null);
    // Different formulas → generally different reservation (unless params collide)
    // Guéant ξ is typically larger than γσ²T for short T → more skew.
    try std.testing.expect(r_g < mid);
    try std.testing.expect(r_as < mid);
    _ = &cfg_as;
    _ = &cfg_g;
}

test "portfolio delta tilt lowers quotes when long delta" {
    const cfg = QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.4,
        .portfolio_delta_penalty = 0.05,
        .quote_size = 1,
    };
    const q0 = makeQuote(4.0, 0, &cfg, null, null, 1.0, 1.0);
    const q1 = applyPortfolioDeltaTilt(q0, 10.0, &cfg);
    try std.testing.expect(q1.reservation < q0.reservation);
    try std.testing.expect(q1.bid < q0.bid);
}
