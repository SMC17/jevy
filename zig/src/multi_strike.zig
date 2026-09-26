//! Desk-shaped multi-strike quote loop.
//!
//! Holds a small strip of strikes around spot with shared underlying /
//! portfolio delta. Each strike gets an AS or Guéant quote; reservation is
//! further tilted by `portfolio_delta_penalty * portfolio_delta`.
//!
//! Event-log compatible: callers should append one Quote event per strike
//! (with `strike` field) via `event_log.appendQuoteStrike`.

const std = @import("std");
const types = @import("types.zig");
const asq = @import("as_quoter.zig");
const bs = @import("black_scholes.zig");
const surface = @import("surface.zig");

const QuoterConfig = types.QuoterConfig;
const Greeks = types.Greeks;
const Quote = types.Quote;
const StrikeQuote = types.StrikeQuote;
const StrikeSlot = types.StrikeSlot;
const MAX_STRIP = types.MAX_STRIP;

pub const StripConfig = struct {
    /// Half-width in strikes (total = 2*half + 1, capped at MAX_STRIP).
    half_width: u32 = 2,
    /// Strike spacing in underlying points (e.g. 1.0 or 5.0).
    strike_step: f64 = 1.0,
    is_call: bool = true,
};

/// Build centered strike grid around `spot`, snapped to `strike_step`.
pub fn buildStrikeGrid(
    spot: f64,
    cfg: StripConfig,
    out_strikes: []f64,
) usize {
    const max_hw: usize = (MAX_STRIP - 1) / 2;
    var hw: usize = @intCast(cfg.half_width);
    if (hw > max_hw) hw = max_hw;
    const n: usize = 2 * hw + 1;
    if (n > out_strikes.len) return 0;
    const step = @max(cfg.strike_step, 1e-9);
    const atm = @round(spot / step) * step;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const offset: i32 = @as(i32, @intCast(i)) - @as(i32, @intCast(hw));
        out_strikes[i] = atm + @as(f64, @floatFromInt(offset)) * step;
    }
    return n;
}

pub const PortfolioGreeks = struct {
    delta: f64 = 0.0,
    gamma: f64 = 0.0,
    vega: f64 = 0.0,
    theta: f64 = 0.0,
    net_inventory: i32 = 0,
};

/// Aggregate inventory-weighted greeks across the strip.
pub fn portfolioGreeks(slots: []const StrikeSlot, per_contract: []const Greeks) PortfolioGreeks {
    var out: PortfolioGreeks = .{};
    const n = @min(slots.len, per_contract.len);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (!slots[i].active) continue;
        const q = @as(f64, @floatFromInt(slots[i].inventory));
        out.delta += q * per_contract[i].delta;
        out.gamma += q * per_contract[i].gamma;
        out.vega += q * per_contract[i].vega;
        out.theta += q * per_contract[i].theta;
        out.net_inventory += slots[i].inventory;
    }
    return out;
}

/// Quote one strike with optional portfolio-delta tilt.
pub fn quoteOne(
    mid: f64,
    inventory: i32,
    cfg: *const QuoterConfig,
    t_remaining: ?f64,
    greeks_opt: ?*const Greeks,
    portfolio_delta: f64,
    spread_mult: f64,
    size_mult: f64,
) Quote {
    var q = asq.makeQuote(mid, inventory, cfg, t_remaining, greeks_opt, spread_mult, size_mult);
    if (cfg.portfolio_delta_penalty != 0.0 and portfolio_delta != 0.0) {
        q = asq.applyPortfolioDeltaTilt(q, portfolio_delta, cfg);
    }
    return q;
}

/// Price + quote an entire strip. Writes up to `out.len` StrikeQuotes.
/// Returns number of quotes written.
pub fn quoteStrip(
    spot: f64,
    t_rem: f64,
    rate: f64,
    div_yield: f64,
    sabr: surface.SabrParams,
    slots: []const StrikeSlot,
    quoter: *const QuoterConfig,
    strip_cfg: StripConfig,
    spread_mult: f64,
    size_mult: f64,
    out: []StrikeQuote,
) usize {
    const n = @min(slots.len, out.len);
    // First pass: greeks for portfolio aggregation
    var greeks_buf: [MAX_STRIP]Greeks = [_]Greeks{.{}} ** MAX_STRIP;
    var mids: [MAX_STRIP]f64 = [_]f64{0} ** MAX_STRIP;
    var ivs: [MAX_STRIP]f64 = [_]f64{0} ** MAX_STRIP;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (!slots[i].active) continue;
        const K = slots[i].strike;
        const forward = spot * @exp((rate - div_yield) * t_rem);
        const iv = surface.sabrImpliedVol(sabr, forward, K, t_rem);
        const pg = bs.priceAndGreeks(spot, K, t_rem, rate, div_yield, iv, strip_cfg.is_call);
        greeks_buf[i] = pg.greeks;
        mids[i] = pg.price;
        ivs[i] = iv;
    }
    const port = portfolioGreeks(slots[0..n], greeks_buf[0..n]);

    var written: usize = 0;
    i = 0;
    while (i < n) : (i += 1) {
        if (!slots[i].active) continue;
        const q = quoteOne(
            mids[i],
            slots[i].inventory,
            quoter,
            quoter.t_horizon,
            &greeks_buf[i],
            port.delta,
            spread_mult,
            size_mult,
        );
        out[written] = .{
            .strike = slots[i].strike,
            .quote = q,
            .mid = mids[i],
            .iv = ivs[i],
            .greeks = greeks_buf[i],
            .inventory = slots[i].inventory,
        };
        written += 1;
    }
    return written;
}

/// Initialize active slots from a strike grid (flat inventory).
pub fn initFlatSlots(strikes: []const f64, out: []StrikeSlot) usize {
    const n = @min(strikes.len, out.len);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        out[i] = .{ .strike = strikes[i], .inventory = 0, .active = true };
    }
    // deactivate remainder
    var j = n;
    while (j < out.len) : (j += 1) {
        out[j] = .{};
    }
    return n;
}

test "buildStrikeGrid centers on spot" {
    var buf: [MAX_STRIP]f64 = undefined;
    const n = buildStrikeGrid(100.4, .{ .half_width = 2, .strike_step = 1.0 }, &buf);
    try std.testing.expect(n == 5);
    try std.testing.expect(buf[2] == 100.0); // snapped ATM
    try std.testing.expect(buf[0] == 98.0);
    try std.testing.expect(buf[4] == 102.0);
}

test "portfolio greeks aggregate inventory" {
    const slots = [_]StrikeSlot{
        .{ .strike = 99.0, .inventory = 2, .active = true },
        .{ .strike = 100.0, .inventory = -1, .active = true },
    };
    const g = [_]Greeks{
        .{ .delta = 0.4, .gamma = 0.02, .vega = 5.0, .theta = -0.1 },
        .{ .delta = 0.5, .gamma = 0.03, .vega = 6.0, .theta = -0.2 },
    };
    const p = portfolioGreeks(&slots, &g);
    try std.testing.expect(@abs(p.delta - (2.0 * 0.4 + (-1.0) * 0.5)) < 1e-12);
    try std.testing.expect(p.net_inventory == 1);
}

test "quoteStrip writes five quotes" {
    var strikes: [MAX_STRIP]f64 = undefined;
    const n = buildStrikeGrid(100.0, .{ .half_width = 2, .strike_step = 1.0 }, &strikes);
    var slots: [MAX_STRIP]StrikeSlot = undefined;
    _ = initFlatSlots(strikes[0..n], &slots);
    const quoter = QuoterConfig{
        .gamma = 0.12,
        .kappa = 1.5,
        .sigma = 0.45,
        .A = 140.0,
        .mode = .gueant_asymptotic,
        .portfolio_delta_penalty = 0.02,
        .quote_size = 1,
    };
    const sabr = surface.SabrParams{ .alpha = 0.22, .beta = 1.0, .rho = -0.3, .nu = 0.4 };
    var out: [MAX_STRIP]StrikeQuote = undefined;
    const written = quoteStrip(
        100.0,
        30.0 / 365.25,
        0.05,
        0.0,
        sabr,
        slots[0..n],
        &quoter,
        .{ .half_width = 2, .strike_step = 1.0 },
        1.0,
        1.0,
        &out,
    );
    try std.testing.expect(written == 5);
    try std.testing.expect(out[0].quote.ask > out[0].quote.bid);
    try std.testing.expect(out[2].strike == 100.0);
}

test "portfolio delta tilt moves strip reservation" {
    const cfg = QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.4,
        .A = 140.0,
        .mode = .as_finite_horizon,
        .portfolio_delta_penalty = 0.1,
        .quote_size = 1,
    };
    const q_flat = quoteOne(5.0, 0, &cfg, null, null, 0.0, 1.0, 1.0);
    const q_long = quoteOne(5.0, 0, &cfg, null, null, 20.0, 1.0, 1.0);
    try std.testing.expect(q_long.reservation < q_flat.reservation);
}
