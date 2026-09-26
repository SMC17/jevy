//! Multi-expiry option book and term-structure risk.
//!
//! Each leg is a European call or put with its own expiry, strike, IV, and
//! signed quantity. Portfolio greeks sum qty × per-contract BS greeks.
//!
//! Conventions (σ in decimal, t in years; vega is ∂V/∂σ not "per vol point"):
//!
//!   * parallel vega  — Σ vega_i. PnL for a flat +dσ shift is vega · dσ,
//!     ignoring vanna/volga cross terms.
//!   * bucket vega    — same sum restricted to one expiry.
//!   * term-structure vega (slope) — Σ vega_i · (T_i − T_front).
//!     A tilt dσ(T) = φ · (T − T_front) has first-order vega PnL
//!     `term_vega_slope * φ` (vega × years × vol).
//!   * vanna — ∂²V/∂S∂σ, aggregated. Cross PnL ≈ vanna · dS · dσ.
//!   * volga — ∂²V/∂σ², aggregated. Second-order PnL ≈ ½ volga · (dσ)².
//!
//! Scenario cells can shock spot once and apply a parallel IV move plus a
//! term tilt. Risk limits cover parallel vega, per-bucket vega, |vanna|,
//! |volga|, and |term slope|.
//!
//! Quotes: each expiry is a strike strip marked with SSVI
//! (θ = σ_atm² T, shared ρ, η, γ) and quoted by the configured mode
//! (including `gueant_ode`). SABR-lite remains available on the single-expiry
//! strip in `multi_strike.zig`.

const std = @import("std");
const types = @import("types.zig");
const bs = @import("black_scholes.zig");
const svi = @import("svi.zig");
const asq = @import("as_quoter.zig");

const Greeks = types.Greeks;
const QuoterConfig = types.QuoterConfig;
const StrikeQuote = types.StrikeQuote;

pub const MAX_LEGS: usize = 32;
pub const MAX_EXPIRIES: usize = 6;
pub const MAX_STRIP: usize = types.MAX_STRIP;

pub const Leg = struct {
    expiry: f64 = 0.0,
    strike: f64 = 0.0,
    is_call: bool = true,
    qty: f64 = 0.0,
    iv: f64 = 0.2,
};

pub const Bucket = struct {
    expiry: f64 = 0.0,
    delta: f64 = 0.0,
    gamma: f64 = 0.0,
    vega: f64 = 0.0,
    theta: f64 = 0.0,
    vanna: f64 = 0.0,
    volga: f64 = 0.0,
    n_legs: u32 = 0,
};

pub const TermRisk = struct {
    delta: f64 = 0.0,
    gamma: f64 = 0.0,
    vega: f64 = 0.0,
    theta: f64 = 0.0,
    vanna: f64 = 0.0,
    volga: f64 = 0.0,
    n_buckets: usize = 0,
    buckets: [MAX_EXPIRIES]Bucket = [_]Bucket{.{}} ** MAX_EXPIRIES,
    t_front: f64 = 0.0,
    /// Σ vega_i (T_i − T_front). Units: vega · years.
    term_vega_slope: f64 = 0.0,
};

pub const TermBreach = enum {
    none,
    parallel_vega,
    bucket_vega,
    vanna,
    volga,
    term_slope,
};

pub const TermLimitConfig = struct {
    max_abs_parallel_vega: f64 = 1.0e9,
    max_abs_bucket_vega: f64 = 1.0e9,
    max_abs_vanna: f64 = 1.0e9,
    max_abs_volga: f64 = 1.0e9,
    max_abs_term_slope: f64 = 1.0e9,
};

pub const ExpiryStrip = struct {
    expiry: f64 = 0.0,
    theta: f64 = 0.0,
    n: usize = 0,
    quotes: [MAX_STRIP]StrikeQuote = [_]StrikeQuote{.{}} ** MAX_STRIP,
};

fn sameExpiry(a: f64, b: f64) bool {
    return @abs(a - b) <= 1e-10 * @max(1.0, @max(@abs(a), @abs(b)));
}

pub fn aggregate(spot: f64, rate: f64, div_yield: f64, legs: []const Leg) TermRisk {
    var risk = TermRisk{};
    for (legs) |leg| {
        if (leg.qty == 0.0 or !(leg.expiry > 0.0) or !(leg.strike > 0.0)) continue;
        const g = bs.greeks(spot, leg.strike, leg.expiry, rate, div_yield, leg.iv, leg.is_call);
        const scaled = g.scale(leg.qty);
        risk.delta += scaled.delta;
        risk.gamma += scaled.gamma;
        risk.vega += scaled.vega;
        risk.theta += scaled.theta;
        risk.vanna += scaled.vanna;
        risk.volga += scaled.volga;

        var slot: ?usize = null;
        var b: usize = 0;
        while (b < risk.n_buckets) : (b += 1) {
            if (sameExpiry(risk.buckets[b].expiry, leg.expiry)) {
                slot = b;
                break;
            }
        }
        if (slot == null) {
            if (risk.n_buckets >= MAX_EXPIRIES) continue;
            slot = risk.n_buckets;
            risk.buckets[risk.n_buckets].expiry = leg.expiry;
            risk.n_buckets += 1;
        }
        const s = slot.?;
        risk.buckets[s].delta += scaled.delta;
        risk.buckets[s].gamma += scaled.gamma;
        risk.buckets[s].vega += scaled.vega;
        risk.buckets[s].theta += scaled.theta;
        risk.buckets[s].vanna += scaled.vanna;
        risk.buckets[s].volga += scaled.volga;
        risk.buckets[s].n_legs += 1;
    }

    // Order buckets by expiry (insertion sort, n ≤ 6).
    var i: usize = 1;
    while (i < risk.n_buckets) : (i += 1) {
        const key = risk.buckets[i];
        var j: usize = i;
        while (j > 0 and risk.buckets[j - 1].expiry > key.expiry) : (j -= 1) {
            risk.buckets[j] = risk.buckets[j - 1];
        }
        risk.buckets[j] = key;
    }
    if (risk.n_buckets > 0) {
        risk.t_front = risk.buckets[0].expiry;
        var s: usize = 0;
        while (s < risk.n_buckets) : (s += 1) {
            risk.term_vega_slope += risk.buckets[s].vega * (risk.buckets[s].expiry - risk.t_front);
        }
    }
    return risk;
}

pub fn evaluateLimits(risk: *const TermRisk, cfg: *const TermLimitConfig) TermBreach {
    if (@abs(risk.vega) > cfg.max_abs_parallel_vega) return .parallel_vega;
    if (@abs(risk.vanna) > cfg.max_abs_vanna) return .vanna;
    if (@abs(risk.volga) > cfg.max_abs_volga) return .volga;
    if (@abs(risk.term_vega_slope) > cfg.max_abs_term_slope) return .term_slope;
    var i: usize = 0;
    while (i < risk.n_buckets) : (i += 1) {
        if (@abs(risk.buckets[i].vega) > cfg.max_abs_bucket_vega) return .bucket_vega;
    }
    return .none;
}

/// Taylor PnL for one expiry bucket. d_iv is the absolute vol shock at that expiry.
pub fn bucketPnl(b: *const Bucket, spot: f64, d_spot_frac: f64, d_iv: f64) f64 {
    const d_s = spot * d_spot_frac;
    return b.delta * d_s + 0.5 * b.gamma * d_s * d_s + b.vega * d_iv + b.vanna * d_s * d_iv + 0.5 * b.volga * d_iv * d_iv;
}

/// Book PnL. IV shock at expiry T is `d_iv_parallel + slope_phi * (T − T_front)`.
pub fn scenarioPnl(risk: *const TermRisk, spot: f64, d_spot_frac: f64, d_iv_parallel: f64, slope_phi: f64) f64 {
    var pnl: f64 = 0.0;
    var i: usize = 0;
    while (i < risk.n_buckets) : (i += 1) {
        const d_iv = d_iv_parallel + slope_phi * (risk.buckets[i].expiry - risk.t_front);
        pnl += bucketPnl(&risk.buckets[i], spot, d_spot_frac, d_iv);
    }
    return pnl;
}

/// Full reprice of every leg under a parallel spot fraction and per-expiry IV bump.
pub fn scenarioReprice(
    spot: f64,
    rate: f64,
    div_yield: f64,
    legs: []const Leg,
    d_spot_frac: f64,
    d_iv_parallel: f64,
    slope_phi: f64,
    t_front: f64,
) f64 {
    var pnl: f64 = 0.0;
    const s2 = spot * (1.0 + d_spot_frac);
    for (legs) |leg| {
        if (leg.qty == 0.0) continue;
        const d_iv = d_iv_parallel + slope_phi * (leg.expiry - t_front);
        const iv2 = @max(leg.iv + d_iv, 1e-6);
        const p0 = bs.price(spot, leg.strike, leg.expiry, rate, div_yield, leg.iv, leg.is_call);
        const p1 = bs.price(s2, leg.strike, leg.expiry, rate, div_yield, iv2, leg.is_call);
        pnl += leg.qty * (p1 - p0);
    }
    return pnl;
}

/// Quote one strike strip per expiry. θ(T) = atm_iv² · T on a shared SSVI.
pub fn quoteExpiries(
    spot: f64,
    rate: f64,
    div_yield: f64,
    smile: svi.SsviParams,
    atm_iv: f64,
    expiries: []const f64,
    strikes: []const f64,
    quoter: *const QuoterConfig,
    is_call: bool,
    out: []ExpiryStrip,
) usize {
    const n_exp = @min(expiries.len, out.len);
    var e: usize = 0;
    while (e < n_exp) : (e += 1) {
        const t = expiries[e];
        const theta = atm_iv * atm_iv * @max(t, 0.0);
        const forward = spot * @exp((rate - div_yield) * t);
        out[e] = .{ .expiry = t, .theta = theta };
        const n_k = @min(strikes.len, MAX_STRIP);
        var i: usize = 0;
        while (i < n_k) : (i += 1) {
            const strike = strikes[i];
            const k = if (forward > 0.0 and strike > 0.0) @log(strike / forward) else 0.0;
            const w = svi.ssviTotalVar(k, theta, smile);
            var iv = svi.ivFromTotalVar(w, t);
            if (!(iv > 1e-6)) iv = @max(atm_iv, 1e-4);
            const pg = bs.priceAndGreeks(spot, strike, t, rate, div_yield, iv, is_call);
            const q = asq.makeQuote(pg.price, 0, quoter, quoter.t_horizon, &pg.greeks, 1.0, 1.0);
            out[e].quotes[i] = .{
                .strike = strike,
                .quote = q,
                .mid = pg.price,
                .iv = iv,
                .greeks = pg.greeks,
                .inventory = 0,
            };
            out[e].n += 1;
        }
    }
    return n_exp;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "two expiries aggregate bucket vega and term slope" {
    const spot: f64 = 100.0;
    const legs = [_]Leg{
        .{ .expiry = 0.25, .strike = 100.0, .is_call = true, .qty = 10.0, .iv = 0.2 },
        .{ .expiry = 1.0, .strike = 100.0, .is_call = true, .qty = -4.0, .iv = 0.2 },
    };
    const risk = aggregate(spot, 0.05, 0.0, &legs);
    try std.testing.expect(risk.n_buckets == 2);
    try std.testing.expect(@abs(risk.buckets[0].expiry - 0.25) < 1e-12);
    try std.testing.expect(@abs(risk.buckets[1].expiry - 1.0) < 1e-12);
    const g_front = bs.greeks(spot, 100.0, 0.25, 0.05, 0.0, 0.2, true);
    const g_back = bs.greeks(spot, 100.0, 1.0, 0.05, 0.0, 0.2, true);
    try std.testing.expect(@abs(risk.buckets[0].vega - 10.0 * g_front.vega) < 1e-8);
    try std.testing.expect(@abs(risk.buckets[1].vega - (-4.0) * g_back.vega) < 1e-8);
    try std.testing.expect(@abs(risk.vega - (risk.buckets[0].vega + risk.buckets[1].vega)) < 1e-8);
    try std.testing.expect(@abs(risk.vanna - (10.0 * g_front.vanna + -4.0 * g_back.vanna)) < 1e-6);
    try std.testing.expect(@abs(risk.volga - (10.0 * g_front.volga + -4.0 * g_back.volga)) < 1e-6);
    const slope = risk.buckets[1].vega * (1.0 - 0.25);
    try std.testing.expect(@abs(risk.term_vega_slope - slope) < 1e-8);
    // Short the back month ⇒ negative bucket vega ⇒ negative term slope.
    try std.testing.expect(risk.term_vega_slope < 0.0);
}

test "term limits trip on bucket vega and vanna" {
    const spot: f64 = 100.0;
    const legs = [_]Leg{
        .{ .expiry = 0.5, .strike = 100.0, .qty = 50.0, .iv = 0.22 },
    };
    const risk = aggregate(spot, 0.01, 0.0, &legs);
    const ok = evaluateLimits(&risk, &.{ .max_abs_parallel_vega = 1e9, .max_abs_bucket_vega = 1e9 });
    try std.testing.expect(ok == .none);
    const breach = evaluateLimits(&risk, &.{ .max_abs_bucket_vega = 1.0 });
    try std.testing.expect(breach == .bucket_vega);
    const breach_v = evaluateLimits(&risk, &.{ .max_abs_vanna = 1e-9 });
    // ATM vanna is near 0 (d2 small) — use a wing strike if needed.
    _ = breach_v;
    const wing = [_]Leg{
        .{ .expiry = 0.5, .strike = 80.0, .qty = 40.0, .iv = 0.3 },
    };
    const wr = aggregate(spot, 0.01, 0.0, &wing);
    try std.testing.expect(@abs(wr.vanna) > 1.0);
    try std.testing.expect(@abs(wr.volga) > 1.0);
    try std.testing.expect(evaluateLimits(&wr, &.{ .max_abs_vanna = 0.5 }) == .vanna);
    try std.testing.expect(evaluateLimits(&wr, &.{ .max_abs_volga = 0.5 }) == .volga);
}

test "scenario tilt pnl tracks term slope; reprice agrees on a small shock" {
    const spot: f64 = 100.0;
    const legs = [_]Leg{
        .{ .expiry = 0.25, .strike = 100.0, .qty = 0.0, .iv = 0.2 },
        .{ .expiry = 1.0, .strike = 100.0, .qty = 8.0, .iv = 0.2 },
    };
    // Drop the flat front leg (qty 0) — only the back month.
    const back = [_]Leg{
        .{ .expiry = 0.25, .strike = 100.0, .qty = 1.0, .iv = 0.2 },
        .{ .expiry = 1.0, .strike = 100.0, .qty = 8.0, .iv = 0.2 },
    };
    _ = legs;
    const risk = aggregate(spot, 0.0, 0.0, &back);
    const phi: f64 = 0.01;
    const pnl = scenarioPnl(&risk, spot, 0.0, 0.0, phi);
    // First order is term_vega_slope * φ plus ½ volga (dσ)² per bucket.
    var expected: f64 = 0.0;
    var i: usize = 0;
    while (i < risk.n_buckets) : (i += 1) {
        const d_iv = phi * (risk.buckets[i].expiry - risk.t_front);
        expected += risk.buckets[i].vega * d_iv + 0.5 * risk.buckets[i].volga * d_iv * d_iv;
    }
    try std.testing.expect(@abs(pnl - expected) < 1e-8);
    try std.testing.expect(pnl > 0.0); // long back vega, positive tilt

    const repr = scenarioReprice(spot, 0.0, 0.0, &back, 0.0, 0.01, 0.0, risk.t_front);
    const taylor = scenarioPnl(&risk, spot, 0.0, 0.01, 0.0);
    // Small parallel shock: Taylor (vega + ½ volga) tracks the reprice.
    try std.testing.expect(repr > 0.0);
    try std.testing.expect(@abs(repr - taylor) / @max(@abs(repr), 1e-6) < 0.05);
}

test "multi-expiry ssvi strip quotes two tenors" {
    const smile = svi.SsviParams{ .rho = -0.35, .eta = 0.9, .gamma = 0.4 };
    try std.testing.expect(svi.ssviParamsCalendarSafe(smile));
    const expiries = [_]f64{ 30.0 / 365.25, 120.0 / 365.25 };
    const strikes = [_]f64{ 95.0, 100.0, 105.0 };
    const quoter = QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.4,
        .A = 100.0,
        .mode = .gueant_asymptotic,
        .quote_size = 1,
        .min_half_spread = 0.01,
        .max_half_spread = 5.0,
    };
    var out: [2]ExpiryStrip = undefined;
    const n = quoteExpiries(100.0, 0.05, 0.0, smile, 0.22, &expiries, &strikes, &quoter, true, &out);
    try std.testing.expect(n == 2);
    try std.testing.expect(out[0].n == 3 and out[1].n == 3);
    try std.testing.expect(out[0].quotes[1].quote.ask > out[0].quotes[1].quote.bid);
    // Longer expiry has more total variance (calendar) and a higher mid for the ATM call.
    try std.testing.expect(out[1].theta > out[0].theta);
    try std.testing.expect(out[1].quotes[1].mid > out[0].quotes[1].mid);
    // Put-wing IV above call-wing IV under negative rho (sticky log-moneyness smile).
    try std.testing.expect(out[0].quotes[0].iv > out[0].quotes[2].iv);
}
