//! Raw SVI and SSVI total-variance surfaces, arbitrage checks, sticky regimes.
//!
//! Primary research surface. SABR-lite stays in `surface.zig` as a one-slice
//! alternative; this module owns smile dynamics and no-arb flags.
//!
//! Raw SVI (Gatheral), total implied variance w(k) = σ²(k) T, k = ln(K/F):
//!
//!   w(k) = a + b { ρ (k − m) + √((k − m)² + σ²) }
//!
//! SSVI (Gatheral–Jacquier), power-law φ:
//!
//!   w(k, θ) = (θ/2) { 1 + ρ φ(θ) k + √((φ(θ) k + ρ)² + (1 − ρ²)) }
//!   φ(θ) = η / (θ^γ (1+θ)^{1−γ})
//!
//! Butterfly (density) gate samples Gatheral–Jacquier g(k) on a log-moneyness
//! grid and Roger Lee's wing bound b(1+|ρ|) ≤ 2. Calendar gate requires total
//! variance non-decreasing in expiry. A failing gate, or a large fit residual,
//! sets `surface_suspect` (same flag the decision layer already understands).
//!
//! Sticky regimes when the forward moves F0 → F1 at fixed strike K:
//!   * sticky_strike — IV(K) unchanged (evaluate w at k0 = ln(K/F0))
//!   * sticky_delta  — IV at fixed log-moneyness unchanged (k1 = ln(K/F1));
//!     this is the natural SVI coordinate
//!
//! Cite: Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces",
//! https://arxiv.org/abs/1204.0646

const std = @import("std");

pub const SviParams = struct {
    a: f64 = 0.04,
    b: f64 = 0.1,
    rho: f64 = -0.4,
    m: f64 = 0.0,
    /// SVI "sigma" — smile curvature, not Black vol.
    sigma: f64 = 0.2,
};

pub const SsviParams = struct {
    rho: f64 = -0.4,
    eta: f64 = 0.8,
    /// Power-law exponent γ ∈ [0, 1].
    gamma: f64 = 0.4,
};

pub const StickyRegime = enum(u8) {
    sticky_strike = 0,
    sticky_delta = 1,
};

pub const ButterflyReport = struct {
    ok: bool = false,
    min_g: f64 = 0.0,
    min_w: f64 = 0.0,
    lee_slope: f64 = 0.0,
    lee_ok: bool = false,
    w_positive: bool = false,
};

pub const SurfaceCheck = struct {
    butterfly_ok: bool = false,
    calendar_ok: bool = true,
    min_g: f64 = 0.0,
    rmse: f64 = 0.0,
    max_abs_residual: f64 = 0.0,
    /// True when butterfly/calendar fails or the fit residual is large.
    surface_suspect: bool = false,
};

const K_LO: f64 = -1.5;
const K_HI: f64 = 1.5;
const K_STEPS: usize = 61;
const LEE_MAX: f64 = 2.0;
const G_EPS: f64 = -1e-8;
const W_EPS: f64 = 1e-10;

fn clampRho(rho: f64) f64 {
    return std.math.clamp(rho, -0.999999, 0.999999);
}

pub fn totalVar(p: SviParams, k: f64) f64 {
    const km = k - p.m;
    const s = @sqrt(km * km + p.sigma * p.sigma);
    return p.a + p.b * (p.rho * km + s);
}

pub fn minTotalVar(p: SviParams) f64 {
    const disc = @max(1.0 - p.rho * p.rho, 0.0);
    return p.a + p.b * p.sigma * @sqrt(disc);
}

const Derivs = struct { w: f64, wp: f64, wpp: f64 };

pub fn derivatives(p: SviParams, k: f64) Derivs {
    const km = k - p.m;
    const sig2 = p.sigma * p.sigma;
    const s = @sqrt(km * km + sig2);
    const s_safe = @max(s, 1e-14);
    const w = p.a + p.b * (p.rho * km + s);
    const wp = p.b * (p.rho + km / s_safe);
    const wpp = p.b * sig2 / (s_safe * s_safe * s_safe);
    return .{ .w = w, .wp = wp, .wpp = wpp };
}

/// Gatheral–Jacquier density function. g(k) ≥ 0 ⇔ no butterfly arbitrage at k.
pub fn densityG(p: SviParams, k: f64) f64 {
    const d = derivatives(p, k);
    if (d.w <= 1e-12) return -1.0e9;
    const term1 = 1.0 - k * d.wp / (2.0 * d.w);
    return term1 * term1 - (d.wp * d.wp) / 4.0 * (1.0 / d.w + 0.25) + d.wpp / 2.0;
}

pub fn leeSlope(p: SviParams) f64 {
    return p.b * (1.0 + @abs(p.rho));
}

pub fn butterflyCheck(p: SviParams) ButterflyReport {
    var rep = ButterflyReport{};
    rep.lee_slope = leeSlope(p);
    rep.lee_ok = rep.lee_slope <= LEE_MAX + 1e-9 and p.b >= 0.0 and @abs(p.rho) < 1.0 and p.sigma > 0.0;
    rep.min_w = std.math.inf(f64);
    rep.min_g = std.math.inf(f64);
    var i: usize = 0;
    while (i < K_STEPS) : (i += 1) {
        const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(K_STEPS - 1));
        const k = K_LO + (K_HI - K_LO) * t;
        const w = totalVar(p, k);
        const g = densityG(p, k);
        if (w < rep.min_w) rep.min_w = w;
        if (g < rep.min_g) rep.min_g = g;
    }
    const wmin_closed = minTotalVar(p);
    if (wmin_closed < rep.min_w) rep.min_w = wmin_closed;
    rep.w_positive = rep.min_w > W_EPS;
    rep.ok = rep.lee_ok and rep.w_positive and rep.min_g >= G_EPS;
    return rep;
}

pub fn ivFromTotalVar(w: f64, t: f64) f64 {
    if (!(t > 0.0) or !(w > 0.0)) return 0.0;
    return @sqrt(w / t);
}

pub fn impliedVol(p: SviParams, k: f64, t: f64) f64 {
    return ivFromTotalVar(totalVar(p, k), t);
}

pub fn ssviPhi(theta: f64, p: SsviParams) f64 {
    if (!(theta > 0.0) or !(p.eta > 0.0)) return 0.0;
    const g = std.math.clamp(p.gamma, 0.0, 1.0);
    const denom = std.math.pow(f64, theta, g) * std.math.pow(f64, 1.0 + theta, 1.0 - g);
    if (!(denom > 0.0)) return 0.0;
    return p.eta / denom;
}

pub fn ssviTotalVar(k: f64, theta: f64, p: SsviParams) f64 {
    if (!(theta > 0.0)) return 0.0;
    const phi = ssviPhi(theta, p);
    const rho = clampRho(p.rho);
    const x = phi * k + rho;
    const disc = x * x + (1.0 - rho * rho);
    return 0.5 * theta * (1.0 + rho * phi * k + @sqrt(@max(disc, 0.0)));
}

pub fn ssviImpliedVol(k: f64, theta: f64, t: f64, p: SsviParams) f64 {
    return ivFromTotalVar(ssviTotalVar(k, theta, p), t);
}

/// Sufficient calendar condition used by Gatheral–Jacquier for power-law SSVI.
pub fn ssviParamsCalendarSafe(p: SsviParams) bool {
    if (!(p.eta >= 0.0)) return false;
    if (p.gamma < 0.0 or p.gamma > 1.0) return false;
    if (@abs(p.rho) >= 1.0) return false;
    return p.eta * (1.0 + @abs(p.rho)) <= LEE_MAX + 1e-12;
}

/// Pointwise calendar check: w(k; θ2) ≥ w(k; θ1) when θ2 ≥ θ1.
pub fn ssviCalendarOk(theta1: f64, theta2: f64, p: SsviParams) bool {
    const lo = @min(theta1, theta2);
    const hi = @max(theta1, theta2);
    var i: usize = 0;
    while (i < K_STEPS) : (i += 1) {
        const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(K_STEPS - 1));
        const k = K_LO + (K_HI - K_LO) * t;
        const w_lo = ssviTotalVar(k, lo, p);
        const w_hi = ssviTotalVar(k, hi, p);
        if (w_hi + 1e-9 < w_lo) return false;
    }
    return true;
}

/// Two raw-SVI slices. `t2` should be the later expiry; total variance must not fall.
pub fn rawCalendarOk(earlier: SviParams, later: SviParams) bool {
    var i: usize = 0;
    while (i < K_STEPS) : (i += 1) {
        const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(K_STEPS - 1));
        const k = K_LO + (K_HI - K_LO) * t;
        if (totalVar(later, k) + 1e-9 < totalVar(earlier, k)) return false;
    }
    return true;
}

pub fn impliedVolAfterMove(
    p: SviParams,
    forward0: f64,
    forward1: f64,
    strike: f64,
    t: f64,
    regime: StickyRegime,
) f64 {
    if (!(forward0 > 0.0) or !(forward1 > 0.0) or !(strike > 0.0)) return 0.0;
    const k_used: f64 = switch (regime) {
        .sticky_strike => @log(strike / forward0),
        .sticky_delta => @log(strike / forward1),
    };
    return impliedVol(p, k_used, t);
}

fn unpackTheta(y: *const [5]f64) SviParams {
    return .{
        .a = y[0],
        .b = @exp(std.math.clamp(y[1], -12.0, 4.0)),
        .rho = std.math.tanh(y[2]),
        .m = y[3],
        .sigma = @exp(std.math.clamp(y[4], -12.0, 3.0)),
    };
}

fn packTheta(p: SviParams) [5]f64 {
    const b = @max(p.b, 1e-8);
    const sig = @max(p.sigma, 1e-8);
    const rho = clampRho(p.rho);
    return .{ p.a, @log(b), 0.5 * @log((1.0 + rho) / (1.0 - rho)), p.m, @log(sig) };
}

fn sliceLoss(y: *const [5]f64, ks: []const f64, ws: []const f64) f64 {
    const p = unpackTheta(y);
    var sse: f64 = 0.0;
    var i: usize = 0;
    while (i < ks.len and i < ws.len) : (i += 1) {
        const d = totalVar(p, ks[i]) - ws[i];
        sse += d * d;
    }
    const lee = leeSlope(p);
    if (lee > LEE_MAX) {
        const over = lee - LEE_MAX;
        sse += over * over * 10.0;
    }
    const wmin = minTotalVar(p);
    if (wmin < 0.0) sse += wmin * wmin * 100.0;
    return sse;
}

fn nelderMead(ks: []const f64, ws: []const f64, x0: [5]f64) [5]f64 {
    const n: usize = 5;
    var simplex: [6][5]f64 = undefined;
    var vals: [6]f64 = undefined;
    simplex[0] = x0;
    vals[0] = sliceLoss(&simplex[0], ks, ws);
    var j: usize = 0;
    while (j < n) : (j += 1) {
        var y = x0;
        const bump = 0.08 * @max(1.0, @abs(y[j]));
        y[j] += if (@abs(y[j]) < 1e-8) 0.08 else bump;
        simplex[j + 1] = y;
        vals[j + 1] = sliceLoss(&y, ks, ws);
    }

    const iters: usize = 450;
    var it: usize = 0;
    while (it < iters) : (it += 1) {
        // insertion sort — 6 points
        var a: usize = 0;
        while (a < n + 1) : (a += 1) {
            var b: usize = a + 1;
            while (b < n + 1) : (b += 1) {
                if (vals[b] < vals[a]) {
                    const tv = vals[a];
                    vals[a] = vals[b];
                    vals[b] = tv;
                    const ts = simplex[a];
                    simplex[a] = simplex[b];
                    simplex[b] = ts;
                }
            }
        }
        var centroid = [_]f64{0} ** 5;
        var c: usize = 0;
        while (c < n) : (c += 1) {
            var d: usize = 0;
            while (d < n) : (d += 1) centroid[d] += simplex[c][d];
        }
        var d: usize = 0;
        while (d < n) : (d += 1) centroid[d] /= @as(f64, @floatFromInt(n));

        var xr: [5]f64 = undefined;
        d = 0;
        while (d < n) : (d += 1) xr[d] = centroid[d] + 1.0 * (centroid[d] - simplex[n][d]);
        const fr = sliceLoss(&xr, ks, ws);
        if (vals[0] <= fr and fr < vals[n - 1]) {
            simplex[n] = xr;
            vals[n] = fr;
            continue;
        }
        if (fr < vals[0]) {
            var xe: [5]f64 = undefined;
            d = 0;
            while (d < n) : (d += 1) xe[d] = centroid[d] + 2.0 * (xr[d] - centroid[d]);
            const fe = sliceLoss(&xe, ks, ws);
            if (fe < fr) {
                simplex[n] = xe;
                vals[n] = fe;
            } else {
                simplex[n] = xr;
                vals[n] = fr;
            }
            continue;
        }
        var xc: [5]f64 = undefined;
        d = 0;
        while (d < n) : (d += 1) xc[d] = centroid[d] + 0.5 * (simplex[n][d] - centroid[d]);
        const fc = sliceLoss(&xc, ks, ws);
        if (fc < vals[n]) {
            simplex[n] = xc;
            vals[n] = fc;
            continue;
        }
        var s: usize = 1;
        while (s < n + 1) : (s += 1) {
            d = 0;
            while (d < n) : (d += 1) {
                simplex[s][d] = simplex[0][d] + 0.5 * (simplex[s][d] - simplex[0][d]);
            }
            vals[s] = sliceLoss(&simplex[s], ks, ws);
        }
    }
    var best: usize = 0;
    var bi: usize = 1;
    while (bi < n + 1) : (bi += 1) {
        if (vals[bi] < vals[best]) best = bi;
    }
    return simplex[best];
}

pub const FitResult = struct {
    params: SviParams = .{},
    rmse: f64 = 0.0,
    max_abs_residual: f64 = 0.0,
    butterfly: ButterflyReport = .{},
    surface_suspect: bool = true,
};

/// Robust derivative-free fit of raw SVI to total-variance smile points.
pub fn calibrate(ks: []const f64, ws: []const f64) FitResult {
    var out = FitResult{};
    if (ks.len == 0 or ws.len == 0 or ks.len != ws.len) return out;
    var imin: usize = 0;
    var wmin_obs = ws[0];
    var i: usize = 1;
    while (i < ws.len) : (i += 1) {
        if (ws[i] < wmin_obs) {
            wmin_obs = ws[i];
            imin = i;
        }
    }
    const guess = SviParams{
        .a = @max(wmin_obs * 0.5, 1e-4),
        .b = 0.15,
        .rho = -0.25,
        .m = ks[imin],
        .sigma = 0.2,
    };
    const y = nelderMead(ks, ws, packTheta(guess));
    out.params = unpackTheta(&y);
    var sse: f64 = 0.0;
    var max_abs: f64 = 0.0;
    i = 0;
    while (i < ks.len) : (i += 1) {
        const resid = totalVar(out.params, ks[i]) - ws[i];
        sse += resid * resid;
        max_abs = @max(max_abs, @abs(resid));
    }
    out.rmse = @sqrt(sse / @as(f64, @floatFromInt(ks.len)));
    out.max_abs_residual = max_abs;
    out.butterfly = butterflyCheck(out.params);
    out.surface_suspect = !out.butterfly.ok or out.rmse > 0.02 or max_abs > 0.05;
    return out;
}

pub fn assess(p: SviParams, ks: []const f64, ws: []const f64, calendar_ok: bool) SurfaceCheck {
    const bf = butterflyCheck(p);
    var sse: f64 = 0.0;
    var max_abs: f64 = 0.0;
    if (ks.len == ws.len and ks.len > 0) {
        var i: usize = 0;
        while (i < ks.len) : (i += 1) {
            const resid = totalVar(p, ks[i]) - ws[i];
            sse += resid * resid;
            max_abs = @max(max_abs, @abs(resid));
        }
    }
    const rmse = if (ks.len > 0) @sqrt(sse / @as(f64, @floatFromInt(ks.len))) else 0.0;
    const suspect = !bf.ok or !calendar_ok or rmse > 0.02 or max_abs > 0.05;
    return .{
        .butterfly_ok = bf.ok,
        .calendar_ok = calendar_ok,
        .min_g = bf.min_g,
        .rmse = rmse,
        .max_abs_residual = max_abs,
        .surface_suspect = suspect,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "no-arb svi smile passes butterfly" {
    const p = SviParams{ .a = 0.04, .b = 0.1, .rho = -0.4, .m = 0.0, .sigma = 0.2 };
    const rep = butterflyCheck(p);
    try std.testing.expect(rep.ok);
    try std.testing.expect(rep.min_g > 0.0);
    try std.testing.expect(rep.w_positive);
    try std.testing.expect(rep.lee_ok);
    const chk = assess(p, &.{}, &.{}, true);
    try std.testing.expect(!chk.surface_suspect);
}

test "injected butterfly arb fails density and lee" {
    const p = SviParams{ .a = 0.04, .b = 3.0, .rho = 0.9, .m = 0.0, .sigma = 0.2 };
    const rep = butterflyCheck(p);
    try std.testing.expect(!rep.ok);
    try std.testing.expect(!rep.lee_ok);
    try std.testing.expect(rep.min_g < 0.0);
    const chk = assess(p, &.{}, &.{}, true);
    try std.testing.expect(chk.surface_suspect);
}

test "tight curvature arb fails g even if lee holds" {
    const p = SviParams{ .a = 0.01, .b = 0.6, .rho = -0.7, .m = 0.05, .sigma = 0.05 };
    const rep = butterflyCheck(p);
    try std.testing.expect(!rep.ok);
    try std.testing.expect(rep.min_g < 0.0);
}

test "ssvi atm total variance equals theta" {
    const p = SsviParams{ .rho = -0.4, .eta = 0.8, .gamma = 0.5 };
    const theta: f64 = 0.04;
    try std.testing.expect(@abs(ssviTotalVar(0.0, theta, p) - theta) < 1e-12);
}

test "ssvi calendar holds for safe params and fails if inverted" {
    const p = SsviParams{ .rho = -0.3, .eta = 1.0, .gamma = 0.4 };
    try std.testing.expect(ssviParamsCalendarSafe(p));
    try std.testing.expect(ssviCalendarOk(0.04, 0.09, p));
    // Hand-built raw slices: later expiry richer passes; inverted fails.
    const near = SviParams{ .a = 0.02, .b = 0.05, .rho = -0.2, .m = 0.0, .sigma = 0.15 };
    const far = SviParams{ .a = 0.05, .b = 0.05, .rho = -0.2, .m = 0.0, .sigma = 0.15 };
    try std.testing.expect(rawCalendarOk(near, far));
    try std.testing.expect(!rawCalendarOk(far, near));
    const chk = assess(near, &.{}, &.{}, false);
    try std.testing.expect(chk.surface_suspect);
    try std.testing.expect(!chk.calendar_ok);
}

test "calibration recovers a planted no-arb smile" {
    const planted = SviParams{ .a = 0.04, .b = 0.15, .rho = -0.3, .m = 0.02, .sigma = 0.25 };
    var ks: [17]f64 = undefined;
    var ws: [17]f64 = undefined;
    var i: usize = 0;
    while (i < ks.len) : (i += 1) {
        const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(ks.len - 1));
        ks[i] = -0.8 + 1.6 * t;
        ws[i] = totalVar(planted, ks[i]);
    }
    const fit = calibrate(&ks, &ws);
    try std.testing.expect(fit.rmse < 1e-4);
    try std.testing.expect(fit.butterfly.ok);
    try std.testing.expect(!fit.surface_suspect);
    try std.testing.expect(@abs(fit.params.a - planted.a) < 0.01);
    try std.testing.expect(@abs(fit.params.b - planted.b) < 0.02);
    try std.testing.expect(@abs(fit.params.rho - planted.rho) < 0.05);
}

test "sticky strike holds iv at K; sticky delta holds iv at k" {
    const p = SviParams{ .a = 0.04, .b = 0.15, .rho = -0.4, .m = 0.0, .sigma = 0.2 };
    const f0: f64 = 100.0;
    const f1: f64 = 110.0;
    const strike: f64 = 100.0;
    const t: f64 = 0.25;
    const iv0 = impliedVol(p, @log(strike / f0), t);
    const sticky_k = impliedVolAfterMove(p, f0, f1, strike, t, .sticky_strike);
    const sticky_d = impliedVolAfterMove(p, f0, f1, strike, t, .sticky_delta);
    try std.testing.expect(@abs(sticky_k - iv0) < 1e-12);
    try std.testing.expect(@abs(sticky_d - impliedVol(p, @log(strike / f1), t)) < 1e-12);
    // Skewed smile: fixed-strike IV moves under sticky-delta when spot moves.
    try std.testing.expect(@abs(sticky_d - sticky_k) > 1e-4);
    // Fixed log-moneyness (strike scales with forward) is unchanged under sticky-delta.
    const k_fixed = @log(strike / f0);
    const strike_scaled = f1 * @exp(k_fixed);
    const iv_scaled = impliedVolAfterMove(p, f0, f1, strike_scaled, t, .sticky_delta);
    try std.testing.expect(@abs(iv_scaled - iv0) < 1e-12);
}
