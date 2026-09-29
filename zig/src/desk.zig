//! Multi-sleeve desk kernels: factor strip and inverse-vol allocation.
//!
//! Paper / simulation only. No orders.
//!
//! Gamma factor matches `hedge.greekPnlStep`: 0.5 * Γ * (ΔS)^2.
//! Beta factor is the simple spot return ΔS/S (caller passes that series).
//! Vega factor matches greek PnL: ν * Δσ.
//!
//! Slopes are OLS on demeaned factors (an intercept is estimated and then
//! kept). The stored residual is
//!
//!   r − β̂ f_β − γ̂ f_Γ − ν̂ f_ν
//!
//! so the intercept, a constant premium, stays in the residual mean.
//! A ridge of 1e-12 sits on the diagonal of the centered Gram matrix.
//! R² uses the centered residual versus the centered raw series.

const std = @import("std");
const hedge = @import("hedge.zig");
const types = @import("types.zig");

pub const RIDGE: f64 = 1e-12;
pub const MAX_SLEEVES: usize = 8;

pub const StripResult = struct {
    beta: f64 = 0.0,
    gamma_coef: f64 = 0.0,
    vega_coef: f64 = 0.0,
    r2: f64 = 0.0,
    n_factors: usize = 0,
};

pub fn betaFactor(d_spot: f64, spot: f64) f64 {
    if (!(spot > 0.0) or !std.math.isFinite(spot)) return 0.0;
    return d_spot / spot;
}

/// Same expression as `hedge.greekPnlStep` gamma bucket.
pub fn gammaFactor(gamma: f64, d_spot: f64) f64 {
    return 0.5 * gamma * d_spot * d_spot;
}

pub fn vegaFactor(vega: f64, d_sigma: f64) f64 {
    return vega * d_sigma;
}

fn dot(a: []const f64, b: []const f64) f64 {
    var s: f64 = 0.0;
    for (a, b) |x, y| s += x * y;
    return s;
}

fn meanOf(x: []const f64) f64 {
    if (x.len == 0) return 0.0;
    var s: f64 = 0.0;
    for (x) |v| s += v;
    return s / @as(f64, @floatFromInt(x.len));
}

fn solveLinear(a_in: [3][3]f64, b_in: [3]f64, n: usize) [3]f64 {
    var a = a_in;
    var b = b_in;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var piv = i;
        var best = @abs(a[i][i]);
        var r: usize = i + 1;
        while (r < n) : (r += 1) {
            const v = @abs(a[r][i]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (piv != i) {
            var c: usize = 0;
            while (c < n) : (c += 1) {
                const tmp = a[i][c];
                a[i][c] = a[piv][c];
                a[piv][c] = tmp;
            }
            const tb = b[i];
            b[i] = b[piv];
            b[piv] = tb;
        }
        const diag = a[i][i];
        if (@abs(diag) < 1e-18) continue;
        var r2: usize = i + 1;
        while (r2 < n) : (r2 += 1) {
            const f = a[r2][i] / diag;
            var c2: usize = i;
            while (c2 < n) : (c2 += 1) a[r2][c2] -= f * a[i][c2];
            b[r2] -= f * b[i];
        }
    }
    var x: [3]f64 = .{ 0, 0, 0 };
    var k = n;
    while (k > 0) {
        k -= 1;
        var s = b[k];
        var c3: usize = k + 1;
        while (c3 < n) : (c3 += 1) s -= a[k][c3] * x[c3];
        if (@abs(a[k][k]) < 1e-18) x[k] = 0 else x[k] = s / a[k][k];
    }
    return x;
}

pub fn stripResidual(
    r: []const f64,
    f_beta: []const f64,
    f_gamma: []const f64,
    f_vega: ?[]const f64,
    out_resid: []f64,
) StripResult {
    const n = r.len;
    if (n == 0 or f_beta.len != n or f_gamma.len != n or out_resid.len != n) return .{};
    const use_v = f_vega != null and f_vega.?.len == n;
    const k: usize = if (use_v) 3 else 2;
    var cols: [3][]const f64 = .{ f_beta, f_gamma, f_beta };
    if (use_v) cols[2] = f_vega.?;

    const nf: f64 = @floatFromInt(n);
    const mean_r = meanOf(r);
    var means: [3]f64 = .{ 0, 0, 0 };
    var i: usize = 0;
    while (i < k) : (i += 1) means[i] = meanOf(cols[i]);
    var xtx: [3][3]f64 = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
    var xty: [3]f64 = .{ 0, 0, 0 };
    i = 0;
    while (i < k) : (i += 1) {
        // Centered factor against raw r: cov(f, r) * n, slopes match an intercept model.
        xty[i] = dot(cols[i], r) - nf * means[i] * mean_r;
        var j: usize = 0;
        while (j < k) : (j += 1) {
            xtx[i][j] = dot(cols[i], cols[j]) - nf * means[i] * means[j];
        }
        xtx[i][i] += RIDGE;
    }
    const coef = solveLinear(xtx, xty, k);
    var ss_res: f64 = 0.0;
    var sum_e: f64 = 0.0;
    for (r, 0..) |y, t| {
        var yhat = coef[0] * f_beta[t] + coef[1] * f_gamma[t];
        if (use_v) yhat += coef[2] * cols[2][t];
        const e = y - yhat;
        out_resid[t] = e;
        sum_e += e;
    }
    const mean_e = sum_e / nf;
    var ss_tot: f64 = 0.0;
    for (r, 0..) |y, t| {
        const d = y - mean_r;
        ss_tot += d * d;
        const er = out_resid[t] - mean_e;
        ss_res += er * er;
    }
    const r2: f64 = if (ss_tot > 1e-18) 1.0 - ss_res / ss_tot else 0.0;
    return .{
        .beta = coef[0],
        .gamma_coef = coef[1],
        .vega_coef = if (use_v) coef[2] else 0.0,
        .r2 = r2,
        .n_factors = k,
    };
}

pub fn pearson(a: []const f64, b: []const f64) f64 {
    if (a.len != b.len or a.len < 2) return 0.0;
    const n: f64 = @floatFromInt(a.len);
    var sa: f64 = 0.0;
    var sb: f64 = 0.0;
    for (a, b) |x, y| {
        sa += x;
        sb += y;
    }
    const ma = sa / n;
    const mb = sb / n;
    var cov: f64 = 0.0;
    var va: f64 = 0.0;
    var vb: f64 = 0.0;
    for (a, b) |x, y| {
        const dx = x - ma;
        const dy = y - mb;
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    if (va <= 1e-24 or vb <= 1e-24) return 0.0;
    return cov / (@sqrt(va) * @sqrt(vb));
}

fn meanStd(x: []const f64) struct { mean: f64, std: f64 } {
    if (x.len == 0) return .{ .mean = 0, .std = 1 };
    var s: f64 = 0;
    for (x) |v| s += v;
    const m = s / @as(f64, @floatFromInt(x.len));
    if (x.len < 2) return .{ .mean = m, .std = 1 };
    var ss: f64 = 0;
    for (x) |v| {
        const d = v - m;
        ss += d * d;
    }
    const variance = ss / @as(f64, @floatFromInt(x.len - 1));
    var stdv = @sqrt(@max(variance, 0));
    if (stdv < 1e-8) stdv = 1e-8;
    return .{ .mean = m, .std = stdv };
}

/// Inverse-vol weights on residual PnL.
///
/// 1. Sample standard deviation (n−1). Disabled sleeves stay at weight 0.
/// 2. Raw weight ∝ 1/σ.
/// 3. For each enabled pair with |ρ| > `corr_cap`, multiply both raw weights
///    by `corr_cap/|ρ|` (pairs compound).
/// 4. A sleeve with negative residual mean is cut to 0 when another enabled
///    sleeve has |ρ| > `corr_cap` and a strictly higher mean.
/// 5. Renormalize the survivors to sum to 1.
/// 6. Cap any weight at `max_weight` and push the excess onto uncapped
///    positive weights. If every survivor is capped, the leftover is cash
///    (weights sum to less than 1).
pub fn allocateInverseVol(
    resid: []const []const f64,
    enabled: []const bool,
    max_weight: f64,
    corr_cap: f64,
    out_w: []f64,
) void {
    const n = @min(resid.len, out_w.len);
    var inv: [MAX_SLEEVES]f64 = .{0} ** MAX_SLEEVES;
    var means: [MAX_SLEEVES]f64 = .{0} ** MAX_SLEEVES;
    var rho: [MAX_SLEEVES][MAX_SLEEVES]f64 = .{.{0} ** MAX_SLEEVES} ** MAX_SLEEVES;
    var on: [MAX_SLEEVES]bool = .{false} ** MAX_SLEEVES;
    const m = @min(n, MAX_SLEEVES);
    var i: usize = 0;
    while (i < n) : (i += 1) out_w[i] = 0;
    i = 0;
    while (i < m) : (i += 1) {
        on[i] = i < enabled.len and enabled[i];
        const ms = meanStd(resid[i]);
        means[i] = ms.mean;
        rho[i][i] = 1;
        if (on[i]) inv[i] = 1.0 / ms.std;
    }
    i = 0;
    while (i < m) : (i += 1) {
        var j: usize = i + 1;
        while (j < m) : (j += 1) {
            const p = pearson(resid[i], resid[j]);
            rho[i][j] = p;
            rho[j][i] = p;
        }
    }
    i = 0;
    while (i < m) : (i += 1) {
        var j: usize = i + 1;
        while (j < m) : (j += 1) {
            if (!on[i] or !on[j]) continue;
            const ar = @abs(rho[i][j]);
            if (ar > corr_cap and ar > 0.0) {
                const scale = corr_cap / ar;
                inv[i] *= scale;
                inv[j] *= scale;
            }
        }
    }
    i = 0;
    while (i < m) : (i += 1) {
        if (!on[i] or means[i] >= 0.0) continue;
        var j: usize = 0;
        while (j < m) : (j += 1) {
            if (i == j or !on[j]) continue;
            if (@abs(rho[i][j]) > corr_cap and means[j] > means[i]) {
                inv[i] = 0;
                break;
            }
        }
    }
    var sum: f64 = 0;
    i = 0;
    while (i < m) : (i += 1) sum += inv[i];
    if (!(sum > 0.0)) return;
    i = 0;
    while (i < m) : (i += 1) out_w[i] = inv[i] / sum;

    var capped: [MAX_SLEEVES]bool = .{false} ** MAX_SLEEVES;
    var iter: usize = 0;
    while (iter < MAX_SLEEVES) : (iter += 1) {
        var excess: f64 = 0;
        var free_sum: f64 = 0;
        i = 0;
        while (i < m) : (i += 1) {
            if (out_w[i] > max_weight + 1e-15) {
                excess += out_w[i] - max_weight;
                out_w[i] = max_weight;
                capped[i] = true;
            } else if (out_w[i] > 0.0 and !capped[i]) {
                free_sum += out_w[i];
            }
        }
        if (excess <= 1e-15 or !(free_sum > 0.0)) break;
        i = 0;
        while (i < m) : (i += 1) {
            if (out_w[i] > 0.0 and !capped[i]) out_w[i] += excess * (out_w[i] / free_sum);
        }
    }
}

test "gamma factor matches greek pnl bucket" {
    const g = types.Greeks{ .delta = 0.0, .gamma = 0.04, .vega = 1.0, .theta = 0.0 };
    const step = hedge.greekPnlStep(&g, 2.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    try std.testing.expectApproxEqAbs(gammaFactor(0.04, 2.0), step.gamma_pnl, 1e-12);
    try std.testing.expectApproxEqAbs(vegaFactor(1.0, 0.01), step.vega_pnl, 1e-12);
    try std.testing.expectApproxEqAbs(betaFactor(2.0, 100.0), 0.02, 1e-12);
}

test "strip recovers planted beta gamma vega and keeps the constant" {
    const fb = [_]f64{ 0.01, -0.02, 0.015, 0.0, -0.01, 0.008 };
    const fg = [_]f64{ 0.001, 0.004, 0.0002, 0.003, 0.0015, 0.0004 };
    const fv = [_]f64{ 0.10, -0.20, 0.0, 0.05, -0.04, 0.02 };
    var r: [6]f64 = undefined;
    var i: usize = 0;
    while (i < 6) : (i += 1) {
        r[i] = 0.5 * fb[i] + 1.0 * fg[i] + 0.25 * fv[i] + 0.01;
    }
    var resid: [6]f64 = undefined;
    const fit = stripResidual(&r, &fb, &fg, &fv, &resid);
    try std.testing.expectApproxEqAbs(0.5, fit.beta, 1e-5);
    try std.testing.expectApproxEqAbs(1.0, fit.gamma_coef, 1e-5);
    try std.testing.expectApproxEqAbs(0.25, fit.vega_coef, 1e-5);
    try std.testing.expect(fit.r2 > 0.99);
    var mean_e: f64 = 0;
    for (resid) |e| mean_e += e;
    mean_e /= 6.0;
    try std.testing.expectApproxEqAbs(0.01, mean_e, 1e-6);
    // Residual is the constant premium. Pearson of a flat series is noise.
    var max_dev: f64 = 0;
    for (resid) |e| max_dev = @max(max_dev, @abs(e - mean_e));
    try std.testing.expect(max_dev < 1e-6);
}

test "toxic sleeve allocator cuts the correlated loser and caps the rest" {
    const good = [_]f64{ 0.03, 0.04, 0.02, 0.05, 0.03, 0.04 };
    const toxic = [_]f64{ -0.04, -0.03, -0.05, -0.02, -0.04, -0.03 };
    try std.testing.expectApproxEqAbs(pearson(&good, &toxic), 1.0, 1e-12);
    const series = [_][]const f64{ &good, &toxic };
    const enabled = [_]bool{ true, true };
    var w: [2]f64 = .{ 0, 0 };
    allocateInverseVol(series[0..], &enabled, 0.40, 0.50, &w);
    try std.testing.expectApproxEqAbs(w[0], 0.40, 1e-12);
    try std.testing.expectApproxEqAbs(w[1], 0.0, 1e-12);
}

test "disabled sleeve stays at weight zero" {
    const a = [_]f64{ 0.02, 0.01, 0.03, 0.02 };
    const b = [_]f64{ 0.01, 0.02, 0.01, 0.015 };
    const series = [_][]const f64{ &a, &b };
    const enabled = [_]bool{ true, false };
    var w: [2]f64 = .{ -1, -1 };
    allocateInverseVol(series[0..], &enabled, 1.0, 0.50, &w);
    try std.testing.expectApproxEqAbs(w[0], 1.0, 1e-12);
    try std.testing.expectApproxEqAbs(w[1], 0.0, 1e-12);
}
