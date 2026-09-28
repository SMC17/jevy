//! Exponential Hawkes intensity for toxic order-flow bursts.
//!
//! λ(t) = μ + Σ_{t_i ≤ t} α exp(−β (t − t_i)).
//! Stationary when the branching ratio α/β < 1.
//! Excitation (λ−μ)/μ feeds Decision `flow.hawkes_*`. It never emits orders.
//!
//! Cite: Hawkes, Biometrika 58(1), 1971, https://doi.org/10.1093/biomet/58.1.83.
//! Finance survey: Bacry, Mastromatteo, Muzy, https://arxiv.org/abs/1502.04592.

const std = @import("std");

pub const HawkesParams = struct {
    mu: f64 = 1.0,
    alpha: f64 = 0.6,
    beta: f64 = 2.0,
};

pub fn intensity(p: HawkesParams, t: f64, events: []const f64) f64 {
    var lam = @max(p.mu, 0.0);
    const a = @max(p.alpha, 0.0);
    const b = p.beta;
    for (events) |ti| {
        if (ti <= t) {
            lam += a * @exp(-b * (t - ti));
        }
    }
    return lam;
}

/// Relative excitation (λ − μ) / μ. Zero when the process sits on its baseline.
pub fn excitation(p: HawkesParams, t: f64, events: []const f64) f64 {
    const mu = @max(p.mu, 1e-12);
    const lam = intensity(p, t, events);
    return @max(0.0, (lam - mu) / mu);
}

pub fn branchingRatio(p: HawkesParams) f64 {
    if (!(p.beta > 0.0)) return std.math.inf(f64);
    return @max(p.alpha, 0.0) / p.beta;
}

/// Scale a baseline fill intensity by excitation. Research fill model only.
pub fn fillIntensity(base: f64, exc: f64) f64 {
    return @max(base, 0.0) * (1.0 + @max(exc, 0.0));
}

test "burst raises intensity then the kernel decays" {
    const p = HawkesParams{ .mu = 1.0, .alpha = 0.8, .beta = 2.0 };
    const ev = [_]f64{ 0.0, 0.1 };
    const at_burst = intensity(p, 0.1, &ev);
    const later = intensity(p, 3.0, &ev);
    try std.testing.expect(at_burst > p.mu + p.alpha);
    try std.testing.expect(later < at_burst);
    try std.testing.expect(later > p.mu);
    try std.testing.expect(excitation(p, 0.1, &ev) > excitation(p, 3.0, &ev));
}

test "branching ratio and fill scale" {
    const p = HawkesParams{ .mu = 2.0, .alpha = 0.5, .beta = 2.0 };
    try std.testing.expectApproxEqAbs(branchingRatio(p), 0.25, 1e-12);
    try std.testing.expect(branchingRatio(p) < 1.0);
    try std.testing.expectApproxEqAbs(fillIntensity(4.0, 1.5), 10.0, 1e-12);
    const none = [_]f64{};
    try std.testing.expectApproxEqAbs(intensity(p, 1.0, &none), 2.0, 1e-12);
}
