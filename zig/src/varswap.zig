//! Stylized variance- and vol-swap replication.
//!
//! Fair variance (r discounted, OTM strip), Demeterfi–Derman–Kamal–Zou,
//! https://emanuelderman.com/wp-content/uploads/1999/02/gs-volatility_swaps.pdf :
//!   K_var = (2 / T) e^{rT} ∫ OTM(K) / K² dK
//! with OTM = put below the forward and call above.
//!
//! Flat-smile greeks: ∂(σ²)/∂σ = 2σ. Vol-swap strike from a variance strike
//! uses the second-order sqrt convexity
//!   E[√X] ≈ √(E X) (1 − Var(X) / (8 (E X)²)),
//! the adjustment in Bossu–Strasser–Guichard,
//! http://docs.sbossu.com/bossu-strasser-guichard-varswap.pdf.
//!
//! Research weights only — not a listed-variance execution model.

const std = @import("std");

/// Trapezoid replication. `strikes` must be strictly increasing and line up
/// with `otm` (put if K<forward, call if K>forward; ATM either).
pub fn fairVariance(
    strikes: []const f64,
    otm: []const f64,
    t: f64,
    df: f64,
) f64 {
    if (strikes.len < 2 or strikes.len != otm.len or !(t > 0.0)) return 0.0;
    var acc: f64 = 0.0;
    var i: usize = 0;
    while (i + 1 < strikes.len) : (i += 1) {
        const k0 = strikes[i];
        const k1 = strikes[i + 1];
        if (!(k0 > 0.0) or !(k1 > k0)) continue;
        const y0 = otm[i] / (k0 * k0);
        const y1 = otm[i + 1] / (k1 * k1);
        acc += 0.5 * (y0 + y1) * (k1 - k0);
    }
    const disc = if (df > 0.0) df else 1.0;
    return (2.0 / t) * disc * acc;
}

pub fn varianceSwapVega(sigma: f64) f64 {
    return 2.0 * sigma;
}

/// Vol strike ≈ √K_var × (1 − var(variance) / (8 K_var²)).
pub fn volSwapFromVariance(k_var: f64, var_of_var: f64) f64 {
    if (!(k_var > 0.0)) return 0.0;
    const conv = @max(var_of_var, 0.0) / (8.0 * k_var * k_var);
    const adj = @max(0.0, 1.0 - conv);
    return @sqrt(k_var) * adj;
}

test "flat zero prices replicate zero variance" {
    const k = [_]f64{ 80.0, 100.0, 120.0 };
    const p = [_]f64{ 0.0, 0.0, 0.0 };
    try std.testing.expectApproxEqAbs(fairVariance(&k, &p, 1.0, 1.0), 0.0, 1e-15);
}

test "trapezoid of a known strip" {
    // Two panels, OTM = K² so OTM/K² = 1, integral of 1 from 80 to 120 = 40.
    // K_var = 2/T * 40 = 80 when T=1, df=1.
    const k = [_]f64{ 80.0, 100.0, 120.0 };
    const p = [_]f64{ 80.0 * 80.0, 100.0 * 100.0, 120.0 * 120.0 };
    try std.testing.expectApproxEqAbs(fairVariance(&k, &p, 1.0, 1.0), 80.0, 1e-9);
}

test "stylized vega and vol convexity" {
    try std.testing.expectApproxEqAbs(varianceSwapVega(0.2), 0.4, 1e-15);
    try std.testing.expectApproxEqAbs(volSwapFromVariance(0.04, 0.0), 0.2, 1e-12);
    const shaved = volSwapFromVariance(0.04, 0.000032);
    // var/(8 K²) = 0.000032 / (8 * 0.0016) = 0.000032 / 0.0128 = 0.0025
    try std.testing.expectApproxEqAbs(shaved, 0.2 * 0.9975, 1e-12);
    try std.testing.expect(shaved < 0.2);
}
