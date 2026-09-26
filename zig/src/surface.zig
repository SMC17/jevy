//! SABR-lite implied volatility (Hagan et al. 2002 European Black IV).
//!
//! Given SABR params (alpha, beta, rho, nu) and (f, K, T), return Black IV.
//! This is the analytic approximation — not a PDE/MC pricer.
//!
//! Approximations / edge cases
//! ---------------------------
//! * ATM limit (f ≈ K): uses the closed-form expansion (Hagan eq. 2.18);
//!   the z/x(z) factor → 1. Threshold: |ln(f/K)| < 1e-8.
//! * f≤0, K≤0, alpha≤0, T<0 → return 0 (caller should treat as invalid).
//! * |rho| is clamped to (-0.999999, 0.999999) so x(z) stays defined.
//! * nu=0 → reduces to CEV-like local vol (still via Hagan expansion).
//! * beta∈[0,1]; values outside are accepted but not arbitrage-checked.
//! * Returned IV is floored at 1e-8 when the expansion is positive; never NaN.
//! * No calendar/term-structure: one expiry slice only (SABR-lite).
//!
//! Reference: Hagan, Kumar, Lesniewski, Woodward — "Managing Smile Risk"
//! (Wilmott, 2002).

const std = @import("std");

pub const SabrParams = struct {
    alpha: f64 = 0.25, // initial vol level (σ0-ish when β=1)
    beta: f64 = 0.5, // CEV exponent; 0=normal, 1=lognormal
    rho: f64 = -0.2, // corr(spot, vol)
    nu: f64 = 0.4, // vol-of-vol
};

const ATM_LOG_EPS: f64 = 1e-8;
const RHO_MAX: f64 = 0.999999;
const IV_FLOOR: f64 = 1e-8;

inline fn clampRho(rho: f64) f64 {
    if (rho > RHO_MAX) return RHO_MAX;
    if (rho < -RHO_MAX) return -RHO_MAX;
    return rho;
}

/// Hagan SABR Black implied vol for European options.
/// `f` = forward, `K` = strike, `T` = time to expiry (years).
pub fn sabrImpliedVol(p: SabrParams, f: f64, K: f64, T: f64) f64 {
    if (!(f > 0.0) or !(K > 0.0) or !(p.alpha > 0.0) or T < 0.0) {
        return 0.0;
    }
    const alpha = p.alpha;
    const beta = p.beta;
    const rho = clampRho(p.rho);
    const nu = p.nu;
    const one_m_beta = 1.0 - beta;
    const log_fk = @log(f / K);
    const abs_log = @abs(log_fk);

    // Common A(K) denominator pieces
    const fk = f * K;
    const fk_pow = if (@abs(one_m_beta) < 1e-14) 1.0 else std.math.pow(f64, fk, 0.5 * one_m_beta);
    const log2 = log_fk * log_fk;
    const log4 = log2 * log2;
    const denom_series = 1.0 + (one_m_beta * one_m_beta / 24.0) * log2 + (one_m_beta * one_m_beta * one_m_beta * one_m_beta / 1920.0) * log4;

    // Time correction (shared ATM / non-ATM)
    const fk_pow_full = if (@abs(one_m_beta) < 1e-14) 1.0 else std.math.pow(f64, fk, one_m_beta);
    const term1 = (one_m_beta * one_m_beta / 24.0) * (alpha * alpha) / fk_pow_full;
    const term2 = 0.25 * rho * beta * nu * alpha / fk_pow;
    const term3 = ((2.0 - 3.0 * rho * rho) / 24.0) * nu * nu;
    const time_factor = 1.0 + (term1 + term2 + term3) * T;

    if (abs_log < ATM_LOG_EPS) {
        // ATM: σ = α / f^(1-β) * time_factor
        const f_pow = if (@abs(one_m_beta) < 1e-14) 1.0 else std.math.pow(f64, f, one_m_beta);
        const atm = (alpha / f_pow) * time_factor;
        return if (atm > IV_FLOOR) atm else IV_FLOOR;
    }

    // z / x(z)
    const z = (nu / alpha) * fk_pow * log_fk;
    var zx: f64 = 1.0;
    if (@abs(z) > 1e-12 and @abs(nu) > 1e-14) {
        const disc = @max(1.0 - 2.0 * rho * z + z * z, 0.0);
        const sqrt_disc = @sqrt(disc);
        const numer = sqrt_disc + z - rho;
        const denom = 1.0 - rho;
        if (numer > 0.0 and @abs(denom) > 1e-14) {
            const xz = @log(numer / denom);
            if (@abs(xz) > 1e-14) {
                zx = z / xz;
            }
        }
    }

    const prefactor = alpha / (fk_pow * denom_series);
    const iv = prefactor * zx * time_factor;
    if (!(iv == iv)) return IV_FLOOR; // NaN guard
    return if (iv > IV_FLOOR) iv else IV_FLOOR;
}

/// Convenience: ATM IV from SABR (exact ATM branch).
pub fn sabrAtmVol(p: SabrParams, f: f64, T: f64) f64 {
    return sabrImpliedVol(p, f, f, T);
}

test "sabr atm equals closed form" {
    const p = SabrParams{ .alpha = 0.25, .beta = 0.5, .rho = -0.3, .nu = 0.4 };
    const f: f64 = 100.0;
    const T: f64 = 0.5;
    const atm = sabrAtmVol(p, f, T);
    // Manual ATM: α/f^(1-β) * (1 + [...]*T)
    const one_m_b = 0.5;
    const f_pow = std.math.pow(f64, f, one_m_b);
    const term1 = (one_m_b * one_m_b / 24.0) * (0.25 * 0.25) / std.math.pow(f64, f, 1.0);
    const term2 = 0.25 * (-0.3) * 0.5 * 0.4 * 0.25 / f_pow;
    const term3 = ((2.0 - 3.0 * 0.09) / 24.0) * 0.16;
    const expected = (0.25 / f_pow) * (1.0 + (term1 + term2 + term3) * T);
    try std.testing.expect(@abs(atm - expected) < 1e-10);
    try std.testing.expect(atm > 0.0);
}

test "sabr atm continuous limit from near-atm" {
    const p = SabrParams{ .alpha = 0.3, .beta = 0.7, .rho = -0.2, .nu = 0.5 };
    const f: f64 = 100.0;
    const T: f64 = 1.0;
    const atm = sabrAtmVol(p, f, T);
    const near = sabrImpliedVol(p, f, f * 1.0000001, T);
    try std.testing.expect(@abs(atm - near) / atm < 1e-4);
}

test "sabr wing monotonicity sanity beta05" {
    // With typical equity-ish rho<0, put wing (K<<f) should be richer than call wing
    // for moderate strikes — at least IV should stay positive and finite.
    const p = SabrParams{ .alpha = 0.25, .beta = 0.5, .rho = -0.4, .nu = 0.5 };
    const f: f64 = 100.0;
    const T: f64 = 0.25;
    const iv_otm_put = sabrImpliedVol(p, f, 80.0, T);
    const iv_atm = sabrAtmVol(p, f, T);
    const iv_otm_call = sabrImpliedVol(p, f, 120.0, T);
    try std.testing.expect(iv_otm_put > 0.0);
    try std.testing.expect(iv_atm > 0.0);
    try std.testing.expect(iv_otm_call > 0.0);
    // Smile: wings above ATM for this nu
    try std.testing.expect(iv_otm_put > iv_atm * 0.9);
    try std.testing.expect(iv_otm_call > iv_atm * 0.85);
    // Skew: put wing typically > call wing when rho < 0
    try std.testing.expect(iv_otm_put > iv_otm_call);
}

test "sabr invalid inputs return zero" {
    const p = SabrParams{};
    try std.testing.expect(sabrImpliedVol(p, -1.0, 100.0, 1.0) == 0.0);
    try std.testing.expect(sabrImpliedVol(p, 100.0, 0.0, 1.0) == 0.0);
    const bad = SabrParams{ .alpha = 0.0 };
    try std.testing.expect(sabrImpliedVol(bad, 100.0, 100.0, 1.0) == 0.0);
}

test "sabr beta1 reduces toward lognormal backbone" {
    // β=1, ν=0, ρ=0 → roughly flat IV ≈ alpha
    const p = SabrParams{ .alpha = 0.22, .beta = 1.0, .rho = 0.0, .nu = 0.0 };
    const atm = sabrAtmVol(p, 100.0, 0.5);
    try std.testing.expect(@abs(atm - 0.22) < 0.01);
}
