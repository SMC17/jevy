//! Option-inventory market making under a constant-vega approximation.
//!
//! Primary cite: Baldacci, Bergault, Guéant, "Algorithmic market making for
//! options", https://arxiv.org/abs/1907.12433.
//! Portfolio vega is frozen per contract,
//!   V^π = Σ q_i V^i,   V^i = ∂_{√ν} O^i
//! and the risk-adjusted objective (their mean-variance reduction) is
//!   E[ ∫ (spread income + V^π (a_P − a_Q)/(2√ν)) dt − (γ ξ² / 8) ∫ (V^π)² dt ].
//! With the appendix's optimal underlying hedge the penalty picks up (1−ρ²).
//! Intensity is exponential Λ(δ)=A e^{−kδ} or the paper's logistic
//!   Λ(δ)=λ / (1 + exp(α + (β / V^i) δ)).
//! The value w(τ, V^π) is an explicit Euler grid in portfolio vega
//! (terminal w(0,·)=0). Quotes are the pointwise Hamiltonian argmax.
//!
//! Stoikov–Sağlam (2009), Review of Derivatives Research,
//! https://doi.org/10.1007/s11147-009-9036-3, Theorem 4 (linear intensity,
//! one-period, delta-hedged, stochastic implied vol) is exposed separately.
//!
//! Lucic–Tse vol view (SSRN 4729290): `iv_alpha` = theo IV − market IV shifts
//! the reservation by V^i · α. It does not replace the HJB premiums.
//!
//! Simulation / paper only. This module does not emit live orders.

const std = @import("std");
const types = @import("types.zig");

pub const MAX_GRID: usize = 65;
pub const MAX_NAMES: usize = 8;

pub const Intensity = enum(u8) {
    exponential = 0,
    logistic = 1,
};

pub const OptionMmConfig = struct {
    gamma: f64 = 0.5,
    xi: f64 = 1.0,
    A: f64 = 40.0,
    kappa: f64 = 2.0,
    lambda0: f64 = 100.0,
    alpha: f64 = 0.7,
    beta: f64 = 150.0,
    intensity: Intensity = .exponential,
    vega_limit: f64 = 40.0,
    horizon: f64 = 0.25,
    grid_n: u32 = 31,
    n_steps: u32 = 60,
    delta_inf: f64 = 0.0,
    /// (a_P − a_Q) / (2 √ν). Units: pnl per unit portfolio vega per year.
    vol_edge: f64 = 0.0,
    rho: f64 = 0.0,
    trade_size: f64 = 1.0,
    /// Theo IV minus market IV (decimal). Reservation += contract_vega * iv_alpha.
    iv_alpha: f64 = 0.0,
    min_premium: f64 = 0.0,
    max_premium: f64 = 50.0,
    quote_size: i32 = 1,
};

pub const OptionQuote = struct {
    bid: f64 = 0.0,
    ask: f64 = 0.0,
    bid_size: i32 = 0,
    ask_size: i32 = 0,
    reservation: f64 = 0.0,
    half_spread: f64 = 0.0,
    delta_b: f64 = 0.0,
    delta_a: f64 = 0.0,
    portfolio_vega: f64 = 0.0,
    bid_blocked: bool = false,
    ask_blocked: bool = false,
};

pub const ValueGrid = struct {
    n: usize = 0,
    vega_limit: f64 = 0.0,
    w: [MAX_GRID]f64 = [_]f64{0} ** MAX_GRID,
};

pub const StoikovPremiums = struct {
    eps_ask: f64 = 0.0,
    eps_bid: f64 = 0.0,
};

fn gridCount(cfg: *const OptionMmConfig) usize {
    var n: usize = @intCast(std.math.clamp(cfg.grid_n, 5, MAX_GRID));
    if (n % 2 == 0) n -= 1;
    return n;
}

/// Quadratic vega penalty coefficient γ ξ² (1−ρ²) / 8.
pub fn penaltyCoeff(cfg: *const OptionMmConfig) f64 {
    const rho = std.math.clamp(cfg.rho, -0.999999, 0.999999);
    return cfg.gamma * cfg.xi * cfg.xi * (1.0 - rho * rho) / 8.0;
}

pub fn exponentialIntensity(A: f64, k: f64, delta: f64) f64 {
    if (!(A > 0.0) or !(k > 0.0)) return 0.0;
    return A * @exp(-k * delta);
}

/// H(p) = sup_{δ≥δ∞} Λ(δ)(δ−p) for Λ(δ)=A e^{−kδ}.
pub fn exponentialHamiltonian(A: f64, k: f64, p: f64, delta_inf: f64) f64 {
    if (!(A > 0.0) or !(k > 0.0)) return 0.0;
    const d_star = p + 1.0 / k;
    if (d_star >= delta_inf) {
        return (A / k) * @exp(-1.0 - k * p);
    }
    const lam = exponentialIntensity(A, k, delta_inf);
    return lam * (delta_inf - p);
}

pub fn exponentialPremium(k: f64, p: f64, delta_inf: f64) f64 {
    if (!(k > 0.0)) return delta_inf;
    return @max(delta_inf, p + 1.0 / k);
}

/// Paper numerical intensity. Distance β δ / V^i is in vol units when V^i=∂_{√ν}O.
pub fn logisticIntensity(lambda: f64, alpha: f64, beta: f64, contract_vega: f64, delta: f64) f64 {
    if (!(lambda > 0.0)) return 0.0;
    const nu = @max(@abs(contract_vega), 1e-8);
    const theta = @max(beta, 1e-12) / nu;
    const u = alpha + theta * delta;
    return lambda / (1.0 + @exp(u));
}

pub fn logisticPremium(
    lambda: f64,
    alpha: f64,
    beta: f64,
    contract_vega: f64,
    p: f64,
    delta_inf: f64,
) f64 {
    _ = lambda;
    const nu = @max(@abs(contract_vega), 1e-8);
    const theta = @max(beta, 1e-12) / nu;
    var d = p + 1.0 / theta;
    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        const e = @exp(-(alpha + theta * d));
        const g = d - p - (1.0 / theta) * (1.0 + e);
        const gp = 1.0 + e;
        d -= g / gp;
    }
    return @max(delta_inf, d);
}

pub fn logisticHamiltonian(
    lambda: f64,
    alpha: f64,
    beta: f64,
    contract_vega: f64,
    p: f64,
    delta_inf: f64,
) f64 {
    const d = logisticPremium(lambda, alpha, beta, contract_vega, p, delta_inf);
    return logisticIntensity(lambda, alpha, beta, contract_vega, d) * (d - p);
}

fn interp(limit: f64, w: []const f64, x: f64) f64 {
    const n = w.len;
    if (n == 0) return 0.0;
    if (n == 1 or x <= -limit) return w[0];
    if (x >= limit) return w[n - 1];
    const d_v = (2.0 * limit) / @as(f64, @floatFromInt(n - 1));
    if (!(d_v > 0.0)) return w[0];
    const pos = (x + limit) / d_v;
    const i: usize = @intFromFloat(@floor(pos));
    if (i >= n - 1) return w[n - 1];
    const frac = pos - @as(f64, @floatFromInt(i));
    return w[i] * (1.0 - frac) + w[i + 1] * frac;
}

fn vAt(limit: f64, n: usize, j: usize) f64 {
    if (n <= 1) return 0.0;
    const d_v = (2.0 * limit) / @as(f64, @floatFromInt(n - 1));
    return -limit + d_v * @as(f64, @floatFromInt(j));
}

fn sidePremium(cfg: *const OptionMmConfig, contract_vega: f64, p: f64) f64 {
    return switch (cfg.intensity) {
        .logistic => logisticPremium(cfg.lambda0, cfg.alpha, cfg.beta, contract_vega, p, cfg.delta_inf),
        .exponential => exponentialPremium(cfg.kappa, p, cfg.delta_inf),
    };
}

fn sideH(cfg: *const OptionMmConfig, contract_vega: f64, p: f64) f64 {
    return switch (cfg.intensity) {
        .logistic => logisticHamiltonian(cfg.lambda0, cfg.alpha, cfg.beta, contract_vega, p, cfg.delta_inf),
        .exponential => exponentialHamiltonian(cfg.A, cfg.kappa, p, cfg.delta_inf),
    };
}

/// Solve w(τ, V) on [−V̄, V̄]. `vegas` are the constant per-contract vegas
/// of the names the maker quotes (each contributes a bid and an ask control).
pub fn solveGrid(cfg: *const OptionMmConfig, vegas: []const f64, out: *ValueGrid) void {
    const n = gridCount(cfg);
    const limit = @max(cfg.vega_limit, 1e-8);
    const steps: usize = @intCast(@max(cfg.n_steps, 1));
    const horizon = @max(cfg.horizon, 1e-8);
    const dtau = horizon / @as(f64, @floatFromInt(steps));
    const z = @max(cfg.trade_size, 1e-8);
    const pen = penaltyCoeff(cfg);
    out.n = n;
    out.vega_limit = limit;
    var w = [_]f64{0} ** MAX_GRID;
    var wn = [_]f64{0} ** MAX_GRID;
    var step: usize = 0;
    while (step < steps) : (step += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            const v = vAt(limit, n, j);
            var ham: f64 = 0.0;
            for (vegas) |nu| {
                if (nu == 0.0) continue;
                const buy = v + z * nu;
                if (@abs(buy) <= limit + 1e-9) {
                    const p_b = (w[j] - interp(limit, w[0..n], buy)) / z;
                    ham += z * sideH(cfg, nu, p_b);
                }
                const sell = v - z * nu;
                if (@abs(sell) <= limit + 1e-9) {
                    const p_a = (w[j] - interp(limit, w[0..n], sell)) / z;
                    ham += z * sideH(cfg, nu, p_a);
                }
            }
            const growth = ham + cfg.vol_edge * v - pen * v * v;
            wn[j] = w[j] + dtau * growth;
        }
        var c: usize = 0;
        while (c < n) : (c += 1) w[c] = wn[c];
    }
    var c: usize = 0;
    while (c < n) : (c += 1) out.w[c] = w[c];
}

pub fn valueAt(grid: *const ValueGrid, portfolio_vega: f64) f64 {
    if (grid.n == 0) return 0.0;
    return interp(grid.vega_limit, grid.w[0..grid.n], portfolio_vega);
}

/// Premiums and prices for one name given portfolio vega and that name's vega.
pub fn quoteOnGrid(
    cfg: *const OptionMmConfig,
    grid: *const ValueGrid,
    mid_in: f64,
    portfolio_vega: f64,
    contract_vega: f64,
    spread_mult: f64,
    size_mult: f64,
) OptionQuote {
    var mid = mid_in;
    if (mid <= 0.0) mid = @max(mid, 0.01);
    const limit = if (grid.vega_limit > 0.0) grid.vega_limit else @max(cfg.vega_limit, 1e-8);
    const z = @max(cfg.trade_size, 1e-8);
    const nu = if (contract_vega == 0.0) 1e-8 else contract_vega;
    const v = std.math.clamp(portfolio_vega, -limit, limit);
    const w_here = valueAt(grid, v);
    const buy = v + z * nu;
    const sell = v - z * nu;
    const bid_blocked = @abs(buy) > limit + 1e-8;
    const ask_blocked = @abs(sell) > limit + 1e-8;
    var p_b: f64 = 0.0;
    var p_a: f64 = 0.0;
    if (!bid_blocked) p_b = (w_here - valueAt(grid, buy)) / z;
    if (!ask_blocked) p_a = (w_here - valueAt(grid, sell)) / z;
    var d_b = if (bid_blocked) cfg.max_premium else sidePremium(cfg, nu, p_b);
    var d_a = if (ask_blocked) cfg.max_premium else sidePremium(cfg, nu, p_a);
    const sm = @max(spread_mult, 0.0);
    d_b *= sm;
    d_a *= sm;
    d_b = std.math.clamp(d_b, cfg.min_premium, cfg.max_premium);
    d_a = std.math.clamp(d_a, cfg.min_premium, cfg.max_premium);
    const shift = nu * cfg.iv_alpha;
    const center = mid + shift + 0.5 * (d_a - d_b);
    const half = 0.5 * (d_a + d_b);
    const bid = @max(0.01, mid + shift - d_b);
    const ask = @max(bid + 0.01, mid + shift + d_a);
    var size: i32 = @intFromFloat(@round(@as(f64, @floatFromInt(cfg.quote_size)) * @max(size_mult, 0.0)));
    if (size < 0) size = 0;
    if (size_mult > 0.0 and size < 1 and !bid_blocked and !ask_blocked) size = 1;
    return .{
        .bid = bid,
        .ask = ask,
        .bid_size = if (bid_blocked or size_mult <= 0.0) 0 else size,
        .ask_size = if (ask_blocked or size_mult <= 0.0) 0 else size,
        .reservation = center,
        .half_spread = half,
        .delta_b = d_b,
        .delta_a = d_a,
        .portfolio_vega = v,
        .bid_blocked = bid_blocked,
        .ask_blocked = ask_blocked,
    };
}

pub fn solveAndQuote(
    cfg: *const OptionMmConfig,
    mid: f64,
    portfolio_vega: f64,
    contract_vega: f64,
    spread_mult: f64,
    size_mult: f64,
) OptionQuote {
    var grid: ValueGrid = .{};
    const vegas = [_]f64{contract_vega};
    solveGrid(cfg, &vegas, &grid);
    return quoteOnGrid(cfg, &grid, mid, portfolio_vega, contract_vega, spread_mult, size_mult);
}

/// Map the shared quoter config (`mode = option_vega`) onto this module.
/// Portfolio vega = inventory × per-contract vega (constant-vega approx).
pub fn quoteFromQuoter(
    mid: f64,
    inventory: i32,
    cfg: *const types.QuoterConfig,
    greeks_opt: ?*const types.Greeks,
    spread_mult: f64,
    size_mult: f64,
) types.Quote {
    const per = if (greeks_opt) |g| (if (g.vega != 0.0) g.vega else cfg.contract_vega) else cfg.contract_vega;
    const ocfg = OptionMmConfig{
        .gamma = cfg.gamma,
        .xi = cfg.xi,
        .A = cfg.A,
        .kappa = cfg.kappa,
        .lambda0 = cfg.logistic_lambda,
        .alpha = cfg.logistic_alpha,
        .beta = cfg.logistic_beta,
        .intensity = if (cfg.intensity_kind == 1) .logistic else .exponential,
        .vega_limit = cfg.vega_limit,
        .horizon = cfg.t_horizon,
        .grid_n = cfg.option_grid_n,
        .n_steps = cfg.option_grid_steps,
        .vol_edge = cfg.vol_edge,
        .rho = cfg.option_rho,
        .iv_alpha = cfg.iv_alpha,
        .min_premium = cfg.min_half_spread,
        .max_premium = cfg.max_half_spread,
        .quote_size = cfg.quote_size,
    };
    const v_pi = @as(f64, @floatFromInt(inventory)) * per;
    const q = solveAndQuote(&ocfg, mid, v_pi, per, spread_mult, size_mult);
    return .{
        .bid = q.bid,
        .ask = q.ask,
        .bid_size = q.bid_size,
        .ask_size = q.ask_size,
        .reservation = q.reservation,
        .half_spread = q.half_spread,
    };
}

/// Theorem 4, Stoikov–Sağlam. Linear intensity λ(ε)=C−Dε on (0, C/D).
/// Premiums are distances from the option mid (ask premium above, bid below).
pub fn stoikovSaglamPremiums(gamma: f64, q: f64, C: f64, D: f64, k_risk: f64) StoikovPremiums {
    if (!(D > 0.0)) return .{};
    const cap = C / D;
    const rev = C / (2.0 * D);
    const tilt = gamma * k_risk;
    return .{
        .eps_ask = std.math.clamp(rev - tilt * (q - 0.5), 0.0, cap),
        .eps_bid = std.math.clamp(rev + tilt * (q + 0.5), 0.0, cap),
    };
}

/// Printed grouping of the Theorem 4 risk scale k (overnight variance × Γ² S⁴ σ²).
pub fn stoikovRiskScale(
    sigma: f64,
    overnight: f64,
    alpha: f64,
    t_mat: f64,
    gamma_greek: f64,
    spot: f64,
) f64 {
    const disc = 0.5 * sigma * sigma * overnight + alpha * alpha * t_mat * t_mat;
    const s2 = spot * spot;
    return disc * gamma_greek * gamma_greek * s2 * s2 * sigma * sigma * overnight;
}

/// Research toy used by unit tests (scaled so the vega skew is visible).
/// Not the euro-notional grid in Baldacci §4 (V̄ = 10^7, γ = 10^{−3}).
pub fn researchToy() OptionMmConfig {
    return .{
        .gamma = 0.5,
        .xi = 1.0,
        .A = 40.0,
        .kappa = 2.0,
        .vega_limit = 40.0,
        .horizon = 0.25,
        .grid_n = 31,
        .n_steps = 60,
        .intensity = .exponential,
        .trade_size = 1.0,
        .delta_inf = 0.0,
        .vol_edge = 0.0,
        .rho = 0.0,
    };
}

test "exponential hamiltonian at flat indifference is A/k * exp(-1)" {
    const h = exponentialHamiltonian(40.0, 2.0, 0.0, 0.0);
    try std.testing.expectApproxEqAbs(h, 20.0 / @exp(1.0), 1e-12);
    try std.testing.expectApproxEqAbs(exponentialPremium(2.0, 0.0, 0.0), 0.5, 1e-12);
}

test "logistic touch intensity matches Baldacci 1/(1+e^alpha)" {
    // λ fraction at δ=0 is 1/(1+e^α). Paper: α=0.7 ⇒ about 33%.
    const frac = logisticIntensity(1.0, 0.7, 150.0, 10.0, 0.0);
    try std.testing.expectApproxEqAbs(frac, 1.0 / (1.0 + @exp(0.7)), 1e-12);
    const richer = logisticIntensity(1.0, 0.7, 150.0, 10.0, -0.1);
    try std.testing.expect(richer > frac);
}

test "long portfolio vega widens the bid and tightens the ask" {
    const cfg = researchToy();
    const nu: f64 = 5.0;
    const flat = solveAndQuote(&cfg, 10.0, 0.0, nu, 1.0, 1.0);
    const long = solveAndQuote(&cfg, 10.0, 20.0, nu, 1.0, 1.0);
    const short = solveAndQuote(&cfg, 10.0, -20.0, nu, 1.0, 1.0);
    try std.testing.expectApproxEqAbs(flat.delta_b, flat.delta_a, 1e-6);
    try std.testing.expect(long.delta_b > long.delta_a);
    try std.testing.expect(short.delta_a > short.delta_b);
    try std.testing.expect(long.reservation < flat.reservation);
    try std.testing.expect(short.reservation > flat.reservation);
    try std.testing.expect(long.bid < short.bid);
    try std.testing.expect(long.ask < short.ask);
}

test "vega limit blocks the side that would breach Vbar" {
    var cfg = researchToy();
    cfg.vega_limit = 12.0;
    const q = solveAndQuote(&cfg, 10.0, 10.0, 5.0, 1.0, 1.0);
    // buy would take V to 15 > 12
    try std.testing.expect(q.bid_blocked);
    try std.testing.expect(q.bid_size == 0);
    try std.testing.expect(!q.ask_blocked);
    try std.testing.expect(q.ask_size > 0);
}

test "positive vol edge at flat vega prefers to buy" {
    var cfg = researchToy();
    cfg.vol_edge = 4.0;
    const q = solveAndQuote(&cfg, 10.0, 0.0, 5.0, 1.0, 1.0);
    try std.testing.expect(q.delta_b < q.delta_a);
}

test "iv alpha lifts the reservation" {
    var cfg = researchToy();
    const base = solveAndQuote(&cfg, 10.0, 0.0, 5.0, 1.0, 1.0);
    cfg.iv_alpha = 0.02;
    const rich = solveAndQuote(&cfg, 10.0, 0.0, 5.0, 1.0, 1.0);
    try std.testing.expectApproxEqAbs(rich.reservation - base.reservation, 5.0 * 0.02, 1e-8);
    try std.testing.expect(rich.bid > base.bid and rich.ask > base.ask);
}

test "spot-vol hedge factor (1-rho^2) shrinks the vega penalty" {
    var a = researchToy();
    var b = researchToy();
    b.rho = 0.6;
    try std.testing.expect(penaltyCoeff(&b) < penaltyCoeff(&a));
    try std.testing.expectApproxEqAbs(penaltyCoeff(&a), 0.5 * 1.0 / 8.0, 1e-12);
}

test "stoikov-saglam risk-neutral premiums are C/(2D) and tilt with inventory" {
    const flat = stoikovSaglamPremiums(0.0, 5.0, 40.0, 200.0, 1.0);
    try std.testing.expectApproxEqAbs(flat.eps_ask, 0.1, 1e-12);
    try std.testing.expectApproxEqAbs(flat.eps_bid, 0.1, 1e-12);
    const zero_q = stoikovSaglamPremiums(0.1, 0.0, 40.0, 200.0, 1.0);
    try std.testing.expectApproxEqAbs(zero_q.eps_ask, 0.15, 1e-12);
    try std.testing.expectApproxEqAbs(zero_q.eps_bid, 0.15, 1e-12);
    const long = stoikovSaglamPremiums(0.1, 5.0, 40.0, 200.0, 1.0);
    try std.testing.expect(long.eps_ask < long.eps_bid);
    try std.testing.expectApproxEqAbs(long.eps_ask, 0.0, 1e-12);
    try std.testing.expectApproxEqAbs(long.eps_bid, 0.2, 1e-12);
}
