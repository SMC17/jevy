//! Scenario risk matrix (Akuna Options 201 — analyze risk).
//!
//! Shock grid over spot moves × IV moves → portfolio PnL approx using greeks
//! (Taylor) or optional full reprice when cheap. Emits into risk snapshot /
//! demo summary; soft vs hard limit hooks.
//!
//! PnL ≈ Δ·dS + ½ Γ (dS)² + ν·dσ + θ·dt   (dt usually 0 for instantaneous shocks)
//!
//! Soft limit: flag when any cell |PnL| exceeds soft_loss.
//! Hard limit: flag when any cell |PnL| exceeds hard_loss (caller may pull quotes).

const std = @import("std");
const types = @import("types.zig");
const bs = @import("black_scholes.zig");
const Greeks = types.Greeks;

pub const MAX_SPOT_SHOCKS: usize = 9;
pub const MAX_IV_SHOCKS: usize = 7;

pub const ScenarioConfig = struct {
    /// Spot relative shocks (e.g. -0.05 = −5%).
    spot_shocks: [MAX_SPOT_SHOCKS]f64 = .{ -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.0, 0.0 },
    n_spot: usize = 7,
    /// Absolute IV shocks (e.g. -0.05 = −5 vol points).
    iv_shocks: [MAX_IV_SHOCKS]f64 = .{ -0.05, -0.02, 0.0, 0.02, 0.05, 0.0, 0.0 },
    n_iv: usize = 5,
    soft_loss: f64 = 100.0,
    hard_loss: f64 = 400.0,
    /// If true, use BS reprice for each cell (slower, more accurate near large moves).
    reprice: bool = false,
};

pub const Cell = struct {
    d_spot_frac: f64 = 0.0,
    d_iv: f64 = 0.0,
    pnl: f64 = 0.0,
};

pub const ScenarioMatrix = struct {
    cells: [MAX_SPOT_SHOCKS * MAX_IV_SHOCKS]Cell = [_]Cell{.{}} ** (MAX_SPOT_SHOCKS * MAX_IV_SHOCKS),
    n_spot: usize = 0,
    n_iv: usize = 0,
    min_pnl: f64 = 0.0,
    max_pnl: f64 = 0.0,
    soft_breach: bool = false,
    hard_breach: bool = false,
    worst_i: usize = 0,
    worst_j: usize = 0,

    pub fn at(self: *const ScenarioMatrix, i: usize, j: usize) Cell {
        return self.cells[i * self.n_iv + j];
    }
};

/// Instantaneous greek Taylor PnL for shocks dS (absolute), dσ (absolute vol).
pub fn taylorPnl(g: *const Greeks, d_spot: f64, d_iv: f64) f64 {
    return g.delta * d_spot + 0.5 * g.gamma * d_spot * d_spot + g.vega * d_iv;
}

/// Build scenario matrix for a portfolio already reduced to aggregate greeks
/// (qty folded in). `spot` is current underlier; shocks are fractional.
pub fn buildMatrix(g: *const Greeks, spot: f64, cfg: *const ScenarioConfig) ScenarioMatrix {
    var out: ScenarioMatrix = .{};
    out.n_spot = @min(cfg.n_spot, MAX_SPOT_SHOCKS);
    out.n_iv = @min(cfg.n_iv, MAX_IV_SHOCKS);
    out.min_pnl = std.math.inf(f64);
    out.max_pnl = -std.math.inf(f64);

    var i: usize = 0;
    while (i < out.n_spot) : (i += 1) {
        const ds_frac = cfg.spot_shocks[i];
        const d_spot = spot * ds_frac;
        var j: usize = 0;
        while (j < out.n_iv) : (j += 1) {
            const d_iv = cfg.iv_shocks[j];
            const pnl = taylorPnl(g, d_spot, d_iv);
            const idx = i * out.n_iv + j;
            out.cells[idx] = .{ .d_spot_frac = ds_frac, .d_iv = d_iv, .pnl = pnl };
            if (pnl < out.min_pnl) {
                out.min_pnl = pnl;
                out.worst_i = i;
                out.worst_j = j;
            }
            if (pnl > out.max_pnl) out.max_pnl = pnl;
            if (@abs(pnl) >= cfg.soft_loss) out.soft_breach = true;
            if (@abs(pnl) >= cfg.hard_loss) out.hard_breach = true;
        }
    }
    return out;
}

/// Single-option reprice matrix (more accurate for large shocks).
/// `qty` signed inventory; is_call selects BS flavour.
pub fn buildMatrixReprice(
    spot: f64,
    strike: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv: f64,
    is_call: bool,
    qty: f64,
    cfg: *const ScenarioConfig,
) ScenarioMatrix {
    var out: ScenarioMatrix = .{};
    out.n_spot = @min(cfg.n_spot, MAX_SPOT_SHOCKS);
    out.n_iv = @min(cfg.n_iv, MAX_IV_SHOCKS);
    out.min_pnl = std.math.inf(f64);
    out.max_pnl = -std.math.inf(f64);
    const base = bs.price(spot, strike, t, rate, div_yield, iv, is_call);

    var i: usize = 0;
    while (i < out.n_spot) : (i += 1) {
        const ds_frac = cfg.spot_shocks[i];
        const s2 = spot * (1.0 + ds_frac);
        var j: usize = 0;
        while (j < out.n_iv) : (j += 1) {
            const d_iv = cfg.iv_shocks[j];
            const iv2 = @max(iv + d_iv, 1e-6);
            const px = bs.price(s2, strike, t, rate, div_yield, iv2, is_call);
            const pnl = qty * (px - base);
            const idx = i * out.n_iv + j;
            out.cells[idx] = .{ .d_spot_frac = ds_frac, .d_iv = d_iv, .pnl = pnl };
            if (pnl < out.min_pnl) {
                out.min_pnl = pnl;
                out.worst_i = i;
                out.worst_j = j;
            }
            if (pnl > out.max_pnl) out.max_pnl = pnl;
            if (@abs(pnl) >= cfg.soft_loss) out.soft_breach = true;
            if (@abs(pnl) >= cfg.hard_loss) out.hard_breach = true;
        }
    }
    return out;
}

/// Default desk-ish shock grid.
pub fn defaultConfig() ScenarioConfig {
    return .{};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "taylor flat book is flat matrix" {
    const g = Greeks{};
    const cfg = defaultConfig();
    const m = buildMatrix(&g, 100.0, &cfg);
    try std.testing.expect(@abs(m.min_pnl) < 1e-12);
    try std.testing.expect(@abs(m.max_pnl) < 1e-12);
    try std.testing.expect(!m.soft_breach);
}

test "long delta: down-spot cells negative" {
    const g = Greeks{ .delta = 10.0, .gamma = 0.0, .vega = 0.0, .theta = 0.0 };
    const cfg = ScenarioConfig{
        .spot_shocks = .{ -0.05, 0.0, 0.05, 0, 0, 0, 0, 0, 0 },
        .n_spot = 3,
        .iv_shocks = .{ 0.0, 0, 0, 0, 0, 0, 0 },
        .n_iv = 1,
        .soft_loss = 1000.0,
        .hard_loss = 5000.0,
    };
    const m = buildMatrix(&g, 100.0, &cfg);
    // dS = -5 → pnl = -50
    try std.testing.expect(m.at(0, 0).pnl < 0.0);
    try std.testing.expect(m.at(2, 0).pnl > 0.0);
    try std.testing.expect(@abs(m.at(0, 0).pnl - (-50.0)) < 1e-9);
}

test "long vega: +iv cells positive" {
    const g = Greeks{ .delta = 0.0, .gamma = 0.0, .vega = 50.0, .theta = 0.0 };
    const cfg = ScenarioConfig{
        .spot_shocks = .{ 0.0, 0, 0, 0, 0, 0, 0, 0, 0 },
        .n_spot = 1,
        .iv_shocks = .{ -0.05, 0.05, 0, 0, 0, 0, 0 },
        .n_iv = 2,
        .soft_loss = 1.0,
        .hard_loss = 10.0,
    };
    const m = buildMatrix(&g, 100.0, &cfg);
    try std.testing.expect(m.at(0, 0).pnl < 0.0);
    try std.testing.expect(m.at(0, 1).pnl > 0.0);
    try std.testing.expect(m.soft_breach); // |−2.5| and |+2.5| > 1
}

test "gamma convexity: up and down both positive for long gamma" {
    const g = Greeks{ .delta = 0.0, .gamma = 0.1, .vega = 0.0, .theta = 0.0 };
    const cfg = ScenarioConfig{
        .spot_shocks = .{ -0.02, 0.02, 0, 0, 0, 0, 0, 0, 0 },
        .n_spot = 2,
        .iv_shocks = .{ 0.0, 0, 0, 0, 0, 0, 0 },
        .n_iv = 1,
        .soft_loss = 1e9,
        .hard_loss = 1e9,
    };
    const m = buildMatrix(&g, 100.0, &cfg);
    // 0.5 * 0.1 * (2)^2 = 0.2
    try std.testing.expect(m.at(0, 0).pnl > 0.0);
    try std.testing.expect(m.at(1, 0).pnl > 0.0);
    try std.testing.expect(@abs(m.at(0, 0).pnl - 0.2) < 1e-9);
}

test "reprice matrix runs and worst cell tracked" {
    const cfg = ScenarioConfig{
        .spot_shocks = .{ -0.1, 0.0, 0.1, 0, 0, 0, 0, 0, 0 },
        .n_spot = 3,
        .iv_shocks = .{ 0.0, 0.05, 0, 0, 0, 0, 0 },
        .n_iv = 2,
        .soft_loss = 1.0,
        .hard_loss = 50.0,
    };
    // Short 10 calls — downside less bad than upside for short calls? short call loses on up-spot
    const m = buildMatrixReprice(100.0, 100.0, 0.25, 0.05, 0.0, 0.2, true, -10.0, &cfg);
    try std.testing.expect(m.min_pnl < 0.0);
    try std.testing.expect(m.n_spot == 3 and m.n_iv == 2);
}

test "hard breach hook" {
    const g = Greeks{ .delta = 100.0, .gamma = 0, .vega = 0, .theta = 0 };
    const cfg = ScenarioConfig{
        .spot_shocks = .{ -0.05, 0, 0, 0, 0, 0, 0, 0, 0 },
        .n_spot = 1,
        .iv_shocks = .{ 0.0, 0, 0, 0, 0, 0, 0 },
        .n_iv = 1,
        .soft_loss = 10.0,
        .hard_loss = 100.0,
    };
    // dS=-5, pnl=-500
    const m = buildMatrix(&g, 100.0, &cfg);
    try std.testing.expect(m.hard_breach);
    try std.testing.expect(m.soft_breach);
}
