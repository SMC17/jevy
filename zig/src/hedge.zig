//! Banded delta hedge + underlier slippage + greek / inventory PnL attribution.
//!
//! Curriculum: Akuna Options 101 (delta hedge, gamma, theta, vol/vega) +
//! Natenberg dynamic hedge — gamma PnL ≈ ½ Γ (ΔS)² vs theta; realized vs implied.
//! Desk practice: hedge in *bands* (not continuous), pay half-spread / bps slippage.
//!
//! Whalley–Wilmott-style practical bandwidth (document + implement):
//!   For proportional transaction cost ε and risk aversion γ_risk, a common
//!   asymptotic no-trade band half-width around delta-neutral is on the order of
//!   Δ_band ≈ (3/2 · ε · S · Γ² / γ_risk)^{1/3}   (WW / Zakamouline-style).
//! We expose `whalleyWilmottBand` as a *helper* to suggest `delta_band` from
//! (γ_opt, σ, cost); the runtime hedge uses the configured `delta_band` directly.
//!
//! Hedge policy (documented + tested): when |net_delta| > band, hedge qty =
//! −net_delta (flatten to zero, not to band edge). Desk variants that hedge only
//! to the band edge are common; we pick full flatten for simplicity and clear tests.

const std = @import("std");
const types = @import("types.zig");
const Greeks = types.Greeks;

pub const HedgeConfig = struct {
    /// Absolute delta units; hedge fires when |net_delta| > delta_band.
    delta_band: f64 = 5.0,
    /// Half-spread in underlier price units (absolute $). Used if > 0.
    half_spread: f64 = 0.0,
    /// Proportional slippage in bps of mid (1 bp = 1e-4). Used if half_spread == 0.
    slip_bps: f64 = 1.0,
    /// If true, hedge qty = −net_delta (flatten). If false, hedge to band edge.
    flatten: bool = true,
};

pub const HedgeOrder = struct {
    /// Signed underlier qty to trade (+ = buy underlier).
    underlier_qty: f64 = 0.0,
    /// Delta being neutralized (pre-hedge net delta).
    delta_to_hedge: f64 = 0.0,
    /// True if |net_delta| exceeded band.
    fire: bool = false,
    reason_len: usize = 0,
    reason: [128]u8 = [_]u8{0} ** 128,

    pub fn reasonSlice(self: *const HedgeOrder) []const u8 {
        return self.reason[0..self.reason_len];
    }
};

pub const HedgeFill = struct {
    time: f64 = 0.0,
    underlier_qty: f64 = 0.0,
    /// Fill price after slippage.
    fill_price: f64 = 0.0,
    mid: f64 = 0.0,
    /// Cash impact: −qty * fill_price (buy spends cash).
    cash_delta: f64 = 0.0,
    slippage_cost: f64 = 0.0,
};

/// Greek / inventory PnL buckets for one step (Natenberg-style explain).
pub const GreekPnlStep = struct {
    spread_capture: f64 = 0.0,
    hedge_slippage: f64 = 0.0,
    /// ≈ ½ Γ (ΔS)² · multiplier  (option qty already folded into Γ).
    gamma_pnl: f64 = 0.0,
    /// θ · Δt  (annualized BS theta × year-fraction step).
    theta_pnl: f64 = 0.0,
    /// ν · Δσ   (vega × change in vol; or finite-diff).
    vega_pnl: f64 = 0.0,
    /// Inventory MTM: qty · Δoption_mid + underlier_pos · ΔS (cash-ex).
    inventory_mtm: f64 = 0.0,
    /// Delta PnL residual: Δ · ΔS (should be ~0 if hedged; useful diagnostic).
    delta_pnl: f64 = 0.0,

    pub fn total(self: GreekPnlStep) f64 {
        return self.spread_capture + self.hedge_slippage + self.gamma_pnl + self.theta_pnl + self.vega_pnl + self.inventory_mtm;
    }

    pub fn add(self: GreekPnlStep, o: GreekPnlStep) GreekPnlStep {
        return .{
            .spread_capture = self.spread_capture + o.spread_capture,
            .hedge_slippage = self.hedge_slippage + o.hedge_slippage,
            .gamma_pnl = self.gamma_pnl + o.gamma_pnl,
            .theta_pnl = self.theta_pnl + o.theta_pnl,
            .vega_pnl = self.vega_pnl + o.vega_pnl,
            .inventory_mtm = self.inventory_mtm + o.inventory_mtm,
            .delta_pnl = self.delta_pnl + o.delta_pnl,
        };
    }
};

/// Practical WW-style bandwidth from option gamma, spot, vol, and proportional cost.
///
/// Formula (asymptotic no-trade band half-width in *delta* units, research-grade):
///   ε = slip as fraction of spot (e.g. half_spread/S or bps·1e-4)
///   Λ = |Γ| · S² · σ    (dollar-gamma scale · vol; Γ here is portfolio gamma)
///   band ≈ c · (ε² · |Γ| · S / γ_risk)^{1/3} · (optional σ factor)
///
/// We use a simplified practical form common in desk code:
///   band = c_ww · (1.5 · ε · S · Γ_abs² / γ_risk)^{1/3}
/// with c_ww ≈ 1, Γ_abs = max(|Γ|, ε_floor). Returns absolute delta band.
pub fn whalleyWilmottBand(
    spot: f64,
    gamma_abs: f64,
    sigma: f64,
    slip_frac: f64,
    risk_aversion: f64,
) f64 {
    _ = sigma; // reserved for extensions (vol enters fuller WW asymptotics)
    const g = @max(gamma_abs, 1e-8);
    const eps = @max(slip_frac, 1e-8);
    const gamma_risk = @max(risk_aversion, 1e-8);
    const inside = 1.5 * eps * spot * g * g / gamma_risk;
    return std.math.pow(f64, @max(inside, 0.0), 1.0 / 3.0);
}

fn setReason(out: *HedgeOrder, msg: []const u8) void {
    const n = @min(msg.len, out.reason.len);
    @memcpy(out.reason[0..n], msg[0..n]);
    out.reason_len = n;
}

/// Propose a paper hedge. Does not mutate inventory — caller applies fill.
pub fn proposeDeltaHedge(net_delta: f64, cfg: *const HedgeConfig) HedgeOrder {
    var out: HedgeOrder = .{};
    const abs_d = @abs(net_delta);
    if (abs_d <= cfg.delta_band) {
        setReason(&out, "within_band");
        return out;
    }
    out.fire = true;
    out.delta_to_hedge = net_delta;
    if (cfg.flatten) {
        out.underlier_qty = -net_delta;
        setReason(&out, "flatten");
    } else {
        // Hedge only back to band edge (same sign as net_delta, magnitude = band).
        const edge = if (net_delta > 0.0) cfg.delta_band else -cfg.delta_band;
        out.underlier_qty = edge - net_delta;
        setReason(&out, "to_band_edge");
    }
    return out;
}

/// Slippage-adjusted underlier fill price.
/// Buy → mid*(1+slip) or mid+half_spread; sell → mid*(1−slip) or mid−half_spread.
pub fn fillPrice(mid: f64, qty: f64, cfg: *const HedgeConfig) f64 {
    if (qty == 0.0) return mid;
    const buying = qty > 0.0;
    if (cfg.half_spread > 0.0) {
        return if (buying) mid + cfg.half_spread else mid - cfg.half_spread;
    }
    const slip = cfg.slip_bps * 1e-4;
    return if (buying) mid * (1.0 + slip) else mid * (1.0 - slip);
}

/// Apply hedge: compute fill + cash + slippage cost bucket.
/// `slippage_cost` is always ≤ 0 (cost): −|qty| · |fill − mid|.
pub fn applyHedge(
    time: f64,
    mid: f64,
    order: *const HedgeOrder,
    cfg: *const HedgeConfig,
) HedgeFill {
    if (!order.fire or order.underlier_qty == 0.0) {
        return .{ .time = time, .mid = mid, .fill_price = mid };
    }
    const px = fillPrice(mid, order.underlier_qty, cfg);
    const cash = -order.underlier_qty * px;
    const slip_cost = -@abs(order.underlier_qty) * @abs(px - mid);
    return .{
        .time = time,
        .underlier_qty = order.underlier_qty,
        .fill_price = px,
        .mid = mid,
        .cash_delta = cash,
        .slippage_cost = slip_cost,
    };
}

/// Baldacci–Bergault–Guéant appendix, arXiv 1907.12433:
///   q^{S*} = −Δ^π − ρ ξ V^π / (2 √ν S)
/// `variance` is the Heston variance ν (not Black vol). `portfolio_vega` is
/// Σ q_i ∂_{√ν} O^i, which matches Black vega ∂V/∂σ when σ=√ν.
/// Returns the target underlier position (shares), not a banded ticket.
pub fn spotVolHedgeQty(
    net_delta: f64,
    rho: f64,
    xi: f64,
    portfolio_vega: f64,
    variance: f64,
    spot: f64,
) f64 {
    const nu = @max(variance, 1e-16);
    const s = @max(@abs(spot), 1e-16);
    const tilt = rho * xi * portfolio_vega / (2.0 * @sqrt(nu) * s);
    return -net_delta - tilt;
}

/// One-step greek PnL attribution (Natenberg).
///
/// Inputs are *portfolio* greeks (qty already folded in) at start of step,
/// and market moves over the step. `option_qty` / `underlier_pos` for MTM.
///
/// gamma_pnl = 0.5 * Γ * (ΔS)^2
/// theta_pnl = θ * dt
/// vega_pnl  = ν * Δσ
/// delta_pnl = Δ * ΔS   (diagnostic; hedged book ≈ 0 + gamma)
/// inventory_mtm = option_qty * Δoption_mid + underlier_pos * ΔS
///                 (cash-style; excludes the pure greek Taylor terms if you
///                  prefer additive buckets — demos usually keep both labeled).
pub fn greekPnlStep(
    greeks: *const Greeks,
    d_spot: f64,
    d_sigma: f64,
    dt: f64,
    option_qty: f64,
    d_option_mid: f64,
    underlier_pos: f64,
    spread_capture: f64,
    hedge_slippage: f64,
) GreekPnlStep {
    return .{
        .spread_capture = spread_capture,
        .hedge_slippage = hedge_slippage,
        .gamma_pnl = 0.5 * greeks.gamma * d_spot * d_spot,
        .theta_pnl = greeks.theta * dt,
        .vega_pnl = greeks.vega * d_sigma,
        .delta_pnl = greeks.delta * d_spot,
        .inventory_mtm = option_qty * d_option_mid + underlier_pos * d_spot,
    };
}

/// Net portfolio delta including underlier hedge position (1.0 delta per share).
pub fn netDelta(option_delta: f64, underlier_pos: f64) f64 {
    return option_delta + underlier_pos;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "within band: no hedge" {
    const cfg = HedgeConfig{ .delta_band = 5.0 };
    const o = proposeDeltaHedge(3.0, &cfg);
    try std.testing.expect(!o.fire);
    try std.testing.expect(o.underlier_qty == 0.0);
}

test "above band: flatten to zero" {
    const cfg = HedgeConfig{ .delta_band = 5.0, .flatten = true };
    const o = proposeDeltaHedge(12.0, &cfg);
    try std.testing.expect(o.fire);
    try std.testing.expect(@abs(o.underlier_qty - (-12.0)) < 1e-12);
}

test "above band: to band edge" {
    const cfg = HedgeConfig{ .delta_band = 5.0, .flatten = false };
    const o = proposeDeltaHedge(12.0, &cfg);
    try std.testing.expect(o.fire);
    // edge = +5, qty = 5 - 12 = -7
    try std.testing.expect(@abs(o.underlier_qty - (-7.0)) < 1e-12);
    const o2 = proposeDeltaHedge(-12.0, &cfg);
    // edge = -5, qty = -5 - (-12) = +7
    try std.testing.expect(@abs(o2.underlier_qty - 7.0) < 1e-12);
}

test "slippage buy at mid+half / sell at mid-half" {
    const cfg = HedgeConfig{ .half_spread = 0.05, .slip_bps = 0.0 };
    try std.testing.expect(@abs(fillPrice(100.0, 10.0, &cfg) - 100.05) < 1e-12);
    try std.testing.expect(@abs(fillPrice(100.0, -10.0, &cfg) - 99.95) < 1e-12);
}

test "slippage bps path" {
    const cfg = HedgeConfig{ .half_spread = 0.0, .slip_bps = 10.0 }; // 10 bp
    try std.testing.expect(@abs(fillPrice(100.0, 1.0, &cfg) - 100.10) < 1e-9);
    try std.testing.expect(@abs(fillPrice(100.0, -1.0, &cfg) - 99.90) < 1e-9);
}

test "apply hedge records negative slippage cost" {
    const cfg = HedgeConfig{ .delta_band = 1.0, .half_spread = 0.10, .flatten = true };
    const o = proposeDeltaHedge(5.0, &cfg);
    const f = applyHedge(0.0, 100.0, &o, &cfg);
    try std.testing.expect(f.underlier_qty == -5.0);
    try std.testing.expect(@abs(f.fill_price - 99.90) < 1e-12); // sell
    try std.testing.expect(f.slippage_cost < 0.0);
    try std.testing.expect(@abs(f.slippage_cost - (-5.0 * 0.10)) < 1e-12);
}

test "gamma pnl = 1/2 Gamma (dS)^2" {
    const g = Greeks{ .delta = 0.0, .gamma = 0.04, .vega = 0.0, .theta = 0.0 };
    const step = greekPnlStep(&g, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    // 0.5 * 0.04 * 4 = 0.08
    try std.testing.expect(@abs(step.gamma_pnl - 0.08) < 1e-12);
}

test "theta and vega buckets" {
    const g = Greeks{ .delta = 10.0, .gamma = 0.0, .vega = 20.0, .theta = -5.0 };
    const dt = 1.0 / 252.0;
    const step = greekPnlStep(&g, 1.0, 0.01, dt, 0.0, 0.0, 0.0, 0.0, 0.0);
    try std.testing.expect(@abs(step.theta_pnl - (-5.0 * dt)) < 1e-12);
    try std.testing.expect(@abs(step.vega_pnl - 0.20) < 1e-12);
    try std.testing.expect(@abs(step.delta_pnl - 10.0) < 1e-12);
}

test "WW band positive and increases with cost" {
    const b1 = whalleyWilmottBand(100.0, 0.05, 0.2, 0.0001, 1e-3);
    const b2 = whalleyWilmottBand(100.0, 0.05, 0.2, 0.001, 1e-3);
    try std.testing.expect(b1 > 0.0);
    try std.testing.expect(b2 > b1);
}

test "net delta includes underlier" {
    try std.testing.expect(@abs(netDelta(8.0, -8.0)) < 1e-12);
    try std.testing.expect(@abs(netDelta(8.0, -3.0) - 5.0) < 1e-12);
}

test "spot-vol hedge matches Baldacci appendix on the toy numbers" {
    // qS* = -Δ - ρ ξ V / (2 √ν S)
    // ρ=-0.5, ξ=0.2, V=10, ν=0.04, S=100, Δ=0.5
    // tilt = -0.5*0.2*10 / (2*0.2*100) = -0.025
    // qS = -0.5 - (-0.025) = -0.475
    const q = spotVolHedgeQty(0.5, -0.5, 0.2, 10.0, 0.04, 100.0);
    try std.testing.expectApproxEqAbs(q, -0.475, 1e-12);
    const flat = spotVolHedgeQty(0.5, 0.0, 0.2, 10.0, 0.04, 100.0);
    try std.testing.expectApproxEqAbs(flat, -0.5, 1e-12);
}
