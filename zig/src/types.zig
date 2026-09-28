//! Core domain types (plain structs, hot-path friendly).

const std = @import("std");

pub const Side = enum(u8) {
    bid = 0,
    ask = 1,
};

/// Quoter closed-form family.
/// - `as_finite_horizon`: classic Avellaneda–Stoikov reservation/spread with T−t.
/// - `gueant_asymptotic`: Guéant–Lehalle–Fernandez-Tapia stationary approx (arXiv 1105.3115).
/// - `gueant_ode`: finite-horizon / spectral ODE on the same intensity (arXiv 1105.3115 §3).
/// - `option_vega`: Baldacci–Bergault–Guéant constant-vega HJB (arXiv 1907.12433).
pub const QuoteMode = enum(u8) {
    as_finite_horizon = 0,
    gueant_asymptotic = 1,
    gueant_ode = 2,
    option_vega = 3,
};

pub const Greeks = struct {
    delta: f64 = 0.0,
    gamma: f64 = 0.0,
    /// ∂V/∂σ with σ in decimal (not per vol point).
    vega: f64 = 0.0,
    theta: f64 = 0.0,
    /// ∂²V/∂S∂σ = ∂Δ/∂σ. Same for calls and puts.
    vanna: f64 = 0.0,
    /// ∂²V/∂σ² = ∂ν/∂σ (volga / vomma). Same for calls and puts.
    volga: f64 = 0.0,

    pub const zero: Greeks = .{};

    pub fn add(self: Greeks, other: Greeks) Greeks {
        return .{
            .delta = self.delta + other.delta,
            .gamma = self.gamma + other.gamma,
            .vega = self.vega + other.vega,
            .theta = self.theta + other.theta,
            .vanna = self.vanna + other.vanna,
            .volga = self.volga + other.volga,
        };
    }

    pub fn scale(self: Greeks, w: f64) Greeks {
        return .{
            .delta = self.delta * w,
            .gamma = self.gamma * w,
            .vega = self.vega * w,
            .theta = self.theta * w,
            .vanna = self.vanna * w,
            .volga = self.volga * w,
        };
    }
};

pub const Quote = struct {
    bid: f64 = 0.0,
    ask: f64 = 0.0,
    bid_size: i32 = 0,
    ask_size: i32 = 0,
    reservation: f64 = 0.0,
    half_spread: f64 = 0.0,
};

/// Per-strike quote tagged for multi-strike / event-log strip.
pub const StrikeQuote = struct {
    strike: f64 = 0.0,
    quote: Quote = .{},
    mid: f64 = 0.0,
    iv: f64 = 0.0,
    greeks: Greeks = .{},
    inventory: i32 = 0,
};

pub const Fill = struct {
    time: f64 = 0.0,
    side: Side = .bid,
    price: f64 = 0.0,
    size: i32 = 0,
    mid_at_fill: f64 = 0.0,
};

pub const RiskSnapshot = struct {
    inventory: i32 = 0,
    delta: f64 = 0.0,
    gamma: f64 = 0.0,
    vega: f64 = 0.0,
    cash_pnl: f64 = 0.0,
    quoting_allowed: bool = true,
    /// Fixed buffer for breach reason (C-ABI friendly; empty if none).
    breach_reason: [128]u8 = [_]u8{0} ** 128,
    breach_len: usize = 0,

    pub fn reasonSlice(self: *const RiskSnapshot) []const u8 {
        return self.breach_reason[0..self.breach_len];
    }
};

pub const QuoterConfig = struct {
    gamma: f64 = 0.1,
    kappa: f64 = 1.5,
    sigma: f64 = 0.5,
    t_horizon: f64 = 1.0 / 252.0,
    /// Guéant mid-touch arrival intensity A (1/year). Unused by classic AS.
    A: f64 = 140.0,
    /// Inventory bound Q for `gueant_ode` (state q ∈ [−Q, Q]).
    inventory_cap: i32 = 10,
    /// RK4 steps for the Guéant ODE over `t_horizon`.
    ode_steps: u32 = 800,
    gamma_penalty: f64 = 0.0,
    vega_penalty: f64 = 0.0,
    /// Extra reservation tilt per unit of *portfolio* delta (multi-strike desk).
    portfolio_delta_penalty: f64 = 0.0,
    min_half_spread: f64 = 0.05,
    max_half_spread: f64 = 5.0,
    quote_size: i32 = 1,
    mode: QuoteMode = .as_finite_horizon,
    /// Constant per-contract vega ∂V/∂σ for `option_vega` (overridden by greeks.vega).
    contract_vega: f64 = 10.0,
    /// Vol-of-vol ξ in the quadratic penalty γ ξ² (1−ρ²) / 8.
    xi: f64 = 1.0,
    /// Hard |portfolio vega| cap V̄.
    vega_limit: f64 = 40.0,
    /// P-vs-Q variance drift, (a_P − a_Q) / (2 √ν).
    vol_edge: f64 = 0.0,
    /// Spot–vol correlation used to scale the vega penalty by (1−ρ²).
    option_rho: f64 = 0.0,
    /// Theo IV − market IV (decimal). Reservation shifts by contract_vega * iv_alpha.
    iv_alpha: f64 = 0.0,
    /// 0 = exponential Λ(δ)=A e^{−kδ}, 1 = logistic Λ.
    intensity_kind: u8 = 0,
    logistic_lambda: f64 = 100.0,
    logistic_alpha: f64 = 0.7,
    logistic_beta: f64 = 150.0,
    option_grid_n: u32 = 31,
    option_grid_steps: u32 = 60,
};

pub const RiskConfig = struct {
    max_abs_inventory: i32 = 25,
    max_abs_delta: f64 = 50.0,
    max_abs_vega: f64 = 200.0,
    max_abs_gamma: f64 = 5.0,
    max_loss: f64 = 500.0,
    /// <= 0 means the limit is unset (no-op).
    max_abs_notional: f64 = 0.0,
    max_abs_per_strike: i32 = 0,
    max_quotes_outstanding: i32 = 0,
};

/// Optional book objects for the hard-risk gate.
/// NaN notional, or a negative per-strike / quotes count, means the object
/// was not supplied and that check is skipped even if the limit is set.
pub const RiskBook = struct {
    extra_delta: f64 = 0.0,
    notional: f64 = std.math.nan(f64),
    per_strike_abs: i32 = -1,
    quotes_outstanding: i32 = -1,
};

/// Fixed-size strike strip (desk-shaped: typically 5 strikes around spot).
pub const MAX_STRIP: usize = 8;

pub const StrikeSlot = struct {
    strike: f64 = 0.0,
    inventory: i32 = 0,
    active: bool = false,
};
