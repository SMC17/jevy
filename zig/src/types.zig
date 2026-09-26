//! Core domain types (plain structs, hot-path friendly).

pub const Side = enum(u8) {
    bid = 0,
    ask = 1,
};

/// Quoter closed-form family.
/// - `as_finite_horizon`: classic Avellaneda–Stoikov reservation/spread with T−t.
/// - `gueant_asymptotic`: Guéant–Lehalle–Fernandez-Tapia stationary approx (arXiv 1105.3115).
/// - `gueant_ode`: finite-horizon / spectral ODE on the same intensity (arXiv 1105.3115 §3).
pub const QuoteMode = enum(u8) {
    as_finite_horizon = 0,
    gueant_asymptotic = 1,
    gueant_ode = 2,
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
};

pub const RiskConfig = struct {
    max_abs_inventory: i32 = 25,
    max_abs_delta: f64 = 50.0,
    max_abs_vega: f64 = 200.0,
    max_abs_gamma: f64 = 5.0,
    max_loss: f64 = 500.0,
};

/// Fixed-size strike strip (desk-shaped: typically 5 strikes around spot).
pub const MAX_STRIP: usize = 8;

pub const StrikeSlot = struct {
    strike: f64 = 0.0,
    inventory: i32 = 0,
    active: bool = false,
};
