//! Synthetic queue / LOB-aware fill model (simulation only — no live market data).
//!
//! Extends the Poisson touch model in `fills.zig`. A resting quote sits behind
//! `ahead` contracts at its price. Two flows consume that queue:
//!
//!   * aggressive trades at rate `trade_intensity` (contracts / year), which
//!     both advance the queue and can fill us;
//!   * cancels of size ahead of us at rate `cancel_ahead`, which advance the
//!     queue without filling us.
//!
//! Fluid limit (used by the comparative-static tests):
//!
//!   t_clear = ahead / (trade_intensity + cancel_ahead)
//!   fills   = min(our_size, trade_intensity * max(0, exposure − t_clear))
//!
//! `exposure` is the horizon, or `cancel_latency` when a cancel is working.
//! A cancel is effective only after `cancel_latency`, so a late cancel still
//! cuts off the rest of the toxic window (fewer adverse fills than never
//! cancelling) while a longer latency leaves more residual adverse fills.
//!
//! When a trade prints through our size, the mid jumps against us by
//! `adverse_jump * toxic_flow * size`. Markout is the signed (mid_after −
//! price) from the liquidity provider's perspective, so a larger `toxic_flow`
//! worsens markout.
//!
//! The stochastic stepper emits sequenced LobAdd / LobExecute / LobCancel
//! events (add / cancel / execute), including partial executes.

const std = @import("std");
const types = @import("types.zig");
const Side = types.Side;

pub const MAX_EVENTS: usize = 64;

pub const LobEventKind = enum(u8) {
    add = 0,
    cancel = 1,
    execute = 2,

    pub fn jsonName(self: LobEventKind) []const u8 {
        return switch (self) {
            .add => "LobAdd",
            .cancel => "LobCancel",
            .execute => "LobExecute",
        };
    }
};

pub const LobEvent = struct {
    time: f64 = 0.0,
    kind: LobEventKind = .add,
    side: Side = .bid,
    price: f64 = 0.0,
    size: f64 = 0.0,
    ahead: f64 = 0.0,
    partial: bool = false,
    adverse: bool = false,
};

pub const LobConfig = struct {
    trade_intensity: f64 = 40.0,
    cancel_ahead: f64 = 0.0,
    adverse_jump: f64 = 0.05,
    toxic_flow: f64 = 1.0,
    /// Sim time at which fills become toxic (adverse jump applies).
    toxic_from: f64 = 0.0,
    dt: f64 = 0.005,
    horizon: f64 = 1.0,
    /// null = never cancel. Otherwise the cancel requested at t=0 is effective
    /// at this time (latency).
    cancel_latency: ?f64 = null,
    ahead: f64 = 0.0,
    our_size: f64 = 1.0,
    price: f64 = 1.0,
    mid: f64 = 1.05,
    side: Side = .bid,
};

pub const LobResult = struct {
    filled: f64 = 0.0,
    adverse_filled: f64 = 0.0,
    markout: f64 = 0.0,
    time_to_first: f64 = std.math.inf(f64),
    n_events: usize = 0,
    events: [MAX_EVENTS]LobEvent = [_]LobEvent{.{}} ** MAX_EVENTS,
};

pub fn timeToFirstFill(ahead: f64, trade_intensity: f64, cancel_ahead: f64) f64 {
    if (ahead <= 0.0) return 0.0;
    const rate = trade_intensity + cancel_ahead;
    if (!(rate > 0.0)) return std.math.inf(f64);
    return ahead / rate;
}

/// Fluid-limit expected contracts filled over `horizon`, optionally cut by cancel latency.
pub fn expectedFills(
    ahead: f64,
    our_size: f64,
    trade_intensity: f64,
    cancel_ahead: f64,
    horizon: f64,
    cancel_latency: ?f64,
) f64 {
    if (!(our_size > 0.0) or !(horizon > 0.0)) return 0.0;
    var exposure = horizon;
    if (cancel_latency) |lat| exposure = @min(horizon, @max(lat, 0.0));
    if (!(exposure > 0.0)) return 0.0;
    const t_clear = timeToFirstFill(@max(ahead, 0.0), trade_intensity, cancel_ahead);
    if (t_clear >= exposure) return 0.0;
    const traded = @max(trade_intensity, 0.0) * (exposure - t_clear);
    return @min(our_size, traded);
}

/// Liquidity-provider markout after an instantaneous adverse jump.
/// Bid (we bought): (mid_after − price) * size, mid_after = mid − jump.
/// Ask (we sold): (price − mid_after) * size, mid_after = mid + jump.
pub fn fillMarkout(side: Side, price: f64, mid: f64, size: f64, adverse_jump: f64, toxic_flow: f64) f64 {
    const jump = @max(adverse_jump, 0.0) * @max(toxic_flow, 0.0) * @max(size, 0.0);
    return switch (side) {
        .bid => (mid - jump - price) * size,
        .ask => (price - (mid + jump)) * size,
    };
}

fn pushEvent(res: *LobResult, ev: LobEvent) void {
    if (res.n_events >= MAX_EVENTS) return;
    res.events[res.n_events] = ev;
    res.n_events += 1;
}

fn poissonSample(rng: std.Random, lambda: f64) i32 {
    if (lambda <= 0.0) return 0;
    if (lambda > 30.0) {
        const z = rng.floatNorm(f64);
        const x = lambda + @sqrt(lambda) * z;
        const rounded = @round(x);
        return if (rounded > 0.0) @intFromFloat(rounded) else 0;
    }
    const L = @exp(-lambda);
    var k: i32 = 0;
    var p: f64 = 1.0;
    while (true) {
        k += 1;
        p *= rng.float(f64);
        if (p <= L) break;
        if (k > 1_000_000) break;
    }
    return k - 1;
}

/// Step a single resting order. Trades and cancels-ahead are Poisson on `dt`.
pub fn simulate(rng: std.Random, cfg: *const LobConfig) LobResult {
    var res = LobResult{};
    var ahead = @max(cfg.ahead, 0.0);
    var remaining = @max(cfg.our_size, 0.0);
    var t: f64 = 0.0;
    const cancel_at: f64 = if (cfg.cancel_latency) |lat| @max(lat, 0.0) else std.math.inf(f64);
    var live = remaining > 0.0;
    pushEvent(&res, .{
        .time = 0.0,
        .kind = .add,
        .side = cfg.side,
        .price = cfg.price,
        .size = remaining,
        .ahead = ahead,
        .partial = false,
        .adverse = false,
    });

    const dt = @max(cfg.dt, 1e-6);
    while (t < cfg.horizon and live) {
        const t_next = @min(t + dt, cfg.horizon);
        if (t_next >= cancel_at and t < cancel_at) {
            // Consume the queue only up to the cancel instant, then pull.
            const span = @max(cancel_at - t, 0.0);
            if (span > 0.0) consume(&res, rng, cfg, &ahead, &remaining, t, span);
            live = false;
            pushEvent(&res, .{
                .time = cancel_at,
                .kind = .cancel,
                .side = cfg.side,
                .price = cfg.price,
                .size = remaining,
                .ahead = ahead,
                .partial = remaining > 0.0 and remaining < cfg.our_size,
                .adverse = false,
            });
            break;
        }
        const span = t_next - t;
        consume(&res, rng, cfg, &ahead, &remaining, t, span);
        if (remaining <= 1e-12) live = false;
        t = t_next;
    }
    return res;
}

fn consume(
    res: *LobResult,
    rng: std.Random,
    cfg: *const LobConfig,
    ahead: *f64,
    remaining: *f64,
    t: f64,
    span: f64,
) void {
    if (!(span > 0.0) or remaining.* <= 0.0) return;
    const n_cancel = poissonSample(rng, @max(cfg.cancel_ahead, 0.0) * span);
    if (n_cancel > 0) ahead.* = @max(0.0, ahead.* - @as(f64, @floatFromInt(n_cancel)));
    const n_trade = poissonSample(rng, @max(cfg.trade_intensity, 0.0) * span);
    var left: f64 = @floatFromInt(@max(n_trade, 0));
    if (ahead.* > 0.0 and left > 0.0) {
        const eat = @min(ahead.*, left);
        ahead.* -= eat;
        left -= eat;
    }
    if (left > 0.0 and remaining.* > 0.0) {
        const fill = @min(remaining.*, left);
        const toxic = (t + span) >= cfg.toxic_from and cfg.toxic_flow > 0.0 and cfg.adverse_jump > 0.0;
        const flow: f64 = if (toxic) cfg.toxic_flow else 0.0;
        const mo = fillMarkout(cfg.side, cfg.price, cfg.mid, fill, cfg.adverse_jump, flow);
        res.filled += fill;
        res.markout += mo;
        if (toxic) res.adverse_filled += fill;
        if (res.time_to_first == std.math.inf(f64)) res.time_to_first = t + span;
        const partial = fill + 1e-9 < cfg.our_size;
        remaining.* -= fill;
        pushEvent(res, .{
            .time = t + span,
            .kind = .execute,
            .side = cfg.side,
            .price = cfg.price,
            .size = fill,
            .ahead = ahead.*,
            .partial = partial and remaining.* > 1e-12,
            .adverse = toxic,
        });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "deeper queue is slower and fills less inside a window" {
    const shallow_t = timeToFirstFill(5.0, 20.0, 0.0);
    const deep_t = timeToFirstFill(80.0, 20.0, 0.0);
    try std.testing.expect(deep_t > shallow_t * 5.0);
    const shallow = expectedFills(5.0, 10.0, 20.0, 0.0, 1.0, null);
    const deep = expectedFills(80.0, 10.0, 20.0, 0.0, 1.0, null);
    try std.testing.expect(shallow > deep);
    try std.testing.expect(deep == 0.0);
    try std.testing.expect(shallow > 0.0);

    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    const cfg_s = LobConfig{ .ahead = 0.0, .our_size = 5.0, .trade_intensity = 30.0, .horizon = 1.0, .dt = 0.01, .toxic_flow = 0.0, .adverse_jump = 0.0 };
    var cfg_d = cfg_s;
    cfg_d.ahead = 500.0;
    const sim_s = simulate(rng, &cfg_s);
    const sim_d = simulate(rng, &cfg_d);
    try std.testing.expect(sim_s.filled > sim_d.filled);
    try std.testing.expect(sim_d.filled == 0.0);
    try std.testing.expect(sim_s.time_to_first < sim_d.time_to_first);
}

test "late cancel reduces adverse fills versus staying in the book" {
    // Whole window is toxic. Never-cancel exposure is the full horizon.
    const stayed = expectedFills(0.0, 1000.0, 50.0, 0.0, 1.0, null);
    const late = expectedFills(0.0, 1000.0, 50.0, 0.0, 1.0, 0.25);
    const fast = expectedFills(0.0, 1000.0, 50.0, 0.0, 1.0, 0.02);
    try std.testing.expect(late < stayed * 0.5);
    try std.testing.expect(fast < late);

    var prng = std.Random.DefaultPrng.init(19);
    const rng = prng.random();
    const base = LobConfig{
        .ahead = 0.0,
        .our_size = 500.0,
        .trade_intensity = 40.0,
        .horizon = 1.0,
        .dt = 0.01,
        .toxic_from = 0.0,
        .toxic_flow = 1.0,
        .adverse_jump = 0.02,
        .cancel_latency = null,
    };
    var late_cfg = base;
    late_cfg.cancel_latency = 0.2;
    const sim_stay = simulate(rng, &base);
    const sim_late = simulate(rng, &late_cfg);
    try std.testing.expect(sim_late.adverse_filled < sim_stay.adverse_filled);
    var saw_cancel = false;
    var i: usize = 0;
    while (i < sim_late.n_events) : (i += 1) {
        if (sim_late.events[i].kind == .cancel) saw_cancel = true;
    }
    try std.testing.expect(saw_cancel);
    try std.testing.expect(sim_late.events[0].kind == .add);
}

test "toxic flow parameter worsens markout" {
    const calm = fillMarkout(.bid, 1.00, 1.05, 2.0, 0.04, 0.0);
    const toxic = fillMarkout(.bid, 1.00, 1.05, 2.0, 0.04, 3.0);
    try std.testing.expect(toxic < calm);
    const calm_ask = fillMarkout(.ask, 1.10, 1.05, 2.0, 0.04, 0.0);
    const toxic_ask = fillMarkout(.ask, 1.10, 1.05, 2.0, 0.04, 3.0);
    try std.testing.expect(toxic_ask < calm_ask);

    var prng = std.Random.DefaultPrng.init(3);
    const rng = prng.random();
    const cfg0 = LobConfig{
        .ahead = 0.0,
        .our_size = 4.0,
        .trade_intensity = 80.0,
        .horizon = 0.5,
        .dt = 0.01,
        .toxic_flow = 0.0,
        .adverse_jump = 0.05,
        .price = 1.0,
        .mid = 1.04,
        .side = .bid,
    };
    var cfg1 = cfg0;
    cfg1.toxic_flow = 4.0;
    const s0 = simulate(rng, &cfg0);
    const s1 = simulate(rng, &cfg1);
    try std.testing.expect(s0.filled > 0.0 and s1.filled > 0.0);
    // Normalize per contract so a different fill count does not confound the jump.
    try std.testing.expect(s1.markout / s1.filled < s0.markout / s0.filled);
}

test "partial fill when traded volume is inside our size" {
    const exp = expectedFills(0.0, 10.0, 4.0, 0.0, 1.0, null);
    try std.testing.expect(exp > 0.0 and exp < 10.0);
    try std.testing.expect(@abs(exp - 4.0) < 1e-12);

    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    const cfg = LobConfig{
        .ahead = 0.0,
        .our_size = 80.0,
        .trade_intensity = 25.0,
        .horizon = 1.0,
        .dt = 0.02,
        .toxic_flow = 0.0,
        .adverse_jump = 0.0,
    };
    const sim = simulate(rng, &cfg);
    try std.testing.expect(sim.filled > 0.0 and sim.filled < cfg.our_size);
    var partial = false;
    var i: usize = 0;
    while (i < sim.n_events) : (i += 1) {
        if (sim.events[i].kind == .execute) partial = true;
    }
    try std.testing.expect(partial);
}

test "cancels ahead shorten time to first fill without counting as our fill" {
    const with_cancel = timeToFirstFill(20.0, 10.0, 30.0);
    const no_cancel = timeToFirstFill(20.0, 10.0, 0.0);
    try std.testing.expect(with_cancel < no_cancel);
    // After the queue clears, only trades fill us: 10/s * (1 - 0.5) = 5.
    const fills = expectedFills(20.0, 100.0, 10.0, 30.0, 1.0, null);
    try std.testing.expect(@abs(fills - 5.0) < 1e-9);
}
