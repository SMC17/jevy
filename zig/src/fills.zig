//! Exogenous Poisson fill model against posted quotes.
//! Port of jev_omm/execution/fills.py

const std = @import("std");
const types = @import("types.zig");
const Fill = types.Fill;
const Quote = types.Quote;
const Side = types.Side;

inline fn sideIntensity(half_away: f64, base: f64, kappa: f64) f64 {
    return base * @exp(-kappa * @max(half_away, 0.0));
}

fn poissonSample(rng: std.Random, lambda: f64) i32 {
    if (lambda <= 0.0) return 0;
    if (lambda > 30.0) {
        const z = rng.floatNorm(f64);
        const x = lambda + @sqrt(lambda) * z;
        const rounded = @round(x);
        return if (rounded > 0.0) @intFromFloat(rounded) else 0;
    }
    // Knuth
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

/// Sample fills into a caller-provided buffer (max 2: bid+ask). Returns count.
pub fn sampleFillsInto(
    rng: std.Random,
    time: f64,
    mid: f64,
    quote: *const Quote,
    dt: f64,
    base_intensity: f64,
    kappa: f64,
    out: []Fill,
) usize {
    if (out.len == 0) return 0;
    var n: usize = 0;
    if (quote.bid_size <= 0 and quote.ask_size <= 0) return 0;

    const bid_away = @max(0.0, mid - quote.bid);
    const ask_away = @max(0.0, quote.ask - mid);
    const lam_bid = sideIntensity(bid_away, base_intensity, kappa) * dt;
    const lam_ask = sideIntensity(ask_away, base_intensity, kappa) * dt;

    var n_bid: i32 = if (quote.bid_size > 0) poissonSample(rng, lam_bid) else 0;
    var n_ask: i32 = if (quote.ask_size > 0) poissonSample(rng, lam_ask) else 0;
    if (n_bid > quote.bid_size) n_bid = quote.bid_size;
    if (n_ask > quote.ask_size) n_ask = quote.ask_size;

    if (n_bid > 0 and n < out.len) {
        out[n] = .{
            .time = time,
            .side = .bid,
            .price = quote.bid,
            .size = n_bid,
            .mid_at_fill = mid,
        };
        n += 1;
    }
    if (n_ask > 0 and n < out.len) {
        out[n] = .{
            .time = time,
            .side = .ask,
            .price = quote.ask,
            .size = n_ask,
            .mid_at_fill = mid,
        };
        n += 1;
    }
    return n;
}

pub fn applyFillCash(cash: *f64, qty: *i32, side: Side, price: f64, size: i32) void {
    const notional = price * @as(f64, @floatFromInt(size));
    switch (side) {
        .bid => {
            qty.* += size;
            cash.* -= notional;
        },
        .ask => {
            qty.* -= size;
            cash.* += notional;
        },
    }
}

test "zero size no fills" {
    var prng = std.Random.DefaultPrng.init(1);
    const rng = prng.random();
    const q = Quote{ .bid = 1.0, .ask = 1.1, .bid_size = 0, .ask_size = 0, .reservation = 1.05, .half_spread = 0.05 };
    var buf: [2]Fill = undefined;
    const n = sampleFillsInto(rng, 0.0, 1.05, &q, 1.0, 100.0, 1.5, &buf);
    try std.testing.expect(n == 0);
}

test "high intensity can fill" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const q = Quote{ .bid = 5.0, .ask = 5.1, .bid_size = 5, .ask_size = 5, .reservation = 5.05, .half_spread = 0.05 };
    var any = false;
    var trial: usize = 0;
    while (trial < 50) : (trial += 1) {
        var buf: [2]Fill = undefined;
        const n = sampleFillsInto(rng, 0.0, 5.05, &q, 1.0, 10.0, 0.1, &buf);
        if (n > 0) {
            any = true;
            break;
        }
    }
    try std.testing.expect(any);
}

test "apply fill updates cash qty" {
    var cash: f64 = 0.0;
    var qty: i32 = 0;
    applyFillCash(&cash, &qty, .bid, 2.5, 2);
    try std.testing.expect(qty == 2);
    try std.testing.expect(@abs(cash + 5.0) < 1e-12);
    applyFillCash(&cash, &qty, .ask, 3.0, 1);
    try std.testing.expect(qty == 1);
    try std.testing.expect(@abs(cash + 2.0) < 1e-12);
}
