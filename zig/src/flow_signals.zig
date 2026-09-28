//! Signed flow features. No network. Mirrors ``jev_omm/flow/signals.py``.
//!
//! Lee–Ready: https://doi.org/10.1111/j.1540-6261.1991.tb02683.x
//! OFI: Cont–Kukanov–Stoikov, https://doi.org/10.1093/jjfinec/nbt003
//! Layered cancels: phenomenology only, Cartea–Jaimungal–Wang
//! https://doi.org/10.1080/1350486X.2020.1726783

const std = @import("std");

fn clamp(x: f64, lo: f64, hi: f64) f64 {
    return @min(@max(x, lo), hi);
}

pub const LeeReady = struct {
    prev_price: ?f64 = null,
    prev_sign: i32 = 0,

    pub fn sign(self: *LeeReady, price: f64, bid: f64, ask: f64) i32 {
        const mid = 0.5 * (bid + ask);
        var s: i32 = 0;
        if (price > mid) {
            s = 1;
        } else if (price < mid) {
            s = -1;
        } else if (self.prev_price == null) {
            s = self.prev_sign;
        } else if (price > self.prev_price.?) {
            s = 1;
        } else if (price < self.prev_price.?) {
            s = -1;
        } else {
            s = self.prev_sign;
        }
        self.prev_price = price;
        if (s != 0) self.prev_sign = s;
        return s;
    }
};

pub fn ofiIncrement(
    prev_bid: f64,
    prev_bid_size: f64,
    prev_ask: f64,
    prev_ask_size: f64,
    bid: f64,
    bid_size: f64,
    ask: f64,
    ask_size: f64,
) f64 {
    var e_b: f64 = 0.0;
    if (bid >= prev_bid) e_b += bid_size;
    if (bid <= prev_bid) e_b -= prev_bid_size;
    var e_a: f64 = 0.0;
    if (ask <= prev_ask) e_a += ask_size;
    if (ask >= prev_ask) e_a -= prev_ask_size;
    return e_b - e_a;
}

pub fn normalizeOfi(ofi: f64, depth: f64) f64 {
    if (depth <= 0.0) return 0.0;
    return clamp(ofi / depth, -1.0, 1.0);
}

pub const BookEvent = struct {
    kind: u8, // 0 add, 1 cancel, 2 trade
    level: i32,
    size: f64,
    age: f64 = 0.0,
};

pub fn spoofScore(events: []const BookEvent, flicker_age: f64, min_size: f64) f64 {
    var layered: f64 = 0.0;
    var traded: f64 = 0.0;
    for (events) |event| {
        if (event.kind == 2) {
            traded += event.size;
        } else if (event.kind == 1 and event.level >= 1 and event.age <= flicker_age and event.size >= min_size) {
            layered += event.size;
        }
    }
    return layered / (layered + traded + 1e-12);
}

pub fn flowToxicity(vpin: f64, ofi_norm: f64, aggr_imbalance: f64, off_exchange_share: f64, spoof: f64) f64 {
    const tox = 0.50 * @max(vpin, 0.0) + 0.20 * @abs(ofi_norm) + 0.15 * @abs(aggr_imbalance) * @max(off_exchange_share, 0.0) + 0.35 * @max(spoof, 0.0);
    return clamp(tox, 0.0, 1.0);
}

pub const FlowPrior = struct {
    toxicity: f64,
    spread_mult: f64,
    size_mult: f64,
};

pub fn flowPrior(vpin: f64, ofi_norm: f64, aggr_imbalance: f64, off_exchange_share: f64, spoof: f64) FlowPrior {
    const tox = flowToxicity(vpin, ofi_norm, aggr_imbalance, off_exchange_share, spoof);
    return .{
        .toxicity = tox,
        .spread_mult = 1.0 + 1.25 * tox,
        .size_mult = 1.0 / (1.0 + 1.50 * tox),
    };
}

test "flow prior is the identity at zero and widens on vpin" {
    const z = flowPrior(0, 0, 0, 0, 0);
    try std.testing.expectApproxEqAbs(z.toxicity, 0.0, 1e-15);
    try std.testing.expectApproxEqAbs(z.spread_mult, 1.0, 1e-15);
    try std.testing.expectApproxEqAbs(z.size_mult, 1.0, 1e-15);
    const hot = flowPrior(1.0, 0, 0, 0, 0);
    try std.testing.expectApproxEqAbs(hot.toxicity, 0.5, 1e-15);
    try std.testing.expectApproxEqAbs(hot.spread_mult, 1.0 + 1.25 * 0.5, 1e-15);
    try std.testing.expectApproxEqAbs(hot.size_mult, 1.0 / (1.0 + 1.5 * 0.5), 1e-15);
    try std.testing.expect(hot.spread_mult > 1.0);
    try std.testing.expect(hot.size_mult < 1.0);
}

test "lee-ready quote test then tick test" {
    var lr = LeeReady{};
    try std.testing.expectEqual(@as(i32, 1), lr.sign(100.2, 100.0, 100.2));
    try std.testing.expectEqual(@as(i32, -1), lr.sign(99.9, 99.8, 100.2));
    // Midpoint 100.0, previous price 99.9 → uptick buy.
    try std.testing.expectEqual(@as(i32, 1), lr.sign(100.0, 99.9, 100.1));
}

test "ofi bid lift is positive and a layered cancel scores high" {
    const e = ofiIncrement(100.0, 10.0, 100.2, 10.0, 100.1, 12.0, 100.2, 10.0);
    try std.testing.expect(e > 0.0);
    const flicker = [_]BookEvent{
        .{ .kind = 1, .level = 2, .size = 50.0, .age = 0.1 },
        .{ .kind = 2, .level = 0, .size = 1.0, .age = 0.0 },
    };
    try std.testing.expect(spoofScore(&flicker, 0.5, 1.0) > 0.9);
    const traded = [_]BookEvent{
        .{ .kind = 1, .level = 2, .size = 5.0, .age = 0.1 },
        .{ .kind = 2, .level = 0, .size = 40.0, .age = 0.0 },
    };
    try std.testing.expect(spoofScore(&traded, 0.5, 1.0) < 0.2);
}
