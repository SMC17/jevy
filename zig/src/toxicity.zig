//! Flow toxicity features for Decision state (research-grade, not production VPIN).
//!
//! Simplified rolling trade imbalance + bucketed VPIN-style feature from a
//! synthetic tape in sim. Feeds `build_mm_state` / DecisionSnapshot so toxicity
//! Score / informed_flow Noul have real numeric features (fallback client can
//! use them deterministically too).
//!
//! Ref framing: Easley, López de Prado, O’Hara — Flow Toxicity and Liquidity
//! (RFS); intro note https://www.quantresearch.org/From%20PIN%20to%20VPIN.pdf
//! DOI https://doi.org/10.1093/rfs/hhs053
//!
//! THIS IS RESEARCH-GRADE: volume buckets are synthetic, no real tick rule /
//! bulk volume classification, no CDF mapping to VPIN probability. Treat as a
//! feature for System One state, not a production toxicity meter.

const std = @import("std");

pub const MAX_BUCKETS: usize = 50;

pub const ToxicityConfig = struct {
    /// Target volume (contracts / shares) per VPIN-style bucket.
    bucket_volume: f64 = 20.0,
    /// Number of recent completed buckets in the rolling VPIN window.
    window_buckets: usize = 10,
};

pub const TradePrint = struct {
    /// Signed volume: +buy (lift ask), −sell (hit bid). Abs = size.
    signed_volume: f64 = 0.0,
};

/// Rolling toxicity state machine.
pub const Tracker = struct {
    cfg: ToxicityConfig = .{},
    /// Accumulator for current open bucket.
    buy_vol: f64 = 0.0,
    sell_vol: f64 = 0.0,
    /// Ring of completed bucket imbalances |B−S|/(B+S).
    imbalances: [MAX_BUCKETS]f64 = [_]f64{0.0} ** MAX_BUCKETS,
    n_completed: usize = 0,
    write_idx: usize = 0,
    /// Lifetime signed volume (for simple imbalance feature).
    cum_buy: f64 = 0.0,
    cum_sell: f64 = 0.0,
    /// EWMA of signed trade flow (fast toxicity proxy).
    ewma_signed: f64 = 0.0,
    ewma_alpha: f64 = 0.2,

    pub fn init(cfg: ToxicityConfig) Tracker {
        var w = cfg.window_buckets;
        if (w == 0) w = 1;
        if (w > MAX_BUCKETS) w = MAX_BUCKETS;
        var c = cfg;
        c.window_buckets = w;
        if (c.bucket_volume <= 0.0) c.bucket_volume = 1.0;
        return .{ .cfg = c };
    }

    fn closeBucket(self: *Tracker) void {
        const tot = self.buy_vol + self.sell_vol;
        const imb: f64 = if (tot > 0.0) @abs(self.buy_vol - self.sell_vol) / tot else 0.0;
        self.imbalances[self.write_idx] = imb;
        self.write_idx = (self.write_idx + 1) % self.cfg.window_buckets;
        if (self.n_completed < self.cfg.window_buckets) self.n_completed += 1;
        self.buy_vol = 0.0;
        self.sell_vol = 0.0;
    }

    /// Ingest one tape print. `signed_volume` > 0 = buy aggressor.
    pub fn onTrade(self: *Tracker, signed_volume: f64) void {
        if (signed_volume == 0.0) return;
        const abs_v = @abs(signed_volume);
        var remaining = abs_v;
        const is_buy = signed_volume > 0.0;
        if (is_buy) {
            self.cum_buy += abs_v;
        } else {
            self.cum_sell += abs_v;
        }
        self.ewma_signed = self.ewma_alpha * signed_volume + (1.0 - self.ewma_alpha) * self.ewma_signed;

        // Consume volume into fixed-size buckets (may close several on a large print).
        var guard: usize = 0;
        while (remaining > 0.0 and guard < 64) : (guard += 1) {
            const open = self.buy_vol + self.sell_vol;
            const room = self.cfg.bucket_volume - open;
            const take = @min(remaining, room);
            if (is_buy) {
                self.buy_vol += take;
            } else {
                self.sell_vol += take;
            }
            remaining -= take;
            if (self.buy_vol + self.sell_vol + 1e-12 >= self.cfg.bucket_volume) {
                self.closeBucket();
            }
        }
    }

    /// Convenience: classify by side vs mid (tick rule lite).
    pub fn onPrint(self: *Tracker, price: f64, size: i32, mid: f64) void {
        if (size == 0) return;
        const abs_sz: f64 = @floatFromInt(if (size < 0) -size else size);
        const signed: f64 = if (price >= mid) abs_sz else -abs_sz;
        self.onTrade(signed);
    }

    /// Rolling average of completed-bucket imbalances ∈ [0, 1].
    /// Research-grade VPIN proxy (NOT calibrated production VPIN).
    pub fn vpin(self: *const Tracker) f64 {
        if (self.n_completed == 0) return 0.0;
        var sum: f64 = 0.0;
        var i: usize = 0;
        while (i < self.n_completed) : (i += 1) {
            sum += self.imbalances[i];
        }
        return sum / @as(f64, @floatFromInt(self.n_completed));
    }

    /// Lifetime signed imbalance (B−S)/(B+S) ∈ [−1, 1].
    pub fn tradeImbalance(self: *const Tracker) f64 {
        const tot = self.cum_buy + self.cum_sell;
        if (tot <= 0.0) return 0.0;
        return (self.cum_buy - self.cum_sell) / tot;
    }

    /// Absolute imbalance in [0, 1] — good Score feature.
    pub fn absImbalance(self: *const Tracker) f64 {
        return @abs(self.tradeImbalance());
    }

    /// Normalized |EWMA signed flow| feature in [0, 1] (soft).
    pub fn ewmaToxicity(self: *const Tracker) f64 {
        // Map |ewma| through a soft scale relative to bucket size.
        const scale = @max(self.cfg.bucket_volume * 0.25, 1.0);
        return @min(1.0, @abs(self.ewma_signed) / scale);
    }

    /// Composite research feature in [0, 1] blending VPIN + |imbalance| + EWMA.
    pub fn composite(self: *const Tracker) f64 {
        const v = self.vpin();
        const a = self.absImbalance();
        const e = self.ewmaToxicity();
        return @min(1.0, 0.5 * v + 0.3 * a + 0.2 * e);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "balanced flow → low VPIN" {
    var tr = Tracker.init(.{ .bucket_volume = 10.0, .window_buckets = 5 });
    // Alternate +5 / −5 → each bucket balanced
    var i: usize = 0;
    while (i < 20) : (i += 1) {
        tr.onTrade(5.0);
        tr.onTrade(-5.0);
    }
    try std.testing.expect(tr.vpin() < 0.15);
    try std.testing.expect(tr.absImbalance() < 0.05);
}

test "one-sided flow → high VPIN" {
    var tr = Tracker.init(.{ .bucket_volume = 10.0, .window_buckets = 5 });
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        tr.onTrade(10.0); // all buys
    }
    try std.testing.expect(tr.vpin() > 0.9);
    try std.testing.expect(tr.tradeImbalance() > 0.9);
    try std.testing.expect(tr.composite() > 0.5);
}

test "onPrint tick rule vs mid" {
    var tr = Tracker.init(.{ .bucket_volume = 5.0, .window_buckets = 4 });
    tr.onPrint(10.1, 5, 10.0); // buy
    tr.onPrint(9.9, 5, 10.0); // sell
    try std.testing.expect(tr.n_completed >= 1);
}

test "composite in unit interval" {
    var tr = Tracker.init(.{});
    tr.onTrade(3.0);
    tr.onTrade(-1.0);
    const c = tr.composite();
    try std.testing.expect(c >= 0.0 and c <= 1.0);
}
