//! Markout attribution for paper fills.
//!
//! On each fill we record mid_at_fill and later resolve mids at fixed step
//! horizons (default 1 / 5 / 30 steps). Decomposition of maker PnL:
//!
//!   spread_capture  = edge vs mid at fill
//!                     bid fill (we buy):  (mid0 - px) * size
//!                     ask fill (we sell): (px - mid0) * size
//!   markout_h       = signed_size * (mid_h - mid0)
//!                     (= inventory PnL on the filled size over horizon h)
//!   adverse_h       = -markout_h   (positive ⇒ mid moved against us)
//!   inventory_mtm   = open_qty * (mid_now - mid_ref) after resolving pending
//!                     markouts; reported as residual vs total marked PnL.
//!
//! Sim / research only.

const std = @import("std");
const types = @import("types.zig");
const Side = types.Side;
const Fill = types.Fill;

pub const N_HORIZONS: usize = 3;
pub const DEFAULT_HORIZONS: [N_HORIZONS]u32 = .{ 1, 5, 30 };

pub const PendingFill = struct {
    step: u64 = 0,
    side: Side = .bid,
    price: f64 = 0.0,
    size: i32 = 0,
    mid0: f64 = 0.0,
    /// Bitmask: bit i set ⇒ horizon i not yet resolved
    pending_mask: u8 = 0,
};

pub const Attribution = struct {
    n_fills: u32 = 0,
    total_contracts: i64 = 0,
    spread_capture: f64 = 0.0,
    markout: [N_HORIZONS]f64 = .{ 0.0, 0.0, 0.0 },
    adverse: [N_HORIZONS]f64 = .{ 0.0, 0.0, 0.0 },
    /// Resolved count per horizon (fills that lived long enough)
    resolved: [N_HORIZONS]u32 = .{ 0, 0, 0 },
    inventory_mtm: f64 = 0.0,
};

pub const Tracker = struct {
    horizons: [N_HORIZONS]u32 = DEFAULT_HORIZONS,
    pending: [64]PendingFill = [_]PendingFill{.{}} ** 64,
    pending_len: usize = 0,
    attr: Attribution = .{},
    /// Mid at last step (for inventory MTM)
    last_mid: f64 = 0.0,
    open_qty: i32 = 0,

    pub fn init(horizons: [N_HORIZONS]u32) Tracker {
        return .{ .horizons = horizons };
    }

    pub fn spreadEdge(side: Side, price: f64, mid: f64, size: i32) f64 {
        const sz = @as(f64, @floatFromInt(size));
        return switch (side) {
            .bid => (mid - price) * sz, // bought below mid
            .ask => (price - mid) * sz, // sold above mid
        };
    }

    pub fn signedSize(side: Side, size: i32) i32 {
        return switch (side) {
            .bid => size,
            .ask => -size,
        };
    }

    /// Record a fill at `step` with current mid. Accumulates spread capture immediately.
    pub fn onFill(self: *Tracker, step: u64, fill: *const Fill) void {
        const edge = spreadEdge(fill.side, fill.price, fill.mid_at_fill, fill.size);
        self.attr.spread_capture += edge;
        self.attr.n_fills += 1;
        self.attr.total_contracts += fill.size;
        self.open_qty += signedSize(fill.side, fill.size);

        if (self.pending_len >= self.pending.len) {
            // Drop oldest slot to keep bounded (research tracker, not audit log)
            var j: usize = 0;
            while (j + 1 < self.pending_len) : (j += 1) {
                self.pending[j] = self.pending[j + 1];
            }
            self.pending_len -= 1;
        }
        var mask: u8 = 0;
        var h: usize = 0;
        while (h < N_HORIZONS) : (h += 1) {
            mask |= (@as(u8, 1) << @intCast(h));
        }
        self.pending[self.pending_len] = .{
            .step = step,
            .side = fill.side,
            .price = fill.price,
            .size = fill.size,
            .mid0 = fill.mid_at_fill,
            .pending_mask = mask,
        };
        self.pending_len += 1;
    }

    /// Advance to `step` with current option mid; resolve any due horizons.
    pub fn onStep(self: *Tracker, step: u64, mid: f64) void {
        // Inventory MTM since previous mid
        if (self.last_mid != 0.0 and self.open_qty != 0) {
            self.attr.inventory_mtm += @as(f64, @floatFromInt(self.open_qty)) * (mid - self.last_mid);
        }
        self.last_mid = mid;

        var i: usize = 0;
        while (i < self.pending_len) {
            var pf = &self.pending[i];
            var h: usize = 0;
            while (h < N_HORIZONS) : (h += 1) {
                const bit: u8 = @as(u8, 1) << @intCast(h);
                if ((pf.pending_mask & bit) == 0) continue;
                const target = pf.step + self.horizons[h];
                if (step >= target) {
                    const signed = @as(f64, @floatFromInt(signedSize(pf.side, pf.size)));
                    const mo = signed * (mid - pf.mid0);
                    self.attr.markout[h] += mo;
                    self.attr.adverse[h] += -mo;
                    self.attr.resolved[h] += 1;
                    pf.pending_mask &= ~bit;
                }
            }
            if (pf.pending_mask == 0) {
                // compact: swap-remove
                self.pending_len -= 1;
                self.pending[i] = self.pending[self.pending_len];
                // do not advance i
            } else {
                i += 1;
            }
        }
    }
};

test "spread edge signs" {
    try std.testing.expect(@abs(Tracker.spreadEdge(.bid, 9.0, 10.0, 2) - 2.0) < 1e-12);
    try std.testing.expect(@abs(Tracker.spreadEdge(.ask, 11.0, 10.0, 2) - 2.0) < 1e-12);
}

test "markout resolves at horizons" {
    var tr = Tracker.init(DEFAULT_HORIZONS);
    const fill = Fill{ .time = 0.0, .side = .bid, .price = 9.5, .size = 2, .mid_at_fill = 10.0 };
    tr.onFill(0, &fill);
    try std.testing.expect(@abs(tr.attr.spread_capture - 1.0) < 1e-12);

    // Step 0 mid — no horizon due yet
    tr.onStep(0, 10.0);
    try std.testing.expect(tr.attr.resolved[0] == 0);

    // Mid drops to 9.0 at step 1 → markout = +2*(9-10) = -2 (adverse +2)
    tr.onStep(1, 9.0);
    try std.testing.expect(tr.attr.resolved[0] == 1);
    try std.testing.expect(@abs(tr.attr.markout[0] - (-2.0)) < 1e-12);
    try std.testing.expect(@abs(tr.attr.adverse[0] - 2.0) < 1e-12);

    // Advance to step 5
    var s: u64 = 2;
    while (s <= 5) : (s += 1) tr.onStep(s, 9.0);
    try std.testing.expect(tr.attr.resolved[1] == 1);

    // Advance to step 30
    s = 6;
    while (s <= 30) : (s += 1) tr.onStep(s, 9.0);
    try std.testing.expect(tr.attr.resolved[2] == 1);
    try std.testing.expect(tr.pending_len == 0);
}
