//! Append-only sequenced event log (Jane Street / exchange-inspired).
//!
//! Events carry monotonic `seq` + sim `ts`. Preferred research format: JSONL
//! (one JSON object per line). Optional compact binary can come later.
//!
//! Types: BookTop | UnderlyingTick | Quote | Fill | Cancel | DecisionSnapshot | RiskBreach
//!
//! Determinism: same seed → identical JSONL bytes → identical SHA-256.

const std = @import("std");
const types = @import("types.zig");
const markout = @import("markout.zig");
const pnl = @import("pnl.zig");
const Side = types.Side;
const Fill = types.Fill;
const Quote = types.Quote;

pub const EventKind = enum {
    book_top,
    underlying_tick,
    quote,
    fill,
    cancel,
    decision_snapshot,
    risk_breach,
    hedge_fill,
    greek_pnl,

    pub fn jsonName(self: EventKind) []const u8 {
        return switch (self) {
            .book_top => "BookTop",
            .underlying_tick => "UnderlyingTick",
            .quote => "Quote",
            .fill => "Fill",
            .cancel => "Cancel",
            .decision_snapshot => "DecisionSnapshot",
            .risk_breach => "RiskBreach",
            .hedge_fill => "HedgeFill",
            .greek_pnl => "GreekPnl",
        };
    }
};

/// Max length of a single JSONL record (research-sized payloads).
pub const MAX_LINE: usize = 4096;

pub const Log = struct {
    allocator: std.mem.Allocator,
    buf: std.ArrayList(u8) = .empty,
    seq: u64 = 0,
    hasher: std.crypto.hash.sha2.Sha256 = std.crypto.hash.sha2.Sha256.init(.{}),
    event_count: u64 = 0,

    pub fn init(allocator: std.mem.Allocator) Log {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Log) void {
        self.buf.deinit(self.allocator);
    }

    pub fn bytes(self: *const Log) []const u8 {
        return self.buf.items;
    }

    /// SHA-256 of the exact JSONL bytes (including newlines), lowercase hex.
    pub fn sha256Hex(self: *const Log) [64]u8 {
        var h = self.hasher;
        const dig = h.finalResult();
        return std.fmt.bytesToHex(dig, .lower);
    }

    fn commitLine(self: *Log, line: []const u8) !void {
        try self.buf.appendSlice(self.allocator, line);
        if (line.len == 0 or line[line.len - 1] != '\n') {
            try self.buf.append(self.allocator, '\n');
            self.hasher.update(line);
            self.hasher.update("\n");
        } else {
            self.hasher.update(line);
        }
        self.event_count += 1;
    }

    fn nextSeq(self: *Log) u64 {
        self.seq += 1;
        return self.seq;
    }

    pub fn appendUnderlyingTick(self: *Log, ts: f64, step: u64, spot: f64) !void {
        const seq = self.nextSeq();
        var line_buf: [MAX_LINE]u8 = undefined;
        const line = try std.fmt.bufPrint(&line_buf, "{{\"seq\":{d},\"ts\":{d:.10},\"type\":\"UnderlyingTick\",\"step\":{d},\"spot\":{d:.10}}}", .{ seq, ts, step, spot });
        try self.commitLine(line);
    }

    pub fn appendBookTop(
        self: *Log,
        ts: f64,
        step: u64,
        bid: f64,
        ask: f64,
        bid_sz: i32,
        ask_sz: i32,
        mid: f64,
    ) !void {
        const seq = self.nextSeq();
        var line_buf: [MAX_LINE]u8 = undefined;
        const line = try std.fmt.bufPrint(&line_buf, "{{\"seq\":{d},\"ts\":{d:.10},\"type\":\"BookTop\",\"step\":{d},\"bid\":{d:.10},\"ask\":{d:.10},\"bid_sz\":{d},\"ask_sz\":{d},\"mid\":{d:.10}}}", .{ seq, ts, step, bid, ask, bid_sz, ask_sz, mid });
        try self.commitLine(line);
    }

    pub fn appendQuote(self: *Log, ts: f64, step: u64, q: *const Quote) !void {
        const seq = self.nextSeq();
        var line_buf: [MAX_LINE]u8 = undefined;
        const line = try std.fmt.bufPrint(&line_buf, "{{\"seq\":{d},\"ts\":{d:.10},\"type\":\"Quote\",\"step\":{d},\"bid\":{d:.10},\"ask\":{d:.10},\"bid_sz\":{d},\"ask_sz\":{d},\"reservation\":{d:.10},\"half_spread\":{d:.10}}}", .{ seq, ts, step, q.bid, q.ask, q.bid_size, q.ask_size, q.reservation, q.half_spread });
        try self.commitLine(line);
    }

    /// Quote with strike tag (multi-strike strip). Backward-compatible extra fields.
    pub fn appendQuoteStrike(self: *Log, ts: f64, step: u64, strike: f64, q: *const Quote) !void {
        const seq = self.nextSeq();
        var line_buf: [MAX_LINE]u8 = undefined;
        const line = try std.fmt.bufPrint(&line_buf, "{{\"seq\":{d},\"ts\":{d:.10},\"type\":\"Quote\",\"step\":{d},\"strike\":{d:.10},\"bid\":{d:.10},\"ask\":{d:.10},\"bid_sz\":{d},\"ask_sz\":{d},\"reservation\":{d:.10},\"half_spread\":{d:.10}}}", .{ seq, ts, step, strike, q.bid, q.ask, q.bid_size, q.ask_size, q.reservation, q.half_spread });
        try self.commitLine(line);
    }

    /// DecisionSnapshot with optional extra `state` JSON object (multi-strike portfolio greeks etc.).
    /// `state_json` must be a JSON object string, e.g. `{...}` (not null).
    pub fn appendDecisionSnapshotState(
        self: *Log,
        ts: f64,
        step: u64,
        source: []const u8,
        model: []const u8,
        answers_json: []const u8,
        confidence_json: []const u8,
        state_json: []const u8,
    ) !void {
        const seq = self.nextSeq();
        var line_buf: [MAX_LINE]u8 = undefined;
        const line = try std.fmt.bufPrint(
            &line_buf,
            "{{\"seq\":{d},\"ts\":{d:.10},\"type\":\"DecisionSnapshot\",\"step\":{d},\"source\":\"{s}\",\"model\":\"{s}\",\"answers\":{s},\"confidence\":{s},\"state\":{s}}}",
            .{ seq, ts, step, source, model, answers_json, confidence_json, state_json },
        );
        try self.commitLine(line);
    }

    pub fn appendFill(self: *Log, ts: f64, step: u64, f: *const Fill) !void {
        const seq = self.nextSeq();
        const side = @tagName(f.side);
        var line_buf: [MAX_LINE]u8 = undefined;
        const line = try std.fmt.bufPrint(&line_buf, "{{\"seq\":{d},\"ts\":{d:.10},\"type\":\"Fill\",\"step\":{d},\"side\":\"{s}\",\"price\":{d:.10},\"size\":{d},\"mid\":{d:.10}}}", .{ seq, ts, step, side, f.price, f.size, f.mid_at_fill });
        try self.commitLine(line);
    }

    pub fn appendCancel(self: *Log, ts: f64, step: u64, reason: []const u8) !void {
        const seq = self.nextSeq();
        var line_buf: [MAX_LINE]u8 = undefined;
        const line = try std.fmt.bufPrint(&line_buf, "{{\"seq\":{d},\"ts\":{d:.10},\"type\":\"Cancel\",\"step\":{d},\"reason\":\"{s}\"}}", .{ seq, ts, step, reason });
        try self.commitLine(line);
    }

    pub fn appendRiskBreach(self: *Log, ts: f64, step: u64, reason: []const u8, inventory: i32) !void {
        const seq = self.nextSeq();
        var line_buf: [MAX_LINE]u8 = undefined;
        const line = try std.fmt.bufPrint(&line_buf, "{{\"seq\":{d},\"ts\":{d:.10},\"type\":\"RiskBreach\",\"step\":{d},\"reason\":\"{s}\",\"inventory\":{d}}}", .{ seq, ts, step, reason, inventory });
        try self.commitLine(line);
    }

    /// Compact DecisionSnapshot: answers JSON object already serialized + source + model.
    /// `answers_json` is a JSON object string (no surrounding whitespace required).
    /// `confidence_json` is a JSON object of question → confidence (Choice/Score only).
    pub fn appendDecisionSnapshot(
        self: *Log,
        ts: f64,
        step: u64,
        source: []const u8,
        model: []const u8,
        answers_json: []const u8,
        confidence_json: []const u8,
    ) !void {
        const seq = self.nextSeq();
        var line_buf: [MAX_LINE]u8 = undefined;
        const line = try std.fmt.bufPrint(
            &line_buf,
            "{{\"seq\":{d},\"ts\":{d:.10},\"type\":\"DecisionSnapshot\",\"step\":{d},\"source\":\"{s}\",\"model\":\"{s}\",\"answers\":{s},\"confidence\":{s}}}",
            .{ seq, ts, step, source, model, answers_json, confidence_json },
        );
        try self.commitLine(line);
    }

    pub fn appendHedgeFill(
        self: *Log,
        ts: f64,
        step: u64,
        underlier_qty: f64,
        fill_price: f64,
        mid: f64,
        cash_delta: f64,
        slippage_cost: f64,
    ) !void {
        const seq = self.nextSeq();
        var line_buf: [MAX_LINE]u8 = undefined;
        const line = try std.fmt.bufPrint(&line_buf, "{{\"seq\":{d},\"ts\":{d:.10},\"type\":\"HedgeFill\",\"step\":{d},\"underlier_qty\":{d:.10},\"fill_price\":{d:.10},\"mid\":{d:.10},\"cash_delta\":{d:.10},\"slippage_cost\":{d:.10}}}", .{ seq, ts, step, underlier_qty, fill_price, mid, cash_delta, slippage_cost });
        try self.commitLine(line);
    }

    pub fn appendGreekPnl(
        self: *Log,
        ts: f64,
        step: u64,
        spread_capture: f64,
        hedge_slippage: f64,
        gamma_pnl: f64,
        theta_pnl: f64,
        vega_pnl: f64,
        inventory_mtm: f64,
        delta_pnl: f64,
    ) !void {
        const seq = self.nextSeq();
        var line_buf: [MAX_LINE]u8 = undefined;
        const line = try std.fmt.bufPrint(&line_buf, "{{\"seq\":{d},\"ts\":{d:.10},\"type\":\"GreekPnl\",\"step\":{d},\"spread_capture\":{d:.10},\"hedge_slippage\":{d:.10},\"gamma_pnl\":{d:.10},\"theta_pnl\":{d:.10},\"vega_pnl\":{d:.10},\"inventory_mtm\":{d:.10},\"delta_pnl\":{d:.10}}}", .{ seq, ts, step, spread_capture, hedge_slippage, gamma_pnl, theta_pnl, vega_pnl, inventory_mtm, delta_pnl });
        try self.commitLine(line);
    }

        pub fn writeJsonl(self: *const Log, path: []const u8) !void {
        const io = std.Io.Threaded.global_single_threaded.io();
        try std.Io.Dir.cwd().writeFile(io, .{
            .sub_path = path,
            .data = self.buf.items,
        });
    }
};

pub const ReplayResult = struct {
    n_events: u64 = 0,
    n_fills: u64 = 0,
    n_breaches: u64 = 0,
    n_decisions: u64 = 0,
    last_seq: u64 = 0,
    cash: f64 = 0.0,
    qty: i32 = 0,
    last_mid: f64 = 0.0,
    marked_pnl: f64 = 0.0,
    attr: markout.Attribution = .{},
    log_sha256_hex: [64]u8 = [_]u8{'0'} ** 64,
};

fn findKey(hay: []const u8, key: []const u8) ?usize {
    // naive search for "key":
    var i: usize = 0;
    while (i + key.len + 3 < hay.len) : (i += 1) {
        if (hay[i] == '"' and std.mem.startsWith(u8, hay[i + 1 ..], key) and hay[i + 1 + key.len] == '"' and hay[i + 2 + key.len] == ':') {
            return i + 3 + key.len;
        }
    }
    return null;
}

fn parseF64After(hay: []const u8, key: []const u8) ?f64 {
    const start = findKey(hay, key) orelse return null;
    var i = start;
    while (i < hay.len and (hay[i] == ' ' or hay[i] == '\t')) : (i += 1) {}
    const rem = hay[i..];
    const end = std.mem.indexOfAny(u8, rem, ",}") orelse rem.len;
    return std.fmt.parseFloat(f64, rem[0..end]) catch null;
}

fn parseI64After(hay: []const u8, key: []const u8) ?i64 {
    const start = findKey(hay, key) orelse return null;
    var i = start;
    while (i < hay.len and (hay[i] == ' ' or hay[i] == '\t')) : (i += 1) {}
    const rem = hay[i..];
    const end = std.mem.indexOfAny(u8, rem, ",}") orelse rem.len;
    return std.fmt.parseInt(i64, rem[0..end], 10) catch null;
}

fn parseU64After(hay: []const u8, key: []const u8) ?u64 {
    const v = parseI64After(hay, key) orelse return null;
    if (v < 0) return null;
    return @intCast(v);
}

fn parseStringAfter(hay: []const u8, key: []const u8, out: []u8) ?[]const u8 {
    const start = findKey(hay, key) orelse return null;
    var i = start;
    while (i < hay.len and (hay[i] == ' ' or hay[i] == '\t')) : (i += 1) {}
    if (i >= hay.len or hay[i] != '"') return null;
    i += 1;
    const s0 = i;
    while (i < hay.len and hay[i] != '"') : (i += 1) {}
    const slice = hay[s0..i];
    if (slice.len > out.len) return null;
    @memcpy(out[0..slice.len], slice);
    return out[0..slice.len];
}

fn eventTypeOf(line: []const u8) ?EventKind {
    var tmp: [32]u8 = undefined;
    const name = parseStringAfter(line, "type", &tmp) orelse return null;
    if (std.mem.eql(u8, name, "BookTop")) return .book_top;
    if (std.mem.eql(u8, name, "UnderlyingTick")) return .underlying_tick;
    if (std.mem.eql(u8, name, "Quote")) return .quote;
    if (std.mem.eql(u8, name, "Fill")) return .fill;
    if (std.mem.eql(u8, name, "Cancel")) return .cancel;
    if (std.mem.eql(u8, name, "DecisionSnapshot")) return .decision_snapshot;
    if (std.mem.eql(u8, name, "RiskBreach")) return .risk_breach;
    if (std.mem.eql(u8, name, "HedgeFill")) return .hedge_fill;
    if (std.mem.eql(u8, name, "GreekPnl")) return .greek_pnl;
    return null;
}

/// Replay JSONL → recompute markout / PnL from Fill + mid timeline (BookTop.mid).
pub fn replayJsonl(jsonl: []const u8) ReplayResult {
    var result: ReplayResult = .{};
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(jsonl);
    result.log_sha256_hex = std.fmt.bytesToHex(hasher.finalResult(), .lower);

    var tracker = markout.Tracker.init(markout.DEFAULT_HORIZONS);
    // last_step unused; BookTop drives markout steps

    var iter = std.mem.splitScalar(u8, jsonl, '\n');
    while (iter.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0) continue;
        result.n_events += 1;
        const kind = eventTypeOf(line) orelse continue;
        if (parseU64After(line, "seq")) |s| result.last_seq = s;

        switch (kind) {
            .book_top => {
                const step = parseU64After(line, "step") orelse 0;
                const mid = parseF64After(line, "mid") orelse continue;
                tracker.onStep(step, mid);
                result.last_mid = mid;
            },
            .underlying_tick => {
                // spot-only; mid comes from BookTop
            },
            .fill => {
                const step = parseU64After(line, "step") orelse 0;
                const price = parseF64After(line, "price") orelse continue;
                const size_i = parseI64After(line, "size") orelse continue;
                const mid = parseF64After(line, "mid") orelse continue;
                var side_buf: [8]u8 = undefined;
                const side_s = parseStringAfter(line, "side", &side_buf) orelse continue;
                const side: Side = if (std.mem.eql(u8, side_s, "ask")) .ask else .bid;
                const size: i32 = @intCast(size_i);
                const f = Fill{
                    .time = parseF64After(line, "ts") orelse 0.0,
                    .side = side,
                    .price = price,
                    .size = size,
                    .mid_at_fill = mid,
                };
                // Apply cash/qty
                const cash_ptr = &result.cash;
                const qty_ptr = &result.qty;
                // inline apply (same as fills.applyFillCash)
                const notional = price * @as(f64, @floatFromInt(size));
                switch (side) {
                    .bid => {
                        cash_ptr.* -= notional;
                        qty_ptr.* += size;
                    },
                    .ask => {
                        cash_ptr.* += notional;
                        qty_ptr.* -= size;
                    },
                }
                tracker.onFill(step, &f);
                result.n_fills += 1;
                result.last_mid = mid;
            },
            .risk_breach => {
                result.n_breaches += 1;
            },
            .decision_snapshot => {
                result.n_decisions += 1;
            },
            .quote, .cancel, .hedge_fill, .greek_pnl => {},
        }
    }

    result.attr = tracker.attr;
    result.marked_pnl = pnl.markedPnl(result.cash, result.qty, result.last_mid);
    return result;
}

pub fn replayFile(path: []const u8, allocator: std.mem.Allocator) !ReplayResult {
    const io = std.Io.Threaded.global_single_threaded.io();
    const data = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(data);
    return replayJsonl(data);
}

test "event log append + hash deterministic" {
    const a = std.testing.allocator;
    var log1 = Log.init(a);
    defer log1.deinit();
    var log2 = Log.init(a);
    defer log2.deinit();

    try log1.appendUnderlyingTick(0.0, 0, 100.0);
    try log1.appendBookTop(0.0, 0, 3.9, 4.1, 2, 2, 4.0);
    const fill = Fill{ .time = 0.0, .side = .bid, .price = 3.9, .size = 1, .mid_at_fill = 4.0 };
    try log1.appendFill(0.0, 0, &fill);
    try log1.appendDecisionSnapshot(0.0, 0, "fallback", "fallback-heuristic", "{\"regime\":{\"type\":\"choice\",\"choice\":\"calm\",\"confidence\":0.8}}", "{\"regime\":0.8}");
    try log1.appendRiskBreach(0.0, 1, "max_abs_inventory", 21);
    try log1.appendCancel(0.0, 1, "pull");

    try log2.appendUnderlyingTick(0.0, 0, 100.0);
    try log2.appendBookTop(0.0, 0, 3.9, 4.1, 2, 2, 4.0);
    try log2.appendFill(0.0, 0, &fill);
    try log2.appendDecisionSnapshot(0.0, 0, "fallback", "fallback-heuristic", "{\"regime\":{\"type\":\"choice\",\"choice\":\"calm\",\"confidence\":0.8}}", "{\"regime\":0.8}");
    try log2.appendRiskBreach(0.0, 1, "max_abs_inventory", 21);
    try log2.appendCancel(0.0, 1, "pull");

    try std.testing.expectEqualStrings(log1.bytes(), log2.bytes());
    const h1 = log1.sha256Hex();
    const h2 = log2.sha256Hex();
    try std.testing.expectEqualStrings(&h1, &h2);
    try std.testing.expect(log1.seq == 6);
}

test "replay recomputes markout from fills" {
    const a = std.testing.allocator;
    var log = Log.init(a);
    defer log.deinit();

    // step 0 mid 10, fill bid @ 9.5 size 2, then mid drops to 9 at step 1
    try log.appendBookTop(0.0, 0, 9.5, 10.5, 2, 2, 10.0);
    const fill = Fill{ .time = 0.0, .side = .bid, .price = 9.5, .size = 2, .mid_at_fill = 10.0 };
    try log.appendFill(0.0, 0, &fill);
    try log.appendBookTop(0.001, 1, 8.5, 9.5, 2, 2, 9.0);

    const r = replayJsonl(log.bytes());
    try std.testing.expect(r.n_fills == 1);
    try std.testing.expect(r.qty == 2);
    try std.testing.expect(@abs(r.attr.spread_capture - 1.0) < 1e-9);
    try std.testing.expect(r.attr.resolved[0] == 1);
    try std.testing.expect(@abs(r.attr.markout[0] - (-2.0)) < 1e-9);
    try std.testing.expect(@abs(r.marked_pnl - (r.cash + @as(f64, @floatFromInt(r.qty)) * r.last_mid)) < 1e-9);

    // hash of bytes matches log.sha256Hex
    const h = log.sha256Hex();
    try std.testing.expectEqualStrings(&h, &r.log_sha256_hex);
}

test "same seed path produces stable hash string length" {
    const a = std.testing.allocator;
    var log = Log.init(a);
    defer log.deinit();
    try log.appendUnderlyingTick(0.0, 0, 100.0);
    const hex = log.sha256Hex();
    try std.testing.expect(hex.len == 64);
}
