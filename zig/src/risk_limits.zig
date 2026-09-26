//! Hard risk limits that stop quoting when breached.
//! Port of jev_omm/risk/limits.py

const std = @import("std");
const types = @import("types.zig");
const Greeks = types.Greeks;
const RiskConfig = types.RiskConfig;
const RiskSnapshot = types.RiskSnapshot;

fn setReason(snap: *RiskSnapshot, comptime fmt: []const u8, args: anytype) void {
    const written = std.fmt.bufPrint(&snap.breach_reason, fmt, args) catch {
        const msg = "breach";
        @memcpy(snap.breach_reason[0..msg.len], msg);
        snap.breach_len = msg.len;
        return;
    };
    snap.breach_len = written.len;
}

pub fn evaluateRisk(
    inventory: i32,
    greeks_per_contract: *const Greeks,
    cash_pnl: f64,
    cfg: *const RiskConfig,
) RiskSnapshot {
    const inv = @as(f64, @floatFromInt(inventory));
    const delta = inv * greeks_per_contract.delta;
    const gamma = inv * greeks_per_contract.gamma;
    const vega = inv * greeks_per_contract.vega;

    var snap = RiskSnapshot{
        .inventory = inventory,
        .delta = delta,
        .gamma = gamma,
        .vega = vega,
        .cash_pnl = cash_pnl,
        .quoting_allowed = true,
    };

    if (@abs(inventory) > cfg.max_abs_inventory) {
        setReason(&snap, "inventory |{d}| > {d}", .{ inventory, cfg.max_abs_inventory });
        snap.quoting_allowed = false;
    } else if (@abs(delta) > cfg.max_abs_delta) {
        setReason(&snap, "delta |{d:.4}| > {d}", .{ delta, cfg.max_abs_delta });
        snap.quoting_allowed = false;
    } else if (@abs(vega) > cfg.max_abs_vega) {
        setReason(&snap, "vega |{d:.4}| > {d}", .{ vega, cfg.max_abs_vega });
        snap.quoting_allowed = false;
    } else if (@abs(gamma) > cfg.max_abs_gamma) {
        setReason(&snap, "gamma |{d:.6}| > {d}", .{ gamma, cfg.max_abs_gamma });
        snap.quoting_allowed = false;
    } else if (cash_pnl < -@abs(cfg.max_loss)) {
        setReason(&snap, "cash_pnl {d:.2} below -{d}", .{ cash_pnl, @abs(cfg.max_loss) });
        snap.quoting_allowed = false;
    }
    return snap;
}

test "inventory limit trips" {
    const cfg = RiskConfig{ .max_abs_inventory = 10 };
    const g = Greeks{ .delta = 0.5, .gamma = 0.02, .vega = 10.0, .theta = -5.0 };
    const snap = evaluateRisk(11, &g, 0.0, &cfg);
    try std.testing.expect(!snap.quoting_allowed);
    try std.testing.expect(std.mem.indexOf(u8, snap.reasonSlice(), "inventory") != null);
}

test "within limits allows quoting" {
    const cfg = RiskConfig{};
    const g = Greeks{ .delta = 0.4, .gamma = 0.01, .vega = 8.0, .theta = -2.0 };
    const snap = evaluateRisk(3, &g, 10.0, &cfg);
    try std.testing.expect(snap.quoting_allowed);
    try std.testing.expect(snap.breach_len == 0);
}

test "pnl loss limit trips" {
    const cfg = RiskConfig{ .max_loss = 100.0 };
    const g = Greeks{ .delta = 0.1, .gamma = 0.01, .vega = 1.0, .theta = -1.0 };
    const snap = evaluateRisk(0, &g, -150.0, &cfg);
    try std.testing.expect(!snap.quoting_allowed);
    try std.testing.expect(std.mem.indexOf(u8, snap.reasonSlice(), "cash_pnl") != null);
}
