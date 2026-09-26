//! Mark-to-model PnL helpers.
//! Port of jev_omm/pnl/mark.py

const std = @import("std");

pub fn markedPnl(cash: f64, qty: i32, option_mid: f64) f64 {
    return cash + @as(f64, @floatFromInt(qty)) * option_mid;
}

test "marked pnl basic" {
    try std.testing.expect(@abs(markedPnl(10.0, 2, 5.0) - 20.0) < 1e-12);
    try std.testing.expect(@abs(markedPnl(-8.0, -1, 3.0) - (-11.0)) < 1e-12);
}
