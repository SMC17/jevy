//! Paper demo of the five Citadel-style training cases.
//!   zig build training

const std = @import("std");
const root = @import("jev_omm");
const training = root.training;

pub fn main() void {
    std.debug.print("jevy training desk — simulation / paper only\n", .{});
    std.debug.print("cases: location_arb, etf_ap_arb, liability_facilitator, mm_inventory, vol_surface_mm\n", .{});
    std.debug.print("Decision layer is not on this binary; Python demo logs fallback Choice/Score/Noul.\n\n", .{});
    for (training.CASES) |name| {
        const naive = training.runCase(name, .naive, 0.0);
        const desk = training.runCase(name, .desk, naive.absolute_pnl);
        std.debug.print("=== {s} ===\n", .{name});
        std.debug.print(
            "  naive  pnl={d:.4}  risk_adj={d:.4}  inv_pen={d:.4}  beta_pen={d:.4}  |beta|={d:.4}  |q|={d:.4}\n",
            .{ naive.absolute_pnl, naive.risk_adjusted, naive.inventory_path_penalty, naive.beta_penalty, naive.mean_abs_beta, naive.mean_abs_inventory },
        );
        std.debug.print(
            "  desk   pnl={d:.4}  risk_adj={d:.4}  relative={d:.4}  inv_pen={d:.4}  beta_pen={d:.4}  |beta|={d:.4}  |q|={d:.4}\n\n",
            .{ desk.absolute_pnl, desk.risk_adjusted, desk.relative_score, desk.inventory_path_penalty, desk.beta_penalty, desk.mean_abs_beta, desk.mean_abs_inventory },
        );
    }
}
