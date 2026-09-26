//! Replay harness: read JSONL event log → recompute markout / PnL.
//! Usage: jev_omm_replay [path.jsonl]
//! Default path: jev_omm_events.jsonl

const std = @import("std");
const root = @import("jev_omm");
const event_log = root.event_log;

pub fn main(init: std.process.Init) !void {
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer it.deinit();
    _ = it.skip(); // exe
    const path = it.next() orelse "jev_omm_events.jsonl";

    std.debug.print("Jev Options MM — event log replay\n", .{});
    std.debug.print("path={s}\n\n", .{path});

    const r = try event_log.replayFile(path, init.gpa);
    std.debug.print("events={d}  fills={d}  breaches={d}  decisions={d}  last_seq={d}\n", .{
        r.n_events, r.n_fills, r.n_breaches, r.n_decisions, r.last_seq,
    });
    std.debug.print("inventory={d}  cash={d:.4}  last_mid={d:.4}  marked_pnl={d:.4}\n", .{
        r.qty, r.cash, r.last_mid, r.marked_pnl,
    });
    const a = r.attr;
    std.debug.print("\n=== Recomputed markout attribution ===\n", .{});
    std.debug.print("fills={d}  contracts={d}\n", .{ a.n_fills, a.total_contracts });
    std.debug.print("spread_capture     {d:>10.4}\n", .{a.spread_capture});
    std.debug.print("markout_1step      {d:>10.4}  (adverse {d:>10.4})  n={d}\n", .{ a.markout[0], a.adverse[0], a.resolved[0] });
    std.debug.print("markout_5step      {d:>10.4}  (adverse {d:>10.4})  n={d}\n", .{ a.markout[1], a.adverse[1], a.resolved[1] });
    std.debug.print("markout_30step     {d:>10.4}  (adverse {d:>10.4})  n={d}\n", .{ a.markout[2], a.adverse[2], a.resolved[2] });
    std.debug.print("inventory_mtm      {d:>10.4}\n", .{a.inventory_mtm});
    std.debug.print("\nlog_sha256={s}\n", .{r.log_sha256_hex});
}
