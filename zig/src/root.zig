//! jev_omm Zig hot-path module root.
//! Import this module from demo/bench; C ABI is in c_abi.zig (compiled into shared lib).

pub const types = @import("types.zig");
pub const black_scholes = @import("black_scholes.zig");
pub const as_quoter = @import("as_quoter.zig");
pub const gueant = @import("gueant.zig");
pub const multi_strike = @import("multi_strike.zig");
pub const risk_limits = @import("risk_limits.zig");
pub const fills = @import("fills.zig");
pub const pnl = @import("pnl.zig");
pub const surface = @import("surface.zig");
pub const markout = @import("markout.zig");
pub const event_log = @import("event_log.zig");
pub const parity = @import("parity.zig");
pub const combos = @import("combos.zig");
pub const hedge = @import("hedge.zig");
pub const toxicity = @import("toxicity.zig");
pub const scenario = @import("scenario.zig");
pub const svi = @import("svi.zig");
pub const gueant_ode = @import("gueant_ode.zig");
pub const lob = @import("lob.zig");
pub const term_book = @import("term_book.zig");
pub const option_mm = @import("option_mm.zig");
pub const hawkes = @import("hawkes.zig");
pub const varswap = @import("varswap.zig");
pub const training = @import("training.zig");
pub const flow_signals = @import("flow_signals.zig");
pub const positioning = @import("positioning.zig");
pub const state_os = @import("state_os.zig");
pub const c_abi = @import("c_abi.zig");

test {
    _ = types;
    _ = black_scholes;
    _ = as_quoter;
    _ = gueant;
    _ = multi_strike;
    _ = risk_limits;
    _ = fills;
    _ = pnl;
    _ = surface;
    _ = markout;
    _ = event_log;
    _ = parity;
    _ = combos;
    _ = hedge;
    _ = toxicity;
    _ = scenario;
    _ = svi;
    _ = gueant_ode;
    _ = lob;
    _ = term_book;
    _ = option_mm;
    _ = hawkes;
    _ = varswap;
    _ = training;
    _ = flow_signals;
    _ = positioning;
    _ = state_os;
    _ = c_abi;
}
