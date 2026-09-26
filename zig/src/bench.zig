//! Microbenchmarks for BS greeks + quote cycle.
//! `zig build bench -Doptimize=ReleaseFast`

const std = @import("std");
const linux = std.os.linux;
const root = @import("jev_omm");
const bs = root.black_scholes;
const asq = root.as_quoter;
const risk = root.risk_limits;
const pnl = root.pnl;
const types = root.types;
const surface = root.surface;

fn nowNs() i128 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(linux.CLOCK.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

fn blackBox(x: f64) f64 {
    return @as(*const volatile f64, @ptrCast(&x)).*;
}

fn benchOne(comptime name: []const u8, iters: usize, func: *const fn (i: usize) f64) void {
    var w: usize = 0;
    while (w < iters / 20 + 1) : (w += 1) {
        std.mem.doNotOptimizeAway(func(w));
    }
    const t0 = nowNs();
    var acc: f64 = 0.0;
    var i: usize = 0;
    while (i < iters) : (i += 1) {
        acc += func(i);
    }
    const t1 = nowNs();
    std.mem.doNotOptimizeAway(acc);
    const total_ns = t1 - t0;
    const per_ns = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iters));
    const thrpt = 1e9 / per_ns;
    std.debug.print("{s}\n", .{name});
    std.debug.print("  iters={d}  total_ns={d}  per={d:.3} ns  thrpt={d:.3} Melem/s\n", .{
        iters,
        total_ns,
        per_ns,
        thrpt / 1e6,
    });
}

fn doPriceAndGreeks(i: usize) f64 {
    const spot = blackBox(100.0 + @as(f64, @floatFromInt(i % 50)) * 0.01);
    const r = bs.priceAndGreeks(spot, 100.0, 0.25, 0.05, 0.0, 0.22, true);
    return r.price + r.greeks.delta + r.greeks.gamma + r.greeks.vega;
}

fn doBatch64(i: usize) f64 {
    _ = i;
    var acc: f64 = 0.0;
    var k: u32 = 0;
    while (k < 64) : (k += 1) {
        const spot = blackBox(95.0 + @as(f64, @floatFromInt(k)) * 0.15);
        const r = bs.priceAndGreeks(spot, 100.0, 0.25, 0.05, 0.0, 0.22, true);
        acc += r.price + r.greeks.delta;
    }
    return acc;
}

const quoter_cfg = types.QuoterConfig{
    .gamma = 0.12,
    .kappa = 1.5,
    .sigma = 0.45,
    .quote_size = 2,
};
const risk_cfg = types.RiskConfig{};

fn doQuoteCycle(i: usize) f64 {
    const spot = blackBox(100.0 + @as(f64, @floatFromInt(i % 40)) * 0.02);
    const r = bs.priceAndGreeks(spot, 100.0, 30.0 / 365.25, 0.05, 0.0, 0.22, true);
    const marked = pnl.markedPnl(10.0, 3, r.price);
    const snap = risk.evaluateRisk(3, &r.greeks, marked, &risk_cfg);
    if (!snap.quoting_allowed) return r.price;
    const q = asq.makeQuote(r.price, 3, &quoter_cfg, quoter_cfg.t_horizon, &r.greeks, 1.0, 1.0);
    return q.bid + q.ask + q.reservation;
}

fn doQuoteCycle256(i: usize) f64 {
    _ = i;
    var acc: f64 = 0.0;
    var k: u32 = 0;
    while (k < 256) : (k += 1) {
        const spot = blackBox(98.0 + @as(f64, @floatFromInt(k)) * 0.02);
        const inv: i32 = @as(i32, @intCast(k % 11)) - 5;
        const r = bs.priceAndGreeks(spot, 100.0, 30.0 / 365.25, 0.05, 0.0, 0.22, true);
        const marked = pnl.markedPnl(0.0, inv, r.price);
        const snap = risk.evaluateRisk(inv, &r.greeks, marked, &risk_cfg);
        if (snap.quoting_allowed) {
            const q = asq.makeQuote(r.price, inv, &quoter_cfg, quoter_cfg.t_horizon, &r.greeks, 1.0, 1.0);
            acc += q.bid + q.ask;
        } else {
            acc += r.price;
        }
    }
    return acc;
}


const sabr_params = surface.SabrParams{
    .alpha = 0.22,
    .beta = 1.0,
    .rho = -0.3,
    .nu = 0.4,
};

fn doSabrIv(i: usize) f64 {
    const k = blackBox(90.0 + @as(f64, @floatFromInt(i % 40)) * 0.5);
    return surface.sabrImpliedVol(sabr_params, 100.0, k, 0.25);
}

fn doSabrIvBatch64(i: usize) f64 {
    _ = i;
    var acc: f64 = 0.0;
    var k: u32 = 0;
    while (k < 64) : (k += 1) {
        const strike = blackBox(80.0 + @as(f64, @floatFromInt(k)) * 0.6);
        acc += surface.sabrImpliedVol(sabr_params, 100.0, strike, 0.25);
    }
    return acc;
}


const gueant = root.gueant;
const multi = root.multi_strike;

const gueant_cfg = types.QuoterConfig{
    .gamma = 0.12,
    .kappa = 1.5,
    .sigma = 0.45,
    .A = 140.0,
    .mode = .gueant_asymptotic,
    .quote_size = 2,
};

fn doGueantQuote(i: usize) f64 {
    const mid = blackBox(4.0 + @as(f64, @floatFromInt(i % 30)) * 0.01);
    const inv: i32 = @as(i32, @intCast(i % 11)) - 5;
    const q = gueant.makeQuote(mid, inv, &gueant_cfg, null, 1.0, 1.0);
    return q.bid + q.ask + q.reservation + gueant.inventoryScale(&gueant_cfg);
}

fn doMultiStrikeStrip(i: usize) f64 {
    const spot = blackBox(100.0 + @as(f64, @floatFromInt(i % 20)) * 0.05);
    var strikes: [types.MAX_STRIP]f64 = undefined;
    const n = multi.buildStrikeGrid(spot, .{ .half_width = 2, .strike_step = 1.0 }, &strikes);
    var slots: [types.MAX_STRIP]types.StrikeSlot = undefined;
    _ = multi.initFlatSlots(strikes[0..n], &slots);
    // give ATM a small inventory so portfolio path is non-trivial
    if (n > 2) slots[n / 2].inventory = @as(i32, @intCast(i % 5)) - 2;
    const sabr = surface.SabrParams{ .alpha = 0.22, .beta = 1.0, .rho = -0.3, .nu = 0.4 };
    var cfg = gueant_cfg;
    cfg.portfolio_delta_penalty = 0.02;
    var out: [types.MAX_STRIP]types.StrikeQuote = undefined;
    const w = multi.quoteStrip(spot, 30.0 / 365.25, 0.05, 0.0, sabr, slots[0..n], &cfg, .{ .half_width = 2, .strike_step = 1.0 }, 1.0, 1.0, &out);
    var acc: f64 = @floatFromInt(w);
    var k: usize = 0;
    while (k < w) : (k += 1) acc += out[k].quote.bid + out[k].quote.ask;
    return acc;
}


const parity = root.parity;
const combos = root.combos;
const hedge = root.hedge;
const scenario = root.scenario;

fn doParityBox(i: usize) f64 {
    const k1 = blackBox(95.0 + @as(f64, @floatFromInt(i % 5)) * 0.1);
    const k2 = k1 + 10.0;
    const theo = parity.boxTheo(k1, k2, 0.5, 0.05);
    const c1 = parity.SideQuotes{ .bid = theo * 0.6, .ask = theo * 0.6 + 0.05 };
    const c2 = parity.SideQuotes{ .bid = 1.0, .ask = 1.05 };
    const p1 = parity.SideQuotes{ .bid = 1.0, .ask = 1.05 };
    const p2 = parity.SideQuotes{ .bid = theo * 0.4, .ask = theo * 0.4 + 0.05 };
    const box = parity.boxSpread(c1, c2, p1, p2, k1, k2, 0.5, 0.05);
    return box.theo_pv + box.buy_edge + box.implied_rate_mid;
}

fn doStraddleTheo(i: usize) f64 {
    const spot = blackBox(100.0 + @as(f64, @floatFromInt(i % 20)) * 0.05);
    const pkg = combos.straddleTheo(spot, 100.0, 0.25, 0.05, 0.0, 0.2);
    return pkg.theo + pkg.greeks.vega + pkg.greeks.gamma;
}

fn doHedgeProposeApply(i: usize) f64 {
    const d = blackBox(@as(f64, @floatFromInt(@as(i32, @intCast(i % 21)) - 10)) * 1.5);
    const cfg = hedge.HedgeConfig{ .delta_band = 5.0, .half_spread = 0.02, .flatten = true };
    const o = hedge.proposeDeltaHedge(d, &cfg);
    const f = hedge.applyHedge(0.0, 100.0, &o, &cfg);
    const g = types.Greeks{ .delta = d, .gamma = 0.05, .vega = 10.0, .theta = -2.0 };
    const step = hedge.greekPnlStep(&g, 0.1, 0.001, 1.0 / 252.0, 1.0, 0.05, -d, 0.01, f.slippage_cost);
    return o.underlier_qty + f.fill_price + step.gamma_pnl + step.total();
}

fn doScenarioMatrix(i: usize) f64 {
    const g = types.Greeks{
        .delta = blackBox(@as(f64, @floatFromInt(i % 10)) - 5.0),
        .gamma = 0.04,
        .vega = 25.0,
        .theta = -3.0,
    };
    const cfg = scenario.defaultConfig();
    const m = scenario.buildMatrix(&g, 100.0, &cfg);
    return m.min_pnl + m.max_pnl + @as(f64, @floatFromInt(m.n_spot * m.n_iv));
}

pub fn main() void {
    std.debug.print("Jev Options MM — Zig benches (ReleaseFast)\n", .{});
    std.debug.print("host=linux  CLOCK_MONOTONIC  inputs varied to defeat DCE\n\n", .{});

    benchOne("bs_greeks/price_and_greeks_atm_call", 2_000_000, &doPriceAndGreeks);
    benchOne("bs_greeks/price_and_greeks_batch_64", 50_000, &doBatch64);
    benchOne("quote_cycle/bs_risk_as_quote", 1_000_000, &doQuoteCycle);
    benchOne("quote_cycle/quote_cycle_x256", 10_000, &doQuoteCycle256);
    benchOne("surface/sabr_iv_single", 2_000_000, &doSabrIv);
    benchOne("surface/sabr_iv_batch_64", 50_000, &doSabrIvBatch64);
    benchOne("gueant/make_quote", 1_000_000, &doGueantQuote);
    benchOne("multi_strike/quote_strip_5", 100_000, &doMultiStrikeStrip);
    benchOne("parity/box_spread", 1_000_000, &doParityBox);
    benchOne("combos/straddle_theo", 1_000_000, &doStraddleTheo);
    benchOne("hedge/propose_apply_greek_pnl", 1_000_000, &doHedgeProposeApply);
    benchOne("scenario/matrix_7x5", 200_000, &doScenarioMatrix);
}
