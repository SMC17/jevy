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

const svi = root.svi;
const ode = root.gueant_ode;
const term = root.term_book;

const svi_params = svi.SviParams{ .a = 0.04, .b = 0.12, .rho = -0.3, .m = 0.0, .sigma = 0.2 };

fn doSviIv(i: usize) f64 {
    const k = blackBox(-0.5 + @as(f64, @floatFromInt(i % 40)) * 0.025);
    return svi.impliedVol(svi_params, k, 0.25) + svi.densityG(svi_params, k);
}

fn doGueantOde(i: usize) f64 {
    const cfg = types.QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.45,
        .A = 120.0,
        .t_horizon = 0.25,
        .inventory_cap = 8,
        .ode_steps = 200,
        .min_half_spread = 0.01,
        .max_half_spread = 10.0,
        .mode = .gueant_ode,
    };
    const o = ode.optimalOffsets(&cfg, @as(i32, @intCast(i % 7)) - 3);
    return o.delta_b + o.delta_a;
}

fn doTermAggregate(i: usize) f64 {
    const spot = blackBox(100.0 + @as(f64, @floatFromInt(i % 10)) * 0.1);
    const legs = [_]term.Leg{
        .{ .expiry = 0.1, .strike = 100.0, .qty = 3.0, .iv = 0.22 },
        .{ .expiry = 0.4, .strike = 105.0, .qty = -2.0, .iv = 0.24 },
        .{ .expiry = 1.0, .strike = 95.0, .qty = 1.0, .iv = 0.2 },
    };
    const book = term.aggregate(spot, 0.03, 0.0, &legs);
    return book.vega + book.vanna + book.volga + book.term_vega_slope;
}

const option_mm = root.option_mm;
const hawkes = root.hawkes;
const training = root.training;

fn doOptionMmSolveQuote(i: usize) f64 {
    var cfg = option_mm.researchToy();
    cfg.grid_n = 21;
    cfg.n_steps = 20;
    const v = blackBox(@as(f64, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) * 4.0);
    const q = option_mm.solveAndQuote(&cfg, 10.0, v, 5.0, 1.0, 1.0);
    return q.delta_b + q.delta_a + q.reservation;
}

fn doHawkesIntensity(i: usize) f64 {
    const p = hawkes.HawkesParams{ .mu = 1.2, .alpha = 0.5, .beta = 1.8 };
    var ev: [8]f64 = undefined;
    var k: usize = 0;
    while (k < ev.len) : (k += 1) {
        ev[k] = @as(f64, @floatFromInt(k)) * 0.05 + @as(f64, @floatFromInt(i % 3)) * 0.01;
    }
    const t = 0.5 + @as(f64, @floatFromInt(i % 5)) * 0.02;
    return hawkes.intensity(p, t, &ev) + hawkes.excitation(p, t, &ev);
}

const desk_k = root.desk;

fn doResidualStrip(i: usize) f64 {
    var r: [256]f64 = undefined;
    var fb: [256]f64 = undefined;
    var fg: [256]f64 = undefined;
    var fv: [256]f64 = undefined;
    var out: [256]f64 = undefined;
    var k: usize = 0;
    while (k < 256) : (k += 1) {
        const x = @as(f64, @floatFromInt((i + k) % 17)) * 0.01;
        fb[k] = x - 0.08;
        fg[k] = x * x;
        fv[k] = @as(f64, @floatFromInt(k % 9)) * 0.02 - 0.08;
        r[k] = 0.4 * fb[k] + 1.1 * fg[k] + 0.2 * fv[k] + 0.01;
    }
    const fit = desk_k.stripResidual(&r, &fb, &fg, &fv, &out);
    return fit.r2 + fit.beta + out[i % 256];
}

fn doResidualStripK(i: usize) f64 {
    var r: [128]f64 = undefined;
    var factors: [6 * 128]f64 = undefined;
    var out: [128]f64 = undefined;
    var coef: [6]f64 = .{0} ** 6;
    var t: usize = 0;
    while (t < 128) : (t += 1) {
        const x = @as(f64, @floatFromInt((i + t) % 19)) * 0.01;
        factors[0 * 128 + t] = x - 0.08;
        factors[1 * 128 + t] = x * x;
        factors[2 * 128 + t] = @as(f64, @floatFromInt(t % 7)) * 0.01;
        factors[3 * 128 + t] = x * 0.2;
        factors[4 * 128 + t] = -x * 0.1;
        factors[5 * 128 + t] = x * x * 0.5;
        r[t] = 0.3 * factors[t] + 0.8 * factors[128 + t] + 0.1 * factors[2 * 128 + t] + 0.01;
    }
    const fit = desk_k.stripResidualFactors(&r, 6, &factors, &out, &coef);
    return fit.r2 + coef[0] + out[i % 128];
}

fn doTrainingLocation(i: usize) f64 {
    _ = i;
    const naive = training.runCase("location_arb", .naive, 0.0);
    const desk = training.runCase("location_arb", .desk, naive.absolute_pnl);
    return desk.risk_adjusted - naive.risk_adjusted;
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
    benchOne("surface/svi_iv_and_density_g", 1_000_000, &doSviIv);
    benchOne("gueant/ode_offsets_q8_200steps", 2_000, &doGueantOde);
    benchOne("term/aggregate_3_expiries", 200_000, &doTermAggregate);
    benchOne("option_mm/solve_and_quote_grid21x20", 2_000, &doOptionMmSolveQuote);
    benchOne("hawkes/intensity_8_events", 500_000, &doHawkesIntensity);
    benchOne("training/location_arb_pair", 2_000, &doTrainingLocation);
    benchOne("desk/residual_strip_n256", 50_000, &doResidualStrip);
    benchOne("desk/residual_strip_k6_n128", 50_000, &doResidualStripK);
}
