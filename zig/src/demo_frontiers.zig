//! Paper demo for frontiers 1–4.
//! `zig build frontiers`
//!
//! 1. SVI / SSVI fit, butterfly + calendar gates, sticky-strike vs sticky-delta
//! 2. Multi-expiry strip quotes + term-structure vega / vanna / volga
//! 3. Queue-aware LOB fills (depth, cancel latency, toxic markout)
//! 4. Guéant ODE / spectral offsets vs the asymptotic closed form, and (A, k) from a synthetic tape
//!
//! Simulation only. No exchange SDKs.

const std = @import("std");
const root = @import("jev_omm");
const svi = root.svi;
const term = root.term_book;
const lob = root.lob;
const ode = root.gueant_ode;
const gueant = root.gueant;
const types = root.types;
const event_log = root.event_log;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    std.debug.print("jevy frontiers demo — simulation / paper only\n", .{});
    std.debug.print("SVI+SSVI · multi-expiry term risk · LOB queue · Guéant ODE\n\n", .{});

    // --- 1. Surface ---
    std.debug.print("=== 1. SVI / SSVI ===\n", .{});
    const planted = svi.SviParams{ .a = 0.04, .b = 0.14, .rho = -0.35, .m = 0.01, .sigma = 0.22 };
    var ks: [15]f64 = undefined;
    var ws: [15]f64 = undefined;
    var i: usize = 0;
    while (i < ks.len) : (i += 1) {
        const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(ks.len - 1));
        ks[i] = -0.6 + 1.2 * t;
        ws[i] = svi.totalVar(planted, ks[i]);
    }
    const fit = svi.calibrate(&ks, &ws);
    const smile = svi.SsviParams{ .rho = -0.35, .eta = 0.9, .gamma = 0.4 };
    std.debug.print("fit rmse={d:.6}  butterfly_ok={s}  suspect={s}  a={d:.4} b={d:.4} rho={d:.3}\n", .{
        fit.rmse,
        if (fit.butterfly.ok) "true" else "false",
        if (fit.surface_suspect) "true" else "false",
        fit.params.a,
        fit.params.b,
        fit.params.rho,
    });
    const arb = svi.SviParams{ .a = 0.04, .b = 3.0, .rho = 0.9, .m = 0.0, .sigma = 0.2 };
    const arb_rep = svi.butterflyCheck(arb);
    std.debug.print("injected arb: butterfly_ok={s} min_g={d:.3} lee={d:.2}\n", .{
        if (arb_rep.ok) "true" else "false",
        arb_rep.min_g,
        arb_rep.lee_slope,
    });
    std.debug.print("ssvi calendar-safe={s}  w(0,0.04)={d:.4}  w(0,0.09)={d:.4}\n", .{
        if (svi.ssviParamsCalendarSafe(smile)) "true" else "false",
        svi.ssviTotalVar(0.0, 0.04, smile),
        svi.ssviTotalVar(0.0, 0.09, smile),
    });
    const f0: f64 = 100.0;
    const f1: f64 = 108.0;
    const strike: f64 = 100.0;
    const tex: f64 = 0.3;
    std.debug.print("sticky-strike IV(K=100)={d:.4}  sticky-delta IV(K=100)={d:.4}\n", .{
        svi.impliedVolAfterMove(planted, f0, f1, strike, tex, .sticky_strike),
        svi.impliedVolAfterMove(planted, f0, f1, strike, tex, .sticky_delta),
    });

    // --- 2. Multi-expiry ---
    std.debug.print("\n=== 2. Multi-expiry book ===\n", .{});
    const expiries = [_]f64{ 30.0 / 365.25, 90.0 / 365.25, 180.0 / 365.25 };
    const strikes = [_]f64{ 95.0, 100.0, 105.0 };
    const quoter = types.QuoterConfig{
        .gamma = 0.12,
        .kappa = 1.5,
        .sigma = 0.35,
        .A = 120.0,
        .t_horizon = 1.0 / 252.0,
        .inventory_cap = 8,
        .ode_steps = 400,
        .mode = .gueant_ode,
        .quote_size = 1,
        .min_half_spread = 0.02,
        .max_half_spread = 3.0,
    };
    var strips: [3]term.ExpiryStrip = undefined;
    const n_exp = term.quoteExpiries(100.0, 0.05, 0.0, smile, 0.22, &expiries, &strikes, &quoter, true, &strips);
    var e: usize = 0;
    while (e < n_exp) : (e += 1) {
        const atm = strips[e].quotes[1];
        std.debug.print("T={d:.3}y  theta={d:.5}  K={d:.0} iv={d:.3} mid={d:.3} bid={d:.3} ask={d:.3}\n", .{
            strips[e].expiry,
            strips[e].theta,
            atm.strike,
            atm.iv,
            atm.mid,
            atm.quote.bid,
            atm.quote.ask,
        });
    }
    const legs = [_]term.Leg{
        .{ .expiry = expiries[0], .strike = 100.0, .qty = 12.0, .iv = strips[0].quotes[1].iv },
        .{ .expiry = expiries[1], .strike = 95.0, .qty = -5.0, .iv = strips[1].quotes[0].iv },
        .{ .expiry = expiries[2], .strike = 105.0, .qty = 4.0, .iv = strips[2].quotes[2].iv, .is_call = true },
    };
    const risk = term.aggregate(100.0, 0.05, 0.0, &legs);
    std.debug.print("portfolio  d={d:.3} g={d:.4} vega={d:.3} vanna={d:.3} volga={d:.3}\n", .{
        risk.delta,
        risk.gamma,
        risk.vega,
        risk.vanna,
        risk.volga,
    });
    std.debug.print("term slope (vega·years)={d:.4}  buckets={d}\n", .{ risk.term_vega_slope, risk.n_buckets });
    var b: usize = 0;
    while (b < risk.n_buckets) : (b += 1) {
        std.debug.print("  bucket T={d:.3} vega={d:.3} vanna={d:.3} volga={d:.3}\n", .{
            risk.buckets[b].expiry,
            risk.buckets[b].vega,
            risk.buckets[b].vanna,
            risk.buckets[b].volga,
        });
    }
    const tilt = term.scenarioPnl(&risk, 100.0, -0.02, 0.0, 0.05);
    std.debug.print("scenario spot-2% + term tilt φ=0.05  pnl={d:.3}\n", .{tilt});
    const limits = term.TermLimitConfig{ .max_abs_parallel_vega = 500.0, .max_abs_bucket_vega = 400.0, .max_abs_term_slope = 200.0 };
    std.debug.print("term limits: {s}\n", .{@tagName(term.evaluateLimits(&risk, &limits))});

    // --- 3. LOB ---
    std.debug.print("\n=== 3. LOB / queue fills ===\n", .{});
    const shallow = lob.expectedFills(2.0, 10.0, 25.0, 0.0, 1.0, null);
    const deep = lob.expectedFills(40.0, 10.0, 25.0, 0.0, 1.0, null);
    const stayed = lob.expectedFills(0.0, 100.0, 40.0, 0.0, 1.0, null);
    const late = lob.expectedFills(0.0, 100.0, 40.0, 0.0, 1.0, 0.2);
    std.debug.print("E[fills] shallow queue={d:.2}  deep queue={d:.2}  (deeper is slower)\n", .{ shallow, deep });
    std.debug.print("E[adverse fills] stay={d:.2}  late cancel @0.20y={d:.2}\n", .{ stayed, late });
    const mo0 = lob.fillMarkout(.bid, 1.00, 1.05, 1.0, 0.03, 0.0);
    const mo1 = lob.fillMarkout(.bid, 1.00, 1.05, 1.0, 0.03, 2.5);
    std.debug.print("bid markout toxic_flow 0 → {d:.4}   toxic_flow 2.5 → {d:.4}\n", .{ mo0, mo1 });

    var prng = std.Random.DefaultPrng.init(42);
    const sim = lob.simulate(prng.random(), &.{
        .ahead = 3.0,
        .our_size = 2.0,
        .trade_intensity = 15.0,
        .horizon = 1.0,
        .dt = 0.02,
        .cancel_latency = 0.6,
        .toxic_from = 0.0,
        .toxic_flow = 1.0,
        .adverse_jump = 0.02,
        .price = 1.90,
        .mid = 2.00,
        .side = .bid,
    });
    var log = event_log.Log.init(gpa);
    defer log.deinit();
    var ev: usize = 0;
    while (ev < sim.n_events) : (ev += 1) {
        const event = sim.events[ev];
        const kind: event_log.EventKind = switch (event.kind) {
            .add => .lob_add,
            .cancel => .lob_cancel,
            .execute => .lob_execute,
        };
        try log.appendLob(event.time, @intCast(ev), kind, @tagName(event.side), event.price, event.size, event.ahead, event.partial, event.adverse);
    }
    const hex = log.sha256Hex();
    std.debug.print("stochastic path filled={d:.2} adverse={d:.2} markout={d:.4} events={d} log_sha={s}\n", .{
        sim.filled,
        sim.adverse_filled,
        sim.markout,
        sim.n_events,
        hex[0..16],
    });

    // --- 4. Guéant ODE ---
    std.debug.print("\n=== 4. Guéant ODE / spectral + (A,k) ===\n", .{});
    const cfg = types.QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.5,
        .A = 140.0,
        .t_horizon = 2.0,
        .inventory_cap = 12,
        .ode_steps = 2000,
        .min_half_spread = 1e-6,
        .max_half_spread = 50.0,
        .mode = .gueant_ode,
    };
    const o0 = ode.optimalOffsets(&cfg, 0);
    const a0 = gueant.optimalOffsets(&cfg, 0);
    const o_deep = ode.optimalOffsets(&cfg, 11);
    const a_deep = gueant.optimalOffsets(&cfg, 11);
    const spec = ode.spectralOffsets(&cfg, 0);
    std.debug.print("q=0  ODE δb={d:.4} δa={d:.4}   asymptotic δb={d:.4} δa={d:.4}   spectral δb={d:.4}\n", .{
        o0.delta_b,
        o0.delta_a,
        a0.delta_b,
        a0.delta_a,
        spec.delta_b,
    });
    std.debug.print("q=11 ODE δb={d:.4}  asymptotic δb={d:.4}  (cap widens the bid)\n", .{
        o_deep.delta_b,
        a_deep.delta_b,
    });
    const deltas = [_]f64{ 0.2, 0.5, 0.9, 1.3, 1.8 };
    var obs: [5]ode.IntensityObs = undefined;
    const planted_A: f64 = 36.0;
    const planted_k: f64 = 1.35;
    for (deltas, 0..) |d, j| {
        const lam = planted_A * @exp(-planted_k * d);
        obs[j] = .{ .delta = d, .exposure = 400.0, .fills = @round(lam * 400.0) };
    }
    const fit_ak = ode.estimateIntensity(&obs);
    std.debug.print("tape MLE  planted A={d:.2} k={d:.3}   recovered A={d:.2} k={d:.3}\n", .{
        planted_A,
        planted_k,
        fit_ak.A,
        fit_ak.k,
    });
    std.debug.print("\nDone. Decision layer stays in Python (Jev never emits orders).\n", .{});
}