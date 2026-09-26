//! Pure-Zig paper demo (sim only; no exchange SDKs).
//! `zig build demo`
//! `zig build demo -- --multi-strike`   # 5-strike strip around spot
//! `zig build demo -- --gueant`         # Guéant asymptotics (arXiv 1105.3115)
//! `zig build demo -- --multi-strike --gueant`
//! `zig build demo -- --hedge-scenario` # parity/box + banded hedge + greek PnL + scenarios + toxicity
//!
//! Writes append-only JSONL event log (default: jev_omm_events.jsonl) and
//! prints fill markout attribution + log SHA-256 for deterministic replay.

const std = @import("std");
const root = @import("jev_omm");
const bs = root.black_scholes;
const asq = root.as_quoter;
const risk = root.risk_limits;
const fills = root.fills;
const pnl = root.pnl;
const surface = root.surface;
const markout = root.markout;
const event_log = root.event_log;
const multi = root.multi_strike;
const types = root.types;
const parity = root.parity;
const combos = root.combos;
const hedge = root.hedge;
const toxicity = root.toxicity;
const scenario = root.scenario;

const DEFAULT_LOG_PATH = "jev_omm_events.jsonl";

const DemoFlags = struct {
    multi_strike: bool = false,
    gueant: bool = false,
    hedge_scenario: bool = false,
};

fn parseFlags(init: std.process.Init) DemoFlags {
    var flags: DemoFlags = .{};
    var it = std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa) catch return flags;
    defer it.deinit();
    _ = it.skip(); // exe
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--multi-strike") or std.mem.eql(u8, arg, "-m")) {
            flags.multi_strike = true;
        } else if (std.mem.eql(u8, arg, "--gueant") or std.mem.eql(u8, arg, "-g")) {
            flags.gueant = true;
        } else if (std.mem.eql(u8, arg, "--hedge-scenario") or std.mem.eql(u8, arg, "-H")) {
            flags.hedge_scenario = true;
        }
    }
    return flags;
}


fn runHedgeScenario(gpa: std.mem.Allocator) void {
    std.debug.print("Jev Options MM — desk hedge / parity / scenario demo\n", .{});
    std.debug.print("Simulation only. Akuna 101/201 curriculum depth layer.\n\n", .{});

    const spot: f64 = 100.0;
    const strike: f64 = 100.0;
    const t: f64 = 30.0 / 365.25;
    const rate: f64 = 0.05;
    const div_yield: f64 = 0.0;
    const iv: f64 = 0.22;

    // --- A. Parity / box ---
    const theo_pc = parity.parityDiffSpot(spot, strike, t, rate, div_yield);
    const call_px = bs.price(spot, strike, t, rate, div_yield, iv, true);
    const put_px = bs.price(spot, strike, t, rate, div_yield, iv, false);
    std.debug.print("=== Put-call parity ===\n", .{});
    std.debug.print("C={d:.4}  P={d:.4}  C-P={d:.4}  theo={d:.4}  resid={d:.6}\n", .{
        call_px, put_px, call_px - put_px, theo_pc, (call_px - put_px) - theo_pc,
    });

    const half: f64 = 0.08;
    const put_q = parity.SideQuotes{ .bid = put_px - half, .ask = put_px + half };
    // Make call 0.40 rich → conversion edge
    const call_rich = parity.SideQuotes{ .bid = call_px + 0.40 - half, .ask = call_px + 0.40 + half };
    const synth = parity.syntheticForwardEdge(call_rich, put_q, spot, strike, t, rate, div_yield, null);
    std.debug.print("synth (call+0.40 rich): conversion_edge={d:.4}  reversal_edge={d:.4}\n", .{
        synth.conversion_edge, synth.reversal_edge,
    });

    const k1: f64 = 95.0;
    const k2: f64 = 105.0;
    const box_theo = parity.boxTheo(k1, k2, t, rate);
    // Fair-ish quotes with a 0.25 cheap buy
    const c1 = parity.SideQuotes{ .bid = 7.0, .ask = 7.10 };
    const c2 = parity.SideQuotes{ .bid = 2.90, .ask = 3.00 };
    const p1 = parity.SideQuotes{ .bid = 1.90, .ask = 2.00 };
    const p2 = parity.SideQuotes{ .bid = 5.70, .ask = 5.80 };
    const box = parity.boxSpread(c1, c2, p1, p2, k1, k2, t, rate);
    std.debug.print("\n=== Box {d:.0}/{d:.0} ===\n", .{ k1, k2 });
    std.debug.print("theo_pv={d:.4}  buy_debit={d:.4}  buy_edge={d:.4}\n", .{ box.theo_pv, box.package_debit, box.buy_edge });
    std.debug.print("implied_rate mid={d:.4} buy={d:.4}  (r={d:.4})\n", .{ box.implied_rate_mid, box.implied_rate_buy, rate });
    _ = box_theo;

    // --- B. Combos ---
    const stradd = combos.straddleTheo(spot, strike, t, rate, div_yield, iv);
    const fly = combos.butterflyCallTheo(spot, 95.0, 100.0, 105.0, t, rate, div_yield, iv, iv, iv);
    std.debug.print("\n=== Combos ===\n", .{});
    std.debug.print("ATM straddle theo={d:.4}  delta={d:.4}  gamma={d:.4}  vega={d:.4}\n", .{
        stradd.theo, stradd.greeks.delta, stradd.greeks.gamma, stradd.greeks.vega,
    });
    std.debug.print("call fly 95/100/105 theo={d:.4}  gamma={d:.4} (pin: short body gamma)\n", .{
        fly.theo, fly.greeks.gamma,
    });

    // --- C. Banded hedge sim ---
    std.debug.print("\n=== Banded delta hedge + greek PnL ===\n", .{});
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    const hcfg = hedge.HedgeConfig{ .delta_band = 2.0, .half_spread = 0.02, .flatten = true };
    var underlier_pos: f64 = 0.0;
    var cash: f64 = 0.0;
    const opt_qty: f64 = 12.0; // long 12 calls → |delta| exceeds band
    var s: f64 = spot;
    var cum = hedge.GreekPnlStep{};
    var tox = toxicity.Tracker.init(.{ .bucket_volume = 8.0, .window_buckets = 8 });
    var log = event_log.Log.init(gpa);
    defer log.deinit();

    const n_steps: usize = 80;
    const dt: f64 = 1.0 / (252.0 * 6.5 * 60.0);
    const spot_vol: f64 = 0.25;
    var prev_mid = bs.price(s, strike, t, rate, div_yield, iv, true);
    var hedge_fills: usize = 0;

    var step: usize = 0;
    while (step < n_steps) : (step += 1) {
        const time = @as(f64, @floatFromInt(step)) * dt;
        const prev_s = s;
        const z = rng.floatNorm(f64);
        s *= @exp((-0.5 * spot_vol * spot_vol) * dt + spot_vol * @sqrt(dt) * z);
        s = @max(s, 1e-6);
        const t_rem = @max(t - time, 1e-6);
        const g_one = bs.greeks(s, strike, t_rem, rate, div_yield, iv, true);
        const mid = bs.price(s, strike, t_rem, rate, div_yield, iv, true);
        const port_g = g_one.scale(opt_qty);
        const net_d = hedge.netDelta(port_g.delta, underlier_pos);

        // Synthetic tape: sign from spot move
        const tape_sign: f64 = if (s >= prev_s) 1.0 else -1.0;
        tox.onTrade(tape_sign * 2.0);

        const order = hedge.proposeDeltaHedge(net_d, &hcfg);
        var slip: f64 = 0.0;
        if (order.fire) {
            const hf = hedge.applyHedge(time, s, &order, &hcfg);
            underlier_pos += hf.underlier_qty;
            cash += hf.cash_delta;
            slip = hf.slippage_cost;
            hedge_fills += 1;
            log.appendHedgeFill(time, @intCast(step), hf.underlier_qty, hf.fill_price, hf.mid, hf.cash_delta, hf.slippage_cost) catch {};
        }

        const d_spot = s - prev_s;
        const d_mid = mid - prev_mid;
        // Portfolio greeks at *start* of move approx: use previous spot greeks
        const g_prev = bs.greeks(prev_s, strike, t_rem, rate, div_yield, iv, true).scale(opt_qty);
        const step_pnl = hedge.greekPnlStep(&g_prev, d_spot, 0.0, dt, opt_qty, d_mid, underlier_pos - (if (order.fire) order.underlier_qty else 0.0), 0.0, slip);
        cum = cum.add(step_pnl);
        log.appendGreekPnl(time, @intCast(step), step_pnl.spread_capture, step_pnl.hedge_slippage, step_pnl.gamma_pnl, step_pnl.theta_pnl, step_pnl.vega_pnl, step_pnl.inventory_mtm, step_pnl.delta_pnl) catch {};
        prev_mid = mid;
    }

    std.debug.print("steps={d}  hedge_fills={d}  underlier_pos={d:.2}  cash={d:.2}\n", .{ n_steps, hedge_fills, underlier_pos, cash });
    std.debug.print("greek buckets: gamma={d:.4}  theta={d:.4}  hedge_slip={d:.4}  inv_mtm={d:.4}  delta_diag={d:.4}\n", .{
        cum.gamma_pnl, cum.theta_pnl, cum.hedge_slippage, cum.inventory_mtm, cum.delta_pnl,
    });
    std.debug.print("toxicity: vpin={d:.3}  imbalance={d:.3}  composite={d:.3}  (research-grade)\n", .{
        tox.vpin(), tox.tradeImbalance(), tox.composite(),
    });

    // --- E. Scenario matrix ---
    const g_final = bs.greeks(s, strike, @max(t - @as(f64, @floatFromInt(n_steps)) * dt, 1e-6), rate, div_yield, iv, true).scale(opt_qty);
    // Include underlier hedge in net delta for scenario
    var g_sc = g_final;
    g_sc.delta = hedge.netDelta(g_final.delta, underlier_pos);
    const scfg = scenario.defaultConfig();
    const mat = scenario.buildMatrix(&g_sc, s, &scfg);
    std.debug.print("\n=== Scenario matrix (spot × IV shocks, greek Taylor) ===\n", .{});
    std.debug.print("min_pnl={d:.2}  max_pnl={d:.2}  soft_breach={}  hard_breach={}\n", .{
        mat.min_pnl, mat.max_pnl, mat.soft_breach, mat.hard_breach,
    });
    std.debug.print("worst cell: dS={d:.1}%  dIV={d:.1}pts  pnl={d:.2}\n", .{
        mat.at(mat.worst_i, mat.worst_j).d_spot_frac * 100.0,
        mat.at(mat.worst_i, mat.worst_j).d_iv * 100.0,
        mat.at(mat.worst_i, mat.worst_j).pnl,
    });
    // Print compact grid header
    std.debug.print("PnL grid (rows=spot shock, cols=iv shock):\n", .{});
    var j: usize = 0;
    std.debug.print("{s:>8}", .{"dS\\dIV"});
    while (j < mat.n_iv) : (j += 1) {
        std.debug.print(" {d:>8.0}", .{scfg.iv_shocks[j] * 100.0});
    }
    std.debug.print("\n", .{});
    var i: usize = 0;
    while (i < mat.n_spot) : (i += 1) {
        std.debug.print("{d:>7.1}%", .{scfg.spot_shocks[i] * 100.0});
        j = 0;
        while (j < mat.n_iv) : (j += 1) {
            std.debug.print(" {d:>8.2}", .{mat.at(i, j).pnl});
        }
        std.debug.print("\n", .{});
    }

    const ww = hedge.whalleyWilmottBand(s, @abs(g_final.gamma), spot_vol, 0.0002, 1e-3);
    std.debug.print("\nWW-style suggested band≈{d:.2} (cfg band={d:.1})\n", .{ ww, hcfg.delta_band });

    log.writeJsonl("jev_omm_events_hedge.jsonl") catch {};
    const hex = log.sha256Hex();
    std.debug.print("event_log=jev_omm_events_hedge.jsonl  events={d}  sha256={s}\n", .{ log.event_count, hex });
}

pub fn main(init: std.process.Init) void {
    const flags = parseFlags(init);
    const gpa = init.gpa;
    if (flags.hedge_scenario) {
        runHedgeScenario(gpa);
        return;
    }
    std.debug.print("Jev Options MM — Zig paper demo\n", .{});
    std.debug.print("Simulation only. No live brokers / exchange SDKs.\n", .{});
    std.debug.print("mode={s}  strip={s}\n\n", .{
        if (flags.gueant) "gueant_asymptotic" else "as_finite_horizon",
        if (flags.multi_strike) "multi-strike(5)" else "single",
    });

    var quoter = types.QuoterConfig{
        .gamma = 0.12,
        .kappa = 1.5,
        .sigma = 0.45,
        .A = 140.0,
        .quote_size = 2,
        .portfolio_delta_penalty = if (flags.multi_strike) 0.02 else 0.0,
        .mode = if (flags.gueant) .gueant_asymptotic else .as_finite_horizon,
    };
    const risk_cfg = types.RiskConfig{
        .max_abs_inventory = 20,
        .max_loss = 400.0,
    };
    const sabr = surface.SabrParams{
        .alpha = 0.22,
        .beta = 1.0,
        .rho = -0.3,
        .nu = 0.4,
    };

    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();

    const n_steps: usize = 150;
    const fill_intensity: f64 = 5.0e4;
    const rate: f64 = 0.05;
    const div_yield: f64 = 0.0;
    const expiry: f64 = 30.0 / 365.25;
    const dt: f64 = 1.0 / (252.0 * 6.5 * 60.0);
    const spot_vol: f64 = 0.20;
    const drift: f64 = 0.0;

    var spot: f64 = 100.0;
    var cash: f64 = 0.0;
    var fill_count: usize = 0;
    var breach_count: usize = 0;
    var last_mid: f64 = 0.0;
    var last_fills: [8]types.Fill = undefined;
    var last_fill_n: usize = 0;
    var tracker = markout.Tracker.init(markout.DEFAULT_HORIZONS);
    var log = event_log.Log.init(gpa);
    defer log.deinit();

    const answers_json =
        \\{"regime":{"type":"choice","choice":"calm","confidence":0.80},"toxicity":{"type":"score","score":0.0,"confidence":0.80},"informed_flow":{"type":"noul","noul":0.15},"widen_quotes":{"type":"noul","noul":0.20},"pull_quotes":{"type":"noul","noul":0.08},"hedge_now":{"type":"noul","noul":0.10},"size_tier":{"type":"choice","choice":"normal","confidence":0.75},"surface_suspect":{"type":"noul","noul":0.08}}
    ;
    const confidence_json =
        \\{"regime":0.80,"toxicity":0.80,"size_tier":0.75}
    ;

    // --- multi-strike strip state ---
    const strip_cfg = multi.StripConfig{ .half_width = 2, .strike_step = 1.0, .is_call = true };
    var strike_buf: [types.MAX_STRIP]f64 = undefined;
    var slots: [types.MAX_STRIP]types.StrikeSlot = undefined;
    var n_slots: usize = 0;
    var single_qty: i32 = 0;
    const single_strike: f64 = 100.0;

    if (flags.multi_strike) {
        n_slots = multi.buildStrikeGrid(spot, strip_cfg, &strike_buf);
        _ = multi.initFlatSlots(strike_buf[0..n_slots], &slots);
    }

    var step: usize = 0;
    while (step < n_steps) : (step += 1) {
        const time = @as(f64, @floatFromInt(step)) * dt;
        const z = rng.floatNorm(f64);
        spot *= @exp((drift - 0.5 * spot_vol * spot_vol) * dt + spot_vol * @sqrt(dt) * z);
        spot = @max(spot, 1e-6);

        const t_rem = @max(expiry - time, 1e-6);
        log.appendUnderlyingTick(time, @intCast(step), spot) catch {};

        if (flags.multi_strike) {
            // Re-center strip infrequently (desk: fixed strikes within a session).
            // Keep initial grid; only price/quote each active strike.
            var out: [types.MAX_STRIP]types.StrikeQuote = undefined;
            const written = multi.quoteStrip(
                spot,
                t_rem,
                rate,
                div_yield,
                sabr,
                slots[0..n_slots],
                &quoter,
                strip_cfg,
                1.0,
                1.0,
                &out,
            );

            // Portfolio greeks for risk + DecisionSnapshot.state
            var greeks_buf: [types.MAX_STRIP]types.Greeks = [_]types.Greeks{.{}} ** types.MAX_STRIP;
            var i: usize = 0;
            while (i < written) : (i += 1) greeks_buf[i] = out[i].greeks;
            // Rebuild slot-aligned greeks for portfolio
            var slot_greeks: [types.MAX_STRIP]types.Greeks = [_]types.Greeks{.{}} ** types.MAX_STRIP;
            i = 0;
            while (i < n_slots) : (i += 1) {
                var j: usize = 0;
                while (j < written) : (j += 1) {
                    if (out[j].strike == slots[i].strike) {
                        slot_greeks[i] = out[j].greeks;
                        break;
                    }
                }
            }
            const port = multi.portfolioGreeks(slots[0..n_slots], slot_greeks[0..n_slots]);
            const atm_mid = if (written > 0) out[written / 2].mid else 0.0;
            last_mid = atm_mid;
            tracker.onStep(@intCast(step), atm_mid);

            const half0 = asq.optimalHalfSpread(&quoter, null);
            log.appendBookTop(time, @intCast(step), atm_mid - half0, atm_mid + half0, quoter.quote_size, quoter.quote_size, atm_mid) catch {};

            const marked = pnl.markedPnl(cash, port.net_inventory, atm_mid);
            // evaluateRisk multiplies greeks × inventory — pass per-contract equiv.
            const inv_f = @as(f64, @floatFromInt(if (port.net_inventory != 0) port.net_inventory else 1));
            var port_g = types.Greeks{
                .delta = port.delta / inv_f,
                .gamma = port.gamma / inv_f,
                .vega = port.vega / inv_f,
                .theta = port.theta / inv_f,
            };
            const snap = risk.evaluateRisk(if (port.net_inventory != 0) port.net_inventory else 0, &port_g, marked, &risk_cfg);
            if (!snap.quoting_allowed) {
                breach_count += 1;
                const reason = if (snap.breach_len > 0) snap.reasonSlice() else "risk_breach";
                log.appendRiskBreach(time, @intCast(step), reason, port.net_inventory) catch {};
                log.appendCancel(time, @intCast(step), "risk_breach") catch {};
                continue;
            }

            var state_buf: [512]u8 = undefined;
            const state_json = std.fmt.bufPrint(&state_buf, "{{\"portfolio_delta\":{d:.6},\"portfolio_gamma\":{d:.6},\"portfolio_vega\":{d:.6},\"net_inventory\":{d},\"n_strikes\":{d},\"quoter_mode\":\"{s}\"}}", .{
                port.delta,
                port.gamma,
                port.vega,
                port.net_inventory,
                written,
                if (flags.gueant) "gueant_asymptotic" else "as_finite_horizon",
            }) catch "{\"n_strikes\":0}";
            log.appendDecisionSnapshotState(time, @intCast(step), "fallback", "fallback-heuristic", answers_json, confidence_json, state_json) catch {};

            i = 0;
            while (i < written) : (i += 1) {
                log.appendQuoteStrike(time, @intCast(step), out[i].strike, &out[i].quote) catch {};
                // Sample fills per strike; apply to that strike's inventory
                var fill_buf: [2]types.Fill = undefined;
                const nf = fills.sampleFillsInto(rng, time, out[i].mid, &out[i].quote, dt, fill_intensity / @as(f64, @floatFromInt(@max(written, 1))), quoter.kappa, &fill_buf);
                var fi: usize = 0;
                while (fi < nf) : (fi += 1) {
                    const f = fill_buf[fi];
                    // Find slot
                    var si: usize = 0;
                    while (si < n_slots) : (si += 1) {
                        if (slots[si].active and slots[si].strike == out[i].strike) {
                            fills.applyFillCash(&cash, &slots[si].inventory, f.side, f.price, f.size);
                            break;
                        }
                    }
                    tracker.onFill(@intCast(step), &f);
                    log.appendFill(time, @intCast(step), &f) catch {};
                    fill_count += 1;
                    if (last_fill_n < 8) {
                        last_fills[last_fill_n] = f;
                        last_fill_n += 1;
                    } else {
                        var j: usize = 0;
                        while (j < 7) : (j += 1) last_fills[j] = last_fills[j + 1];
                        last_fills[7] = f;
                    }
                }
            }
        } else {
            // --- single-strike path (original) ---
            const strike = single_strike;
            const forward = spot * @exp((rate - div_yield) * t_rem);
            const iv = surface.sabrImpliedVol(sabr, forward, strike, t_rem);
            const opt_mid = bs.price(spot, strike, t_rem, rate, div_yield, iv, true);
            const g = bs.greeks(spot, strike, t_rem, rate, div_yield, iv, true);
            last_mid = opt_mid;

            tracker.onStep(@intCast(step), opt_mid);

            const half0 = asq.optimalHalfSpread(&quoter, null);
            log.appendBookTop(time, @intCast(step), opt_mid - half0, opt_mid + half0, quoter.quote_size, quoter.quote_size, opt_mid) catch {};

            const marked = pnl.markedPnl(cash, single_qty, opt_mid);
            const snap = risk.evaluateRisk(single_qty, &g, marked, &risk_cfg);
            if (!snap.quoting_allowed) {
                breach_count += 1;
                const reason = if (snap.breach_len > 0) snap.reasonSlice() else "risk_breach";
                log.appendRiskBreach(time, @intCast(step), reason, single_qty) catch {};
                log.appendCancel(time, @intCast(step), "risk_breach") catch {};
                continue;
            }

            log.appendDecisionSnapshot(time, @intCast(step), "fallback", "fallback-heuristic", answers_json, confidence_json) catch {};

            const quote = asq.makeQuote(opt_mid, single_qty, &quoter, quoter.t_horizon, &g, 1.0, 1.0);
            log.appendQuote(time, @intCast(step), &quote) catch {};

            var fill_buf: [2]types.Fill = undefined;
            const n = fills.sampleFillsInto(rng, time, opt_mid, &quote, dt, fill_intensity, quoter.kappa, &fill_buf);
            var i: usize = 0;
            while (i < n) : (i += 1) {
                const f = fill_buf[i];
                fills.applyFillCash(&cash, &single_qty, f.side, f.price, f.size);
                tracker.onFill(@intCast(step), &f);
                log.appendFill(time, @intCast(step), &f) catch {};
                fill_count += 1;
                if (last_fill_n < 8) {
                    last_fills[last_fill_n] = f;
                    last_fill_n += 1;
                } else {
                    var j: usize = 0;
                    while (j < 7) : (j += 1) last_fills[j] = last_fills[j + 1];
                    last_fills[7] = f;
                }
            }
        }
    }

    var final_qty: i32 = single_qty;
    if (flags.multi_strike) {
        final_qty = 0;
        var i: usize = 0;
        while (i < n_slots) : (i += 1) {
            if (slots[i].active) final_qty += slots[i].inventory;
        }
    }
    const final_pnl = pnl.markedPnl(cash, final_qty, last_mid);
    const atm_iv = surface.sabrAtmVol(sabr, 100.0, expiry);
    std.debug.print("steps={d}  fills={d}  breaches={d}\n", .{ n_steps, fill_count, breach_count });
    std.debug.print("SABR atm_iv≈{d:.4} (alpha={d:.2} beta={d:.1} rho={d:.1} nu={d:.1})\n", .{
        atm_iv, sabr.alpha, sabr.beta, sabr.rho, sabr.nu,
    });
    std.debug.print("final inventory={d}  cash={d:.4}  marked_pnl={d:.4}\n", .{ final_qty, cash, final_pnl });
    std.debug.print("last option mid={d:.4}  last spot={d:.4}\n", .{ last_mid, spot });

    if (flags.multi_strike) {
        std.debug.print("\nStrip inventories:\n", .{});
        var i: usize = 0;
        while (i < n_slots) : (i += 1) {
            if (!slots[i].active) continue;
            std.debug.print("  K={d:.1}  qty={d}\n", .{ slots[i].strike, slots[i].inventory });
        }
    }

    const a = tracker.attr;
    std.debug.print("\n=== Markout attribution (horizons in steps: 1 / 5 / 30) ===\n", .{});
    std.debug.print("fills={d}  contracts={d}\n", .{ a.n_fills, a.total_contracts });
    std.debug.print("spread_capture     {d:>10.4}\n", .{a.spread_capture});
    std.debug.print("markout_1step      {d:>10.4}  (adverse {d:>10.4})  n={d}\n", .{ a.markout[0], a.adverse[0], a.resolved[0] });
    std.debug.print("markout_5step      {d:>10.4}  (adverse {d:>10.4})  n={d}\n", .{ a.markout[1], a.adverse[1], a.resolved[1] });
    std.debug.print("markout_30step     {d:>10.4}  (adverse {d:>10.4})  n={d}\n", .{ a.markout[2], a.adverse[2], a.resolved[2] });
    std.debug.print("inventory_mtm      {d:>10.4}\n", .{a.inventory_mtm});
    std.debug.print("note: markout = signed_size*(mid_h-mid0); adverse=-markout\n", .{});
    std.debug.print("      ~1 step ≈ 1 trading minute at default dt\n", .{});

    if (last_fill_n > 0) {
        std.debug.print("\nLast fills (up to 8):\n", .{});
        var i: usize = 0;
        while (i < last_fill_n) : (i += 1) {
            const f = last_fills[i];
            std.debug.print("  t={d:.6} side={s} px={d:.4} sz={d} mid={d:.4}\n", .{
                f.time,
                @tagName(f.side),
                f.price,
                f.size,
                f.mid_at_fill,
            });
        }
    } else {
        std.debug.print("\nNo fills this seed — raise fill intensity.\n", .{});
    }

    log.writeJsonl(DEFAULT_LOG_PATH) catch |err| {
        std.debug.print("\nWARN: failed to write {s}: {s}\n", .{ DEFAULT_LOG_PATH, @errorName(err) });
        return;
    };
    const hex = log.sha256Hex();
    std.debug.print("\nevent_log={s}  events={d}  sha256={s}\n", .{ DEFAULT_LOG_PATH, log.event_count, hex });
    std.debug.print("replay: zig build replay -- {s}\n", .{DEFAULT_LOG_PATH});
    std.debug.print("     or: zig-out/bin/jev_omm_replay {s}\n", .{DEFAULT_LOG_PATH});

    const rr = event_log.replayJsonl(log.bytes());
    std.debug.print("replay_check fills={d} spread={d:.4} marked_pnl={d:.4}\n", .{
        rr.n_fills, rr.attr.spread_capture, rr.marked_pnl,
    });
}
