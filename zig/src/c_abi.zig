//! C ABI exports for Python ctypes/cffi.
//! Keep signatures POD / extern-friendly.

const bs = @import("black_scholes.zig");
const asq = @import("as_quoter.zig");
const gueant = @import("gueant.zig");
const risk = @import("risk_limits.zig");
const pnl = @import("pnl.zig");
const types = @import("types.zig");
const surface = @import("surface.zig");
const markout = @import("markout.zig");
const multi = @import("multi_strike.zig");
const parity = @import("parity.zig");
const combos = @import("combos.zig");
const hedge = @import("hedge.zig");
const scenario = @import("scenario.zig");
const desk = @import("desk.zig");
const lob = @import("lob.zig");

pub const CGreeks = extern struct {
    delta: f64,
    gamma: f64,
    vega: f64,
    theta: f64,
};

pub const CQuote = extern struct {
    bid: f64,
    ask: f64,
    bid_size: i32,
    ask_size: i32,
    reservation: f64,
    half_spread: f64,
};

pub const CQuoterConfig = extern struct {
    gamma: f64,
    kappa: f64,
    sigma: f64,
    t_horizon: f64,
    gamma_penalty: f64,
    vega_penalty: f64,
    min_half_spread: f64,
    max_half_spread: f64,
    quote_size: i32,
    /// 0 = as_finite_horizon, 1 = gueant_asymptotic, 2 = gueant_ode, 3 = option_vega
    mode: i32 = 0,
    /// Guéant mid-touch intensity A
    A: f64 = 140.0,
    portfolio_delta_penalty: f64 = 0.0,
};

pub const CRiskConfig = extern struct {
    max_abs_inventory: i32,
    _pad: i32 = 0,
    max_abs_delta: f64,
    max_abs_vega: f64,
    max_abs_gamma: f64,
    max_loss: f64,
};

pub const CRiskSnapshot = extern struct {
    inventory: i32,
    quoting_allowed: i32, // bool as i32
    delta: f64,
    gamma: f64,
    vega: f64,
    cash_pnl: f64,
    breach_reason: [128]u8,
};

pub const CStrikeQuote = extern struct {
    strike: f64,
    bid: f64,
    ask: f64,
    bid_size: i32,
    ask_size: i32,
    reservation: f64,
    half_spread: f64,
    mid: f64,
    iv: f64,
    inventory: i32,
    _pad: i32 = 0,
};

fn toQuoterConfig(cfg: *const CQuoterConfig) types.QuoterConfig {
    return .{
        .gamma = cfg.gamma,
        .kappa = cfg.kappa,
        .sigma = cfg.sigma,
        .t_horizon = cfg.t_horizon,
        .gamma_penalty = cfg.gamma_penalty,
        .vega_penalty = cfg.vega_penalty,
        .min_half_spread = cfg.min_half_spread,
        .max_half_spread = cfg.max_half_spread,
        .quote_size = cfg.quote_size,
        .A = cfg.A,
        .portfolio_delta_penalty = cfg.portfolio_delta_penalty,
        .mode = switch (cfg.mode) {
            1 => .gueant_asymptotic,
            2 => .gueant_ode,
            3 => .option_vega,
            else => .as_finite_horizon,
        },
    };
}

export fn jev_omm_price(
    spot: f64,
    strike: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv: f64,
    is_call: i32,
) callconv(.c) f64 {
    return bs.price(spot, strike, t, rate, div_yield, iv, is_call != 0);
}

export fn jev_omm_greeks(
    spot: f64,
    strike: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv: f64,
    is_call: i32,
    out: *CGreeks,
) callconv(.c) void {
    const g = bs.greeks(spot, strike, t, rate, div_yield, iv, is_call != 0);
    out.* = .{ .delta = g.delta, .gamma = g.gamma, .vega = g.vega, .theta = g.theta };
}

export fn jev_omm_price_and_greeks(
    spot: f64,
    strike: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv: f64,
    is_call: i32,
    out_greeks: *CGreeks,
) callconv(.c) f64 {
    const r = bs.priceAndGreeks(spot, strike, t, rate, div_yield, iv, is_call != 0);
    out_greeks.* = .{ .delta = r.greeks.delta, .gamma = r.greeks.gamma, .vega = r.greeks.vega, .theta = r.greeks.theta };
    return r.price;
}

export fn jev_omm_reservation_price(
    mid: f64,
    inventory: i32,
    cfg: *const CQuoterConfig,
    t_remaining: f64,
    use_t_remaining: i32,
    g_gamma: f64,
    g_vega: f64,
    use_greeks: i32,
) callconv(.c) f64 {
    const qcfg = toQuoterConfig(cfg);
    const t_opt: ?f64 = if (use_t_remaining != 0) t_remaining else null;
    var greeks_storage: types.Greeks = .{ .gamma = g_gamma, .vega = g_vega };
    const g_opt: ?*const types.Greeks = if (use_greeks != 0) &greeks_storage else null;
    return asq.reservationPrice(mid, inventory, &qcfg, t_opt, g_opt);
}

export fn jev_omm_optimal_half_spread(
    cfg: *const CQuoterConfig,
    t_remaining: f64,
    use_t_remaining: i32,
) callconv(.c) f64 {
    const qcfg = toQuoterConfig(cfg);
    const t_opt: ?f64 = if (use_t_remaining != 0) t_remaining else null;
    return asq.optimalHalfSpread(&qcfg, t_opt);
}

export fn jev_omm_make_quote(
    mid: f64,
    inventory: i32,
    cfg: *const CQuoterConfig,
    t_remaining: f64,
    use_t_remaining: i32,
    g_gamma: f64,
    g_vega: f64,
    use_greeks: i32,
    spread_mult: f64,
    size_mult: f64,
    out: *CQuote,
) callconv(.c) void {
    const qcfg = toQuoterConfig(cfg);
    const t_opt: ?f64 = if (use_t_remaining != 0) t_remaining else null;
    var greeks_storage: types.Greeks = .{ .gamma = g_gamma, .vega = g_vega };
    const g_opt: ?*const types.Greeks = if (use_greeks != 0) &greeks_storage else null;
    const q = asq.makeQuote(mid, inventory, &qcfg, t_opt, g_opt, spread_mult, size_mult);
    out.* = .{
        .bid = q.bid,
        .ask = q.ask,
        .bid_size = q.bid_size,
        .ask_size = q.ask_size,
        .reservation = q.reservation,
        .half_spread = q.half_spread,
    };
}

export fn jev_omm_gueant_inventory_scale(cfg: *const CQuoterConfig) callconv(.c) f64 {
    const qcfg = toQuoterConfig(cfg);
    return gueant.inventoryScale(&qcfg);
}

export fn jev_omm_gueant_half_spread(cfg: *const CQuoterConfig) callconv(.c) f64 {
    const qcfg = toQuoterConfig(cfg);
    return gueant.optimalHalfSpread(&qcfg);
}

/// Quote a centered strike strip. `strikes`/`inventories` length `n_strikes` (max 8).
/// Writes up to `out_cap` CStrikeQuote records; returns count written.
export fn jev_omm_quote_strip(
    spot: f64,
    t_rem: f64,
    rate: f64,
    div_yield: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
    strikes: [*]const f64,
    inventories: [*]const i32,
    n_strikes: i32,
    cfg: *const CQuoterConfig,
    spread_mult: f64,
    size_mult: f64,
    is_call: i32,
    out: [*]CStrikeQuote,
    out_cap: i32,
) callconv(.c) i32 {
    const n_in: usize = @intCast(@max(n_strikes, 0));
    const cap: usize = @intCast(@max(out_cap, 0));
    const n = @min(n_in, @min(cap, types.MAX_STRIP));
    var slots: [types.MAX_STRIP]types.StrikeSlot = undefined;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        slots[i] = .{ .strike = strikes[i], .inventory = inventories[i], .active = true };
    }
    const qcfg = toQuoterConfig(cfg);
    const sabr = surface.SabrParams{ .alpha = alpha, .beta = beta, .rho = rho, .nu = nu };
    const strip_cfg = multi.StripConfig{ .half_width = 2, .strike_step = 1.0, .is_call = is_call != 0 };
    var sq: [types.MAX_STRIP]types.StrikeQuote = undefined;
    const written = multi.quoteStrip(spot, t_rem, rate, div_yield, sabr, slots[0..n], &qcfg, strip_cfg, spread_mult, size_mult, &sq);
    i = 0;
    while (i < written) : (i += 1) {
        out[i] = .{
            .strike = sq[i].strike,
            .bid = sq[i].quote.bid,
            .ask = sq[i].quote.ask,
            .bid_size = sq[i].quote.bid_size,
            .ask_size = sq[i].quote.ask_size,
            .reservation = sq[i].quote.reservation,
            .half_spread = sq[i].quote.half_spread,
            .mid = sq[i].mid,
            .iv = sq[i].iv,
            .inventory = sq[i].inventory,
        };
    }
    return @intCast(written);
}

export fn jev_omm_marked_pnl(cash: f64, qty: i32, option_mid: f64) callconv(.c) f64 {
    return pnl.markedPnl(cash, qty, option_mid);
}

export fn jev_omm_evaluate_risk(
    inventory: i32,
    delta_pc: f64,
    gamma_pc: f64,
    vega_pc: f64,
    cash_pnl: f64,
    cfg: *const CRiskConfig,
    out: *CRiskSnapshot,
) callconv(.c) void {
    const g = types.Greeks{ .delta = delta_pc, .gamma = gamma_pc, .vega = vega_pc, .theta = 0.0 };
    const rcfg = types.RiskConfig{
        .max_abs_inventory = cfg.max_abs_inventory,
        .max_abs_delta = cfg.max_abs_delta,
        .max_abs_vega = cfg.max_abs_vega,
        .max_abs_gamma = cfg.max_abs_gamma,
        .max_loss = cfg.max_loss,
    };
    const snap = risk.evaluateRisk(inventory, &g, cash_pnl, &rcfg);
    out.* = .{
        .inventory = snap.inventory,
        .quoting_allowed = if (snap.quoting_allowed) 1 else 0,
        .delta = snap.delta,
        .gamma = snap.gamma,
        .vega = snap.vega,
        .cash_pnl = snap.cash_pnl,
        .breach_reason = snap.breach_reason,
    };
}

export fn jev_omm_sabr_iv(
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
    forward: f64,
    strike: f64,
    t: f64,
) callconv(.c) f64 {
    const p = surface.SabrParams{ .alpha = alpha, .beta = beta, .rho = rho, .nu = nu };
    return surface.sabrImpliedVol(p, forward, strike, t);
}

export fn jev_omm_sabr_atm(
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
    forward: f64,
    t: f64,
) callconv(.c) f64 {
    const p = surface.SabrParams{ .alpha = alpha, .beta = beta, .rho = rho, .nu = nu };
    return surface.sabrAtmVol(p, forward, t);
}

export fn jev_omm_spread_edge(
    is_bid: i32,
    price: f64,
    mid: f64,
    size: i32,
) callconv(.c) f64 {
    const side: types.Side = if (is_bid != 0) .bid else .ask;
    return markout.Tracker.spreadEdge(side, price, mid, size);
}


pub const CSideQuotes = extern struct {
    bid: f64,
    ask: f64,
};

pub const CSyntheticEdge = extern struct {
    package_debit: f64,
    theo_debit: f64,
    edge: f64,
    conversion_edge: f64,
    reversal_edge: f64,
};

/// Inputs travel in one struct. Zig 0.16 mis-reads C float arguments past
/// the ninth when they are passed individually (the stack slots repeat the
/// first xmm values). Python ctypes uses the same struct.
pub const CBoxInputs = extern struct {
    c1_bid: f64,
    c1_ask: f64,
    c2_bid: f64,
    c2_ask: f64,
    p1_bid: f64,
    p1_ask: f64,
    p2_bid: f64,
    p2_ask: f64,
    k1: f64,
    k2: f64,
    t: f64,
    rate: f64,
};

pub const CBoxResult = extern struct {
    theo_pv: f64,
    package_debit: f64,
    package_credit: f64,
    buy_edge: f64,
    sell_edge: f64,
    implied_rate_mid: f64,
    implied_rate_buy: f64,
    implied_rate_sell: f64,
};

pub const CHedgeConfig = extern struct {
    delta_band: f64,
    half_spread: f64,
    slip_bps: f64,
    flatten: i32,
};

pub const CHedgeOrder = extern struct {
    underlier_qty: f64,
    delta_to_hedge: f64,
    fire: i32,
    _pad: i32 = 0,
};

pub const CHedgeFill = extern struct {
    time: f64,
    underlier_qty: f64,
    fill_price: f64,
    mid: f64,
    cash_delta: f64,
    slippage_cost: f64,
};

pub const CGreekPnl = extern struct {
    spread_capture: f64,
    hedge_slippage: f64,
    gamma_pnl: f64,
    theta_pnl: f64,
    vega_pnl: f64,
    inventory_mtm: f64,
    delta_pnl: f64,
};

pub const CPackageTheo = extern struct {
    theo: f64,
    delta: f64,
    gamma: f64,
    vega: f64,
    theta: f64,
    package_bid: f64,
    package_ask: f64,
    buy_edge: f64,
    sell_edge: f64,
};

export fn jev_omm_parity_diff(
    spot: f64,
    strike: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
) callconv(.c) f64 {
    return parity.parityDiffSpot(spot, strike, t, rate, div_yield);
}

export fn jev_omm_parity_residual(
    call_mid: f64,
    put_mid: f64,
    spot: f64,
    strike: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
) callconv(.c) f64 {
    return parity.parityResidual(call_mid, put_mid, spot, strike, t, rate, div_yield);
}

export fn jev_omm_synthetic_edge(
    call_bid: f64,
    call_ask: f64,
    put_bid: f64,
    put_ask: f64,
    spot: f64,
    strike: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    use_underlier: i32,
    und_bid: f64,
    und_ask: f64,
    out: *CSyntheticEdge,
) callconv(.c) void {
    const call = parity.SideQuotes{ .bid = call_bid, .ask = call_ask };
    const put = parity.SideQuotes{ .bid = put_bid, .ask = put_ask };
    const und_opt: ?parity.SideQuotes = if (use_underlier != 0) parity.SideQuotes{ .bid = und_bid, .ask = und_ask } else null;
    const e = parity.syntheticForwardEdge(call, put, spot, strike, t, rate, div_yield, und_opt);
    out.* = .{
        .package_debit = e.package_debit,
        .theo_debit = e.theo_debit,
        .edge = e.edge,
        .conversion_edge = e.conversion_edge,
        .reversal_edge = e.reversal_edge,
    };
}

export fn jev_omm_box_spread(inp: *const CBoxInputs, out: *CBoxResult) callconv(.c) void {
    const box = parity.boxSpread(
        .{ .bid = inp.c1_bid, .ask = inp.c1_ask },
        .{ .bid = inp.c2_bid, .ask = inp.c2_ask },
        .{ .bid = inp.p1_bid, .ask = inp.p1_ask },
        .{ .bid = inp.p2_bid, .ask = inp.p2_ask },
        inp.k1,
        inp.k2,
        inp.t,
        inp.rate,
    );
    out.* = .{
        .theo_pv = box.theo_pv,
        .package_debit = box.package_debit,
        .package_credit = box.package_credit,
        .buy_edge = box.buy_edge,
        .sell_edge = box.sell_edge,
        .implied_rate_mid = box.implied_rate_mid,
        .implied_rate_buy = box.implied_rate_buy,
        .implied_rate_sell = box.implied_rate_sell,
    };
}

export fn jev_omm_box_theo(k1: f64, k2: f64, t: f64, rate: f64) callconv(.c) f64 {
    return parity.boxTheo(k1, k2, t, rate);
}

export fn jev_omm_hedge_propose(
    net_delta: f64,
    cfg: *const CHedgeConfig,
    out: *CHedgeOrder,
) callconv(.c) void {
    const hcfg = hedge.HedgeConfig{
        .delta_band = cfg.delta_band,
        .half_spread = cfg.half_spread,
        .slip_bps = cfg.slip_bps,
        .flatten = cfg.flatten != 0,
    };
    const o = hedge.proposeDeltaHedge(net_delta, &hcfg);
    out.* = .{
        .underlier_qty = o.underlier_qty,
        .delta_to_hedge = o.delta_to_hedge,
        .fire = if (o.fire) 1 else 0,
    };
}

export fn jev_omm_hedge_apply(
    time: f64,
    mid: f64,
    underlier_qty: f64,
    delta_to_hedge: f64,
    fire: i32,
    cfg: *const CHedgeConfig,
    out: *CHedgeFill,
) callconv(.c) void {
    const hcfg = hedge.HedgeConfig{
        .delta_band = cfg.delta_band,
        .half_spread = cfg.half_spread,
        .slip_bps = cfg.slip_bps,
        .flatten = cfg.flatten != 0,
    };
    const order = hedge.HedgeOrder{
        .underlier_qty = underlier_qty,
        .delta_to_hedge = delta_to_hedge,
        .fire = fire != 0,
    };
    const f = hedge.applyHedge(time, mid, &order, &hcfg);
    out.* = .{
        .time = f.time,
        .underlier_qty = f.underlier_qty,
        .fill_price = f.fill_price,
        .mid = f.mid,
        .cash_delta = f.cash_delta,
        .slippage_cost = f.slippage_cost,
    };
}

export fn jev_omm_greek_pnl_step(
    delta: f64,
    gamma: f64,
    vega: f64,
    theta: f64,
    d_spot: f64,
    d_sigma: f64,
    dt: f64,
    option_qty: f64,
    d_option_mid: f64,
    underlier_pos: f64,
    spread_capture: f64,
    hedge_slippage: f64,
    out: *CGreekPnl,
) callconv(.c) void {
    const g = types.Greeks{ .delta = delta, .gamma = gamma, .vega = vega, .theta = theta };
    const step = hedge.greekPnlStep(&g, d_spot, d_sigma, dt, option_qty, d_option_mid, underlier_pos, spread_capture, hedge_slippage);
    out.* = .{
        .spread_capture = step.spread_capture,
        .hedge_slippage = step.hedge_slippage,
        .gamma_pnl = step.gamma_pnl,
        .theta_pnl = step.theta_pnl,
        .vega_pnl = step.vega_pnl,
        .inventory_mtm = step.inventory_mtm,
        .delta_pnl = step.delta_pnl,
    };
}

export fn jev_omm_ww_band(
    spot: f64,
    gamma_abs: f64,
    sigma: f64,
    slip_frac: f64,
    risk_aversion: f64,
) callconv(.c) f64 {
    return hedge.whalleyWilmottBand(spot, gamma_abs, sigma, slip_frac, risk_aversion);
}

export fn jev_omm_straddle_theo(
    spot: f64,
    strike: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv: f64,
    out: *CPackageTheo,
) callconv(.c) void {
    const pkg = combos.straddleTheo(spot, strike, t, rate, div_yield, iv);
    out.* = .{
        .theo = pkg.theo,
        .delta = pkg.greeks.delta,
        .gamma = pkg.greeks.gamma,
        .vega = pkg.greeks.vega,
        .theta = pkg.greeks.theta,
        .package_bid = pkg.package_bid,
        .package_ask = pkg.package_ask,
        .buy_edge = pkg.buy_edge,
        .sell_edge = pkg.sell_edge,
    };
}

export fn jev_omm_vertical_call_theo(
    spot: f64,
    k1: f64,
    k2: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv1: f64,
    iv2: f64,
    out: *CPackageTheo,
) callconv(.c) void {
    const pkg = combos.verticalCallTheo(spot, k1, k2, t, rate, div_yield, iv1, iv2);
    out.* = .{
        .theo = pkg.theo,
        .delta = pkg.greeks.delta,
        .gamma = pkg.greeks.gamma,
        .vega = pkg.greeks.vega,
        .theta = pkg.greeks.theta,
        .package_bid = 0,
        .package_ask = 0,
        .buy_edge = 0,
        .sell_edge = 0,
    };
}

export fn jev_omm_butterfly_call_theo(
    spot: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv1: f64,
    iv2: f64,
    iv3: f64,
    out: *CPackageTheo,
) callconv(.c) void {
    const pkg = combos.butterflyCallTheo(spot, k1, k2, k3, t, rate, div_yield, iv1, iv2, iv3);
    out.* = .{
        .theo = pkg.theo,
        .delta = pkg.greeks.delta,
        .gamma = pkg.greeks.gamma,
        .vega = pkg.greeks.vega,
        .theta = pkg.greeks.theta,
        .package_bid = 0,
        .package_ask = 0,
        .buy_edge = 0,
        .sell_edge = 0,
    };
}

export fn jev_omm_scenario_taylor(
    delta: f64,
    gamma: f64,
    vega: f64,
    theta: f64,
    d_spot: f64,
    d_iv: f64,
) callconv(.c) f64 {
    const g = types.Greeks{ .delta = delta, .gamma = gamma, .vega = vega, .theta = theta };
    return scenario.taylorPnl(&g, d_spot, d_iv);
}

export fn jev_omm_vanna(
    spot: f64,
    strike: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv: f64,
    is_call: i32,
) callconv(.c) f64 {
    return bs.greeks(spot, strike, t, rate, div_yield, iv, is_call != 0).vanna;
}

export fn jev_omm_volga(
    spot: f64,
    strike: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv: f64,
    is_call: i32,
) callconv(.c) f64 {
    return bs.greeks(spot, strike, t, rate, div_yield, iv, is_call != 0).volga;
}

export fn jev_omm_svi_total_var(a: f64, b: f64, rho: f64, m: f64, sigma: f64, k: f64) callconv(.c) f64 {
    const svi = @import("svi.zig");
    return svi.totalVar(.{ .a = a, .b = b, .rho = rho, .m = m, .sigma = sigma }, k);
}

export fn jev_omm_svi_iv(a: f64, b: f64, rho: f64, m: f64, sigma: f64, k: f64, t: f64) callconv(.c) f64 {
    const svi = @import("svi.zig");
    return svi.impliedVol(.{ .a = a, .b = b, .rho = rho, .m = m, .sigma = sigma }, k, t);
}

export fn jev_omm_svi_density_g(a: f64, b: f64, rho: f64, m: f64, sigma: f64, k: f64) callconv(.c) f64 {
    const svi = @import("svi.zig");
    return svi.densityG(.{ .a = a, .b = b, .rho = rho, .m = m, .sigma = sigma }, k);
}

export fn jev_omm_svi_butterfly_ok(a: f64, b: f64, rho: f64, m: f64, sigma: f64) callconv(.c) i32 {
    const svi = @import("svi.zig");
    return if (svi.butterflyCheck(.{ .a = a, .b = b, .rho = rho, .m = m, .sigma = sigma }).ok) 1 else 0;
}

export fn jev_omm_ssvi_total_var(theta: f64, rho: f64, eta: f64, gamma: f64, k: f64) callconv(.c) f64 {
    const svi = @import("svi.zig");
    return svi.ssviTotalVar(k, theta, .{ .rho = rho, .eta = eta, .gamma = gamma });
}

export fn jev_omm_gueant_ode_offsets(
    gamma: f64,
    kappa: f64,
    sigma: f64,
    A: f64,
    inventory: i32,
    inventory_cap: i32,
    horizon: f64,
    n_steps: i32,
    out_delta_b: *f64,
    out_delta_a: *f64,
) callconv(.c) void {
    const ode = @import("gueant_ode.zig");
    const cfg = types.QuoterConfig{
        .gamma = gamma,
        .kappa = kappa,
        .sigma = sigma,
        .A = A,
        .inventory_cap = inventory_cap,
        .t_horizon = horizon,
        .ode_steps = if (n_steps > 0) @intCast(n_steps) else 1,
        .min_half_spread = 1e-8,
        .max_half_spread = 1e6,
        .mode = .gueant_ode,
    };
    const o = ode.optimalOffsets(&cfg, inventory);
    out_delta_b.* = o.delta_b;
    out_delta_a.* = o.delta_a;
}

export fn jev_omm_option_mm_premiums(
    mid: f64,
    portfolio_vega: f64,
    contract_vega: f64,
    gamma: f64,
    xi: f64,
    A: f64,
    kappa: f64,
    vega_limit: f64,
    horizon: f64,
    grid_n: i32,
    n_steps: i32,
    vol_edge: f64,
    rho: f64,
    iv_alpha: f64,
    intensity_kind: i32,
    lambda0: f64,
    alpha: f64,
    beta: f64,
    out_delta_b: *f64,
    out_delta_a: *f64,
    out_bid_size: *i32,
    out_ask_size: *i32,
    out_reservation: *f64,
) callconv(.c) void {
    const omm = @import("option_mm.zig");
    const cfg = omm.OptionMmConfig{
        .gamma = gamma,
        .xi = xi,
        .A = A,
        .kappa = kappa,
        .vega_limit = vega_limit,
        .horizon = horizon,
        .grid_n = if (grid_n > 0) @intCast(grid_n) else 31,
        .n_steps = if (n_steps > 0) @intCast(n_steps) else 60,
        .vol_edge = vol_edge,
        .rho = rho,
        .iv_alpha = iv_alpha,
        .intensity = if (intensity_kind == 1) .logistic else .exponential,
        .lambda0 = lambda0,
        .alpha = alpha,
        .beta = beta,
        .min_premium = 0.0,
        .max_premium = 1e6,
    };
    const q = omm.solveAndQuote(&cfg, mid, portfolio_vega, contract_vega, 1.0, 1.0);
    out_delta_b.* = q.delta_b;
    out_delta_a.* = q.delta_a;
    out_bid_size.* = q.bid_size;
    out_ask_size.* = q.ask_size;
    out_reservation.* = q.reservation;
}

export fn jev_omm_spot_vol_hedge(
    net_delta: f64,
    rho: f64,
    xi: f64,
    portfolio_vega: f64,
    variance: f64,
    spot: f64,
) callconv(.c) f64 {
    return hedge.spotVolHedgeQty(net_delta, rho, xi, portfolio_vega, variance, spot);
}

export fn jev_omm_hawkes_intensity(
    mu: f64,
    alpha: f64,
    beta: f64,
    t: f64,
    n_events: i32,
    events: [*]const f64,
) callconv(.c) f64 {
    const hawkes = @import("hawkes.zig");
    const n: usize = if (n_events > 0) @intCast(n_events) else 0;
    return hawkes.intensity(.{ .mu = mu, .alpha = alpha, .beta = beta }, t, events[0..n]);
}

const flow_signals = @import("flow_signals.zig");
const positioning = @import("positioning.zig");

pub const CFlowPrior = extern struct {
    toxicity: f64,
    spread_mult: f64,
    size_mult: f64,
};

pub const CGexAdjust = extern struct {
    reservation_shift: f64,
    spread_mult: f64,
    size_mult: f64,
    hedge_band_mult: f64,
    hedge_urgency: f64,
};

export fn jev_omm_flow_prior(
    vpin: f64,
    ofi_norm: f64,
    aggr_imbalance: f64,
    off_exchange_share: f64,
    spoof: f64,
    out: *CFlowPrior,
) callconv(.c) void {
    const p = flow_signals.flowPrior(vpin, ofi_norm, aggr_imbalance, off_exchange_share, spoof);
    out.* = .{ .toxicity = p.toxicity, .spread_mult = p.spread_mult, .size_mult = p.size_mult };
}

export fn jev_omm_gex_adjust(
    enabled: i32,
    gex_norm: f64,
    pin_gap: f64,
    spot_return: f64,
    mid: f64,
    out: *CGexAdjust,
) callconv(.c) void {
    const g = positioning.gexAdjust(enabled != 0, gex_norm, pin_gap, spot_return, mid);
    out.* = .{
        .reservation_shift = g.reservation_shift,
        .spread_mult = g.spread_mult,
        .size_mult = g.size_mult,
        .hedge_band_mult = g.hedge_band_mult,
        .hedge_urgency = g.hedge_urgency,
    };
}

export fn jev_omm_cot_fade(
    enabled: i32,
    cot_z: f64,
    mid: f64,
    out_shift: *f64,
    out_size: *f64,
) callconv(.c) void {
    const c = positioning.cotFade(enabled != 0, cot_z, mid);
    out_shift.* = c.shift;
    out_size.* = c.size_mult;
}

const state_os = @import("state_os.zig");

pub const CStateGate = extern struct {
    reservation_shift: f64,
    spread_mult: f64,
    size_mult: f64,
    hedge_urgency: f64,
    pull: i32,
    instability: f64,
};

export fn jev_omm_state_gate(
    enabled: i32,
    instability_value: f64,
    constraint_active: i32,
    parent_remaining: f64,
    f_signed: f64,
    l_exec: f64,
    mid: f64,
    out: *CStateGate,
) callconv(.c) void {
    const g = state_os.stateGate(enabled != 0, instability_value, constraint_active != 0, parent_remaining, f_signed, l_exec, mid);
    out.* = .{
        .reservation_shift = g.reservation_shift,
        .spread_mult = g.spread_mult,
        .size_mult = g.size_mult,
        .hedge_urgency = g.hedge_urgency,
        .pull = if (g.pull) 1 else 0,
        .instability = g.instability,
    };
}

export fn jev_omm_instability(f_value: f64, l_exec: f64) callconv(.c) f64 {
    return state_os.instability(f_value, l_exec);
}

export fn jev_omm_evaluate_risk_ext(
    inventory: i32,
    delta_pc: f64,
    gamma_pc: f64,
    vega_pc: f64,
    cash_pnl: f64,
    max_abs_inventory: i32,
    max_abs_delta: f64,
    max_abs_vega: f64,
    max_abs_gamma: f64,
    max_loss: f64,
    max_abs_notional: f64,
    max_abs_per_strike: i32,
    max_quotes_outstanding: i32,
    extra_delta: f64,
    notional: f64,
    per_strike_abs: i32,
    quotes_outstanding: i32,
    out_allowed: *i32,
    out_reason: *[128]u8,
) callconv(.c) void {
    const g = types.Greeks{ .delta = delta_pc, .gamma = gamma_pc, .vega = vega_pc, .theta = 0.0 };
    const rcfg = types.RiskConfig{
        .max_abs_inventory = max_abs_inventory,
        .max_abs_delta = max_abs_delta,
        .max_abs_vega = max_abs_vega,
        .max_abs_gamma = max_abs_gamma,
        .max_loss = max_loss,
        .max_abs_notional = max_abs_notional,
        .max_abs_per_strike = max_abs_per_strike,
        .max_quotes_outstanding = max_quotes_outstanding,
    };
    const book = types.RiskBook{
        .extra_delta = extra_delta,
        .notional = notional,
        .per_strike_abs = per_strike_abs,
        .quotes_outstanding = quotes_outstanding,
    };
    const snap = risk.evaluateRiskBook(inventory, &g, cash_pnl, &rcfg, book);
    out_allowed.* = if (snap.quoting_allowed) 1 else 0;
    @memset(out_reason, 0);
    if (snap.breach_len > 0) {
        const n = @min(snap.breach_len, out_reason.len);
        @memcpy(out_reason[0..n], snap.breach_reason[0..n]);
    }
}

export fn jev_omm_residual_strip(
    n: usize,
    r: [*]const f64,
    f_beta: [*]const f64,
    f_gamma: [*]const f64,
    f_vega: [*]const f64,
    use_vega: i32,
    out_resid: [*]f64,
    out_beta: *f64,
    out_gamma: *f64,
    out_vega: *f64,
    out_r2: *f64,
) callconv(.c) void {
    const vega: ?[]const f64 = if (use_vega != 0) f_vega[0..n] else null;
    const fit = desk.stripResidual(r[0..n], f_beta[0..n], f_gamma[0..n], vega, out_resid[0..n]);
    out_beta.* = fit.beta;
    out_gamma.* = fit.gamma_coef;
    out_vega.* = fit.vega_coef;
    out_r2.* = fit.r2;
}

export fn jev_omm_residual_strip_k(
    n: usize,
    k: usize,
    r: [*]const f64,
    factors: [*]const f64,
    out_resid: [*]f64,
    out_coef: [*]f64,
    out_r2: *f64,
) callconv(.c) void {
    const kk = @min(k, desk.MAX_FACTORS);
    var coef: [desk.MAX_FACTORS]f64 = .{0} ** desk.MAX_FACTORS;
    const fit = desk.stripResidualFactors(r[0..n], kk, factors[0 .. kk * n], out_resid[0..n], coef[0..kk]);
    var i: usize = 0;
    while (i < kk) : (i += 1) out_coef[i] = coef[i];
    out_r2.* = fit.r2;
}

export fn jev_omm_smoothness(
    n: usize,
    resid: [*]const f64,
    raw: [*]const f64,
    out_ac1: *f64,
    out_dc: *f64,
    out_const_trend: *f64,
    out_low_freq: *f64,
    out_penalty: *f64,
    out_flag: *i32,
) callconv(.c) void {
    const sm = desk.smoothnessPenalty(resid[0..n], raw[0..n]);
    out_ac1.* = sm.ac1;
    out_dc.* = sm.dc_share;
    out_const_trend.* = sm.const_trend_r2;
    out_low_freq.* = sm.low_freq_share;
    out_penalty.* = sm.penalty;
    out_flag.* = @intCast(sm.flag);
}

export fn jev_omm_allocate_inverse_vol(
    n_sleeves: usize,
    n_steps: usize,
    resid_packed: [*]const f64,
    enabled: [*]const u8,
    max_weight: f64,
    corr_cap: f64,
    sharpe_tilt: f64,
    sigma_clip_q: f64,
    out_w: [*]f64,
) callconv(.c) void {
    const m = @min(n_sleeves, desk.MAX_SLEEVES);
    var series: [desk.MAX_SLEEVES][]const f64 = undefined;
    var on: [desk.MAX_SLEEVES]bool = .{false} ** desk.MAX_SLEEVES;
    var i: usize = 0;
    while (i < m) : (i += 1) {
        const start = i * n_steps;
        series[i] = resid_packed[start .. start + n_steps];
        on[i] = enabled[i] != 0;
    }
    var w: [desk.MAX_SLEEVES]f64 = .{0} ** desk.MAX_SLEEVES;
    desk.allocateInverseVolClipped(series[0..m], on[0..m], max_weight, corr_cap, sharpe_tilt, sigma_clip_q, w[0..m]);
    i = 0;
    while (i < m) : (i += 1) out_w[i] = w[i];
}

export fn jev_omm_kappa_for_touch(gamma: f64, target_half: f64, kappa_prior: f64) callconv(.c) f64 {
    return lob.kappaForTouch(gamma, target_half, kappa_prior);
}

export fn jev_omm_queue_decision(
    ahead: f64,
    our_size: f64,
    trade_intensity: f64,
    cancel_ahead: f64,
    horizon: f64,
    spread_capture: f64,
    adverse_per_fill: f64,
    toxic_flow: f64,
    queue_edge: u8,
    out_action: *u8,
    out_ahead: *f64,
    out_latency: *f64,
    out_spread_mult: *f64,
    out_size_mult: *f64,
) callconv(.c) void {
    const d = lob.queueDecision(
        ahead,
        our_size,
        trade_intensity,
        cancel_ahead,
        horizon,
        spread_capture,
        adverse_per_fill,
        toxic_flow,
        queue_edge != 0,
    );
    out_action.* = d.action;
    out_ahead.* = d.ahead;
    out_latency.* = d.cancel_latency;
    out_spread_mult.* = d.spread_mult;
    out_size_mult.* = d.size_mult;
}

export fn jev_omm_version() callconv(.c) [*:0]const u8 {
    return "1.3.0-zig-fills";
}

