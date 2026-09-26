//! Put-call parity, synthetics (conversion/reversal), and box spreads.
//!
//! Curriculum anchors (Akuna Options 101 §2.2 / §3.6 + public sources):
//!   - European PCP: C − P = S e^{-qT} − K e^{-rT}  (= DF · (F − K))
//!   - Synthetic forward / conversion-reversal on *executable* sides (not mids)
//!   - Box K1<K2: theo PV = (K2−K1) e^{-rT}; package edge from four-leg bids/asks
//!   - Implied financing rate from box mid / executable
//!
//! Public refs (do not invent):
//!   https://akunacapital.com/work-with-us/options-101/
//!   https://blog.moontower.ai/implying-the-cost-of-carry-in-options/
//!   optionseducation.org put-call parity; Wikipedia box spread
//!
//! Package edges always use executable sides (buy at ask, sell at bid) —
//! desk practice, never mids for arb detection.

const std = @import("std");

/// Discount factor e^{-rT}.
pub fn discountFactor(rate: f64, t: f64) f64 {
    return @exp(-rate * t);
}

/// Forward from spot: F = S e^{(r−q)T}.
pub fn forward(spot: f64, rate: f64, div_yield: f64, t: f64) f64 {
    return spot * @exp((rate - div_yield) * t);
}

/// European put-call parity theo: C − P = S e^{-qT} − K e^{-rT}.
pub fn parityDiffSpot(spot: f64, strike: f64, t: f64, rate: f64, div_yield: f64) f64 {
    return spot * @exp(-div_yield * t) - strike * @exp(-rate * t);
}

/// Equivalent form: C − P = DF · (F − K).
pub fn parityDiffForward(fwd: f64, strike: f64, t: f64, rate: f64) f64 {
    return discountFactor(rate, t) * (fwd - strike);
}

/// Observed C−P vs theo parity residual (positive ⇒ call rich vs put / synthetic).
pub fn parityResidual(call_mid: f64, put_mid: f64, spot: f64, strike: f64, t: f64, rate: f64, div_yield: f64) f64 {
    return (call_mid - put_mid) - parityDiffSpot(spot, strike, t, rate, div_yield);
}

/// Market quotes for one option (or underlier) — bid/ask only.
pub const SideQuotes = struct {
    bid: f64,
    ask: f64,

    pub fn mid(self: SideQuotes) f64 {
        return 0.5 * (self.bid + self.ask);
    }

    pub fn isValid(self: SideQuotes) bool {
        return self.bid > 0.0 and self.ask >= self.bid;
    }
};

/// Synthetic long forward via long call + short put (buy call ask, sell put bid).
/// Cost to enter = call_ask − put_bid. Theo cost = DF*(F−K) = S e^{-qT} − K e^{-rT}.
/// Edge > 0 means synthetic forward is *cheap* to buy (arb / edge to long synth).
pub const SyntheticEdge = struct {
    /// Cash outlay to buy synthetic forward (call_ask − put_bid), possibly + underlier.
    package_debit: f64 = 0.0,
    /// Theo fair debit (parity).
    theo_debit: f64 = 0.0,
    /// theo − package (positive = package cheap / buy edge).
    edge: f64 = 0.0,
    /// Conversion: short synth + long stock. Edge if conversion credits more than theo.
    conversion_edge: f64 = 0.0,
    /// Reversal: long synth + short stock.
    reversal_edge: f64 = 0.0,
};

/// Conversion/reversal edges using executable sides on call, put, and underlier.
///
/// Conversion = short call + long put + long stock
///   Credit = call_bid − put_ask − spot_ask  (we receive call_bid, pay put_ask, pay spot ask)
///   At expiry locked value ≈ −K (pay K via exercise) … easier:
///   Conversion PnL locked ≈ call_bid − put_ask − (S_ask − K·DF) + carry adjustments.
///
/// Desk view used here (European, continuous dividends):
///   Fair C−P = S·e^{-qT} − K·e^{-rT}.
///   Buy synthetic forward (long C short P) executable debit = C_ask − P_bid.
///   Sell synthetic forward (short C long P) executable credit = C_bid − P_ask.
///   Reversal edge (buy synth vs fair) = fair − (C_ask − P_bid)   [>0 ⇒ buy synth]
///   Conversion edge (sell synth vs fair) = (C_bid − P_ask) − fair [>0 ⇒ sell synth]
///
/// When underlier sides are supplied, stock financing uses ask to buy / bid to sell
/// in a full conversion/reversal package vs discounted strike.
pub fn syntheticForwardEdge(
    call: SideQuotes,
    put: SideQuotes,
    spot: f64,
    strike: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    underlier: ?SideQuotes,
) SyntheticEdge {
    const theo = parityDiffSpot(spot, strike, t, rate, div_yield);
    const buy_synth = call.ask - put.bid; // debit to long synth fwd
    const sell_synth = call.bid - put.ask; // credit to short synth fwd
    var out = SyntheticEdge{
        .package_debit = buy_synth,
        .theo_debit = theo,
        .edge = theo - buy_synth, // + ⇒ synth cheap
        .reversal_edge = theo - buy_synth,
        .conversion_edge = sell_synth - theo,
    };
    // Full package with underlier: conversion = short synth + long stock financed to T
    // Locked value of conversion at T is +K (put exercise / call assigned).
    // PV of locked K is K·e^{-rT}. Cost to put on: −sell_synth + S_ask·e^{-qT} approx
    // (div yield treated as continuous stock carry). Edge = PV(K) − cost.
    if (underlier) |u| {
        const df = discountFactor(rate, t);
        const dq = @exp(-div_yield * t);
        // Conversion: sell call, buy put, buy stock. Cash now = +C_bid − P_ask − S_ask
        // Terminal ≈ +K (assignment/exercise). PV edge = cash_now + K·DF  (ignore stock div separately via dq)
        // More carefully with continuous yield: stock carry uses dq on spot.
        const conv_cash = call.bid - put.ask - u.ask * dq;
        out.conversion_edge = conv_cash + strike * df;
        // Reversal: buy call, sell put, short stock. Cash now = −C_ask + P_bid + S_bid·dq
        // Terminal ≈ −K. PV edge = cash_now − K·DF
        const rev_cash = -call.ask + put.bid + u.bid * dq;
        out.reversal_edge = rev_cash - strike * df;
        out.edge = out.reversal_edge; // prefer full-package long-synth edge when underlier known
        out.package_debit = call.ask - put.bid - u.bid * dq; // net debit to long synth + short stock
        out.theo_debit = -strike * df; // fair cash for that package (should offset)
    }
    return out;
}

/// Box spread: long call K1 + short call K2 + short put K1 + long put K2 (K1 < K2),
/// or equivalently long vertical call + long vertical put of same strikes.
/// European theo PV of long box = (K2 − K1) · e^{-rT}.
pub const BoxResult = struct {
    theo_pv: f64 = 0.0,
    /// Debit to buy the box on executable sides (pay asks on longs, receive bids on shorts).
    package_debit: f64 = 0.0,
    /// Credit to sell the box.
    package_credit: f64 = 0.0,
    /// theo − debit (>0 ⇒ box cheap to buy).
    buy_edge: f64 = 0.0,
    /// credit − theo (>0 ⇒ box rich to sell).
    sell_edge: f64 = 0.0,
    /// Implied continuous rate from box mid: DF = mid/(K2−K1), r = −ln(DF)/T.
    implied_rate_mid: f64 = 0.0,
    /// Implied rate from executable buy debit.
    implied_rate_buy: f64 = 0.0,
    /// Implied rate from executable sell credit.
    implied_rate_sell: f64 = 0.0,
};

fn impliedRateFromPv(pv: f64, width: f64, t: f64) f64 {
    if (t <= 0.0 or width <= 0.0 or pv <= 0.0) return 0.0;
    const df = pv / width;
    if (df <= 0.0) return 0.0;
    return -@log(df) / t;
}

/// Four-leg box from call/put quotes at K1 and K2 (K1 < K2).
/// Long box legs: +C(K1), −C(K2), −P(K1), +P(K2).
pub fn boxSpread(
    call_k1: SideQuotes,
    call_k2: SideQuotes,
    put_k1: SideQuotes,
    put_k2: SideQuotes,
    k1: f64,
    k2: f64,
    t: f64,
    rate: f64,
) BoxResult {
    const width = k2 - k1;
    const theo = width * discountFactor(rate, t);
    // Buy box: pay ask on +C1 +P2, receive bid on −C2 −P1
    const buy_debit = call_k1.ask - call_k2.bid - put_k1.bid + put_k2.ask;
    // Sell box: receive bid on −C1 −P2, pay ask on +C2 +P1
    const sell_credit = call_k1.bid - call_k2.ask - put_k1.ask + put_k2.bid;
    const mid_pv = 0.5 * (buy_debit + sell_credit);
    return .{
        .theo_pv = theo,
        .package_debit = buy_debit,
        .package_credit = sell_credit,
        .buy_edge = theo - buy_debit,
        .sell_edge = sell_credit - theo,
        .implied_rate_mid = impliedRateFromPv(mid_pv, width, t),
        .implied_rate_buy = impliedRateFromPv(buy_debit, width, t),
        .implied_rate_sell = impliedRateFromPv(sell_credit, width, t),
    };
}

/// Convenience: box theo only.
pub fn boxTheo(k1: f64, k2: f64, t: f64, rate: f64) f64 {
    return (k2 - k1) * discountFactor(rate, t);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "parity identity DF*(F-K)" {
    const s: f64 = 100.0;
    const k: f64 = 100.0;
    const t: f64 = 0.25;
    const r: f64 = 0.05;
    const q: f64 = 0.02;
    const f = forward(s, r, q, t);
    const a = parityDiffSpot(s, k, t, r, q);
    const b = parityDiffForward(f, k, t, r);
    try std.testing.expect(@abs(a - b) < 1e-12);
}

test "parity residual zero when on parity" {
    const s: f64 = 100.0;
    const k: f64 = 105.0;
    const t: f64 = 0.5;
    const r: f64 = 0.03;
    const q: f64 = 0.01;
    const theo = parityDiffSpot(s, k, t, r, q);
    // Construct mids exactly on parity: C−P = theo
    const call_mid: f64 = 4.0;
    const put_mid = call_mid - theo;
    const resid = parityResidual(call_mid, put_mid, s, k, t, r, q);
    try std.testing.expect(@abs(resid) < 1e-12);
}

test "no arb when prices on parity (tight quotes around fair)" {
    const s: f64 = 100.0;
    const k: f64 = 100.0;
    const t: f64 = 0.25;
    const r: f64 = 0.05;
    const q: f64 = 0.0;
    const theo = parityDiffSpot(s, k, t, r, q);
    // Call and put mids on parity; half-spread 0.05 each side → no crossed package edge
    const half: f64 = 0.05;
    const c_mid = 5.0;
    const p_mid = c_mid - theo;
    const call = SideQuotes{ .bid = c_mid - half, .ask = c_mid + half };
    const put = SideQuotes{ .bid = p_mid - half, .ask = p_mid + half };
    const edge = syntheticForwardEdge(call, put, s, k, t, r, q, null);
    // Buy synth debit = (c_mid+h) − (p_mid−h) = theo + 2h > theo ⇒ reversal edge negative
    // Sell synth credit = (c_mid−h) − (p_mid+h) = theo − 2h < theo ⇒ conversion edge negative
    try std.testing.expect(edge.reversal_edge < 0.0);
    try std.testing.expect(edge.conversion_edge < 0.0);
    try std.testing.expect(@abs(edge.reversal_edge + 2.0 * half) < 1e-10);
}

test "detectable synth edge when call rich" {
    const s: f64 = 100.0;
    const k: f64 = 100.0;
    const t: f64 = 0.25;
    const r: f64 = 0.05;
    const q: f64 = 0.0;
    const theo = parityDiffSpot(s, k, t, r, q);
    // Call 1.0 rich vs put
    const c_mid = 6.0;
    const p_mid = c_mid - theo - 1.0;
    const half: f64 = 0.05;
    const call = SideQuotes{ .bid = c_mid - half, .ask = c_mid + half };
    const put = SideQuotes{ .bid = p_mid - half, .ask = p_mid + half };
    const edge = syntheticForwardEdge(call, put, s, k, t, r, q, null);
    // Selling the rich synth should show positive conversion edge (~1 − 2*half)
    try std.testing.expect(edge.conversion_edge > 0.5);
}

test "box theo PV equals width * DF" {
    const k1: f64 = 95.0;
    const k2: f64 = 105.0;
    const t: f64 = 0.5;
    const r: f64 = 0.04;
    const theo = boxTheo(k1, k2, t, r);
    try std.testing.expect(@abs(theo - 10.0 * @exp(-r * t)) < 1e-12);
}

test "box on fair: no buy/sell edge beyond spread" {
    const k1: f64 = 100.0;
    const k2: f64 = 110.0;
    const t: f64 = 1.0;
    const r: f64 = 0.05;
    const theo = boxTheo(k1, k2, t, r);
    const half: f64 = 0.02;
    // Build four legs so mid package = theo. Long box mid = C1−C2−P1+P2.
    // Set vertical call mid = 4, vertical put mid = theo−4.
    const vc: f64 = 4.0;
    const vp = theo - vc;
    const c1 = SideQuotes{ .bid = 6.0 - half, .ask = 6.0 + half };
    const c2 = SideQuotes{ .bid = 6.0 - vc - half, .ask = 6.0 - vc + half };
    const p1 = SideQuotes{ .bid = 3.0 - half, .ask = 3.0 + half };
    const p2 = SideQuotes{ .bid = 3.0 + vp - half, .ask = 3.0 + vp + half };
    const box = boxSpread(c1, c2, p1, p2, k1, k2, t, r);
    // Mid PV should be near theo; buy edge negative by ~4*half (crossing four spreads)
    try std.testing.expect(@abs(0.5 * (box.package_debit + box.package_credit) - theo) < 0.05);
    try std.testing.expect(box.buy_edge < 0.0);
    try std.testing.expect(box.sell_edge < 0.0);
}

test "box off fair: detectable buy edge" {
    const k1: f64 = 100.0;
    const k2: f64 = 110.0;
    const t: f64 = 1.0;
    const r: f64 = 0.05;
    const theo = boxTheo(k1, k2, t, r);
    // Make buy debit theo − 0.50 (box 50c cheap), tiny spreads
    const cheap: f64 = 0.50;
    const half: f64 = 0.01;
    // buy_debit = c1.ask - c2.bid - p1.bid + p2.ask = theo - cheap
    // Set c1.ask=5, c2.bid=1, p1.bid=1, p2.ask = theo-cheap -5 +1 +1 = theo-cheap-3
    const target_debit = theo - cheap;
    const c1 = SideQuotes{ .bid = 4.98, .ask = 5.00 };
    const c2 = SideQuotes{ .bid = 1.00, .ask = 1.02 };
    const p1 = SideQuotes{ .bid = 1.00, .ask = 1.02 };
    const p2_ask = target_debit - c1.ask + c2.bid + p1.bid;
    const p2 = SideQuotes{ .bid = p2_ask - 2.0 * half, .ask = p2_ask };
    const box = boxSpread(c1, c2, p1, p2, k1, k2, t, r);
    try std.testing.expect(@abs(box.package_debit - target_debit) < 1e-9);
    try std.testing.expect(box.buy_edge > 0.4);
    // Cheap box = lower PV → lower DF → *higher* implied continuous rate.
    try std.testing.expect(box.implied_rate_buy > r);
}

test "implied rate recovers input when box mid = theo" {
    const k1: f64 = 90.0;
    const k2: f64 = 100.0;
    const t: f64 = 0.5;
    const r: f64 = 0.06;
    const theo = boxTheo(k1, k2, t, r);
    // Zero-width quotes at fair vertical contributions
    const c1 = SideQuotes{ .bid = theo, .ask = theo };
    const c2 = SideQuotes{ .bid = 0.0, .ask = 0.0 };
    const p1 = SideQuotes{ .bid = 0.0, .ask = 0.0 };
    const p2 = SideQuotes{ .bid = 0.0, .ask = 0.0 };
    // buy = theo - 0 - 0 + 0 = theo; sell = theo - 0 - 0 + 0 = theo
    const box = boxSpread(c1, c2, p1, p2, k1, k2, t, r);
    try std.testing.expect(@abs(box.implied_rate_mid - r) < 1e-9);
}
