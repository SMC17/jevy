//! Combo / package theos (Akuna theos & spreads/flies).
//!
//! Vertical call/put, butterfly, straddle/strangle package theo from BS legs;
//! package bid/ask from executable leg sides; edge vs theo.
//!
//! Curriculum: Akuna Options 101 (theos & combination spreads, spreads/flies,
//! pin risk on flies). Public: https://akunacapital.com/work-with-us/options-101/

const std = @import("std");
const bs = @import("black_scholes.zig");
const types = @import("types.zig");
const parity = @import("parity.zig");
const SideQuotes = parity.SideQuotes;
const Greeks = types.Greeks;

pub const ComboKind = enum(u8) {
    vertical_call = 0,
    vertical_put = 1,
    butterfly_call = 2,
    butterfly_put = 3,
    straddle = 4,
    strangle = 5,
};

pub const PackageTheo = struct {
    theo: f64 = 0.0,
    greeks: Greeks = .{},
    /// Executable package bid (what you receive selling the package).
    package_bid: f64 = 0.0,
    /// Executable package ask (what you pay buying the package).
    package_ask: f64 = 0.0,
    /// theo − ask (>0 ⇒ cheap to buy).
    buy_edge: f64 = 0.0,
    /// bid − theo (>0 ⇒ rich to sell).
    sell_edge: f64 = 0.0,
    kind: ComboKind = .straddle,
};

fn legPx(spot: f64, k: f64, t: f64, r: f64, q: f64, iv: f64, is_call: bool) f64 {
    return bs.price(spot, k, t, r, q, iv, is_call);
}

fn legG(spot: f64, k: f64, t: f64, r: f64, q: f64, iv: f64, is_call: bool) Greeks {
    return bs.greeks(spot, k, t, r, q, iv, is_call);
}

/// Call vertical: +C(K1) −C(K2), K1 < K2 (bull call spread).
pub fn verticalCallTheo(
    spot: f64,
    k1: f64,
    k2: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv1: f64,
    iv2: f64,
) PackageTheo {
    const p1 = legPx(spot, k1, t, rate, div_yield, iv1, true);
    const p2 = legPx(spot, k2, t, rate, div_yield, iv2, true);
    const g1 = legG(spot, k1, t, rate, div_yield, iv1, true);
    const g2 = legG(spot, k2, t, rate, div_yield, iv2, true);
    return .{
        .theo = p1 - p2,
        .greeks = g1.add(g2.scale(-1.0)),
        .kind = .vertical_call,
    };
}

/// Put vertical: +P(K2) −P(K1), K1 < K2 (bear put spread) — or +P(K1)−P(K2) bull.
/// Here: bull put = −P(K1)+P(K2) credit; we expose debit bull = +P(K1)−P(K2).
pub fn verticalPutTheo(
    spot: f64,
    k1: f64,
    k2: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv1: f64,
    iv2: f64,
) PackageTheo {
    const p1 = legPx(spot, k1, t, rate, div_yield, iv1, false);
    const p2 = legPx(spot, k2, t, rate, div_yield, iv2, false);
    const g1 = legG(spot, k1, t, rate, div_yield, iv1, false);
    const g2 = legG(spot, k2, t, rate, div_yield, iv2, false);
    return .{
        .theo = p1 - p2,
        .greeks = g1.add(g2.scale(-1.0)),
        .kind = .vertical_put,
    };
}

/// Call butterfly: +C(K1) −2 C(K2) +C(K3), K1 < K2 < K3 (typically equal wings).
pub fn butterflyCallTheo(
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
) PackageTheo {
    const p1 = legPx(spot, k1, t, rate, div_yield, iv1, true);
    const p2 = legPx(spot, k2, t, rate, div_yield, iv2, true);
    const p3 = legPx(spot, k3, t, rate, div_yield, iv3, true);
    const g1 = legG(spot, k1, t, rate, div_yield, iv1, true);
    const g2 = legG(spot, k2, t, rate, div_yield, iv2, true);
    const g3 = legG(spot, k3, t, rate, div_yield, iv3, true);
    return .{
        .theo = p1 - 2.0 * p2 + p3,
        .greeks = g1.add(g2.scale(-2.0)).add(g3),
        .kind = .butterfly_call,
    };
}

/// Put butterfly: +P(K1) −2 P(K2) +P(K3).
pub fn butterflyPutTheo(
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
) PackageTheo {
    const p1 = legPx(spot, k1, t, rate, div_yield, iv1, false);
    const p2 = legPx(spot, k2, t, rate, div_yield, iv2, false);
    const p3 = legPx(spot, k3, t, rate, div_yield, iv3, false);
    const g1 = legG(spot, k1, t, rate, div_yield, iv1, false);
    const g2 = legG(spot, k2, t, rate, div_yield, iv2, false);
    const g3 = legG(spot, k3, t, rate, div_yield, iv3, false);
    return .{
        .theo = p1 - 2.0 * p2 + p3,
        .greeks = g1.add(g2.scale(-2.0)).add(g3),
        .kind = .butterfly_put,
    };
}

/// ATM/OTM straddle: +C(K) +P(K).
pub fn straddleTheo(
    spot: f64,
    k: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv: f64,
) PackageTheo {
    const c = legPx(spot, k, t, rate, div_yield, iv, true);
    const p = legPx(spot, k, t, rate, div_yield, iv, false);
    const gc = legG(spot, k, t, rate, div_yield, iv, true);
    const gp = legG(spot, k, t, rate, div_yield, iv, false);
    return .{
        .theo = c + p,
        .greeks = gc.add(gp),
        .kind = .straddle,
    };
}

/// Strangle: +C(K_call) +P(K_put), typically K_put < spot < K_call.
pub fn strangleTheo(
    spot: f64,
    k_put: f64,
    k_call: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv_put: f64,
    iv_call: f64,
) PackageTheo {
    const c = legPx(spot, k_call, t, rate, div_yield, iv_call, true);
    const p = legPx(spot, k_put, t, rate, div_yield, iv_put, false);
    const gc = legG(spot, k_call, t, rate, div_yield, iv_call, true);
    const gp = legG(spot, k_put, t, rate, div_yield, iv_put, false);
    return .{
        .theo = c + p,
        .greeks = gc.add(gp),
        .kind = .strangle,
    };
}

/// Attach executable package bid/ask given per-leg SideQuotes and signed weights.
/// weight > 0 ⇒ long leg (buy at ask / sell at bid); weight < 0 ⇒ short.
pub fn attachPackageSides(pkg: *PackageTheo, legs: []const SideQuotes, weights: []const f64) void {
    std.debug.assert(legs.len == weights.len);
    var buy: f64 = 0.0;
    var sell: f64 = 0.0;
    var i: usize = 0;
    while (i < legs.len) : (i += 1) {
        const w = weights[i];
        if (w > 0.0) {
            buy += w * legs[i].ask;
            sell += w * legs[i].bid;
        } else if (w < 0.0) {
            // short: to buy package we *sell* the short leg → receive bid (|w| * bid),
            // so buy cost decreases; to sell package we *buy back* → pay ask.
            buy += w * legs[i].bid; // w negative, legs[i].bid positive → subtract
            sell += w * legs[i].ask;
        }
    }
    pkg.package_ask = buy;
    pkg.package_bid = sell;
    pkg.buy_edge = pkg.theo - pkg.package_ask;
    pkg.sell_edge = pkg.package_bid - pkg.theo;
}

/// Vertical call with sides.
pub fn verticalCallPackage(
    spot: f64,
    k1: f64,
    k2: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv1: f64,
    iv2: f64,
    leg_k1: SideQuotes,
    leg_k2: SideQuotes,
) PackageTheo {
    var pkg = verticalCallTheo(spot, k1, k2, t, rate, div_yield, iv1, iv2);
    const legs = [_]SideQuotes{ leg_k1, leg_k2 };
    const w = [_]f64{ 1.0, -1.0 };
    attachPackageSides(&pkg, &legs, &w);
    return pkg;
}

/// Straddle with sides.
pub fn straddlePackage(
    spot: f64,
    k: f64,
    t: f64,
    rate: f64,
    div_yield: f64,
    iv: f64,
    call_q: SideQuotes,
    put_q: SideQuotes,
) PackageTheo {
    var pkg = straddleTheo(spot, k, t, rate, div_yield, iv);
    const legs = [_]SideQuotes{ call_q, put_q };
    const w = [_]f64{ 1.0, 1.0 };
    attachPackageSides(&pkg, &legs, &w);
    return pkg;
}

/// Butterfly call with sides (weights +1, −2, +1).
pub fn butterflyCallPackage(
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
    leg1: SideQuotes,
    leg2: SideQuotes,
    leg3: SideQuotes,
) PackageTheo {
    var pkg = butterflyCallTheo(spot, k1, k2, k3, t, rate, div_yield, iv1, iv2, iv3);
    const legs = [_]SideQuotes{ leg1, leg2, leg3 };
    const w = [_]f64{ 1.0, -2.0, 1.0 };
    attachPackageSides(&pkg, &legs, &w);
    return pkg;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "ATM straddle vega and gamma positive and large" {
    const s: f64 = 100.0;
    const t: f64 = 0.25;
    const r: f64 = 0.05;
    const q: f64 = 0.0;
    const iv: f64 = 0.20;
    // Forward-ATM strike so straddle is near delta-neutral
    const k = s * @exp((r - q) * t);
    const pkg = straddleTheo(s, k, t, r, q, iv);
    try std.testing.expect(pkg.theo > 0.0);
    // Forward-ATM straddle ≈ delta-neutral
    try std.testing.expect(@abs(pkg.greeks.delta) < 0.05);
    try std.testing.expect(pkg.greeks.gamma > 0.0);
    try std.testing.expect(pkg.greeks.vega > 0.0);
    // Straddle vega ≈ 2 × call vega
    const call_g = legG(s, k, t, r, q, iv, true);
    try std.testing.expect(@abs(pkg.greeks.vega - 2.0 * call_g.vega) < 1e-9);
    try std.testing.expect(@abs(pkg.greeks.gamma - 2.0 * call_g.gamma) < 1e-9);
}

test "fly pin risk: gamma peaks at body, wings offset" {
    const s: f64 = 100.0;
    const k1: f64 = 95.0;
    const k2: f64 = 100.0;
    const k3: f64 = 105.0;
    const t: f64 = 5.0 / 365.25; // near expiry — pin risk
    const r: f64 = 0.05;
    const q: f64 = 0.0;
    const iv: f64 = 0.20;
    const fly = butterflyCallTheo(s, k1, k2, k3, t, r, q, iv, iv, iv);
    try std.testing.expect(fly.theo > 0.0);
    try std.testing.expect(fly.theo < (k2 - k1)); // max payoff = wing width
    // Long fly: short gamma at body (we are short 2 ATM)
    try std.testing.expect(fly.greeks.gamma < 0.0);
    // Near expiry ATM body has large |gamma|; wings smaller → net short gamma
    const g_body = legG(s, k2, t, r, q, iv, true);
    try std.testing.expect(g_body.gamma > 0.0);
}

test "vertical call theo positive for ITM-ish bull spread" {
    const pkg = verticalCallTheo(100.0, 95.0, 105.0, 0.25, 0.05, 0.0, 0.2, 0.2);
    try std.testing.expect(pkg.theo > 0.0);
    try std.testing.expect(pkg.theo < 10.0); // width
    try std.testing.expect(pkg.greeks.delta > 0.0);
}

test "straddle package sides: buy edge negative on fair mids with spread" {
    const s: f64 = 100.0;
    const k: f64 = 100.0;
    const t: f64 = 0.25;
    const r: f64 = 0.05;
    const iv: f64 = 0.2;
    const theo = straddleTheo(s, k, t, r, 0.0, iv).theo;
    const half: f64 = 0.10;
    const c = SideQuotes{ .bid = theo * 0.5 - half, .ask = theo * 0.5 + half };
    const p = SideQuotes{ .bid = theo * 0.5 - half, .ask = theo * 0.5 + half };
    const pkg = straddlePackage(s, k, t, r, 0.0, iv, c, p);
    try std.testing.expect(pkg.package_ask > pkg.theo);
    try std.testing.expect(pkg.buy_edge < 0.0);
    try std.testing.expect(pkg.sell_edge < 0.0);
}

test "butterfly call package attaches sides" {
    const s: f64 = 100.0;
    const k1: f64 = 95.0;
    const k2: f64 = 100.0;
    const k3: f64 = 105.0;
    const t: f64 = 0.25;
    const r: f64 = 0.05;
    const iv: f64 = 0.2;
    const theo = butterflyCallTheo(s, k1, k2, k3, t, r, 0.0, iv, iv, iv).theo;
    const half: f64 = 0.05;
    // Rough fair mids
    const p1 = legPx(s, k1, t, r, 0.0, iv, true);
    const p2 = legPx(s, k2, t, r, 0.0, iv, true);
    const p3 = legPx(s, k3, t, r, 0.0, iv, true);
    const l1 = SideQuotes{ .bid = p1 - half, .ask = p1 + half };
    const l2 = SideQuotes{ .bid = p2 - half, .ask = p2 + half };
    const l3 = SideQuotes{ .bid = p3 - half, .ask = p3 + half };
    const pkg = butterflyCallPackage(s, k1, k2, k3, t, r, 0.0, iv, iv, iv, l1, l2, l3);
    try std.testing.expect(@abs(pkg.theo - theo) < 1e-12);
    try std.testing.expect(pkg.package_ask >= pkg.package_bid);
}
