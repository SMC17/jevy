//! Guéant–Lehalle–Fernandez-Tapia finite-horizon ODE and spectral quotes.
//!
//! Same intensity as the asymptotic module: λ(δ) = A exp(−k δ).
//! Cite: https://arxiv.org/abs/1105.3115
//!
//! Ansatz u = −exp(−γ(x + q s) + ω_q(t)), terminal ω_q(T) = 0.
//! In time-to-go τ = T − t the inventory system is
//!
//!   dω_q/dτ = (σ² γ² q²)/2 − Σ_{neighbors} C exp(−(k/γ)(ω_nb − ω_q))
//!   C = A · (γ/(k+γ)) · (k/(k+γ))^{k/γ}
//!
//! with the neighbor missing at the inventory caps ±Q (cannot buy at +Q,
//! cannot sell at −Q). The change of variables v_q = exp(−(k/γ) ω_q) turns
//! this into the linear system dv/dτ = M v of the paper (tridiagonal spectral
//! problem). Long-τ ratios of v are the principal eigenmode; `spectralOffsets`
//! isolates that mode by RK4 + renormalization.
//!
//! Optimal distances to the reference mid:
//!
//!   δ^b(q) = (1/γ) ln(1 + γ/k) + (ω_{q+1} − ω_q)/γ
//!   δ^a(q) = (1/γ) ln(1 + γ/k) + (ω_{q−1} − ω_q)/γ
//!
//! (A, k) from a synthetic tape: Poisson MLE for λ(δ) = A e^{−k δ}
//! (Newton on (ln A, k), seeded by weighted log-linear regression).

const std = @import("std");
const types = @import("types.zig");
const gueant = @import("gueant.zig");

const QuoterConfig = types.QuoterConfig;
const Greeks = types.Greeks;
const Quote = types.Quote;

pub const MAX_Q: i32 = 32;
const MAX_N: usize = @as(usize, @intCast(2 * MAX_Q + 1));
pub const NO_QUOTE: f64 = 1.0e6;

pub const Offsets = struct {
    delta_b: f64 = 0.0,
    delta_a: f64 = 0.0,
    half: f64 = 0.0,
    reservation_shift: f64 = 0.0,
};

pub const IntensityObs = struct {
    delta: f64 = 0.0,
    exposure: f64 = 0.0,
    fills: f64 = 0.0,
};

pub const IntensityFit = struct {
    A: f64 = 0.0,
    k: f64 = 0.0,
    loglik: f64 = 0.0,
    n_bins: usize = 0,
};

fn clampQ(q: i32) i32 {
    if (q > MAX_Q) return MAX_Q;
    if (q < 1) return 1;
    return q;
}

pub fn coefficientC(gamma: f64, k: f64, A: f64) f64 {
    if (!(gamma > 0.0) or !(k > 0.0) or !(A > 0.0)) return 0.0;
    const ratio = k / (k + gamma);
    return A * (gamma / (k + gamma)) * std.math.pow(f64, ratio, k / gamma);
}

fn deriv(
    w: *const [MAX_N]f64,
    dw: *[MAX_N]f64,
    q_cap: i32,
    alpha: f64,
    C: f64,
    coef: f64,
) void {
    const n: usize = @intCast(2 * q_cap + 1);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const q: i32 = @as(i32, @intCast(i)) - q_cap;
        var benefit: f64 = 0.0;
        if (q + 1 <= q_cap) {
            const expo = std.math.clamp(-coef * (w[i + 1] - w[i]), -80.0, 80.0);
            benefit += C * @exp(expo);
        }
        if (q - 1 >= -q_cap) {
            const expo = std.math.clamp(-coef * (w[i - 1] - w[i]), -80.0, 80.0);
            benefit += C * @exp(expo);
        }
        const qq: f64 = @floatFromInt(q);
        dw[i] = alpha * qq * qq - benefit;
    }
}

fn addScaled(dst: *[MAX_N]f64, a: *const [MAX_N]f64, b: *const [MAX_N]f64, s: f64, n: usize) void {
    var i: usize = 0;
    while (i < n) : (i += 1) dst[i] = a[i] + s * b[i];
}

fn axpy4(dst: *[MAX_N]f64, y: *const [MAX_N]f64, k1: *const [MAX_N]f64, k2: *const [MAX_N]f64, k3: *const [MAX_N]f64, k4: *const [MAX_N]f64, dt: f64, n: usize) void {
    const c = dt / 6.0;
    var i: usize = 0;
    while (i < n) : (i += 1) dst[i] = y[i] + c * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

/// Integrate ω(τ) from 0 to `horizon` with RK4. `out[q + Q] = ω_q`.
pub fn solveOmega(cfg: *const QuoterConfig, q_cap_in: i32, horizon: f64, n_steps_in: u32, out: *[MAX_N]f64) i32 {
    const q_cap = clampQ(q_cap_in);
    const n: usize = @intCast(2 * q_cap + 1);
    @memset(out[0..n], 0.0);
    const gamma = cfg.gamma;
    const k = cfg.kappa;
    if (!(gamma > 0.0) or !(k > 0.0) or !(cfg.A > 0.0) or horizon <= 0.0) return q_cap;
    const steps: u32 = if (n_steps_in == 0) 1 else n_steps_in;
    const dt = horizon / @as(f64, @floatFromInt(steps));
    const alpha = 0.5 * cfg.sigma * cfg.sigma * gamma * gamma;
    const C = coefficientC(gamma, k, cfg.A);
    const coef = k / gamma;

    var w = [_]f64{0} ** MAX_N;
    var k1 = [_]f64{0} ** MAX_N;
    var k2 = [_]f64{0} ** MAX_N;
    var k3 = [_]f64{0} ** MAX_N;
    var k4 = [_]f64{0} ** MAX_N;
    var tmp = [_]f64{0} ** MAX_N;

    var step: u32 = 0;
    while (step < steps) : (step += 1) {
        deriv(&w, &k1, q_cap, alpha, C, coef);
        addScaled(&tmp, &w, &k1, 0.5 * dt, n);
        deriv(&tmp, &k2, q_cap, alpha, C, coef);
        addScaled(&tmp, &w, &k2, 0.5 * dt, n);
        deriv(&tmp, &k3, q_cap, alpha, C, coef);
        addScaled(&tmp, &w, &k3, dt, n);
        deriv(&tmp, &k4, q_cap, alpha, C, coef);
        axpy4(&w, &w, &k1, &k2, &k3, &k4, dt, n);
    }
    @memcpy(out[0..n], w[0..n]);
    return q_cap;
}

pub fn offsetsAt(cfg: *const QuoterConfig, omega: *const [MAX_N]f64, q_cap: i32, inventory: i32) Offsets {
    var o = Offsets{};
    const gamma = cfg.gamma;
    const k = cfg.kappa;
    if (!(gamma > 0.0) or !(k > 0.0)) return o;
    const psi = (1.0 / gamma) * @log(1.0 + gamma / k);
    const q = std.math.clamp(inventory, -q_cap, q_cap);
    const i: usize = @intCast(q + q_cap);
    if (q + 1 <= q_cap) {
        o.delta_b = psi + (omega[i + 1] - omega[i]) / gamma;
    } else o.delta_b = NO_QUOTE;
    if (q - 1 >= -q_cap) {
        o.delta_a = psi + (omega[i - 1] - omega[i]) / gamma;
    } else o.delta_a = NO_QUOTE;
    const db = if (o.delta_b >= NO_QUOTE * 0.5) cfg.max_half_spread else o.delta_b;
    const da = if (o.delta_a >= NO_QUOTE * 0.5) cfg.max_half_spread else o.delta_a;
    o.half = 0.5 * (da + db);
    o.reservation_shift = 0.5 * (da - db);
    return o;
}

pub fn optimalOffsets(cfg: *const QuoterConfig, inventory: i32) Offsets {
    var omega = [_]f64{0} ** MAX_N;
    const q_cap = solveOmega(cfg, cfg.inventory_cap, @max(cfg.t_horizon, 1e-6), cfg.ode_steps, &omega);
    return offsetsAt(cfg, &omega, q_cap, inventory);
}

/// Principal eigenmode of dv/dτ = M v via RK4 in v-space with renormalization.
/// Equivalent to the long-horizon ODE; ratios v_q / v_{q±1} are the spectral quotes.
pub fn spectralOffsets(cfg: *const QuoterConfig, inventory: i32) Offsets {
    const q_cap = clampQ(cfg.inventory_cap);
    const n: usize = @intCast(2 * q_cap + 1);
    const gamma = cfg.gamma;
    const k = cfg.kappa;
    if (!(gamma > 0.0) or !(k > 0.0) or !(cfg.A > 0.0)) return .{};
    const ratio = k / (k + gamma);
    const eta = cfg.A * std.math.pow(f64, ratio, (k + gamma) / gamma);
    const diag_coef = 0.5 * k * gamma * cfg.sigma * cfg.sigma;

    var v = [_]f64{1} ** MAX_N;
    var k1 = [_]f64{0} ** MAX_N;
    var k2 = [_]f64{0} ** MAX_N;
    var k3 = [_]f64{0} ** MAX_N;
    var k4 = [_]f64{0} ** MAX_N;
    var tmp = [_]f64{0} ** MAX_N;

    const mv = struct {
        fn apply(src: *const [MAX_N]f64, dst: *[MAX_N]f64, qcap: i32, nn: usize, eta_: f64, diag: f64) void {
            var i: usize = 0;
            while (i < nn) : (i += 1) {
                const q: i32 = @as(i32, @intCast(i)) - qcap;
                const qq: f64 = @floatFromInt(q);
                var acc = -diag * qq * qq * src[i];
                if (q + 1 <= qcap) acc += eta_ * src[i + 1];
                if (q - 1 >= -qcap) acc += eta_ * src[i - 1];
                dst[i] = acc;
            }
        }
    }.apply;

    // dt well below 1/η. A few thousand steps isolate the principal mode.
    const dt: f64 = 0.002;
    const steps: usize = 4000;
    var step: usize = 0;
    while (step < steps) : (step += 1) {
        mv(&v, &k1, q_cap, n, eta, diag_coef);
        addScaled(&tmp, &v, &k1, 0.5 * dt, n);
        mv(&tmp, &k2, q_cap, n, eta, diag_coef);
        addScaled(&tmp, &v, &k2, 0.5 * dt, n);
        mv(&tmp, &k3, q_cap, n, eta, diag_coef);
        addScaled(&tmp, &v, &k3, dt, n);
        mv(&tmp, &k4, q_cap, n, eta, diag_coef);
        axpy4(&v, &v, &k1, &k2, &k3, &k4, dt, n);
        var sum: f64 = 0.0;
        var i: usize = 0;
        while (i < n) : (i += 1) sum += @abs(v[i]);
        const scale = sum / @as(f64, @floatFromInt(n));
        if (scale > 0.0) {
            i = 0;
            while (i < n) : (i += 1) v[i] /= scale;
        }
    }

    // ω = −(γ/k) ln v, quotes use differences so the additive constant drops.
    var omega = [_]f64{0} ** MAX_N;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const vv = @max(v[i], 1e-300);
        omega[i] = -(gamma / k) * @log(vv);
    }
    return offsetsAt(cfg, &omega, q_cap, inventory);
}

pub fn reservationPrice(
    mid: f64,
    inventory: i32,
    cfg: *const QuoterConfig,
    greeks_opt: ?*const Greeks,
) f64 {
    const o = optimalOffsets(cfg, inventory);
    var r = mid + o.reservation_shift;
    if (greeks_opt) |g| {
        if (inventory != 0) {
            const inv = @as(f64, @floatFromInt(inventory));
            r -= inv * cfg.gamma_penalty * @abs(g.gamma);
            r -= inv * cfg.vega_penalty * @abs(g.vega) * 0.01;
        }
    }
    return r;
}

pub fn optimalHalfSpread(cfg: *const QuoterConfig, inventory: i32) f64 {
    const o = optimalOffsets(cfg, inventory);
    return std.math.clamp(o.half, cfg.min_half_spread, cfg.max_half_spread);
}

pub fn makeQuote(
    mid_in: f64,
    inventory: i32,
    cfg: *const QuoterConfig,
    greeks_opt: ?*const Greeks,
    spread_mult: f64,
    size_mult: f64,
) Quote {
    var mid = mid_in;
    if (mid <= 0.0) mid = @max(mid, 0.01);
    const o = optimalOffsets(cfg, inventory);
    var half = o.half * @max(spread_mult, 0.25);
    half = std.math.clamp(half, cfg.min_half_spread, cfg.max_half_spread);
    var r = mid + o.reservation_shift;
    if (greeks_opt) |g| {
        if (inventory != 0) {
            const inv = @as(f64, @floatFromInt(inventory));
            r -= inv * cfg.gamma_penalty * @abs(g.gamma);
            r -= inv * cfg.vega_penalty * @abs(g.vega) * 0.01;
        }
    }
    const bid_open = o.delta_b < NO_QUOTE * 0.5;
    const ask_open = o.delta_a < NO_QUOTE * 0.5;
    const bid = if (bid_open) @max(0.01, r - half) else 0.01;
    const ask = if (ask_open) @max(bid + 0.01, r + half) else @max(bid + 0.01, r + half);
    var size: i32 = @intFromFloat(@round(@as(f64, @floatFromInt(cfg.quote_size)) * @max(size_mult, 0.0)));
    if (size < 1) size = 1;
    return .{
        .bid = bid,
        .ask = ask,
        .bid_size = if (bid_open) size else 0,
        .ask_size = if (ask_open) size else 0,
        .reservation = r,
        .half_spread = half,
    };
}

fn logLik(obs: []const IntensityObs, A: f64, k: f64) f64 {
    var ll: f64 = 0.0;
    for (obs) |o| {
        if (!(o.exposure > 0.0)) continue;
        const lam = A * @exp(-k * o.delta);
        const mean = lam * o.exposure;
        if (o.fills > 0.0) ll += o.fills * @log(@max(mean, 1e-300));
        ll -= mean;
    }
    return ll;
}

/// Poisson MLE for λ(δ) = A e^{−k δ}. Bins with exposure ≤ 0 are ignored.
pub fn estimateIntensity(obs: []const IntensityObs) IntensityFit {
    var fit = IntensityFit{};
    var n_used: usize = 0;
    var sw: f64 = 0;
    var swx: f64 = 0;
    var swy: f64 = 0;
    var swxx: f64 = 0;
    var swxy: f64 = 0;
    for (obs) |o| {
        if (!(o.exposure > 0.0)) continue;
        n_used += 1;
        if (!(o.fills > 0.0)) continue;
        const y = @log(o.fills / o.exposure);
        const x = -o.delta;
        const w = o.fills;
        sw += w;
        swx += w * x;
        swy += w * y;
        swxx += w * x * x;
        swxy += w * x * y;
    }
    fit.n_bins = n_used;
    if (n_used == 0 or sw <= 0.0) return fit;
    const det = sw * swxx - swx * swx;
    var ln_a: f64 = @log(1.0);
    var k: f64 = 1.0;
    if (@abs(det) > 1e-12) {
        ln_a = (swxx * swy - swx * swxy) / det;
        // y = ln A + (−δ) k, and x = −δ, so slope on x is k.
        k = (sw * swxy - swx * swy) / det;
    }
    if (!(k > 1e-6)) k = 1e-6;
    if (!(ln_a == ln_a)) ln_a = 0.0;

    var iter: usize = 0;
    while (iter < 16) : (iter += 1) {
        const A = @exp(std.math.clamp(ln_a, -20.0, 20.0));
        var g1: f64 = 0;
        var g2: f64 = 0;
        var h11: f64 = 0;
        var h12: f64 = 0;
        var h22: f64 = 0;
        for (obs) |o| {
            if (!(o.exposure > 0.0)) continue;
            const mean = A * @exp(-k * o.delta) * o.exposure;
            g1 += o.fills - mean;
            g2 += -o.fills * o.delta + mean * o.delta;
            h11 += -mean;
            h12 += mean * o.delta;
            h22 += -mean * o.delta * o.delta;
        }
        const hdet = h11 * h22 - h12 * h12;
        if (!(@abs(hdet) > 1e-18)) break;
        const d_ln = (h22 * g1 - h12 * g2) / hdet;
        const d_k = (-h12 * g1 + h11 * g2) / hdet;
        ln_a -= d_ln;
        k -= d_k;
        if (k < 1e-6) k = 1e-6;
        if (@abs(d_ln) < 1e-10 and @abs(d_k) < 1e-10) break;
    }
    fit.A = @exp(std.math.clamp(ln_a, -20.0, 20.0));
    fit.k = k;
    fit.loglik = logLik(obs, fit.A, fit.k);
    return fit;
}

/// Plant Poisson counts at fixed offsets (synthetic tape, no live market data).
pub fn simulateTape(
    rng: std.Random,
    A: f64,
    k: f64,
    deltas: []const f64,
    exposure: f64,
    out: []IntensityObs,
) usize {
    const n = @min(deltas.len, out.len);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const lam = A * @exp(-k * deltas[i]) * exposure;
        const fills = poissonSample(rng, lam);
        out[i] = .{ .delta = deltas[i], .exposure = exposure, .fills = @floatFromInt(fills) };
    }
    return n;
}

fn poissonSample(rng: std.Random, lambda: f64) i32 {
    if (lambda <= 0.0) return 0;
    if (lambda > 30.0) {
        const z = rng.floatNorm(f64);
        const x = lambda + @sqrt(lambda) * z;
        const rounded = @round(x);
        return if (rounded > 0.0) @intFromFloat(rounded) else 0;
    }
    const L = @exp(-lambda);
    var kk: i32 = 0;
    var p: f64 = 1.0;
    while (true) {
        kk += 1;
        p *= rng.float(f64);
        if (p <= L) break;
        if (kk > 1_000_000) break;
    }
    return kk - 1;
}

fn relClose(a: f64, b: f64, tol: f64) bool {
    const den = @max(@abs(b), 1e-9);
    return @abs(a - b) / den < tol;
}

test "ode near flat inventory matches gueant asymptotic" {
    const cfg = QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.5,
        .A = 140.0,
        .t_horizon = 2.0,
        .inventory_cap = 20,
        .ode_steps = 2000,
        .min_half_spread = 1e-6,
        .max_half_spread = 50.0,
        .mode = .gueant_ode,
    };
    const ode = optimalOffsets(&cfg, 0);
    const asym = gueant.optimalOffsets(&cfg, 0);
    try std.testing.expect(relClose(ode.delta_b, asym.delta_b, 0.02));
    try std.testing.expect(relClose(ode.delta_a, asym.delta_a, 0.02));
    const ode5 = optimalOffsets(&cfg, 2);
    const asym5 = gueant.optimalOffsets(&cfg, 2);
    try std.testing.expect(relClose(ode5.delta_b, asym5.delta_b, 0.08));
    try std.testing.expect(ode5.delta_b > ode.delta_b);
    try std.testing.expect(ode5.delta_a < ode.delta_a);
}

test "long-horizon rk4 offsets stay near the spectral mode" {
    // Guéant–Lehalle–Fernandez-Tapia, arXiv 1105.3115. The spectral
    // eigenmode is the stationary shape; RK4 over a long horizon should
    // land nearby. Tolerance is relative, not a published table digit.
    const cfg = QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.4,
        .A = 80.0,
        .t_horizon = 4.0,
        .inventory_cap = 6,
        .ode_steps = 1600,
        .min_half_spread = 1e-6,
        .max_half_spread = 50.0,
        .mode = .gueant_ode,
    };
    const rk = optimalOffsets(&cfg, 1);
    const sp = spectralOffsets(&cfg, 1);
    try std.testing.expect(relClose(rk.delta_b, sp.delta_b, 0.2));
    try std.testing.expect(relClose(rk.delta_a, sp.delta_a, 0.2));
}

test "ode deep inventory bids wider than the unbounded asymptotic" {
    const cfg = QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.5,
        .A = 140.0,
        .t_horizon = 3.0,
        .inventory_cap = 10,
        .ode_steps = 3000,
        .min_half_spread = 1e-6,
        .max_half_spread = 50.0,
        .mode = .gueant_ode,
    };
    const q: i32 = 9;
    const ode = optimalOffsets(&cfg, q);
    const asym = gueant.optimalOffsets(&cfg, q);
    try std.testing.expect(ode.delta_b > asym.delta_b * 1.15);
    // At the cap the bid is shut.
    const at_cap = optimalOffsets(&cfg, 10);
    try std.testing.expect(at_cap.delta_b >= NO_QUOTE * 0.5);
    const qte = makeQuote(5.0, 10, &cfg, null, 1.0, 1.0);
    try std.testing.expect(qte.bid_size == 0);
    try std.testing.expect(qte.ask_size > 0);
}

test "spectral principal mode agrees with asymptotic at q=0" {
    const cfg = QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.5,
        .A = 140.0,
        .inventory_cap = 12,
        .min_half_spread = 1e-6,
        .max_half_spread = 50.0,
    };
    const spec = spectralOffsets(&cfg, 0);
    const asym = gueant.optimalOffsets(&cfg, 0);
    try std.testing.expect(relClose(spec.delta_b, asym.delta_b, 0.05));
    try std.testing.expect(relClose(spec.delta_a, asym.delta_a, 0.05));
    const spec_long = spectralOffsets(&cfg, 6);
    try std.testing.expect(spec_long.delta_b > spec.delta_b);
    try std.testing.expect(spec_long.delta_a < spec.delta_a);
}

test "intensity mle recovers planted A and k" {
    const A: f64 = 40.0;
    const k: f64 = 1.2;
    const deltas = [_]f64{ 0.2, 0.4, 0.7, 1.0, 1.4, 1.8 };
    const exposure: f64 = 500.0;
    var obs: [6]IntensityObs = undefined;
    for (deltas, 0..) |d, i| {
        const lam = A * @exp(-k * d);
        obs[i] = .{ .delta = d, .exposure = exposure, .fills = @round(lam * exposure) };
    }
    const fit = estimateIntensity(&obs);
    try std.testing.expect(@abs(fit.A - A) / A < 0.05);
    try std.testing.expect(@abs(fit.k - k) / k < 0.05);

    var prng = std.Random.DefaultPrng.init(7);
    var tape: [6]IntensityObs = undefined;
    _ = simulateTape(prng.random(), A, k, &deltas, exposure, &tape);
    const fit2 = estimateIntensity(&tape);
    try std.testing.expect(@abs(fit2.A - A) / A < 0.15);
    try std.testing.expect(@abs(fit2.k - k) / k < 0.15);
}

test "ode quotes skew with inventory" {
    const cfg = QuoterConfig{
        .gamma = 0.1,
        .kappa = 1.5,
        .sigma = 0.4,
        .A = 120.0,
        .t_horizon = 1.0,
        .inventory_cap = 8,
        .ode_steps = 800,
        .quote_size = 1,
        .min_half_spread = 0.01,
        .max_half_spread = 20.0,
        .mode = .gueant_ode,
    };
    const q_long = makeQuote(4.0, 4, &cfg, null, 1.0, 1.0);
    const q_short = makeQuote(4.0, -4, &cfg, null, 1.0, 1.0);
    try std.testing.expect(q_long.reservation < q_short.reservation);
    try std.testing.expect(q_long.bid < q_short.bid);
}
