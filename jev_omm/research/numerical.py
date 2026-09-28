"""Numerical checks that do not add a new model.

- Constant-vega HJB grid vs a finer grid (Baldacci–Bergault–Guéant,
  https://arxiv.org/abs/1907.12433). This is self-convergence. It is not a
  reproduction of their §4 Euro Stoxx table; the code's own comment says the
  toy grid is not that calibration.
- Guéant RK4 vs the principal eigenmode (https://arxiv.org/abs/1105.3115).
- Black–Scholes Greeks vs central finite differences.
- Raw SVI butterfly g(k) on strikes wider than the calibration knots
  (Gatheral–Jacquier, https://arxiv.org/abs/1204.0646).
"""

from __future__ import annotations

import math

import numpy as np

from jev_omm.config import QuoterConfig
from jev_omm.pricing.black_scholes import greeks, price
from jev_omm.quoter.gueant_ode import optimal_offsets, spectral_offsets
from jev_omm.quoter.option_mm import OptionMmConfig, solve_grid, value_at
from jev_omm.surface.svi import SviParams, calibrate, density_g, total_var


def hjb_convergence() -> list[dict[str, float]]:
    rows = []
    for n, steps in ((11, 20), (21, 40), (31, 80), (41, 120)):
        cfg = OptionMmConfig(grid_n=n, n_steps=steps, vega_limit=40.0, A=40.0, kappa=2.0, gamma=0.5)
        grid = solve_grid(cfg, [10.0])
        rows.append({"n_v": float(n), "n_t": float(steps), "w0": float(value_at(grid, 0.0))})
    ref = rows[-1]["w0"]
    for row in rows:
        row["abs_err_vs_finest"] = abs(row["w0"] - ref)
    return rows


def gueant_rk4_vs_spectral() -> list[dict[str, float]]:
    rows = []
    grid = (
        (0.10, 1.5, 80.0, 0.40),
        (0.20, 1.2, 60.0, 0.35),
        (0.05, 2.0, 100.0, 0.30),
    )
    for gamma, kappa, A, sigma in grid:
        cfg = QuoterConfig(
            mode="gueant_ode",
            gamma=gamma,
            kappa=kappa,
            sigma=sigma,
            A=A,
            inventory_cap=4,
            T_horizon=3.0,
            ode_steps=800,
            min_half_spread=1e-6,
            max_half_spread=50.0,
        )
        rk = optimal_offsets(cfg, 1)
        sp = spectral_offsets(cfg, 1)
        rows.append(
            {
                "gamma": gamma,
                "kappa": kappa,
                "A": A,
                "sigma": sigma,
                "rk4_delta_b": float(rk.delta_b),
                "spectral_delta_b": float(sp.delta_b),
                "rk4_delta_a": float(rk.delta_a),
                "spectral_delta_a": float(sp.delta_a),
                "abs_db": abs(float(rk.delta_b) - float(sp.delta_b)),
                "abs_da": abs(float(rk.delta_a) - float(sp.delta_a)),
            }
        )
    return rows


def _fd(spot: float, strike: float, t: float, rate: float, q: float, iv: float, is_call: bool) -> dict[str, float]:
    ds = max(1e-2, 1e-4 * spot)
    dv = 1e-4
    base = price(spot, strike, t, rate, q, iv, is_call)
    up = price(spot + ds, strike, t, rate, q, iv, is_call)
    dn = price(spot - ds, strike, t, rate, q, iv, is_call)
    delta = (up - dn) / (2.0 * ds)
    gamma = (up - 2.0 * base + dn) / (ds * ds)
    vu = price(spot, strike, t, rate, q, iv + dv, is_call)
    vd = price(spot, strike, t, rate, q, iv - dv, is_call)
    vega = (vu - vd) / (2.0 * dv)
    return {"delta": delta, "gamma": gamma, "vega": vega}


def greeks_vs_finite_difference() -> list[dict[str, float]]:
    cases = (
        (100.0, 100.0, 0.25, 0.05, 0.0, 0.20, True),
        (100.0, 110.0, 0.10, 0.01, 0.02, 0.30, False),
        (80.0, 75.0, 1.0, 0.03, 0.01, 0.18, True),
    )
    rows = []
    for spot, strike, t, rate, q, iv, is_call in cases:
        analytic = greeks(spot, strike, t, rate, q, iv, is_call)
        fd = _fd(spot, strike, t, rate, q, iv, is_call)
        rows.append(
            {
                "spot": spot,
                "strike": strike,
                "is_call": 1.0 if is_call else 0.0,
                "delta_an": analytic.delta,
                "delta_fd": fd["delta"],
                "gamma_an": analytic.gamma,
                "gamma_fd": fd["gamma"],
                "vega_an": analytic.vega,
                "vega_fd": fd["vega"],
                "abs_delta": abs(analytic.delta - fd["delta"]),
                "abs_gamma": abs(analytic.gamma - fd["gamma"]),
                "abs_vega": abs(analytic.vega - fd["vega"]),
            }
        )
    return rows


def svi_arb_sweep() -> dict[str, float]:
    """Calibrate on a narrow log-strike knot set; score g(k) out on the wings."""
    truth = SviParams(0.04, 0.1, -0.4, 0.0, 0.2)
    knots = np.linspace(-0.25, 0.25, 7)
    ws = [total_var(truth, float(k)) for k in knots]
    fit = calibrate(knots, ws)
    wing = np.linspace(-1.2, 1.2, 49)
    gs = np.array([density_g(fit.params, float(k)) for k in wing])
    outside = np.array([abs(float(k)) > 0.25 + 1e-12 for k in wing])
    return {
        "fit_a": float(fit.params.a),
        "fit_b": float(fit.params.b),
        "fit_rho": float(fit.params.rho),
        "fit_m": float(fit.params.m),
        "fit_sigma": float(fit.params.sigma),
        "min_g_full": float(gs.min()),
        "min_g_outside_knots": float(gs[outside].min()) if outside.any() else float("nan"),
        "n_outside": float(outside.sum()),
        "max_abs_w_err_on_knots": float(
            max(abs(total_var(fit.params, float(k)) - total_var(truth, float(k))) for k in knots)
        ),
    }


def render_numerical_markdown() -> str:
    hjb = hjb_convergence()
    gue = gueant_rk4_vs_spectral()
    gr = greeks_vs_finite_difference()
    svi = svi_arb_sweep()
    lines = [
        "# Numerical checks",
        "",
        "Self-convergence and finite-difference checks. They are not a claim",
        "that a published calibration table was reproduced. Citations:",
        "",
        "- Baldacci, Bergault, Guéant, https://arxiv.org/abs/1907.12433",
        "  (constant-vega HJB). The Euro Stoxx grid in their §4 is not this toy.",
        "- Guéant, Lehalle, Fernandez-Tapia, https://arxiv.org/abs/1105.3115",
        "  (RK4 on the ODE versus the principal eigenmode).",
        "- Black–Scholes–Merton Greeks versus central differences.",
        "- Gatheral, Jacquier, https://arxiv.org/abs/1204.0646",
        "  (butterfly g(k) outside the knots used to calibrate).",
        "",
        "## HJB value at zero vega",
        "",
        "| N_V | N_t | w(0) | abs err vs finest |",
        "| --- | --- | --- | --- |",
    ]
    for row in hjb:
        lines.append(
            f"| {int(row['n_v'])} | {int(row['n_t'])} | {row['w0']:.6f} | {row['abs_err_vs_finest']:.6f} |"
        )
    lines += [
        "",
        "The last row is the reference, so its error is zero by construction.",
        "A smaller error on a finer grid is the check; the level of w(0) is",
        "not a published digit.",
        "",
        "## Guéant RK4 vs spectral (inventory = 1, cap = 4, horizon = 3y, 800 steps)",
        "",
        "| γ | k | A | σ | RK4 δb | spectral δb | abs | RK4 δa | spectral δa | abs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in gue:
        lines.append(
            "| {gamma:.2f} | {kappa:.2f} | {A:.1f} | {sigma:.2f} | {rk4_delta_b:.4f} | "
            "{spectral_delta_b:.4f} | {abs_db:.4f} | {rk4_delta_a:.4f} | "
            "{spectral_delta_a:.4f} | {abs_da:.4f} |".format(**row)
        )
    lines += [
        "",
        "Agreement is a numerical statement about this implementation.",
        "A large gap would mean the horizon is too short for the eigenmode",
        "or the RK4 step is coarse. It is not a latency number.",
        "",
        "## Greeks vs central differences",
        "",
        "| spot | strike | call | |Δ| | |Γ| | |ν| |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in gr:
        lines.append(
            f"| {row['spot']:.1f} | {row['strike']:.1f} | {int(row['is_call'])} | "
            f"{row['abs_delta']:.3e} | {row['abs_gamma']:.3e} | {row['abs_vega']:.3e} |"
        )
    lines += [
        "",
        "Steps: ΔS = max(1e-2, 1e-4 S), Δσ = 1e-4. Analytic minus central difference.",
        "",
        "## SVI wings past the calibration knots",
        "",
        "Truth `SviParams(0.04, 0.1, -0.4, 0, 0.2)` is sampled on k ∈ [−0.25, 0.25]",
        "(7 knots) and refit. g(k) is then evaluated on k ∈ [−1.2, 1.2].",
        "",
        f"- fit (a, b, ρ, m, σ) = ({svi['fit_a']:.4f}, {svi['fit_b']:.4f}, "
        f"{svi['fit_rho']:.4f}, {svi['fit_m']:.4f}, {svi['fit_sigma']:.4f})",
        f"- max |w error| on the knots: {svi['max_abs_w_err_on_knots']:.6e}",
        f"- min g(k) on the full sweep: {svi['min_g_full']:.6f}",
        f"- min g(k) outside the knots ({int(svi['n_outside'])} points): "
        f"{svi['min_g_outside_knots']:.6f}",
        "",
        "Negative g outside the knots is a real arbitrage the in-sample check can miss.",
        "",
    ]
    return "\n".join(lines) + "\n"
