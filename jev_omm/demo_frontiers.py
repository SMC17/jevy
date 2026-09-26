"""Paper demo for frontiers 1–4. Simulation only.

    python -m jev_omm.demo_frontiers
"""

from __future__ import annotations

import math

from jev_omm.config import QuoterConfig
from jev_omm.execution.lob import expected_fills, fill_markout
from jev_omm.models.types import Side
from jev_omm.quoter.gueant import optimal_offsets as asym_offsets
from jev_omm.quoter.gueant_ode import estimate_intensity, optimal_offsets, spectral_offsets
from jev_omm.quoter.gueant_ode import IntensityObs
from jev_omm.risk.term import Leg, TermLimits, aggregate, evaluate_limits, quote_expiries, scenario_pnl
from jev_omm.surface.svi import (
    SsviParams,
    SviParams,
    butterfly_check,
    calibrate,
    implied_vol_after_move,
    ssvi_params_calendar_safe,
    ssvi_total_var,
    total_var,
)


def main() -> None:
    print("jevy frontiers demo — simulation / paper only")
    print("SVI+SSVI · multi-expiry term risk · LOB queue · Guéant ODE\n")

    print("=== 1. SVI / SSVI ===")
    planted = SviParams(0.04, 0.14, -0.35, 0.01, 0.22)
    ks = [-0.6 + 1.2 * i / 14 for i in range(15)]
    ws = [total_var(planted, k) for k in ks]
    fit = calibrate(ks, ws)
    smile = SsviParams(-0.35, 0.9, 0.4)
    print(
        f"fit rmse={fit.rmse:.6f}  butterfly_ok={fit.butterfly.ok}  "
        f"suspect={fit.surface_suspect}  a={fit.params.a:.4f} b={fit.params.b:.4f} rho={fit.params.rho:.3f}"
    )
    arb = butterfly_check(SviParams(0.04, 3.0, 0.9, 0.0, 0.2))
    print(f"injected arb: butterfly_ok={arb.ok} min_g={arb.min_g:.3f} lee={arb.lee_slope:.2f}")
    print(
        f"ssvi calendar-safe={ssvi_params_calendar_safe(smile)}  "
        f"w(0,0.04)={ssvi_total_var(0.0, 0.04, smile):.4f}"
    )
    print(
        "sticky-strike IV={:.4f}  sticky-delta IV={:.4f}".format(
            implied_vol_after_move(planted, 100.0, 108.0, 100.0, 0.3, "sticky_strike"),
            implied_vol_after_move(planted, 100.0, 108.0, 100.0, 0.3, "sticky_delta"),
        )
    )

    print("\n=== 2. Multi-expiry book ===")
    expiries = [30 / 365.25, 90 / 365.25, 180 / 365.25]
    strikes = [95.0, 100.0, 105.0]
    quoter = QuoterConfig(
        gamma=0.12,
        kappa=1.5,
        sigma=0.35,
        A=120.0,
        T_horizon=1.0 / 252.0,
        inventory_cap=8,
        ode_steps=400,
        mode="gueant_ode",
        quote_size=1,
        min_half_spread=0.02,
        max_half_spread=3.0,
    )
    strips = quote_expiries(100.0, 0.05, 0.0, smile, 0.22, expiries, strikes, quoter, True)
    for row in strips:
        atm = row["quotes"][1]
        q = atm["quote"]
        print(
            f"T={row['expiry']:.3f}y  theta={row['theta']:.5f}  "
            f"iv={atm['iv']:.3f} mid={atm['mid']:.3f} bid={q.bid:.3f} ask={q.ask:.3f}"
        )
    legs = [
        Leg(expiries[0], 100.0, 12.0, strips[0]["quotes"][1]["iv"]),
        Leg(expiries[1], 95.0, -5.0, strips[1]["quotes"][0]["iv"]),
        Leg(expiries[2], 105.0, 4.0, strips[2]["quotes"][2]["iv"]),
    ]
    risk = aggregate(100.0, 0.05, 0.0, legs)
    print(
        f"portfolio d={risk.delta:.3f} vega={risk.vega:.3f} "
        f"vanna={risk.vanna:.3f} volga={risk.volga:.3f} slope={risk.term_vega_slope:.3f}"
    )
    for b in risk.buckets:
        print(f"  bucket T={b.expiry:.3f} vega={b.vega:.3f} vanna={b.vanna:.3f} volga={b.volga:.3f}")
    print(f"scenario pnl={scenario_pnl(risk, 100.0, -0.02, 0.0, 0.05):.3f}")
    print(f"term limits: {evaluate_limits(risk, TermLimits(max_abs_parallel_vega=500, max_abs_bucket_vega=400))}")

    print("\n=== 3. LOB / queue fills ===")
    print(
        f"E[fills] shallow={expected_fills(2, 10, 25, 0, 1):.2f}  "
        f"deep={expected_fills(40, 10, 25, 0, 1):.2f}"
    )
    print(
        f"E[adverse] stay={expected_fills(0, 100, 40, 0, 1):.2f}  "
        f"late cancel={expected_fills(0, 100, 40, 0, 1, 0.2):.2f}"
    )
    print(
        f"markout toxic 0 → {fill_markout(Side.BID, 1.0, 1.05, 1.0, 0.03, 0.0):.4f}  "
        f"toxic 2.5 → {fill_markout(Side.BID, 1.0, 1.05, 1.0, 0.03, 2.5):.4f}"
    )

    print("\n=== 4. Guéant ODE / spectral + (A, k) ===")
    cfg = QuoterConfig(
        gamma=0.1,
        kappa=1.5,
        sigma=0.5,
        A=140.0,
        T_horizon=2.0,
        inventory_cap=12,
        ode_steps=2000,
        min_half_spread=1e-6,
        max_half_spread=50.0,
        mode="gueant_ode",
    )
    o0 = optimal_offsets(cfg, 0)
    a0 = asym_offsets(cfg, 0)
    spec = spectral_offsets(cfg, 0)
    deep = optimal_offsets(cfg, 11)
    adeep = asym_offsets(cfg, 11)
    print(
        f"q=0 ODE δb={o0.delta_b:.4f}  asymptotic δb={a0[0]:.4f}  spectral δb={spec.delta_b:.4f}"
    )
    print(f"q=11 ODE δb={deep.delta_b:.4f}  asymptotic δb={adeep[0]:.4f}")
    planted_A, planted_k = 36.0, 1.35
    obs = []
    for d in (0.2, 0.5, 0.9, 1.3, 1.8):
        lam = planted_A * math.exp(-planted_k * d)
        obs.append(IntensityObs(d, 400.0, round(lam * 400.0)))
    fit_ak = estimate_intensity(obs)
    print(f"tape MLE planted A={planted_A:.2f} k={planted_k:.3f}  recovered A={fit_ak.A:.2f} k={fit_ak.k:.3f}")
    print("\nDone. Jev is the decision layer only — it does not emit orders.")


if __name__ == "__main__":
    main()
