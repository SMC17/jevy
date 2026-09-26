"""Desk curriculum demo — parity, combos, banded hedge, greek PnL, scenarios, toxicity.

Run:  python -m jev_omm.demo_desk

Simulation / paper only. No live brokers.
"""

from __future__ import annotations

import math

from rich.console import Console
from rich.table import Table

from jev_omm.decisions.client import DeterministicFallbackClient
from jev_omm.decisions.schemas import build_mm_questions, build_mm_state
from jev_omm.flow.toxicity import ToxicityTracker
from jev_omm.hedge.delta import (
    apply_hedge,
    greek_pnl_step,
    net_delta,
    propose_delta_hedge,
    whalley_wilmott_band,
)
from jev_omm.models.types import Greeks
from jev_omm.pricing import ZIG_AVAILABLE, native_version
from jev_omm.pricing.black_scholes import greeks as bs_greeks
from jev_omm.pricing.black_scholes import price as bs_price
from jev_omm.pricing.combos import butterfly_call_theo, straddle_theo
from jev_omm.pricing.parity import SideQuotes, box_spread, box_theo, parity_diff, synthetic_forward_edge
from jev_omm.risk.scenario import build_matrix


def main() -> None:
    console = Console()
    console.print("\n[bold cyan]Jev Options MM — desk curriculum demo[/bold cyan]")
    console.print(
        f"Simulation only. native={native_version()}  zig={ZIG_AVAILABLE}\n"
    )

    spot, strike, t, rate, q, iv = 100.0, 100.0, 30.0 / 365.25, 0.05, 0.0, 0.22
    call_px = bs_price(spot, strike, t, rate, q, iv, True)
    put_px = bs_price(spot, strike, t, rate, q, iv, False)
    theo = parity_diff(spot, strike, t, rate, q)

    console.print("[bold]=== Put-call parity ===[/bold]")
    console.print(
        f"C={call_px:.4f}  P={put_px:.4f}  C−P={call_px - put_px:.4f}  "
        f"theo={theo:.4f}  resid={call_px - put_px - theo:.2e}"
    )
    half = 0.08
    put_q = SideQuotes(put_px - half, put_px + half)
    call_rich = SideQuotes(call_px + 0.40 - half, call_px + 0.40 + half)
    synth = synthetic_forward_edge(call_rich, put_q, spot, strike, t, rate, q)
    console.print(
        f"synth (call+0.40 rich): conversion={synth.conversion_edge:.4f}  "
        f"reversal={synth.reversal_edge:.4f}"
    )

    k1, k2 = 95.0, 105.0
    box = box_spread(
        SideQuotes(7.0, 7.10), SideQuotes(2.90, 3.00),
        SideQuotes(1.90, 2.00), SideQuotes(5.70, 5.80),
        k1, k2, t, rate,
    )
    console.print(f"\n[bold]=== Box {k1:.0f}/{k2:.0f} ===[/bold]")
    console.print(
        f"theo={box_theo(k1, k2, t, rate):.4f}  buy_debit={box.package_debit:.4f}  "
        f"buy_edge={box.buy_edge:.4f}  implied_r_buy={box.implied_rate_buy:.4f}"
    )

    stradd = straddle_theo(spot, strike, t, rate, q, iv)
    fly = butterfly_call_theo(spot, 95, 100, 105, t, rate, q, iv, iv, iv)
    console.print("\n[bold]=== Combos ===[/bold]")
    console.print(
        f"ATM straddle theo={stradd.theo:.4f}  δ={stradd.greeks.delta:.4f}  "
        f"Γ={stradd.greeks.gamma:.4f}  ν={stradd.greeks.vega:.4f}"
    )
    console.print(f"call fly 95/100/105 theo={fly.theo:.4f}  Γ={fly.greeks.gamma:.4f}")

    # Banded hedge mini-sim
    console.print("\n[bold]=== Banded delta hedge + greek PnL ===[/bold]")
    import random

    rng = random.Random(11)
    opt_qty = 12.0
    underlier = 0.0
    cash = 0.0
    s = spot
    dt = 1.0 / (252.0 * 6.5 * 60.0)
    spot_vol = 0.25
    band = 2.0
    half_spr = 0.02
    tox = ToxicityTracker(bucket_volume=8.0, window_buckets=8)
    cum_gamma = cum_theta = cum_slip = cum_mtm = 0.0
    hedge_fills = 0
    prev_mid = call_px
    n_steps = 80

    for step in range(n_steps):
        time = step * dt
        prev_s = s
        z = rng.gauss(0.0, 1.0)
        s *= math.exp((-0.5 * spot_vol * spot_vol) * dt + spot_vol * math.sqrt(dt) * z)
        s = max(s, 1e-6)
        t_rem = max(t - time, 1e-6)
        g1 = bs_greeks(s, strike, t_rem, rate, q, iv, True)
        mid = bs_price(s, strike, t_rem, rate, q, iv, True)
        port_delta = g1.delta * opt_qty
        port_gamma = g1.gamma * opt_qty
        nd = net_delta(port_delta, underlier)
        tox.on_trade(2.0 if s >= prev_s else -2.0)

        order = propose_delta_hedge(nd, band=band, half_spread=half_spr, flatten=True)
        slip = 0.0
        if order is not None:
            hf = apply_hedge(time, s, order, half_spread=half_spr, band=band)
            underlier += hf.underlier_qty
            cash += hf.cash_delta
            slip = hf.slippage_cost
            hedge_fills += 1

        g_prev = bs_greeks(prev_s, strike, t_rem, rate, q, iv, True)
        step_pnl = greek_pnl_step(
            delta=g_prev.delta * opt_qty,
            gamma=g_prev.gamma * opt_qty,
            vega=g_prev.vega * opt_qty,
            theta=g_prev.theta * opt_qty,
            d_spot=s - prev_s,
            dt=dt,
            option_qty=opt_qty,
            d_option_mid=mid - prev_mid,
            underlier_pos=underlier,
            hedge_slippage=slip,
        )
        cum_gamma += step_pnl.gamma_pnl
        cum_theta += step_pnl.theta_pnl
        cum_slip += step_pnl.hedge_slippage
        cum_mtm += step_pnl.inventory_mtm
        prev_mid = mid

    console.print(
        f"steps={n_steps}  hedge_fills={hedge_fills}  underlier={underlier:.2f}  cash={cash:.2f}"
    )
    console.print(
        f"greek buckets: gamma={cum_gamma:.4f}  theta={cum_theta:.4f}  "
        f"hedge_slip={cum_slip:.4f}  inv_mtm={cum_mtm:.4f}"
    )
    feats = tox.feature_dict()
    console.print(
        f"toxicity: vpin={feats['vpin']:.3f}  imbalance={feats['trade_imbalance']:.3f}  "
        f"composite={feats['toxicity_composite']:.3f}  (research-grade)"
    )

    # Decision state with toxicity features
    client = DeterministicFallbackClient()
    state = build_mm_state(
        time=n_steps * dt,
        spot=s,
        option_mid=prev_mid,
        iv=iv,
        inventory=int(opt_qty),
        delta=port_delta,
        gamma=port_gamma,
        vega=g1.vega * opt_qty,
        cash_pnl=cash,
        half_spread=0.1,
        quoting_allowed=True,
        toxicity_features=feats,
        underlier_pos=underlier,
        net_delta=net_delta(port_delta, underlier),
    )
    result = client.system_one(state, build_mm_questions())
    tox_ans = result.answers["toxicity"]
    console.print(
        f"Decision fallback toxicity score={getattr(tox_ans, 'score', tox_ans):}  "
        f"informed={result.answers['informed_flow'].noul:.2f}"
    )

    # Scenario matrix
    g_final = Greeks(
        delta=net_delta(g1.delta * opt_qty, underlier),
        gamma=g1.gamma * opt_qty,
        vega=g1.vega * opt_qty,
        theta=g1.theta * opt_qty,
    )
    mat = build_matrix(g_final, s, soft_loss=100.0, hard_loss=400.0)
    console.print("\n[bold]=== Scenario matrix ===[/bold]")
    console.print(
        f"min_pnl={mat.min_pnl:.2f}  max_pnl={mat.max_pnl:.2f}  "
        f"soft={mat.soft_breach}  hard={mat.hard_breach}"
    )
    table = Table(title="PnL grid (rows=spot%, cols=iv pts)")
    ivs = (-5, -2, 0, 2, 5)
    table.add_column("dS\\dIV", justify="right")
    for ivp in ivs:
        table.add_column(f"{ivp}", justify="right")
    spot_pcts = (-5, -2, -1, 0, 1, 2, 5)
    for i, sp in enumerate(spot_pcts):
        row = [f"{sp}%"] + [f"{mat.at(i, j).pnl:.2f}" for j in range(5)]
        table.add_row(*row)
    console.print(table)

    ww = whalley_wilmott_band(s, abs(g_final.gamma), spot_vol, 0.0002, 1e-3)
    console.print(f"\nWW-style suggested band≈{ww:.2f} (demo band={band})")
    console.print("[dim]Paper only — no live venue keys or broker SDKs.[/dim]\n")


if __name__ == "__main__":
    main()
