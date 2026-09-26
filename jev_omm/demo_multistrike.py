"""Multi-strike paper demo (Python mirror of Zig --multi-strike).

Run:
  python -m jev_omm.demo_multistrike
  python -m jev_omm.demo_multistrike --gueant

Simulation / paper only. Logs one Quote event per strike each step.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from rich.console import Console
from rich.table import Table

from jev_omm.config import QuoterConfig
from jev_omm.models.types import Greeks
from jev_omm.obs.event_log import EventLog
from jev_omm.pricing import price_and_greeks
from jev_omm.quoter.multi_strike import (
    StripConfig,
    StrikeSlot,
    build_strike_grid,
    portfolio_greeks,
    quote_strip,
)
from jev_omm.surface.sabr import SabrIVSurface


def main() -> None:
    ap = argparse.ArgumentParser(description="Jev OMM multi-strike paper demo")
    ap.add_argument("--gueant", action="store_true", help="Use Guéant asymptotics")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    console = Console()
    mode = "gueant_asymptotic" if args.gueant else "as_finite_horizon"
    cfg = QuoterConfig(
        gamma=0.12,
        kappa=1.5,
        sigma=0.45,
        A=140.0,
        quote_size=2,
        portfolio_delta_penalty=0.02,
        mode=mode,
    )
    strip = StripConfig(half_width=2, strike_step=1.0, is_call=True)
    sabr = SabrIVSurface(alpha=0.22, beta=1.0, rho=-0.3, nu=0.4, rate=0.05, div_yield=0.0)

    spot = 100.0
    rate, div_y, expiry = 0.05, 0.0, 30.0 / 365.25
    dt = 1.0 / (252.0 * 6.5 * 60.0)
    strikes = build_strike_grid(spot, strip)
    slots = [StrikeSlot(strike=k, inventory=0, active=True) for k in strikes]

    def iv_fn(forward: float, k: float, t: float) -> float:
        # SabrIVSurface.iv takes spot; recover spot from forward for consistency
        s = forward * math.exp(-(rate - div_y) * t) if t > 0 else forward
        return sabr.iv(s, k, t)

    def price_fn(s, k, t, r, q, iv, is_call):
        return price_and_greeks(s, k, t, r, q, iv, is_call)

    log_path = Path("jev_omm_events_multistrike_py.jsonl")
    console.print("[bold cyan]Jev Options MM — multi-strike paper demo[/bold cyan]")
    console.print(f"mode={mode}  backend={sabr.backend_name()}")
    console.print(f"strikes={[f'{k:.0f}' for k in strikes]}  log={log_path}\n")

    rng = random.Random(args.seed)
    elog = EventLog()
    elog.open(log_path)
    try:
        for step in range(args.steps):
            time = step * dt
            z = rng.gauss(0.0, 1.0)
            spot *= math.exp(-0.5 * 0.20**2 * dt + 0.20 * math.sqrt(dt) * z)
            spot = max(spot, 1e-6)
            t_rem = max(expiry - time, 1e-6)

            elog.append_underlying_tick(time, step, spot)
            quoted = quote_strip(
                spot=spot,
                t_rem=t_rem,
                rate=rate,
                div_yield=div_y,
                slots=slots,
                cfg=cfg,
                price_fn=price_fn,
                iv_fn=iv_fn,
                strip=strip,
            )
            greeks_list: list[Greeks] = []
            for slot in slots:
                match = next((sq for sq in quoted if sq.strike == slot.strike), None)
                greeks_list.append(match.greeks if match else Greeks())
            port = portfolio_greeks(slots, greeks_list)
            elog.append_decision_snapshot_raw(
                time,
                step,
                source="fallback",
                model="fallback-heuristic",
                answers={"regime": {"type": "choice", "choice": "calm", "confidence": 0.8}},
                confidence={"regime": 0.8},
                state={
                    "portfolio_delta": port.delta,
                    "portfolio_gamma": port.gamma,
                    "portfolio_vega": port.vega,
                    "net_inventory": port.net_inventory,
                    "n_strikes": len(quoted),
                    "quoter_mode": mode,
                },
            )
            for sq in quoted:
                elog.append_quote_strike(time, step, sq.strike, sq.quote)
                # Light Poisson-ish fill on ATM only for demo movement
                if abs(sq.strike - 100.0) < 0.5 and rng.random() < 0.05:
                    from jev_omm.models.types import Fill, Side

                    side = Side.BID if rng.random() < 0.5 else Side.ASK
                    px = sq.quote.bid if side == Side.BID else sq.quote.ask
                    fill = Fill(time=time, side=side, price=px, size=1, mid_at_fill=sq.mid)
                    for s in slots:
                        if abs(s.strike - sq.strike) < 1e-9:
                            s.inventory += 1 if side == Side.BID else -1
                            break
                    elog.append_fill(time, step, fill)
    finally:
        elog.close()

    table = Table(title="Final strip inventories")
    table.add_column("Strike")
    table.add_column("Inv", justify="right")
    for s in slots:
        table.add_row(f"{s.strike:.1f}", str(s.inventory))
    console.print(table)
    console.print(f"events={elog.seq}  sha256={elog.sha256_hex()}")
    console.print("[dim]Simulation / paper only.[/dim]")


if __name__ == "__main__":
    main()
