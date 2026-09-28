"""Short paper simulation demo.

Run:  python -m jev_omm.demo

Uses DeterministicFallbackClient unless TYPESAFE_API_KEY is set.
No live brokers. Simulation / paper only.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from jev_omm.backtest.simulator import run_simulation
from jev_omm.config import EngineConfig, QuoterConfig, RiskConfig, SimConfig
from jev_omm.decisions.client import make_decision_client
from jev_omm.obs.logging import summarize_run
from pathlib import Path
from jev_omm.surface.sabr import SabrIVSurface


def main() -> None:
    console = Console()
    cfg = EngineConfig(
        quoter=QuoterConfig(gamma=0.12, kappa=1.5, sigma=0.45, quote_size=2),
        risk=RiskConfig(max_abs_inventory=20, max_loss=400.0),
        sim=SimConfig(n_steps=150, seed=7),
    )
    surface = SabrIVSurface(
        alpha=cfg.market.atm_iv,
        beta=1.0,
        rho=-0.3,
        nu=0.4,
        rate=cfg.market.rate,
        div_yield=cfg.market.dividend_yield,
    )
    client = make_decision_client()
    log_path = Path("jev_omm_events_py.jsonl")
    result = run_simulation(
        cfg, client=client, surface=surface, event_log_path=str(log_path)
    )

    summary = summarize_run(
        steps=cfg.sim.n_steps,
        fills=len(result.fills),
        final_inventory=result.position.qty,
        final_pnl=result.final_pnl,
        breaches=result.breach_count,
        hedges=len(result.hedges),
        decision_source=result.decision_source,
    )

    console.print("\n[bold cyan]Jev Options MM — paper demo[/bold cyan]")
    console.print(
        "Research laboratory. Simulation only. No live brokers. "
        f"fill_model={result.fill_model} (LOB is the primary backend; "
        "pass fill_model='poisson' for the touch model).\n"
    )
    console.print(
        f"Surface: {result.surface_label}  backend={surface.backend_name()}  "
        f"atm≈{surface.atm_iv(cfg.market.spot0, cfg.market.expiry_years):.4f}"
    )

    table = Table(title="Run summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in summary.items():
        table.add_row(str(k), str(v))
    console.print(table)

    at = Table(title="Markout attribution (horizons: 1 / 5 / 30 steps ≈ min)")
    at.add_column("Component")
    at.add_column("Value", justify="right")
    for k, v in result.attribution.as_rows():
        at.add_row(k, v)
    console.print(at)
    console.print(
        "[dim]markout = signed_size*(mid_h−mid0); adverse = −markout; "
        "inventory_mtm = open_qty*Δmid[/dim]"
    )

    if result.fills:
        ft = Table(title="Last fills (up to 8)")
        ft.add_column("t")
        ft.add_column("side")
        ft.add_column("px")
        ft.add_column("sz")
        ft.add_column("mid")
        for f in result.fills[-8:]:
            ft.add_row(
                f"{f.time:.6f}",
                f.side.value,
                f"{f.price:.4f}",
                str(f.size),
                f"{f.mid_at_fill:.4f}",
            )
        console.print(ft)
    else:
        console.print(
            "[yellow]No fills this seed. Intensity is events/second × dt_seconds; "
            "raise fill_intensity_per_second or lower lob_ahead.[/yellow]"
        )

    if result.adjustments:
        last = result.adjustments[-1]
        console.print(
            f"\nLast QuoteAdjustments: spread_mult={last.spread_mult:.2f} "
            f"size_mult={last.size_mult:.2f} pull={last.pull} "
            f"hedge_now={last.hedge_now} tox={last.composite_toxicity:.2f} "
            f"({last.reason})"
        )

    console.print(
        f"\nFinal inventory={result.position.qty}  cash={result.position.cash:.2f}  "
        f"marked_pnl={result.final_pnl:.2f}  "
        f"underlier={result.position.underlier_qty:.2f}  "
        f"hedge_slippage={result.position.hedge_slippage:.4f}"
    )
    if result.event_log_path:
        console.print(
            f"\nevent_log={result.event_log_path}  events={result.event_count}  "
            f"sha256={result.event_log_sha256}"
        )
        console.print(
            "[dim]Replay: from jev_omm.obs.event_log import replay_jsonl; "
            f"replay_jsonl({result.event_log_path!r})[/dim]"
        )


if __name__ == "__main__":
    main()
