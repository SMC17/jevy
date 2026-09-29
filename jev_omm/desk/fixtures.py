"""Synthetic multi-underlier surface fixtures.

Three names, three smiles, three betas to the equity index. Every row is
``synthetic_fixture=1``. These are not OPRA prints and not a licensed tape.

Shapes (research labels):

- ``EQ_INDEX`` — equity index, beta 1, negative skew
- ``EQ_SINGLE`` — single name, beta 1.35, steeper skew and a higher wing
- ``FX_PAIR`` — a currency pair, beta 0.15, a near-symmetric smile
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from jev_omm.surface.svi import SviParams, implied_vol

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "fixtures" / "surfaces_synthetic.csv"
)

# Log-moneyness grid used by the fixture and the desk fitter.
LOG_MONEYNESS = (-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40)

EXPIRIES = (30.0 / 365.0, 90.0 / 365.0, 180.0 / 365.0)


@dataclass(frozen=True)
class UnderlierSpec:
    underlier_id: str
    asset_class: str
    beta_to_index: float
    spot: float
    rate: float
    div_yield: float
    idio_vol: float
    iv0: float
    slices: dict[float, SviParams]
    synthetic_fixture: int = 1


def _slices(rho: float, b: float, a0: float, sigma: float) -> dict[float, SviParams]:
    """Increasing total-variance level across the three fixture expiries."""
    out: dict[float, SviParams] = {}
    for i, t in enumerate(EXPIRIES):
        out[t] = SviParams(
            a=a0 * (1.0 + i),
            b=b * (1.0 + 0.04 * i),
            rho=rho,
            m=0.0,
            sigma=sigma,
        )
    return out


UNDERLIERS: dict[str, UnderlierSpec] = {
    "EQ_INDEX": UnderlierSpec(
        underlier_id="EQ_INDEX",
        asset_class="equity_index",
        beta_to_index=1.0,
        spot=5000.0,
        rate=0.04,
        div_yield=0.015,
        idio_vol=0.004,
        iv0=0.18,
        slices=_slices(rho=-0.45, b=0.06, a0=0.004, sigma=0.20),
    ),
    "EQ_SINGLE": UnderlierSpec(
        underlier_id="EQ_SINGLE",
        asset_class="equity_single",
        beta_to_index=1.35,
        spot=180.0,
        rate=0.04,
        div_yield=0.0,
        idio_vol=0.012,
        iv0=0.28,
        slices=_slices(rho=-0.62, b=0.08, a0=0.008, sigma=0.22),
    ),
    "FX_PAIR": UnderlierSpec(
        underlier_id="FX_PAIR",
        asset_class="fx_pair",
        beta_to_index=0.15,
        spot=1.10,
        rate=0.03,
        div_yield=0.02,
        idio_vol=0.005,
        iv0=0.09,
        slices=_slices(rho=-0.05, b=0.04, a0=0.0015, sigma=0.16),
    ),
}


def forward(spec: UnderlierSpec, expiry_years: float) -> float:
    return spec.spot * math.exp((spec.rate - spec.div_yield) * expiry_years)


def fixture_rows(specs: dict[str, UnderlierSpec] | None = None) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    book = specs or UNDERLIERS
    for spec in book.values():
        for expiry, params in spec.slices.items():
            fwd = forward(spec, expiry)
            for k in LOG_MONEYNESS:
                iv = implied_vol(params, k, expiry)
                rows.append(
                    {
                        "synthetic_fixture": spec.synthetic_fixture,
                        "underlier_id": spec.underlier_id,
                        "asset_class": spec.asset_class,
                        "beta_to_index": spec.beta_to_index,
                        "spot": spec.spot,
                        "rate": spec.rate,
                        "div_yield": spec.div_yield,
                        "expiry_years": expiry,
                        "log_moneyness": k,
                        "strike": fwd * math.exp(k),
                        "forward": fwd,
                        "iv_mkt": iv,
                    }
                )
    return rows


_COLUMNS = (
    "synthetic_fixture",
    "underlier_id",
    "asset_class",
    "beta_to_index",
    "spot",
    "rate",
    "div_yield",
    "expiry_years",
    "log_moneyness",
    "strike",
    "forward",
    "iv_mkt",
)


def render_csv(rows: list[dict[str, float | str | int]] | None = None) -> str:
    body = [",".join(_COLUMNS)]
    for row in rows if rows is not None else fixture_rows():
        body.append(",".join(str(row[c]) for c in _COLUMNS))
    return "\n".join(body) + "\n"


def load_csv(path: Path | None = None) -> list[dict[str, str]]:
    text = (path or FIXTURE_PATH).read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0].split(",")
    out = []
    for ln in lines[1:]:
        parts = ln.split(",")
        out.append(dict(zip(header, parts)))
    return out
