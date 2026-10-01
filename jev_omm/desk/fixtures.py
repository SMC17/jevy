"""Synthetic multi-underlier surface fixtures.

Eight names, distinct betas and smiles. Every row is ``synthetic_fixture=1``.
These are not OPRA prints and not a licensed tape. ``CRYPTO_BETA`` is a
fake high-vol path, not a coin.

Shapes (research labels):

- ``EQ_INDEX`` — equity index, beta 1, negative skew
- ``EQ_SINGLE`` — single name, beta 1.35, steeper skew
- ``EQ_LOWBETA`` — defensive single name, beta 0.42
- ``FX_PAIR`` — a currency pair, beta 0.15, a near-symmetric smile
- ``FX_EM`` — higher-vol EM FX smile, beta 0.55
- ``COMMO_ENERGY`` — energy smile plus a seasonality stub, beta 0.28
- ``RATES_STIR`` — short-rate vol, beta 0.06
- ``CRYPTO_BETA`` — synthetic high-vol beta 1.70
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
    season_amp: float = 0.0


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
    "EQ_LOWBETA": UnderlierSpec(
        underlier_id="EQ_LOWBETA",
        asset_class="equity_single",
        beta_to_index=0.42,
        spot=90.0,
        rate=0.04,
        div_yield=0.028,
        idio_vol=0.006,
        iv0=0.14,
        slices=_slices(rho=-0.22, b=0.045, a0=0.0028, sigma=0.16),
    ),
    "FX_EM": UnderlierSpec(
        underlier_id="FX_EM",
        asset_class="fx_em",
        beta_to_index=0.55,
        spot=18.5,
        rate=0.08,
        div_yield=0.06,
        idio_vol=0.011,
        iv0=0.18,
        slices=_slices(rho=-0.28, b=0.07, a0=0.0045, sigma=0.24),
    ),
    "COMMO_ENERGY": UnderlierSpec(
        underlier_id="COMMO_ENERGY",
        asset_class="commodity",
        beta_to_index=0.28,
        spot=75.0,
        rate=0.04,
        div_yield=0.0,
        idio_vol=0.014,
        iv0=0.32,
        slices=_slices(rho=-0.12, b=0.09, a0=0.010, sigma=0.20),
        season_amp=0.04,
    ),
    "RATES_STIR": UnderlierSpec(
        underlier_id="RATES_STIR",
        asset_class="rates",
        beta_to_index=0.06,
        spot=96.5,
        rate=0.045,
        div_yield=0.0,
        idio_vol=0.003,
        iv0=0.07,
        slices=_slices(rho=0.08, b=0.025, a0=0.0008, sigma=0.12),
    ),
    "CRYPTO_BETA": UnderlierSpec(
        underlier_id="CRYPTO_BETA",
        asset_class="crypto_synthetic",
        beta_to_index=1.70,
        spot=40.0,
        rate=0.0,
        div_yield=0.0,
        idio_vol=0.028,
        iv0=0.62,
        slices=_slices(rho=-0.30, b=0.11, a0=0.020, sigma=0.28),
    ),
}


DEFAULT_PRODUCTS: tuple[str, ...] = (
    "EQ_INDEX",
    "EQ_SINGLE",
    "EQ_LOWBETA",
    "FX_PAIR",
    "FX_EM",
    "COMMO_ENERGY",
    "RATES_STIR",
    "CRYPTO_BETA",
)


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
