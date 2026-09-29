"""Multi-product surface book.

Each underlier holds a map of expiry → raw SVI fit, with the existing
butterfly and calendar gates. A failed calendar check damps the marked total
variance up to the previous expiry instead of quoting through the arb.

Spot is an input. A missing spot stays missing: no forward is invented and
no GEX conversion is attempted from this book.

Quotes in this module are synthetic unless the caller says otherwise.
``synthetic_fixture=1`` is the label used by the desk fixtures.

Cite Gatheral & Jacquier, https://arxiv.org/abs/1204.0646
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from jev_omm.surface.svi import (
    FitResult,
    StickyRegime,
    SviParams,
    assess,
    calibrate,
    implied_vol,
    implied_vol_after_move,
    iv_from_total_var,
    raw_calendar_ok,
    total_var,
)


@dataclass
class QuotePoint:
    log_moneyness: float
    strike: float
    forward: float
    iv_mkt: float
    total_var_mkt: float


@dataclass
class ResidualPoint:
    log_moneyness: float
    strike: float
    iv_mkt: float
    iv_fit: float
    residual: float
    damped: bool


@dataclass
class SliceFit:
    underlier_id: str
    expiry_years: float
    forward: float | None
    params: SviParams
    fit: FitResult | None
    butterfly_ok: bool
    calendar_ok: bool
    damped: bool
    surface_suspect: bool
    rmse: float
    residuals: list[ResidualPoint] = field(default_factory=list)
    synthetic_fixture: int = 1


@dataclass
class UnderlierSurface:
    underlier_id: str
    asset_class: str
    beta_to_index: float
    spot: float | None
    rate: float
    div_yield: float
    slices: dict[float, SliceFit] = field(default_factory=dict)
    synthetic_fixture: int = 1

    @property
    def spot_missing(self) -> bool:
        return self.spot is None or not (self.spot > 0.0)


@dataclass
class SurfaceBook:
    """Registry of per-underlier SVI slices. Paper marks only."""

    underliers: dict[str, UnderlierSurface] = field(default_factory=dict)
    synthetic_fixture: int = 1

    def add_underlier(
        self,
        underlier_id: str,
        *,
        asset_class: str,
        beta_to_index: float,
        spot: float | None,
        rate: float,
        div_yield: float,
        synthetic_fixture: int = 1,
    ) -> UnderlierSurface:
        row = UnderlierSurface(
            underlier_id=underlier_id,
            asset_class=asset_class,
            beta_to_index=beta_to_index,
            spot=spot if spot is not None and spot > 0.0 else None,
            rate=rate,
            div_yield=div_yield,
            synthetic_fixture=synthetic_fixture,
        )
        self.underliers[underlier_id] = row
        return row

    def forward(self, underlier_id: str, expiry_years: float) -> float | None:
        u = self.underliers.get(underlier_id)
        if u is None or u.spot_missing or not (expiry_years > 0.0):
            return None
        assert u.spot is not None
        return u.spot * math.exp((u.rate - u.div_yield) * expiry_years)

    def fit_quotes(
        self,
        underlier_id: str,
        expiry_years: float,
        points: list[QuotePoint],
        *,
        earlier: SliceFit | None = None,
    ) -> SliceFit:
        u = self.underliers[underlier_id]
        if u.spot_missing or not points:
            sl = SliceFit(
                underlier_id=underlier_id,
                expiry_years=expiry_years,
                forward=None,
                params=SviParams(),
                fit=None,
                butterfly_ok=False,
                calendar_ok=True,
                damped=False,
                surface_suspect=False,
                rmse=0.0,
                residuals=[],
                synthetic_fixture=u.synthetic_fixture,
            )
            u.slices[expiry_years] = sl
            return sl
        ks = [p.log_moneyness for p in points]
        ws = [p.total_var_mkt for p in points]
        fit = calibrate(ks, ws)
        calendar_ok = True
        if earlier is not None:
            calendar_ok = raw_calendar_ok(earlier.params, fit.params)
        report = assess(fit.params, ks, ws, calendar_ok=calendar_ok)
        damped = not calendar_ok
        residuals: list[ResidualPoint] = []
        for p in points:
            w_fit, was_damped = _marked_total_var(fit.params, earlier, p.log_moneyness, damped)
            iv_fit = iv_from_total_var(w_fit, expiry_years)
            residuals.append(
                ResidualPoint(
                    log_moneyness=p.log_moneyness,
                    strike=p.strike,
                    iv_mkt=p.iv_mkt,
                    iv_fit=iv_fit,
                    residual=p.iv_mkt - iv_fit,
                    damped=was_damped,
                )
            )
        sl = SliceFit(
            underlier_id=underlier_id,
            expiry_years=expiry_years,
            forward=points[0].forward,
            params=fit.params,
            fit=fit,
            butterfly_ok=bool(report["butterfly_ok"]),
            calendar_ok=calendar_ok,
            damped=damped,
            surface_suspect=bool(report["surface_suspect"]),
            rmse=float(report["rmse"]),
            residuals=residuals,
            synthetic_fixture=u.synthetic_fixture,
        )
        u.slices[expiry_years] = sl
        return sl

    def fit_expiries(
        self,
        underlier_id: str,
        quotes_by_expiry: dict[float, list[QuotePoint]],
    ) -> list[SliceFit]:
        """Fit short expiry first so the calendar gate has a predecessor."""
        out: list[SliceFit] = []
        earlier: SliceFit | None = None
        for expiry in sorted(quotes_by_expiry):
            sl = self.fit_quotes(
                underlier_id, expiry, quotes_by_expiry[expiry], earlier=earlier
            )
            out.append(sl)
            earlier = sl
        return out

    def mark_iv(
        self,
        underlier_id: str,
        expiry_years: float,
        strike: float,
        forward_now: float,
        regime: StickyRegime | str = StickyRegime.STICKY_STRIKE,
    ) -> float | None:
        u = self.underliers.get(underlier_id)
        if u is None:
            return None
        sl = u.slices.get(expiry_years)
        if sl is None or sl.forward is None or not (forward_now > 0.0) or not (strike > 0.0):
            return None
        if sl.damped:
            k = math.log(strike / forward_now) if StickyRegime(regime) is StickyRegime.STICKY_DELTA else math.log(strike / sl.forward)
            earlier = _previous_slice(u, expiry_years)
            w, _ = _marked_total_var(sl.params, earlier, k, True)
            return iv_from_total_var(w, expiry_years)
        return implied_vol_after_move(
            sl.params, sl.forward, forward_now, strike, expiry_years, regime
        )

    def wing_residual(self, underlier_id: str, expiry_years: float, wing: float = 0.30) -> float:
        """Mean signed wing residual (call wing minus put wing) / 2, in vol points."""
        sl = self._slice(underlier_id, expiry_years)
        if sl is None or not sl.residuals:
            return 0.0
        call = [p.residual for p in sl.residuals if p.log_moneyness >= wing]
        put = [p.residual for p in sl.residuals if p.log_moneyness <= -wing]
        if not call or not put:
            return 0.0
        return 0.5 * (float(np.mean(call)) - float(np.mean(put)))

    def fly_residual(self, underlier_id: str, expiry_years: float, wing: float = 0.30) -> float:
        """Wing average residual minus the ATM residual. Curvature vs the fit."""
        sl = self._slice(underlier_id, expiry_years)
        if sl is None or not sl.residuals:
            return 0.0
        wings = [p.residual for p in sl.residuals if abs(p.log_moneyness) >= wing]
        atm = [p.residual for p in sl.residuals if abs(p.log_moneyness) <= 0.05]
        if not wings or not atm:
            return 0.0
        return float(np.mean(wings) - np.mean(atm))

    def _slice(self, underlier_id: str, expiry_years: float) -> SliceFit | None:
        u = self.underliers.get(underlier_id)
        if u is None:
            return None
        return u.slices.get(expiry_years)


def _previous_slice(u: UnderlierSurface, expiry_years: float) -> SliceFit | None:
    earlier = [t for t in u.slices if t < expiry_years - 1e-12]
    if not earlier:
        return None
    return u.slices[max(earlier)]


def _marked_total_var(
    params: SviParams,
    earlier: SliceFit | None,
    k: float,
    damp: bool,
) -> tuple[float, bool]:
    w = total_var(params, k)
    if not damp or earlier is None:
        return w, False
    w_prev = total_var(earlier.params, k)
    if w + 1e-9 < w_prev:
        return w_prev, True
    return w, False


def quotes_from_svi(
    params: SviParams,
    *,
    forward: float,
    expiry_years: float,
    ks: list[float],
    skew: float = 0.0,
    fly: float = 0.0,
    iv_noise: float = 0.0,
) -> list[QuotePoint]:
    """Build labeled synthetic mids. ``skew`` and ``fly`` are vol-point bumps."""
    if not (forward > 0.0) or not (expiry_years > 0.0):
        return []
    out: list[QuotePoint] = []
    for k in ks:
        iv = implied_vol(params, k, expiry_years) + skew * k + fly * k * k + iv_noise
        iv = max(iv, 1e-4)
        out.append(
            QuotePoint(
                log_moneyness=k,
                strike=forward * math.exp(k),
                forward=forward,
                iv_mkt=iv,
                total_var_mkt=iv * iv * expiry_years,
            )
        )
    return out
