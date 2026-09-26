"""SVI / SSVI: no-arb smiles pass, injected butterfly arb fails, sticky regimes."""

from __future__ import annotations

import math

from jev_omm.surface.svi import (
    SsviParams,
    SviParams,
    StickyRegime,
    assess,
    butterfly_check,
    calibrate,
    implied_vol,
    implied_vol_after_move,
    raw_calendar_ok,
    ssvi_calendar_ok,
    ssvi_params_calendar_safe,
    ssvi_total_var,
    total_var,
)


def test_no_arb_smile_passes():
    p = SviParams(0.04, 0.1, -0.4, 0.0, 0.2)
    rep = butterfly_check(p)
    assert rep.ok and rep.min_g > 0.0 and rep.lee_ok
    chk = assess(p, calendar_ok=True)
    assert chk["surface_suspect"] is False


def test_injected_butterfly_arb_fails():
    p = SviParams(0.04, 3.0, 0.9, 0.0, 0.2)
    rep = butterfly_check(p)
    assert not rep.ok
    assert not rep.lee_ok
    assert rep.min_g < 0.0
    assert assess(p)["surface_suspect"] is True


def test_ssvi_calendar_and_atm():
    p = SsviParams(-0.3, 1.0, 0.4)
    assert ssvi_params_calendar_safe(p)
    assert abs(ssvi_total_var(0.0, 0.04, p) - 0.04) < 1e-12
    assert ssvi_calendar_ok(0.04, 0.09, p)
    near = SviParams(0.02, 0.05, -0.2, 0.0, 0.15)
    far = SviParams(0.05, 0.05, -0.2, 0.0, 0.15)
    assert raw_calendar_ok(near, far)
    assert not raw_calendar_ok(far, near)
    assert assess(near, calendar_ok=False)["surface_suspect"] is True


def test_calibration_recovers_planted_smile():
    planted = SviParams(0.04, 0.15, -0.3, 0.02, 0.25)
    ks = [-0.8 + 1.6 * i / 16 for i in range(17)]
    ws = [total_var(planted, k) for k in ks]
    fit = calibrate(ks, ws)
    assert fit.rmse < 1e-4
    assert fit.butterfly.ok
    assert not fit.surface_suspect
    assert abs(fit.params.a - planted.a) < 0.01
    assert abs(fit.params.b - planted.b) < 0.02


def test_sticky_strike_vs_sticky_delta():
    p = SviParams(0.04, 0.15, -0.4, 0.0, 0.2)
    f0, f1, strike, t = 100.0, 110.0, 100.0, 0.25
    iv0 = implied_vol(p, math.log(strike / f0), t)
    sticky_k = implied_vol_after_move(p, f0, f1, strike, t, StickyRegime.STICKY_STRIKE)
    sticky_d = implied_vol_after_move(p, f0, f1, strike, t, "sticky_delta")
    assert abs(sticky_k - iv0) < 1e-12
    assert abs(sticky_d - implied_vol(p, math.log(strike / f1), t)) < 1e-12
    assert abs(sticky_d - sticky_k) > 1e-4
    strike_scaled = f1 * math.exp(math.log(strike / f0))
    iv_scaled = implied_vol_after_move(p, f0, f1, strike_scaled, t, StickyRegime.STICKY_DELTA)
    assert abs(iv_scaled - iv0) < 1e-12
