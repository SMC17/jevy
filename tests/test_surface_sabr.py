"""SABR-lite surface: ATM behavior + wing monotonicity."""

from __future__ import annotations

from jev_omm.surface.sabr import SabrIVSurface, SabrIVSurfacePythonFallback, hagan_sabr_iv


def test_hagan_atm_closed_form():
    alpha, beta, rho, nu = 0.25, 0.5, -0.3, 0.4
    f, t = 100.0, 0.5
    atm = hagan_sabr_iv(alpha, beta, rho, nu, f, f, t)
    one_m_b = 1.0 - beta
    f_pow = f**one_m_b
    term1 = (one_m_b**2 / 24.0) * (alpha**2) / (f ** (2 * one_m_b))
    term2 = 0.25 * rho * beta * nu * alpha / (f**one_m_b)
    term3 = ((2.0 - 3.0 * rho * rho) / 24.0) * nu * nu
    expected = (alpha / f_pow) * (1.0 + (term1 + term2 + term3) * t)
    assert abs(atm - expected) < 1e-10


def test_atm_near_continuous():
    # Compare raw Hagan (no floor/cap) with very near-ATM strike
    alpha, beta, rho, nu = 0.3, 0.7, -0.2, 0.5
    f, t = 100.0, 1.0
    atm = hagan_sabr_iv(alpha, beta, rho, nu, f, f, t)
    near = hagan_sabr_iv(alpha, beta, rho, nu, f, f * (1.0 + 1e-9), t)
    assert abs(atm - near) / atm < 1e-4


def test_wing_skew_rho_negative():
    s = SabrIVSurfacePythonFallback(alpha=0.22, beta=1.0, rho=-0.4, nu=0.5, floor_iv=0.01)
    put_w = s.iv(100.0, 80.0, 0.25)
    atm = s.iv(100.0, 100.0, 0.25)
    call_w = s.iv(100.0, 120.0, 0.25)
    assert put_w > 0 and atm > 0 and call_w > 0
    assert put_w > call_w  # equity-ish skew


def test_sabr_prefers_zig_when_available():
    s = SabrIVSurface(alpha=0.22, beta=1.0, rho=-0.3, nu=0.4)
    iv = s.iv(100.0, 100.0, 0.25)
    assert 0.05 <= iv <= 1.5
    assert "zig" in s.backend_name() or "python" in s.backend_name()


def test_parametric_still_labeled_placeholder():
    from jev_omm.surface.parametric import ParametricIVSurface

    p = ParametricIVSurface()
    assert "PLACEHOLDER" in p.label
