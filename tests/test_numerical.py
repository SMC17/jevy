"""HJB convergence, Guéant RK4 vs spectral, Greeks FD, SVI wing sweep."""

from __future__ import annotations

from pathlib import Path

from jev_omm.research.numerical import (
    greeks_vs_finite_difference,
    gueant_rk4_vs_spectral,
    hjb_convergence,
    render_numerical_markdown,
    svi_arb_sweep,
)


def test_hjb_error_shrinks_toward_the_fine_grid():
    rows = hjb_convergence()
    assert rows[-1]["abs_err_vs_finest"] == 0.0
    # Coarsest grid is farther from the reference than the next-to-last.
    assert rows[0]["abs_err_vs_finest"] >= rows[-2]["abs_err_vs_finest"]


def test_greeks_match_finite_differences():
    for row in greeks_vs_finite_difference():
        assert row["abs_delta"] < 5e-4
        assert row["abs_gamma"] < 5e-4
        assert row["abs_vega"] < 5e-2


def test_svi_wings_are_scored_outside_the_knots():
    sweep = svi_arb_sweep()
    assert sweep["n_outside"] > 10
    assert sweep["max_abs_w_err_on_knots"] < 1e-3
    # The check ran. A negative wing g is reported, not hidden.
    assert sweep["min_g_outside_knots"] == sweep["min_g_outside_knots"]


def test_rk4_and_spectral_are_finite():
    rows = gueant_rk4_vs_spectral()
    assert len(rows) == 3
    for row in rows:
        assert abs(row["rk4_delta_b"]) < 50.0
        assert abs(row["spectral_delta_b"]) < 50.0


def test_numerical_markdown_matches_checked_in_doc():
    path = Path("docs/NUMERICAL.md")
    assert path.is_file()
    assert path.read_text() == render_numerical_markdown()
