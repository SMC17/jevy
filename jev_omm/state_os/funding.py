"""Funding capacity beyond the SOFR print.

A higher SOFR is one number. Scarcity shows up as specials versus GC, fails,
an auction tail, and the cross-currency basis (a balance-sheet rent). The
score below puts a small weight on the SOFR change and a large weight on
those four. A calm specials market with a higher SOFR is not the same state
as a blown CIP basis with SOFR unchanged.

- Duffie, *Special Repo Rates*, JF 1996, https://doi.org/10.1111/j.1540-6261.1996.tb02692.x
- Du, Tepper, Verdelhan, *Deviations from Covered Interest Rate Parity*, JF 2018,
  https://doi.org/10.1111/jofi.12620
"""

from __future__ import annotations


def scarcity_rent(
    sofr_change: float,
    special_minus_gc: float,
    fails: float,
    auction_tail: float,
    cip_basis: float,
) -> float:
    """Non-negative stress. Units are the caller's (typically percent or bp).

    ``special_minus_gc`` is the special's rate gap versus general collateral,
    signed so that a more special bond is positive (GC minus specials rate).
    ``cip_basis`` is signed; the rent is its absolute value.
    ``fails`` and ``auction_tail`` are taken as non-negative pressures.
    """
    return (
        0.05 * max(sofr_change, 0.0)
        + 1.00 * max(special_minus_gc, 0.0)
        + 0.50 * max(fails, 0.0)
        + 0.70 * max(auction_tail, 0.0)
        + 0.80 * abs(cip_basis)
    )
