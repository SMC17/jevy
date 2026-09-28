"""Risk hard-limit trip."""

from __future__ import annotations

from jev_omm.config import RiskConfig
from jev_omm.models.types import Greeks
from jev_omm.risk.limits import evaluate_risk


def test_inventory_limit_trips():
    cfg = RiskConfig(max_abs_inventory=10)
    g = Greeks(delta=0.5, gamma=0.02, vega=10.0, theta=-5.0)
    snap = evaluate_risk(inventory=11, greeks_per_contract=g, cash_pnl=0.0, cfg=cfg)
    assert snap.quoting_allowed is False
    assert snap.breach_reason is not None
    assert "inventory" in snap.breach_reason


def test_within_limits_allows_quoting():
    cfg = RiskConfig(max_abs_inventory=25)
    g = Greeks(delta=0.4, gamma=0.01, vega=8.0, theta=-2.0)
    snap = evaluate_risk(inventory=3, greeks_per_contract=g, cash_pnl=10.0, cfg=cfg)
    assert snap.quoting_allowed is True
    assert snap.breach_reason is None


def test_pnl_loss_limit_trips():
    cfg = RiskConfig(max_loss=100.0)
    g = Greeks(delta=0.1, gamma=0.01, vega=1.0, theta=-1.0)
    snap = evaluate_risk(inventory=0, greeks_per_contract=g, cash_pnl=-150.0, cfg=cfg)
    assert snap.quoting_allowed is False
    assert "cash_pnl" in (snap.breach_reason or "")


def test_unset_notional_per_strike_and_quotes_are_identity():
    cfg = RiskConfig()
    g = Greeks(delta=0.5, gamma=0.01, vega=1.0, theta=-1.0)
    snap = evaluate_risk(
        inventory=3,
        greeks_per_contract=g,
        cash_pnl=0.0,
        cfg=cfg,
        notional=1.0e9,
        per_strike_abs=10_000,
        quotes_outstanding=50,
    )
    assert snap.quoting_allowed is True
    assert snap.breach_reason is None


def test_notional_per_strike_and_quotes_trip_when_set():
    g = Greeks(delta=0.2, gamma=0.01, vega=1.0, theta=-1.0)
    n = evaluate_risk(
        1, g, 0.0, RiskConfig(max_abs_notional=1000.0), notional=2500.0
    )
    assert n.quoting_allowed is False and "notional" in (n.breach_reason or "")
    s = evaluate_risk(
        2, g, 0.0, RiskConfig(max_abs_per_strike=4), per_strike_abs=6
    )
    assert s.quoting_allowed is False and "per_strike" in (s.breach_reason or "")
    q = evaluate_risk(
        0, g, 0.0, RiskConfig(max_quotes_outstanding=2), quotes_outstanding=4
    )
    assert q.quoting_allowed is False and "quotes_outstanding" in (q.breach_reason or "")
    # Objects absent: the limit does not trip.
    absent = evaluate_risk(1, g, 0.0, RiskConfig(max_abs_notional=1.0))
    assert absent.quoting_allowed is True


def test_underlier_hedge_counts_in_net_delta():
    g = Greeks(delta=0.5, gamma=0.01, vega=1.0, theta=-1.0)
    cfg = RiskConfig(max_abs_delta=3.0)
    bare = evaluate_risk(10, g, 0.0, cfg)
    hedged = evaluate_risk(10, g, 0.0, cfg, extra_delta=-5.0)
    assert bare.quoting_allowed is False
    assert hedged.quoting_allowed is True
    assert abs(hedged.delta - 0.0) < 1e-12
