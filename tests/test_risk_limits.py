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
