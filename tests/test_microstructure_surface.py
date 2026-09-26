"""Hawkes bursts, Dupire local vol, variance-swap weights, multi-level queue value."""

from __future__ import annotations

import math

from jev_omm.decisions.client import DeterministicFallbackClient
from jev_omm.decisions.schemas import build_mm_state
from jev_omm.execution.lob import depth_ahead, queue_value
from jev_omm.flow.hawkes import HawkesParams, branching_ratio, excitation, fill_intensity, intensity
from jev_omm.pricing.varswap import (
    fair_variance,
    fair_variance_flat_black,
    variance_swap_vega,
    vol_swap_from_variance,
)
from jev_omm.surface.dupire import dupire_local_variance, local_variance_from_total
from jev_omm.surface.rough_vol import fbm_covariance, holder_scale, rough_bergomi_variance


def test_hawkes_burst_then_decay():
    p = HawkesParams(mu=1.0, alpha=0.8, beta=2.0)
    ev = [0.0, 0.1]
    assert intensity(p, 0.1, ev) > p.mu + p.alpha
    assert intensity(p, 3.0, ev) < intensity(p, 0.1, ev)
    assert excitation(p, 0.1, ev) > excitation(p, 3.0, ev)
    assert abs(branching_ratio(p) - 0.4) < 1e-12
    assert abs(fill_intensity(4.0, 1.5) - 10.0) < 1e-12


def test_hawkes_excitation_raises_fallback_toxicity():
    calm = build_mm_state(
        time=0.0, spot=100.0, option_mid=2.0, iv=0.2, inventory=0,
        delta=0.5, gamma=0.01, vega=10.0, cash_pnl=0.0, half_spread=0.1, quoting_allowed=True,
    )
    hot = build_mm_state(
        time=0.0, spot=100.0, option_mid=2.0, iv=0.2, inventory=0,
        delta=0.5, gamma=0.01, vega=10.0, cash_pnl=0.0, half_spread=0.1, quoting_allowed=True,
        toxicity_features={"hawkes_excitation": 2.5, "hawkes_intensity": 8.0},
    )
    client = DeterministicFallbackClient()
    a = client.system_one(calm)
    b = client.system_one(hot)
    assert b.answers["toxicity"].score >= a.answers["toxicity"].score
    assert b.answers["informed_flow"].noul > a.answers["informed_flow"].noul
    assert hot["flow"]["hawkes_excitation"] == 2.5


def test_flat_total_variance_is_flat_local_vol():
    # w = σ² T, derivatives in k vanish, ∂_T w = σ², g = 1.
    sig2 = 0.04
    lv = local_variance_from_total(0.3, sig2 * 0.5, 0.0, 0.0, sig2)
    assert abs(lv - sig2) < 1e-12


def test_dupire_on_flat_black_recovers_iv():
    iv = 0.22
    lv = dupire_local_variance(100.0, 100.0, 0.5, iv, d_k=1.0, d_t=1.0 / 365.0)
    assert abs(math.sqrt(lv) - iv) < 5e-3


def test_fbm_variance_and_rough_bergomi_positive():
    import numpy as np

    times = np.array([0.25, 0.5, 1.0])
    cov = fbm_covariance(times, 0.1)
    for i, t in enumerate(times):
        assert abs(cov[i, i] - t ** (0.2)) < 1e-12
    t, var = rough_bergomi_variance(8, 0.1, 0.4, 0.04, seed=1)
    assert var.shape == (8,)
    assert np.all(var > 0.0)
    assert holder_scale(0.01, 0.1) > holder_scale(0.01, 0.5)


def test_variance_swap_weights_and_flat_replication():
    k = [80.0, 100.0, 120.0]
    otm = [80.0**2, 100.0**2, 120.0**2]
    assert abs(fair_variance(k, otm, 1.0, 1.0) - 80.0) < 1e-9
    assert abs(variance_swap_vega(0.2) - 0.4) < 1e-15
    shaved = vol_swap_from_variance(0.04, 0.000032)
    assert abs(shaved - 0.2 * 0.9975) < 1e-12
    k_var = fair_variance_flat_black(100.0, 1.0, 0.2, n=61)
    # Discrete wings miss some mass; stay in a band around 0.04.
    assert 0.025 < k_var < 0.055


def test_queue_value_falls_with_depth_when_the_spread_pays():
    shallow = queue_value(0.10, 0.02, 1.0, 5.0, 20.0, 0.0, 1.0, None)
    deep = queue_value(0.10, 0.02, 30.0, 5.0, 20.0, 0.0, 1.0, None)
    assert shallow > deep > 0.0 or (shallow > 0.0 and shallow > deep)
    toxic = queue_value(0.02, 0.20, 1.0, 5.0, 20.0, 0.0, 1.0, None)
    toxic_deep = queue_value(0.02, 0.20, 30.0, 5.0, 20.0, 0.0, 1.0, None)
    assert toxic < 0.0 and toxic_deep > toxic
    levels = [(1.00, 4.0), (0.99, 6.0), (0.98, 8.0)]
    assert abs(depth_ahead(levels, 2) - 10.0) < 1e-12
    assert depth_ahead(levels, 0) == 0.0
