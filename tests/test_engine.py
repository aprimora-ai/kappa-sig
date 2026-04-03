"""
Basic smoke tests for kappa_fin.engine
"""
import numpy as np
import pytest

def test_imports():
    from kappa_fin.engine import (
        Config, corr_from_returns, dist_from_corr,
        compute_ph_entropy_and_dominance, compute_v_raw,
        compute_phi_series, persist_crossing
    )
    assert True

def test_corr_from_returns_spearman():
    from kappa_fin.engine import corr_from_returns
    np.random.seed(42)
    R = np.random.randn(22, 5)
    C = corr_from_returns(R, "spearman")
    assert C.shape == (5, 5)
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-10)

def test_dist_from_corr():
    from kappa_fin.engine import dist_from_corr
    C = np.eye(4)
    D = dist_from_corr(C, "corr", 1.0)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)

def test_compute_v_raw():
    from kappa_fin.engine import compute_v_raw
    v = compute_v_raw(1.5, 0.3, "entropy_x_anti_dom")
    assert abs(v - 1.5 * 0.7) < 1e-10

def test_persist_crossing_found():
    from kappa_fin.engine import persist_crossing
    x = np.array([0.5, 0.5, 1.5, 1.5, 1.5, 0.5])
    idx = persist_crossing(x, 1.0, 3, 0)
    assert idx == 2

def test_persist_crossing_not_found():
    from kappa_fin.engine import persist_crossing
    x = np.array([0.5, 1.5, 0.5, 1.5, 0.5])
    idx = persist_crossing(x, 1.0, 3, 0)
    assert idx is None

def test_phi_series_accumulates():
    from kappa_fin.engine import compute_phi_series
    # Need enough calm samples so quantile is meaningful
    calm_oh = np.ones(20) * 0.5
    crisis_oh = np.ones(10) * 3.0
    Oh = np.concatenate([calm_oh, crisis_oh])
    calm_mask = np.array([True] * 20 + [False] * 10)
    phi, Oh_pre = compute_phi_series(Oh, calm_mask, delta=0.1, gamma=0.97, pre_q=0.95, phi_floor=1e-6)
    assert len(phi) == 30
    assert phi[29] > phi[19]  # damage accumulated during crisis vs calm
