"""Tests de temperature scaling (ecg/calibration.py)."""

import numpy as np

from ecg.calibration import (
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_temperature_per_class,
)


def _overconfident(seed=7, n=800, c=12):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, c))
    p_model = 1.0 / (1.0 + np.exp(-z * 2.0))
    y = (rng.random((n, c)) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    return y, p_model


def test_fit_recovers_temperature_and_improves_ece():
    y, p = _overconfident()
    temps = fit_temperature_per_class(y, p)
    assert temps.shape == (12,)
    assert 1.5 < float(temps.mean()) < 2.5
    calibrated = apply_temperature(p, temps)
    assert expected_calibration_error(y, calibrated) < expected_calibration_error(y, p)
    assert brier_score(y, calibrated) < brier_score(y, p)


def test_apply_temperature_identity_and_shape():
    y, p = _overconfident(n=100, c=4)
    same = apply_temperature(p, np.ones(4))
    np.testing.assert_allclose(same, p, rtol=1e-4, atol=1e-4)
    assert same.shape == p.shape
    assert same.dtype == np.float32


def test_ece_bounds():
    y, p = _overconfident(n=100, c=4)
    assert 0.0 <= expected_calibration_error(y, p) <= 1.0
    assert 0.0 <= brier_score(y, p) <= 1.0


def test_degenerate_class_keeps_unit_temperature():
    rng = np.random.default_rng(1)
    y = np.zeros((200, 2))
    y[:, 1] = (rng.random(200) < 0.5).astype(float)
    p = rng.random((200, 2))
    temps = fit_temperature_per_class(y, p)
    assert temps[0] == 1.0
