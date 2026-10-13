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


# --- calibración afín (Platt) por clase: a·logit + b ---------------------------------------------------------------

from ecg.calibration import (  # noqa: E402
    apply_platt,
    apply_platt_logits,
    fit_platt_per_class,
    load_platt_csv,
    save_platt_csv,
    to_logits,
)


def _biased_logits(n=40000, a_true=(0.6, 1.0, 1.4), b_true=(-1.0, 0.0, 0.7), seed=3):
    """Logits crudos z y etiquetas con P(y=1|z) = sigmoid(a_true*z + b_true): se conoce la calibración exacta."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, size=(n, len(a_true)))
    p = 1.0 / (1.0 + np.exp(-(z * np.asarray(a_true) + np.asarray(b_true))))
    return z, (rng.random(z.shape) < p).astype(float)


def test_platt_recovers_the_true_slope_and_bias_per_class():
    z, y = _biased_logits()
    a, b = fit_platt_per_class(y, z)
    np.testing.assert_allclose(a, [0.6, 1.0, 1.4], atol=0.05)
    np.testing.assert_allclose(b, [-1.0, 0.0, 0.7], atol=0.08)


def test_apply_platt_logits_known_values_and_dtype():
    out = apply_platt_logits(np.array([[0.0, 2.0]]), np.array([0.5, 0.5]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(out, [[1 / (1 + np.exp(-1.0)), 1 / (1 + np.exp(-2.0))]], rtol=1e-12)
    assert out.dtype == np.float64


def test_apply_platt_on_probabilities_goes_through_the_clipped_logit():
    p = np.array([[0.2, 1.0]], dtype=np.float32)               # 1.0 exacto: lo que deja el sigmoide en fp16/fp32
    z = to_logits(p)
    assert np.all(np.isfinite(z))
    np.testing.assert_allclose(apply_platt(p, np.array([1.0, 1.0]), np.array([0.0, 0.0])),
                               apply_platt_logits(z, np.array([1.0, 1.0]), np.array([0.0, 0.0])))
    assert apply_platt(p, np.array([1.0, 1.0]), np.array([0.0, 0.0]))[0, 1] < 1.0


def test_platt_fixes_a_prior_shift_that_temperature_scaling_cannot():
    """pos_weight=neg/pos sube la probabilidad base de todos los registros: solo el término b lo corrige."""
    rng = np.random.default_rng(11)
    z_true = rng.normal(-2.0, 1.5, size=(30000, 4))            # prevalencia real baja (~15 %)
    y = (rng.random(z_true.shape) < 1 / (1 + np.exp(-z_true))).astype(float)
    z_model = z_true + 2.3                                     # modelo con el logit desplazado
    p_model = 1 / (1 + np.exp(-z_model))
    temps = fit_temperature_per_class(y, p_model)
    ece_temp = expected_calibration_error(y, apply_temperature(p_model, temps))
    a, b = fit_platt_per_class(y, z_model)
    ece_platt = expected_calibration_error(y, apply_platt_logits(z_model, a, b))
    assert expected_calibration_error(y, p_model) > 0.2
    assert ece_platt < 0.01 < ece_temp and ece_platt < ece_temp / 5
    np.testing.assert_allclose(b, -2.3, atol=0.15)


def test_platt_never_inverts_the_ranking_and_keeps_the_slope_positive():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(500, 1))
    y = (rng.random((500, 1)) < 1 / (1 + np.exp(z * 3))).astype(float)    # puntuaciones anticorrelacionadas
    a, b = fit_platt_per_class(y, z)
    assert a[0] > 0
    order_in = np.argsort(z[:, 0], kind="stable")
    out = apply_platt_logits(z, a, b)[:, 0]
    assert np.all(np.diff(out[order_in]) >= 0)


def test_platt_degenerate_class_is_the_identity():
    z = np.random.default_rng(0).normal(size=(100, 2))
    y = np.zeros((100, 2))
    y[:, 1] = (np.arange(100) % 2).astype(float)
    a, b = fit_platt_per_class(y, z)
    assert (a[0], b[0]) == (1.0, 0.0)


def test_platt_csv_round_trip_and_validation(tmp_path):
    path = tmp_path / "platt.csv"
    save_platt_csv(path, ["A", "B"], np.array([0.9, 1.1]), np.array([-0.5, 0.25]))
    a, b = load_platt_csv(path, ["B", "A"])
    np.testing.assert_allclose(a, [1.1, 0.9]); np.testing.assert_allclose(b, [0.25, -0.5])
    import pytest

    with pytest.raises(ValueError, match="faltantes"):
        load_platt_csv(path, ["A", "C"])
    (tmp_path / "bad.csv").write_text("class,temperature\nA,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="class,a,b"):
        load_platt_csv(tmp_path / "bad.csv", ["A"])
