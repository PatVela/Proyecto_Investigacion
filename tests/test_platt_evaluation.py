"""Calibración afín en la evaluación: evaluate.py --platt y ensemble_evaluate.py (--platt-a/-b, recalibración).

Las predicciones de ``predict_hdf5`` se simulan con logits conocidos para comprobar la aritmética exacta y que ningún
parámetro se ajusta con test.
"""

import json

import numpy as np
import pandas as pd
import pytest

from ecg.calibration import (
    expected_calibration_error,
    fit_platt_per_class,
    load_platt_csv,
    save_platt_csv,
    to_logits,
)
from examples.cinc2020 import ensemble_evaluate, evaluate

CLASSES = ["NSR", "AF", "MI"]


def _fake_predict(shift_val_labels=None, n_val=3000, n_test=6000, seed=0):
    """Logits sesgados (+1.8): P(y=1|z) = sigmoid(z - 1.8). Devuelve la función que reemplaza predict_hdf5."""
    rng = np.random.default_rng(seed)
    store = {}

    def make(split, n):
        z = rng.normal(-1.0, 1.8, size=(n, 3)) + 1.8
        y = (rng.random(z.shape) < 1 / (1 + np.exp(-(z - 1.8)))).astype(np.uint8)
        store[split] = (z.astype(np.float32), y)

    make("val", n_val)
    make("test", n_test)

    def fake(checkpoint, h5_path, **kwargs):
        z, y = store["val" if "val" in str(h5_path) else "test"]
        if shift_val_labels is not None and "test" in str(h5_path):
            y = shift_val_labels(y)
        return {"labels": y, "logits": z.copy(), "probabilities": (1 / (1 + np.exp(-z.astype(np.float64)))).astype(np.float32),
                "class_names": CLASSES, "record_names": [f"R{i:05d}" for i in range(len(y))]}

    return fake, store


CONFIG = {"dev": "val.h5", "test": "test.h5", "batch_size": 8, "num_workers": 0}


def test_evaluate_platt_applies_sigmoid_a_z_plus_b_before_thresholds(tmp_path, monkeypatch):
    fake, store = _fake_predict()
    monkeypatch.setattr("ecg.predict.predict_hdf5", fake)
    save_platt_csv(tmp_path / "platt.csv", CLASSES, [0.5, 1.0, 2.0], [0.0, -1.0, 0.5])
    evaluate.run_evaluation(CONFIG, "x.pt", tmp_path / "out", device="cpu", platt=tmp_path / "platt.csv")
    got = pd.read_csv(tmp_path / "out" / "predictions_validation.csv")
    z = store["val"][0].astype(np.float64)
    expected = 1 / (1 + np.exp(-(z * np.array([0.5, 1.0, 2.0]) + np.array([0.0, -1.0, 0.5]))))
    np.testing.assert_allclose(got[[f"prob_{c}" for c in CLASSES]].to_numpy(), expected, atol=1e-6)
    summary = json.loads((tmp_path / "out" / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["platt"].endswith("platt.csv") and summary["temperatures"] is None


def test_evaluate_refuses_temperature_and_platt_together(tmp_path, monkeypatch):
    fake, _ = _fake_predict()
    monkeypatch.setattr("ecg.predict.predict_hdf5", fake)
    pd.DataFrame({"class": CLASSES, "temperature": [1.0] * 3}).to_csv(tmp_path / "t.csv", index=False)
    save_platt_csv(tmp_path / "p.csv", CLASSES, [1.0] * 3, [0.0] * 3)
    with pytest.raises(ValueError, match="temperatures.*platt|platt.*temperatures"):
        evaluate.run_evaluation(CONFIG, "x.pt", tmp_path / "out", device="cpu",
                                temperatures=tmp_path / "t.csv", platt=tmp_path / "p.csv")


def test_evaluate_without_logits_falls_back_to_the_clipped_logit_of_the_probabilities(tmp_path, monkeypatch):
    fake, store = _fake_predict()

    def no_logits(*a, **k):
        out = fake(*a, **k)
        out.pop("logits")
        return out

    monkeypatch.setattr("ecg.predict.predict_hdf5", no_logits)
    save_platt_csv(tmp_path / "p.csv", CLASSES, [1.0] * 3, [-1.0] * 3)
    evaluate.run_evaluation(CONFIG, "x.pt", tmp_path / "out", device="cpu", platt=tmp_path / "p.csv")
    got = pd.read_csv(tmp_path / "out" / "predictions_validation.csv")["prob_NSR"].to_numpy()
    z = store["val"][0][:, 0].astype(np.float64)
    np.testing.assert_allclose(got, 1 / (1 + np.exp(-(z - 1.0))), atol=1e-4)


def _ensemble(tmp_path, monkeypatch, **kwargs):
    fake, store = _fake_predict(seed=3)
    monkeypatch.setattr("ecg.predict.predict_hdf5", fake)
    save_platt_csv(tmp_path / "pa.csv", CLASSES, [1.0] * 3, [-1.8] * 3)
    save_platt_csv(tmp_path / "pb.csv", CLASSES, [1.0] * 3, [-1.8] * 3)
    summary = ensemble_evaluate.run_ensemble_evaluation(
        CONFIG, "a.pt", "b.pt", tmp_path / "ens", device="cpu", threshold_beta=1.0,
        platt_a=tmp_path / "pa.csv", platt_b=tmp_path / "pb.csv", **kwargs)
    return summary, store


def test_ensemble_platt_per_model_then_average(tmp_path, monkeypatch):
    summary, store = _ensemble(tmp_path, monkeypatch)
    got = pd.read_csv(tmp_path / "ens" / "predictions_validation.csv")[[f"prob_{c}" for c in CLASSES]].to_numpy()
    z = store["val"][0].astype(np.float64)
    np.testing.assert_allclose(got, 1 / (1 + np.exp(-(z - 1.8))), atol=1e-6)     # A y B idénticos: el promedio es igual
    assert summary["platt_a"].endswith("pa.csv") and summary["recalibrate_ensemble"] is False
    assert not (tmp_path / "ens" / "platt_ensemble_validation.csv").exists()


def test_ensemble_recalibration_after_averaging_fits_on_validation_only(tmp_path, monkeypatch):
    summary, store = _ensemble(tmp_path, monkeypatch, recalibrate_ensemble=True)
    ens_csv = tmp_path / "ens" / "platt_ensemble_validation.csv"
    assert ens_csv.exists() and summary["platt_ensemble"].endswith("platt_ensemble_validation.csv")
    a, b = load_platt_csv(ens_csv, CLASSES)
    # Los parámetros salen de validación: reproducibles a partir de las probabilidades promediadas de validación.
    z_val, y_val = store["val"]
    avg_val = 1 / (1 + np.exp(-(z_val.astype(np.float64) - 1.8)))
    a_ref, b_ref = fit_platt_per_class(y_val, to_logits(avg_val))
    np.testing.assert_allclose(a, a_ref, atol=1e-6)
    np.testing.assert_allclose(b, b_ref, atol=1e-6)
    got_test = pd.read_csv(tmp_path / "ens" / "predictions_test.csv")
    ece = expected_calibration_error(got_test[[f"true_{c}" for c in CLASSES]].to_numpy(float),
                                     got_test[[f"prob_{c}" for c in CLASSES]].to_numpy(float))
    assert ece < 0.03


def test_ensemble_parameters_do_not_depend_on_test_labels(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _ensemble(tmp_path / "a", monkeypatch, recalibrate_ensemble=True)
    # Mismas señales; en test se invierten todas las etiquetas. Los parámetros de validación no pueden cambiar.
    fake, _ = _fake_predict(shift_val_labels=lambda y: 1 - y, seed=3)
    monkeypatch.setattr("ecg.predict.predict_hdf5", fake)
    save_platt_csv(tmp_path / "b" / "pa.csv", CLASSES, [1.0] * 3, [-1.8] * 3)
    ensemble_evaluate.run_ensemble_evaluation(
        CONFIG, "a.pt", "b.pt", tmp_path / "b" / "ens", device="cpu", threshold_beta=1.0,
        platt_a=tmp_path / "b" / "pa.csv", platt_b=tmp_path / "b" / "pa.csv", recalibrate_ensemble=True)
    a1, b1 = load_platt_csv(tmp_path / "a" / "ens" / "platt_ensemble_validation.csv", CLASSES)
    a2, b2 = load_platt_csv(tmp_path / "b" / "ens" / "platt_ensemble_validation.csv", CLASSES)
    np.testing.assert_allclose(a1, a2); np.testing.assert_allclose(b1, b2)
    t1 = pd.read_csv(tmp_path / "a" / "ens" / "thresholds_validation.csv")["threshold"]
    t2 = pd.read_csv(tmp_path / "b" / "ens" / "thresholds_validation.csv")["threshold"]
    assert t1.equals(t2)
