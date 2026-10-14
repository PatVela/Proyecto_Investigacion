"""Registros largos: la webapp promedia las ventanas YA calibradas, no toma el máximo de las crudas.

Regresión: los umbrales y la calibración se ajustaron con una ventana centrada. El máximo de hasta 4 ventanas
correlacionadas queda sesgado hacia arriba: en 1019 registros largos de test añadía 85 falsos positivos (426 -> 511)
y el MI de Q1033 pasaba de 30,68 % a 33,04 % (umbral 33 %) solo por eso.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("flask")

from ecg import load, network
from ecg.calibration import apply_platt
from webapp import prediction as pred

NAMES = load.CLASS_NAMES


def _checkpoint(path, seed):
    torch.manual_seed(seed)
    model = network.build_network(num_leads=12, num_classes=12)
    torch.save({"model_state_dict": model.state_dict(),
                "config": {"label_schema": load.LABEL_SCHEMA, "norm_mode": "physical", "bandpass": True},
                "class_names": NAMES.copy(), "label_schema": load.LABEL_SCHEMA, "epoch": 1, "val_loss": 0.5}, str(path))
    return str(path)


def _csv(path, seconds):
    signal = np.random.default_rng(11).normal(0, 0.3, (load.TARGET_FS * seconds, load.NUM_LEADS))
    np.savetxt(str(path), signal, delimiter=",", header=",".join(load.STANDARD_LEAD_ORDER), comments="")
    return path


def _platt(a, b):
    return {name: {"a": a + 0.01 * i, "b": b - 0.01 * i} for i, name in enumerate(NAMES)}


def _vec(platt):
    return np.array([platt[n]["a"] for n in NAMES]), np.array([platt[n]["b"] for n in NAMES])


def _ordered(result, key):
    rows = sorted(result["predictions"], key=lambda r: NAMES.index(r["class"]))
    return np.array([r[key] for r in rows])


def _fake_inference(monkeypatch, *per_model):
    """La red aleatoria de los tests da casi la misma salida en todas las ventanas (dif. ~1e-5): no distingue media de
    máximo. Se sustituye la inferencia por probabilidades crudas con mucha dispersión, una matriz (K, 12) por modelo."""
    calls = iter(per_model)
    monkeypatch.setattr(pred, "_infer_windows_unlocked", lambda model, windows, device: next(calls))


def _raw(seed, k=3):
    return np.random.default_rng(seed).uniform(0.02, 0.98, (k, 12)).astype(np.float32)


def test_one_window_is_returned_unchanged():
    vector = np.linspace(0.05, 0.9, 12)
    probabilities, how = pred.aggregate_window_probabilities(vector[None, :])
    np.testing.assert_array_equal(probabilities, vector)
    assert how == "single"


def test_several_windows_are_averaged_not_maxed():
    per_window = np.array([[0.10, 0.80], [0.30, 0.20], [0.50, 0.50]])
    probabilities, how = pred.aggregate_window_probabilities(per_window)
    np.testing.assert_allclose(probabilities, [0.30, 0.50], atol=1e-7)
    assert how == "mean" and probabilities.dtype == np.float32
    assert (probabilities < per_window.max(axis=0)).all()                  # el máximo daría [0.50, 0.80]


@pytest.mark.parametrize("bad", [np.zeros(12), np.zeros((0, 12)), np.zeros((2, 3, 4))], ids=["1d", "sin_ventanas", "3d"])
def test_aggregate_rejects_malformed_input(bad):
    with pytest.raises(ValueError):
        pred.aggregate_window_probabilities(bad)


def test_long_record_final_probability_is_the_mean_of_the_fully_calibrated_windows(tmp_path, monkeypatch):
    ckpt_a, ckpt_b = _checkpoint(tmp_path / "a.pt", 1), _checkpoint(tmp_path / "b.pt", 2)
    csv = _csv(tmp_path / "largo.csv", seconds=30)
    pa, pb, pe = _platt(0.7, -1.5), _platt(0.8, -1.1), _platt(1.4, -0.7)
    raw_a, raw_b = _raw(1), _raw(2)
    _fake_inference(monkeypatch, raw_a, raw_b)
    run = pred.run_prediction_from_saved_paths(
        [csv], ckpt_a, None, tmp_path / "r", model_path_b=ckpt_b, alpha=0.5, platt=pa, platt_b=pb, platt_ensemble=pe)

    per_window = apply_platt(0.5 * apply_platt(raw_a, *_vec(pa)) + 0.5 * apply_platt(raw_b, *_vec(pb)), *_vec(pe))
    assert run["num_windows"] == 3 and run["window_aggregation"] == "mean"
    np.testing.assert_allclose(run["per_window_probabilities"], per_window, atol=1e-5)
    np.testing.assert_allclose(_ordered(run, "probability"), per_window.mean(axis=0), atol=1e-5)
    assert np.abs(per_window.max(axis=0) - per_window.mean(axis=0)).max() > 0.1          # el caso separa media de máximo


def test_thresholds_are_applied_to_the_averaged_probability(tmp_path, monkeypatch):
    ckpt = _checkpoint(tmp_path / "a.pt", 3)
    csv = _csv(tmp_path / "largo.csv", seconds=30)
    raw = _raw(5)
    mean_probability, max_probability = raw.mean(axis=0), raw.max(axis=0)
    thresholds = {name: float(mean_probability[i]) + (0.001 if i % 2 else -0.001) for i, name in enumerate(NAMES)}
    threshold_vector = np.array([thresholds[n] for n in NAMES])
    assert ((max_probability >= threshold_vector) != (mean_probability >= threshold_vector)).any()   # el máximo decidiría distinto
    _fake_inference(monkeypatch, raw)
    run = pred.run_prediction_from_saved_paths([csv], ckpt, None, tmp_path / "r", thresholds=thresholds,
                                               normal_fallback_min_prob=None)
    np.testing.assert_array_equal(_ordered(run, "prediction"), (mean_probability >= threshold_vector).astype(int))


def test_ten_second_record_is_unchanged_by_the_policy(tmp_path):
    ckpt = _checkpoint(tmp_path / "a.pt", 4)
    csv = _csv(tmp_path / "corto.csv", seconds=10)
    run = pred.run_prediction_from_saved_paths([csv], ckpt, None, tmp_path / "r", platt=_platt(0.8, -0.4))
    assert run["num_windows"] == 1 and run["window_aggregation"] == "single"
    np.testing.assert_allclose(_ordered(run, "probability"), np.asarray(run["per_window_probabilities"])[0], atol=1e-7)
