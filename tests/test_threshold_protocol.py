"""Protocolo de umbrales: se eligen solo con validación, la malla es la documentada y el empate se resuelve de forma fija.

Cubre las brechas P2/P3 de la auditoría: ningún test fijaba valores de ``optimize_thresholds`` (cambiar ``>`` por ``>=`` o no
pasar ``beta`` dejaba la suite en verde) ni comprobaba que ``evaluate.run_evaluation`` no mira las etiquetas de test.
"""

import numpy as np
import pandas as pd
import pytest

from examples.cinc2020 import evaluate

CLASSES = ["NSR", "AF", "MI"]


def test_best_threshold_finds_the_optimum_on_the_documented_grid():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
    p = np.array([0.10, 0.20, 0.30, 0.55, 0.45, 0.60, 0.80, 0.90])
    threshold, score = evaluate.best_threshold_fbeta(y, p, beta=1.0)
    assert threshold == pytest.approx(0.31)                     # primer punto de la malla (paso 0.01) por encima de 0.30
    assert score == pytest.approx(8 / 9)                        # 4 TP, 1 FP (el 0.55 negativo), 0 FN


def test_tie_between_equally_good_thresholds_picks_the_lowest_one():
    """Mesetas: ``>`` estricto conserva el primer máximo. Con ``>=`` esta prueba devolvería 0.69."""
    y = np.array([0, 0, 1, 1], dtype=np.uint8)
    p = np.array([0.10, 0.20, 0.70, 0.80])
    threshold, score = evaluate.best_threshold_fbeta(y, p, beta=1.0)
    assert score == pytest.approx(1.0) and threshold == pytest.approx(0.21)


def test_the_grid_is_0_01_to_0_99_in_steps_of_0_01():
    y = np.array([0, 1], dtype=np.uint8)
    for p_pos in (0.995, 0.005):
        threshold, _ = evaluate.best_threshold_fbeta(y, np.array([0.0, p_pos]), beta=1.0)
        assert 0.01 <= threshold <= 0.99 and round(threshold * 100) == pytest.approx(threshold * 100, abs=1e-9)


def test_beta_changes_the_chosen_threshold_towards_precision_or_recall():
    rng = np.random.default_rng(7)
    y = np.array([0] * 300 + [1] * 100, dtype=np.uint8)
    p = np.concatenate([rng.uniform(0.0, 0.7, 300), rng.uniform(0.3, 1.0, 100)])
    t_precision, _ = evaluate.best_threshold_fbeta(y, p, beta=0.5)
    t_f1, _ = evaluate.best_threshold_fbeta(y, p, beta=1.0)
    t_recall, _ = evaluate.best_threshold_fbeta(y, p, beta=2.0)
    assert t_precision >= t_f1 >= t_recall and t_precision > t_recall


def test_class_without_positives_gets_the_neutral_threshold():
    threshold, score = evaluate.best_threshold_fbeta(np.zeros(10, dtype=np.uint8), np.linspace(0, 1, 10), beta=1.0)
    assert (threshold, score) == (0.5, 0.0)


def _fake_predict(flip_test_labels):
    rng = np.random.default_rng(0)
    z_val = rng.normal(size=(400, 3))
    z_test = rng.normal(size=(300, 3))
    y_val = (rng.random((400, 3)) < 1 / (1 + np.exp(-z_val))).astype(np.uint8)
    y_test = (rng.random((300, 3)) < 1 / (1 + np.exp(-z_test))).astype(np.uint8)
    if flip_test_labels:
        y_test = 1 - y_test

    def fake(checkpoint, h5_path, **kwargs):
        z, y = (z_val, y_val) if "val" in str(h5_path) else (z_test, y_test)
        return {"labels": y, "logits": z, "probabilities": (1 / (1 + np.exp(-z))).astype(np.float32),
                "class_names": CLASSES, "record_names": [f"R{i}" for i in range(len(y))]}

    return fake


def test_run_evaluation_never_looks_at_test_labels_to_choose_thresholds(tmp_path, monkeypatch):
    config = {"dev": "val.h5", "test": "test.h5", "batch_size": 8, "num_workers": 0}
    results = {}
    for flip in (False, True):
        monkeypatch.setattr("ecg.predict.predict_hdf5", _fake_predict(flip))
        out = tmp_path / f"flip{flip}"
        evaluate.run_evaluation(config, "x.pt", out, device="cpu", threshold_beta=1.0)
        results[flip] = (pd.read_csv(out / "thresholds_validation.csv"), pd.read_csv(out / "metrics_global.csv"))
    pd.testing.assert_frame_equal(results[False][0], results[True][0])         # umbrales idénticos con etiquetas de test opuestas
    test_f1 = {flip: results[flip][1].query("split == 'test'")["f1_macro"].iloc[0] for flip in (False, True)}
    assert test_f1[False] != pytest.approx(test_f1[True])                       # y el resultado en test sí cambia
