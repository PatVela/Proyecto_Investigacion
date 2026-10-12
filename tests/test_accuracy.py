"""Tests de las mejoras de accuracy: F-beta, NSR exclusivo y ensemble."""

import numpy as np

from examples.cinc2020 import ensemble_evaluate, evaluate


def test_fbeta_half_is_more_conservative_than_f1():
    rng = np.random.default_rng(7)
    y = np.array([0] * 160 + [1] * 40, dtype=np.uint8)
    # Negativos ruidosos hasta 0.55, positivos desde 0.45 (traslape).
    probs = np.concatenate([rng.uniform(0.0, 0.55, 160), rng.uniform(0.45, 1.0, 40)])
    t_f1, _ = evaluate.best_threshold_fbeta(y, probs, beta=1.0)
    t_f05, _ = evaluate.best_threshold_fbeta(y, probs, beta=0.5)
    assert t_f05 >= t_f1
    fp_f1 = int(((probs >= t_f1) & (y == 0)).sum())
    fp_f05 = int(((probs >= t_f05) & (y == 0)).sum())
    assert fp_f05 <= fp_f1


def test_fbeta_one_matches_f1_wrapper():
    rng = np.random.default_rng(3)
    y = (rng.random(200) < 0.3).astype(np.uint8)
    probs = rng.random(200)
    assert evaluate.best_threshold_f1(y, probs) == evaluate.best_threshold_fbeta(y, probs, beta=1.0)


def test_optimize_thresholds_reports_objective():
    rng = np.random.default_rng(11)
    y = (rng.random((120, 2)) < 0.4).astype(np.uint8)
    probs = rng.random((120, 2))
    thresholds, df = evaluate.optimize_thresholds(y, probs, ["A", "B"], beta=0.5)
    assert thresholds.shape == (2,)
    assert list(df.columns) == ["index", "class", "threshold", "validation_f1",
                                "validation_objective", "validation_positives"]


def test_nsr_exclusive_suppresses_other_positives():
    classes = ["NSR", "AF", "MI"]
    y_prob = np.array([[0.95, 0.80, 0.70],
                       [0.30, 0.80, 0.10]], dtype=np.float32)
    y_pred = np.array([[1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)
    out, count = evaluate.apply_nsr_exclusive(y_pred, y_prob, classes, min_prob=0.90)
    assert count == 1
    assert out[0].tolist() == [1, 0, 0]
    assert out[1].tolist() == [0, 1, 0]
    out2, count2 = evaluate.apply_nsr_exclusive(y_pred, y_prob, classes, min_prob=None)
    assert count2 == 0 and (out2 == y_pred).all()


def test_ensemble_probabilities_weighting():
    a = np.array([[0.9, 0.1]])
    b = np.array([[0.5, 0.5]])
    assert np.allclose(ensemble_evaluate.ensemble_probabilities(a, b, 0.5), [[0.7, 0.3]])
    assert np.allclose(ensemble_evaluate.ensemble_probabilities(a, b, 1.0), a)
    assert np.allclose(ensemble_evaluate.ensemble_probabilities(a, b, 0.0), b)
    try:
        ensemble_evaluate.ensemble_probabilities(a, b, 1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("alpha fuera de [0, 1] debe fallar")


def test_ensemble_end_to_end_with_mocked_predictions(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    classes = ["NSR", "AF", "MI"]

    def fake_predict(checkpoint, h5_path, **kwargs):
        n = 60 if "val" in str(h5_path) else 40
        return {"labels": (rng.random((n, 3)) < 0.4).astype(np.uint8),
                "probabilities": rng.random((n, 3)).astype(np.float32),
                "class_names": classes,
                "record_names": [f"R{i:04d}" for i in range(n)]}

    monkeypatch.setattr("ecg.predict.predict_hdf5", fake_predict)
    summary = ensemble_evaluate.run_ensemble_evaluation(
        {"dev": "val.h5", "test": "test.h5"}, "a.pt", "b.pt", tmp_path, alpha=0.6,
        device="cpu", threshold_beta=0.5, nsr_exclusive_min_prob=0.9,
    )
    assert summary["ensemble_alpha"] == 0.6
    assert summary["threshold_beta"] == 0.5
    for name in ("metrics_global.csv", "metrics_per_class.csv", "thresholds_validation.csv",
                 "predictions_test.csv", "evaluation_summary.json",
                 "roc_curves_test.csv", "pr_curves_test.csv"):
        assert (tmp_path / name).exists(), name


def test_fixed_thresholds_desactiva_optimizacion():
    rng = np.random.default_rng(11)
    y = (rng.random((120, 2)) < 0.4).astype(np.uint8)
    probs = rng.random((120, 2))
    thresholds, df = evaluate.fixed_thresholds(y, probs, ["A", "B"], fixed=0.5)
    assert (thresholds == 0.5).all()
    assert list(df.columns) == ["index", "class", "threshold", "validation_f1",
                                "validation_objective", "validation_positives"]
    assert (df["threshold"] == 0.5).all()
