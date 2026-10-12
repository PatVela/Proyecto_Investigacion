"""Lógica de decisión: umbrales, fallback NSR, búsqueda de umbral F-beta y métricas.

Cubre el núcleo que ningún test protegía (la revisión de código mostró que
invertir ``prob >= umbral`` o la búsqueda de umbral dejaba la suite en verde).
"""

import numpy as np
import pytest
from sklearn.metrics import fbeta_score

from ecg import load
from ecg.calibration import apply_temperature
from examples.cinc2020 import ensemble_evaluate, evaluate
from webapp import prediction as pred

NAMES = load.CLASS_NAMES


# --------------------------------------------------------------------------- #
# webapp: regla de decisión por clase
# --------------------------------------------------------------------------- #
def _rows(probs, thresholds):
    return pred._prediction_rows(np.asarray(probs, dtype=float), NAMES, np.asarray(thresholds, dtype=float))


def test_decision_is_probability_greater_or_equal_to_threshold():
    probs = np.full(12, 0.5)
    thr = np.full(12, 0.5)
    thr[1], thr[2] = 0.5000001, 0.4999999
    rows = _rows(probs, thr)
    assert rows[0]["prediction"] == 1          # igual al umbral -> positivo
    assert rows[1]["prediction"] == 0          # umbral apenas mayor -> negativo
    assert rows[2]["prediction"] == 1          # umbral apenas menor -> positivo


def test_decision_uses_each_class_own_threshold():
    probs = np.full(12, 0.6)
    thr = np.where(np.arange(12) % 2 == 0, 0.5, 0.7)
    rows = _rows(probs, thr)
    assert [r["prediction"] for r in rows] == [1, 0] * 6


def test_margin_and_near_threshold_flags():
    probs = np.full(12, 0.9)
    thr = np.full(12, 0.9)
    thr[0] = 0.5          # lejos del umbral
    thr[1] = 0.93         # dentro de 0.05
    rows = _rows(probs, thr)
    assert rows[0]["margin"] == pytest.approx(0.4) and rows[0]["near_threshold"] is False
    assert rows[1]["margin"] == pytest.approx(-0.03) and rows[1]["near_threshold"] is True


def test_threshold_values_from_dict_array_and_fallback():
    assert np.allclose(pred.threshold_values_for_class_names(NAMES, 0.4, None), 0.4)
    from_dict = pred.threshold_values_for_class_names(NAMES, 0.5, {"AF": 0.8})
    assert from_dict[NAMES.index("AF")] == pytest.approx(0.8)
    assert from_dict[NAMES.index("NSR")] == pytest.approx(0.5)
    arr = pred.threshold_values_for_class_names(NAMES, 0.5, list(np.linspace(0.1, 0.9, 12)))
    assert arr[0] == pytest.approx(0.1) and arr[-1] == pytest.approx(0.9)
    with pytest.raises(ValueError):
        pred.threshold_values_for_class_names(NAMES, 0.5, [0.5] * 11)


def _row_set(probs_by_class, thr=0.5):
    probs = np.array([probs_by_class.get(n, 0.0) for n in NAMES])
    return _rows(probs, np.full(12, thr))


def test_normal_fallback_adds_nsr_only_when_nothing_else_is_positive():
    rows = _row_set({"NSR": 0.45, "AF": 0.1})
    info = pred.apply_normal_fallback(rows, min_nsr_probability=0.40)
    assert info["applied"] is True and info["added_class"] == "NSR"
    assert next(r for r in rows if r["class"] == "NSR")["prediction"] == 1

    rows = _row_set({"NSR": 0.45, "AF": 0.9})                   # ya hay un positivo
    info = pred.apply_normal_fallback(rows, min_nsr_probability=0.40)
    assert info["applied"] is False
    assert next(r for r in rows if r["class"] == "NSR")["prediction"] == 0

    rows = _row_set({"NSR": 0.30})                                # NSR por debajo del mínimo
    assert pred.apply_normal_fallback(rows, min_nsr_probability=0.40)["applied"] is False


def test_label_comparison_counts_tp_fp_fn():
    rows = _row_set({"AF": 0.9, "MI": 0.9, "TAb": 0.9})
    cmp_ = pred.build_label_comparison(["AF", "NSR"], rows, "manual")
    assert cmp_["true_positive"] == ["AF"]
    assert set(cmp_["false_positive"]) == {"MI", "TAb"}
    assert cmp_["false_negative"] == ["NSR"]
    assert cmp_["exact_match"] is False
    assert cmp_["precision"] == pytest.approx(1 / 3) and cmp_["recall"] == pytest.approx(1 / 2)
    assert cmp_["jaccard"] == pytest.approx(1 / 4)


# --------------------------------------------------------------------------- #
# evaluate.py: búsqueda de umbral F-beta y métricas
# --------------------------------------------------------------------------- #
def test_best_threshold_separable_scores_are_perfect():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95])
    thr, score = evaluate.best_threshold_fbeta(y, p, beta=1.0)
    assert score == pytest.approx(1.0)
    assert 0.3 < thr <= 0.7


def test_best_threshold_matches_brute_force_maximum():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.3).astype(np.uint8)
    p = np.clip(0.35 * y + rng.normal(0.35, 0.2, 300), 0, 1)
    for beta in (0.5, 1.0, 2.0):
        thr, score = evaluate.best_threshold_fbeta(y, p, beta=beta)
        grid = [fbeta_score(y, (p >= t).astype(int), beta=beta, zero_division=0) for t in np.linspace(0.01, 0.99, 99)]
        assert score == pytest.approx(max(grid))
        assert score >= fbeta_score(y, (p >= 0.5).astype(int), beta=beta, zero_division=0) - 1e-12


def test_beta_changes_threshold_in_the_expected_direction():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.2).astype(np.uint8)
    p = np.clip(0.3 * y + rng.normal(0.3, 0.18, 2000), 0, 1)
    t_precision, _ = evaluate.best_threshold_fbeta(y, p, beta=0.5)
    t_recall, _ = evaluate.best_threshold_fbeta(y, p, beta=2.0)
    assert t_precision > t_recall            # beta<1 favorece precisión -> umbral más alto


def test_best_threshold_without_positives_defaults_to_half():
    assert evaluate.best_threshold_fbeta(np.zeros(10, dtype=np.uint8), np.linspace(0, 1, 10)) == (0.5, 0.0)


def test_optimize_thresholds_returns_one_per_class_with_support():
    rng = np.random.default_rng(2)
    y = (rng.random((200, 3)) < 0.4).astype(np.uint8)
    p = np.clip(0.4 * y + rng.normal(0.3, 0.15, y.shape), 0, 1)
    thresholds, df = evaluate.optimize_thresholds(y, p, ["a", "b", "c"], beta=1.0)
    assert thresholds.shape == (3,) and list(df["class"]) == ["a", "b", "c"]
    assert list(df["validation_positives"]) == list(y.sum(axis=0))


def test_evaluate_predictions_known_confusion_matrix():
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
    y_prob = np.array([[0.9, 0.1], [0.4, 0.2], [0.2, 0.8], [0.6, 0.1]])
    class_df, global_m, y_pred = evaluate.evaluate_predictions(y_true, y_prob, np.array([0.5, 0.5]), ["x", "y"], "test")
    x = class_df[class_df["class"] == "x"].iloc[0]
    assert (x["TP"], x["FN"], x["FP"], x["TN"]) == (1, 1, 1, 1)
    assert y_pred.tolist() == [[1, 0], [0, 0], [0, 1], [1, 0]]
    assert global_m["f1_macro"] == pytest.approx((0.5 + 1.0) / 2)


def test_normal_fallback_and_nsr_exclusive_in_evaluate():
    names = ["NSR", "AF"]
    y_prob = np.array([[0.45, 0.1], [0.45, 0.9], [0.95, 0.9]])
    y_pred = np.array([[0, 0], [0, 1], [1, 1]])
    out, n = evaluate.apply_normal_fallback(y_pred, y_prob, names, min_prob=0.40)
    assert n == 1 and out.tolist() == [[1, 0], [0, 1], [1, 1]]
    out, n = evaluate.apply_nsr_exclusive(y_pred, y_prob, names, min_prob=0.9)
    assert n == 1 and out.tolist() == [[0, 0], [0, 1], [1, 0]]
    assert evaluate.apply_normal_fallback(y_pred, y_prob, names, min_prob=None)[1] == 0


# --------------------------------------------------------------------------- #
# ensemble y calibración
# --------------------------------------------------------------------------- #
def test_ensemble_rejects_alpha_outside_unit_interval():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            ensemble_evaluate.ensemble_probabilities([0.5], [0.5], bad)


def test_temperature_one_is_identity_and_higher_temperature_softens():
    p = np.array([[0.9, 0.1, 0.5]])
    assert np.allclose(apply_temperature(p, [1.0, 1.0, 1.0]), p, atol=1e-6)
    soft = apply_temperature(p, [2.0, 2.0, 2.0])
    assert soft[0, 0] < 0.9 and soft[0, 1] > 0.1 and soft[0, 2] == pytest.approx(0.5, abs=1e-6)
    sharp = apply_temperature(p, [0.5, 0.5, 0.5])
    assert sharp[0, 0] > 0.9 and sharp[0, 1] < 0.1


# --------------------------------------------------------------------------- #
# casos límite: igualdad exacta con el umbral / el mínimo del fallback
# --------------------------------------------------------------------------- #
def test_evaluate_counts_probability_equal_to_threshold_as_positive():
    y_true = np.array([[1], [0]])
    y_prob = np.array([[0.5], [0.5]])
    _, _, y_pred = evaluate.evaluate_predictions(y_true, y_prob, np.array([0.5]), ["x"], "test")
    assert y_pred.ravel().tolist() == [1, 1]


def test_normal_fallback_applies_when_nsr_probability_equals_minimum():
    rows = _row_set({"NSR": 0.40})
    assert pred.apply_normal_fallback(rows, min_nsr_probability=0.40)["applied"] is True
    out, n = evaluate.apply_normal_fallback(np.zeros((1, 2), dtype=np.uint8), np.array([[0.40, 0.0]]), ["NSR", "AF"], 0.40)
    assert n == 1 and out.tolist() == [[1, 0]]
