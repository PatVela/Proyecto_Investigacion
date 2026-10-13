"""challenge_score debe coincidir exactamente con el evaluador oficial vendored."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from examples.cinc2020 import challenge_score as cs

OFFICIAL = Path(cs.__file__).parent / "official" / "evaluate_12ECG_score.py"
NORMAL = "426783006"


@pytest.fixture(scope="module")
def official():
    spec = importlib.util.spec_from_file_location("official_eval", OFFICIAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tables():
    return cs.load_official_tables()


def _random_case(n_codes, seed):
    rng = np.random.default_rng(seed)
    labels = (rng.random((200, n_codes)) < 0.1).astype(np.uint8)
    outputs = np.where(rng.random((200, n_codes)) < 0.2, 1 - labels, labels).astype(np.uint8)
    return labels, outputs


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_confusion_matrix_and_metric_match_official(official, tables, seed):
    codes, weights = tables
    labels, outputs = _random_case(len(codes), seed)
    assert np.array_equal(cs.modified_confusion_matrix(labels, outputs),
                          official.compute_modified_confusion_matrix(labels, outputs))
    ours = cs.challenge_metric(weights, labels, outputs, codes)["challenge_metric"]
    theirs = official.compute_challenge_metric(weights, labels.astype(bool), outputs.astype(bool), codes, NORMAL)
    assert ours == pytest.approx(theirs, abs=1e-12)


def test_perfect_classifier_scores_one_and_always_normal_scores_zero(tables):
    codes, weights = tables
    labels, _ = _random_case(len(codes), 5)
    assert cs.challenge_metric(weights, labels, labels, codes)["challenge_metric"] == pytest.approx(1.0)
    inactive = np.zeros_like(labels)
    inactive[:, codes.index(NORMAL)] = 1
    assert cs.challenge_metric(weights, labels, inactive, codes)["challenge_metric"] == pytest.approx(0.0)


def test_group_expansion_marks_every_code_of_a_predicted_group(tables):
    codes, _ = tables
    names = ["NSR", "BBB"]
    prob = np.array([[0.1, 0.9]], dtype=np.float32)
    code_prob, code_pred, uncovered = cs.expand_to_scored(prob, (prob >= 0.5).astype(np.uint8), names, codes)
    bbb_codes = {"713427006", "713426002", "164909002", "59118001", "698252002", "251146004"}
    positive = {codes[j] for j in np.where(code_pred[0] == 1)[0]}
    assert bbb_codes <= positive and NORMAL not in positive
    assert "426783006" not in uncovered and len(uncovered) > 0       # el resto de grupos no se pasó


# --- clases equivalentes: el evaluador oficial puntúa 24 clases, no 27 -----------------------------------------------

def _official_equivalent_classes():
    import ast
    import re

    text = OFFICIAL.read_text(encoding="utf-8")
    match = re.search(r"equivalent_classes\s*=\s*(\[\[.*?\]\])", text)
    return ast.literal_eval(match.group(1))


def test_equivalent_classes_are_the_ones_of_the_vendored_official_scorer():
    assert cs.EQUIVALENT_CLASSES == _official_equivalent_classes()


def test_merged_weights_and_codes_match_official_load_weights(official, tables):
    codes, weights = tables
    merged_codes, merged_weights = cs.merge_weights(codes, weights)
    official_codes, official_weights = official.load_weights(str(cs.OFFICIAL_DIR / "weights.csv"),
                                                             _official_equivalent_classes())
    assert merged_codes == official_codes and len(merged_codes) == len(codes) - 3
    assert np.array_equal(merged_weights, official_weights)


def test_merge_columns_is_or_of_the_equivalent_codes(tables):
    codes, _ = tables
    crbbb, rbbb = codes.index("713427006"), codes.index("59118001")
    matrix = np.zeros((3, len(codes)), dtype=np.uint8)
    matrix[0, crbbb] = 1
    matrix[1, rbbb] = 1
    merged, merged_codes = cs.merge_columns(matrix, codes)
    col = merged_codes.index("713427006")
    assert "59118001" not in merged_codes
    assert merged[:, col].tolist() == [1, 1, 0]


def test_merged_metric_matches_official_on_random_cases(official, tables):
    codes, weights = tables
    labels, outputs = _random_case(len(codes), 9)
    merged_codes, merged_weights = cs.merge_weights(codes, weights)
    labels_m, _ = cs.merge_columns(labels, codes)
    outputs_m, _ = cs.merge_columns(outputs, codes)
    ours = cs.challenge_metric(merged_weights, labels_m, outputs_m, merged_codes)["challenge_metric"]
    theirs = official.compute_challenge_metric(merged_weights, labels_m.astype(bool), outputs_m.astype(bool),
                                               merged_codes, NORMAL)
    assert ours == pytest.approx(theirs, abs=1e-12)


def test_score_groups_merges_by_default_and_can_keep_27(tables):
    codes, weights = tables
    names = ["NSR", "BBB"]
    prob = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    pred = (prob >= 0.5).astype(np.uint8)
    dx = ["426783006", "59118001"]            # NSR ; RBBB (equivalente de CRBBB)
    merged = cs.score_groups(prob, pred, names, dx, codes, weights)
    legacy = cs.score_groups(prob, pred, names, dx, codes, weights, merge_equivalent=False)
    assert merged["n_scored_classes"] == len(codes) - 3 and legacy["n_scored_classes"] == len(codes)
    assert merged["merge_equivalent"] is True and legacy["merge_equivalent"] is False


def test_scores_from_predictions_csv_use_the_stored_decisions(tmp_path, tables):
    import pandas as pd

    codes, weights = tables
    csv = tmp_path / "predictions_test.csv"
    pd.DataFrame({"record_name": ["R1", "R2"], "true_NSR": [1, 0], "prob_NSR": [0.9, 0.1], "pred_NSR": [1, 0],
                  "true_BBB": [0, 1], "prob_BBB": [0.1, 0.9], "pred_BBB": [0, 1]}).to_csv(csv, index=False)
    names, record_names, prob, pred = cs.load_group_predictions(csv)
    assert names == ["NSR", "BBB"] and record_names == ["R1", "R2"]
    assert pred.tolist() == [[1, 0], [0, 1]] and prob.dtype == np.float32
