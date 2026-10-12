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
