"""Tests de exportación de curvas ROC/PR (evaluate.save_roc_pr_curves)."""

import numpy as np
import pandas as pd

from evaluate import save_roc_pr_curves


def test_curves_files_columns_and_cap(tmp_path):
    rng = np.random.default_rng(5)
    n, c = 300, 4
    y = (rng.random((n, c)) < 0.4).astype(float)
    p = np.clip(y * 0.6 + rng.random((n, c)) * 0.4, 0.01, 0.99)
    classes = [f"C{i}" for i in range(c)]
    save_roc_pr_curves(tmp_path, y, p, classes, "test", max_points=50)
    roc = pd.read_csv(tmp_path / "roc_curves_test.csv")
    pr = pd.read_csv(tmp_path / "pr_curves_test.csv")
    assert list(roc.columns) == ["split", "class", "fpr", "tpr", "threshold"]
    assert list(pr.columns) == ["split", "class", "precision", "recall", "threshold"]
    assert roc.groupby("class").size().max() <= 50
    assert pr.groupby("class").size().max() <= 50
    assert set(roc["class"].unique()) == set(classes)
    assert roc["fpr"].between(0, 1).all() and roc["tpr"].between(0, 1).all()


def test_degenerate_class_is_skipped(tmp_path):
    rng = np.random.default_rng(5)
    y = np.zeros((100, 2))
    y[:, 1] = (rng.random(100) < 0.5).astype(float)
    p = rng.random((100, 2))
    save_roc_pr_curves(tmp_path, y, p, ["A", "B"], "test")
    roc = pd.read_csv(tmp_path / "roc_curves_test.csv")
    assert set(roc["class"].unique()) == {"B"}
