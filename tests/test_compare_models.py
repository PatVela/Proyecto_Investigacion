"""compare_models.py no debe comparar evaluaciones con distinto punto de operación sin avisar.

Regresión: ``runs/reports/model_comparison.csv`` ponía v1 y v2 (umbrales F0.5) junto al ensemble
(umbrales F1) sin una columna ni un error que lo dijera, y la diferencia v2 − v1 de F0.5 (+0.035) se leía
como una mejora de la receta cuando con F1 común es +0.0125.
"""

import json

import pandas as pd
import pytest

from examples.cinc2020 import compare_models


def _eval_dir(path, f1, beta, with_summary=True):
    path.mkdir(parents=True)
    pd.DataFrame([
        {"split": "validation", "samples": 10, "f1_macro": f1 + 0.01},
        {"split": "test", "samples": 10, "f1_macro": f1},
    ]).to_csv(path / "metrics_global.csv", index=False)
    if with_summary:
        (path / "evaluation_summary.json").write_text(json.dumps({"threshold_beta": beta}), encoding="utf-8")
    return path


def test_same_beta_is_compared_and_the_operating_point_is_reported(tmp_path):
    a = _eval_dir(tmp_path / "a", 0.70, 1.0)
    b = _eval_dir(tmp_path / "b", 0.75, 1.0)
    df = compare_models.compare(a, b, "A", "B", tmp_path / "out.csv")
    assert list(df["model"]) == ["A", "B"]
    assert list(df["threshold_beta"]) == [1.0, 1.0]
    assert (tmp_path / "out.csv").exists()


def test_different_beta_is_refused(tmp_path):
    a = _eval_dir(tmp_path / "a", 0.70, 0.5)
    b = _eval_dir(tmp_path / "b", 0.75, 1.0)
    with pytest.raises(ValueError, match="threshold_beta"):
        compare_models.compare(a, b, "A", "B", tmp_path / "out.csv")
    assert not (tmp_path / "out.csv").exists()


def test_different_beta_can_be_allowed_explicitly_and_stays_visible(tmp_path):
    a = _eval_dir(tmp_path / "a", 0.70, 0.5)
    b = _eval_dir(tmp_path / "b", 0.75, 1.0)
    df = compare_models.compare(a, b, "A", "B", tmp_path / "out.csv", allow_different_beta=True)
    assert list(df["threshold_beta"]) == [0.5, 1.0]


def test_missing_summary_is_reported_as_unknown_not_guessed(tmp_path):
    a = _eval_dir(tmp_path / "a", 0.70, 1.0, with_summary=False)
    b = _eval_dir(tmp_path / "b", 0.75, 1.0)
    df = compare_models.compare(a, b, "A", "B", tmp_path / "out.csv")
    assert pd.isna(df.loc[0, "threshold_beta"]) and df.loc[1, "threshold_beta"] == 1.0
