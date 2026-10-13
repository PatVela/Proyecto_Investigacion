"""error_analysis.py: diagnóstico por origen que no castiga clases que el origen nunca anota.

Regresión: ``errores_origen.csv`` daba 0 de F1 a toda clase ausente de un origen, así que CPSC aparecía en 0.433 con
v2 cuando sus 7 clases anotadas dan 0.769, y los falsos positivos sobre clases que ese origen no etiqueta (se
entrenan como negativas) no se distinguían de los demás.
"""

import json

import numpy as np
import pandas as pd
import pytest

from examples.cinc2020 import error_analysis as ea

CLASSES = ["X", "Y"]


def test_unannotated_cells_are_origin_class_pairs_with_no_train_positives():
    labels = np.array([[1, 0], [0, 0], [1, 1], [0, 1]])
    sources = np.array(["A", "A", "B", "B"])
    assert ea.unannotated_cells(labels, sources, CLASSES) == {("A", "Y")}


def test_prevalence_floor_also_masks_nearly_unannotated_cells():
    labels = np.zeros((1000, 2), dtype=int)
    labels[:, 0] = 1
    labels[0, 1] = 1                       # 1 positivo de Y en 1000 registros: 0.1 %
    sources = np.array(["A"] * 1000)
    assert ea.unannotated_cells(labels, sources, CLASSES) == set()
    assert ea.unannotated_cells(labels, sources, CLASSES, min_prevalence=0.005) == {("A", "Y")}


def _case():
    # Origen A no anota Y: sus FP de Y no son un error del modelo sobre algo que A etiquete.
    sources = np.array(["A"] * 4 + ["B"] * 4)
    y_true = np.array([[1, 0], [1, 0], [0, 0], [0, 0],
                       [1, 1], [1, 1], [0, 0], [0, 0]])
    y_pred = np.array([[1, 1], [1, 1], [0, 0], [0, 0],       # dos FP de Y en A
                       [1, 1], [1, 1], [0, 0], [0, 0]])
    return sources, y_true, y_pred


def test_masked_f1_ignores_records_of_origins_that_do_not_annotate_the_class():
    sources, y_true, y_pred = _case()
    out = ea.masked_summary(y_true, y_pred, sources, CLASSES, {("A", "Y")})
    assert out["f1_macro_unmasked"] == pytest.approx((1.0 + 2 / 3) / 2)       # Y: precisión 2/4, recall 1 -> F1 2/3
    assert out["f1_macro_masked"] == pytest.approx(1.0)
    assert out["fp_total"] == 2 and out["fp_in_masked_cells"] == 2
    assert out["masked_cells"] == [{"source_db": "A", "class": "Y", "n_records": 4}]


def test_per_source_f1_uses_only_the_classes_the_source_annotates():
    sources, y_true, y_pred = _case()
    rows = ea.per_source_rows(pd.DataFrame(), y_true, y_pred, sources, CLASSES, {("A", "Y")}).set_index("source_db")
    assert rows.loc["A", "n_classes_annotated"] == 1 and rows.loc["A", "f1_macro_annotated"] == pytest.approx(1.0)
    assert rows.loc["B", "n_classes_annotated"] == 2
    assert rows.loc["A", "f1_macro"] == pytest.approx(0.5)        # el valor histórico (clase ausente cuenta 0) se conserva


def test_main_writes_the_masked_summary_next_to_the_csv(tmp_path, monkeypatch):
    sources, y_true, y_pred = _case()
    names = [f"R{i}" for i in range(len(sources))]
    payload = {"record_name": names}
    for j, c in enumerate(CLASSES):
        payload[f"true_{c}"], payload[f"pred_{c}"] = y_true[:, j], y_pred[:, j]
    pd.DataFrame(payload).to_csv(tmp_path / "predictions_test.csv", index=False)
    monkeypatch.setattr(ea, "load_sources", lambda _path: dict(zip(names, sources)))
    train_labels = np.array([[1, 0], [0, 0], [1, 1], [0, 1]])
    train_sources = np.array(["A", "A", "B", "B"])
    monkeypatch.setattr(ea, "load_train_labels", lambda _path, classes: (train_labels, train_sources))
    monkeypatch.setattr("sys.argv", ["error_analysis", "--predictions", str(tmp_path / "predictions_test.csv"),
                                     "--test-h5", "x.h5", "--train-h5", "train.h5", "--output", str(tmp_path / "e.csv")])
    ea.main()
    table = pd.read_csv(tmp_path / "e.csv")
    assert {"f1_macro", "f1_macro_annotated", "n_classes_annotated"} <= set(table.columns)
    summary = json.loads((tmp_path / "e.masked.json").read_text(encoding="utf-8"))
    assert summary["f1_macro_masked"] == pytest.approx(1.0) and summary["rule"].startswith("0 positivos")
