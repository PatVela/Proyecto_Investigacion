"""Análisis de errores por subconjunto de origen (hospital/fuente CINC2020).

Cruza ``predictions_test.csv`` con el dataset ``source_dbs`` del HDF5 de test
y reporta exact-match + F1 macro/micro por cada fuente (georgia, cpsc_2018,
ptb, ...). Útil para detectar sesgo por hospital de origen.

Con ``--train-h5`` añade el diagnóstico de anotación parcial: hay orígenes que nunca etiquetan algunas clases
(p. ej. cpsc_2018 no tiene ningún positivo de AxisDev, MI, LVH, STach ni SB) y esas celdas "sin anotar" se entrenan
como negativas. La regla es fija y sale solo de train: una celda origen×clase está sin anotar si no tiene positivos
en train (o menos que ``--min-train-prevalence``). Con ella se reporta:

- ``f1_macro_annotated`` por origen: F1 medio solo sobre las clases que el origen anota (``f1_macro`` conserva el
  valor histórico, en el que una clase ausente cuenta 0);
- un resumen global ``<output>.masked.json``: F1-macro con y sin enmascarar las celdas sin anotar, y cuántos falsos
  positivos caen en ellas.

El F1 sin enmascarar sigue siendo el titular: el enmascarado es un diagnóstico y solo elimina falsos positivos, así
que siempre sube.

Ejemplo:
    python examples/cinc2020/error_analysis.py \\
      --predictions <dir-evaluacion>/predictions_test.csv \\
      --test-h5 data/cinc2020_12/test.h5 \\
      --train-h5 data/cinc2020_12/train.h5 \\
      --output <errores-por-origen>.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecg.util import decode_h5_strings  # noqa: E402


def load_sources(h5_path: str | Path) -> dict[str, str]:
    """Devuelve {record_name: source_db} desde el HDF5 (primera ventana si repite)."""
    mapping: dict[str, str] = {}
    with h5py.File(h5_path, "r") as h5:
        if "source_dbs" not in h5 or "record_names" not in h5:
            raise ValueError(f"{h5_path} no tiene source_dbs/record_names; regenere el HDF5.")
        names = decode_h5_strings(h5["record_names"][:])
        sources = decode_h5_strings(h5["source_dbs"][:])
    for name, src in zip(names, sources):
        mapping.setdefault(name, src)
        mapping.setdefault(name.split(":")[0], src)
    return mapping


def load_train_labels(h5_path: str | Path, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(etiquetas ventana×clase, origen por ventana) de train.h5, en el orden de ``classes``."""
    from ecg import load

    if list(classes) != list(load.CLASS_NAMES):
        raise ValueError(f"Las clases de las predicciones {list(classes)} no coinciden con las del HDF5 "
                         f"{list(load.CLASS_NAMES)}.")
    with h5py.File(h5_path, "r") as h5:
        if "source_dbs" not in h5 or "labels" not in h5:
            raise ValueError(f"{h5_path} no tiene source_dbs/labels; regenere el HDF5.")
        labels = np.asarray(h5["labels"][:])
        sources = np.asarray(decode_h5_strings(h5["source_dbs"][:]))
    return labels, sources


def unannotated_cells(labels: np.ndarray, sources: np.ndarray, classes: list[str],
                      min_prevalence: float = 0.0) -> set[tuple[str, str]]:
    """Celdas (origen, clase) sin positivos en train, o con prevalencia menor que ``min_prevalence``."""
    cells: set[tuple[str, str]] = set()
    labels = np.asarray(labels)
    for source in sorted(set(sources.tolist())):
        block = labels[sources == source]
        for j, name in enumerate(classes):
            positives = float(block[:, j].sum())
            prevalence = positives / max(len(block), 1)
            if positives == 0 or (min_prevalence > 0 and prevalence < min_prevalence):
                cells.add((str(source), str(name)))
    return cells


def _class_f1(y_true, y_pred, keep) -> float:
    return float(f1_score(y_true[keep], y_pred[keep], zero_division=0)) if keep.any() else float("nan")


def masked_summary(y_true: np.ndarray, y_pred: np.ndarray, sources: np.ndarray, classes: list[str],
                   cells: set[tuple[str, str]]) -> dict:
    """F1-macro global con y sin ignorar los registros de los orígenes que no anotan cada clase."""
    f1_unmasked, f1_masked, fp_in_cells = [], [], 0
    for j, name in enumerate(classes):
        masked_sources = [s for s, c in cells if c == name]
        keep = ~np.isin(sources, masked_sources)
        f1_unmasked.append(_class_f1(y_true[:, j], y_pred[:, j], np.ones(len(sources), dtype=bool)))
        f1_masked.append(_class_f1(y_true[:, j], y_pred[:, j], keep))
        fp_in_cells += int(((y_pred[:, j] == 1) & (y_true[:, j] == 0) & ~keep).sum())
    return {
        "f1_macro_unmasked": float(np.nanmean(f1_unmasked)),
        "f1_macro_masked": float(np.nanmean(f1_masked)),
        "fp_total": int(((y_pred == 1) & (y_true == 0)).sum()),
        "fp_in_masked_cells": fp_in_cells,
        "masked_cells": [
            {"source_db": s, "class": c, "n_records": int((sources == s).sum())} for s, c in sorted(cells)
        ],
    }


def per_source_rows(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, sources: np.ndarray,
                    classes: list[str], cells: set[tuple[str, str]] | None = None) -> pd.DataFrame:
    rows = []
    for source in sorted(set(sources.tolist())):
        m = sources == source
        yt, yp = y_true[m], y_pred[m]
        row = {
            "source_db": source,
            "n": int(m.sum()),
            "positives": int(yt.sum()),
            "exact_match": float((yt == yp).all(axis=1).mean()),
            "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(yt, yp, average="micro", zero_division=0)),
        }
        if cells is not None:
            annotated = [j for j, name in enumerate(classes) if (source, name) not in cells]
            row["n_classes_annotated"] = len(annotated)
            row["f1_macro_annotated"] = (
                float(np.mean([f1_score(yt[:, j], yp[:, j], zero_division=0) for j in annotated]))
                if annotated else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Errores por subconjunto de origen.")
    parser.add_argument("--predictions", required=True, help="predictions_test.csv de evaluate.py")
    parser.add_argument("--test-h5", required=True, help="HDF5 de test con source_dbs")
    parser.add_argument("--train-h5", default=None,
                        help="HDF5 de train: activa el diagnóstico de celdas origen×clase sin anotar")
    parser.add_argument("--min-train-prevalence", type=float, default=0.0,
                        help="Una celda cuenta como sin anotar si su prevalencia en train es menor que este valor "
                             "(0 = solo las que no tienen ningún positivo)")
    parser.add_argument("--output", required=True, help="CSV de salida")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    classes = [c[len("true_"):] for c in df.columns if c.startswith("true_")]
    if not classes:
        raise SystemExit(f"{args.predictions} no tiene columnas true_<clase>.")
    by_source = load_sources(args.test_h5)
    df["source_db"] = df["record_name"].map(by_source)
    unknown = int(df["source_db"].isna().sum())
    if unknown:
        print(f"AVISO: {unknown} filas sin fuente conocida (se excluyen).")
        df = df.dropna(subset=["source_db"]).reset_index(drop=True)
    if df.empty:
        raise SystemExit("Sin filas con fuente conocida; revise que el HDF5 corresponda al CSV.")

    y_true = df[[f"true_{c}" for c in classes]].to_numpy(dtype=np.uint8)
    y_pred = df[[f"pred_{c}" for c in classes]].to_numpy(dtype=np.uint8)
    sources = df["source_db"].to_numpy()

    cells = None
    if args.train_h5:
        train_labels, train_sources = load_train_labels(args.train_h5, classes)
        cells = unannotated_cells(train_labels, train_sources, classes, args.min_train_prevalence)
    out = per_source_rows(df, y_true, y_pred, sources, classes, cells)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"\nResultados guardados en: {args.output}")

    if cells is not None:
        summary = masked_summary(y_true, y_pred, sources, classes, cells)
        summary["rule"] = ("0 positivos de la clase en train del origen" if args.min_train_prevalence <= 0 else
                           f"prevalencia en train del origen < {args.min_train_prevalence}")
        summary["train_h5"] = str(args.train_h5)
        summary["predictions"] = str(args.predictions)
        masked_path = Path(args.output).with_suffix(".masked.json")
        masked_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nF1-macro sin enmascarar : {summary['f1_macro_unmasked']:.4f}  (titular)"
              f"\nF1-macro enmascarado     : {summary['f1_macro_masked']:.4f}  (diagnóstico; {summary['rule']})"
              f"\nFP en celdas sin anotar  : {summary['fp_in_masked_cells']} de {summary['fp_total']}"
              f"\nResumen en               : {masked_path}")


if __name__ == "__main__":
    main()
