"""Análisis de errores por subconjunto de origen (hospital/fuente CINC2020).

Cruza ``predictions_test.csv`` con el dataset ``source_dbs`` del HDF5 de test
y reporta exact-match + F1 macro/micro por cada fuente (georgia, cpsc_2018,
ptb, ...). Útil para detectar sesgo por hospital de origen.

Ejemplo:
    python examples/cinc2020/error_analysis.py \\
      --predictions <dir-evaluacion>/predictions_test.csv \\
      --test-h5 data/cinc2020_12/test.h5 \\
      --output <errores-por-origen>.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_sources(h5_path: str | Path) -> dict[str, str]:
    """Devuelve {record_name: source_db} desde el HDF5 (primera ventana si repite)."""
    mapping: dict[str, str] = {}
    with h5py.File(h5_path, "r") as h5:
        if "source_dbs" not in h5 or "record_names" not in h5:
            raise ValueError(f"{h5_path} no tiene source_dbs/record_names; regenere el HDF5.")
        names = [_decode(v) for v in h5["record_names"][:]]
        sources = [_decode(v) for v in h5["source_dbs"][:]]
    for name, src in zip(names, sources):
        mapping.setdefault(name, src)
        mapping.setdefault(name.split(":")[0], src)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Errores por subconjunto de origen.")
    parser.add_argument("--predictions", required=True, help="predictions_test.csv de evaluate.py")
    parser.add_argument("--test-h5", required=True, help="HDF5 de test con source_dbs")
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

    rows = []
    for source, group in df.groupby("source_db"):
        y_true = group[[f"true_{c}" for c in classes]].to_numpy(dtype=np.uint8)
        y_pred = group[[f"pred_{c}" for c in classes]].to_numpy(dtype=np.uint8)
        rows.append({
            "source_db": source,
            "n": int(len(group)),
            "positives": int(y_true.sum()),
            "exact_match": float((y_true == y_pred).all(axis=1).mean()),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        })
    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"\nResultados guardados en: {args.output}")


if __name__ == "__main__":
    main()
