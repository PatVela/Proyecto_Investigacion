"""Diagnóstico por registro: helpers compartidos (biblioteca, sin CLI).

Extraído de ``debug_record_prediction.py``: lo usan tanto ese script como
``diagnose_prediction.py`` para no duplicar lectura de thresholds, búsqueda
en HDF5 ni tablas de probabilidades.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecg import load, util  # noqa: E402
from webapp.prediction import apply_normal_fallback  # noqa: E402


def read_thresholds(path: str | None, class_names: list[str]) -> dict | None:
    if not path:
        return None
    p = util.resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe thresholds: {p}")
    df = pd.read_csv(p)
    if "class" not in df.columns or "threshold" not in df.columns:
        raise ValueError("thresholds_validation.csv debe tener columnas class y threshold")
    values = {str(row["class"]): float(row["threshold"]) for _, row in df.iterrows()}
    missing = [name for name in class_names if name not in values]
    if missing:
        raise ValueError(f"Faltan thresholds para: {missing}")
    return values


def find_record_in_hdf5(config: dict, record_name: str):
    import h5py

    candidates = [
        ("train", config.get("train")),
        ("validation", config.get("dev")),
        ("test", config.get("test")),
    ]
    target = record_name.lower()
    for split, h5_path in candidates:
        if not h5_path:
            continue
        path = util.resolve_path(h5_path)
        if not path.exists():
            continue
        with h5py.File(path, "r") as h5:
            names = h5.get("record_names")
            if names is None:
                continue
            decoded = [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in names[:]]
            h5_classes = h5.attrs.get("classes", None)
            if h5_classes is not None:
                h5_classes = [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in h5_classes]
            else:
                h5_classes = load.CLASS_NAMES
            for i, name in enumerate(decoded):
                if name.lower() == target:
                    signal = np.asarray(h5["signals"][i], dtype=np.float32)
                    label = np.asarray(h5["labels"][i], dtype=np.uint8)
                    return {
                        "split": split,
                        "path": path,
                        "index": i,
                        "signal": signal,
                        "label": label,
                        "h5_classes": list(h5_classes),
                        "true_classes": [h5_classes[j] for j, active in enumerate(label) if active],
                    }
    return None


def rows_from_probabilities(probabilities, class_names, thresholds, fallback_min_prob):
    rows = []
    for i, prob in enumerate(probabilities):
        thr = float(thresholds[i])
        rows.append({
            "index": i,
            "class": class_names[i],
            "probability": float(prob),
            "threshold": thr,
            "prediction": int(float(prob) >= thr),
        })
    raw = sorted([dict(r) for r in rows if r["prediction"]], key=lambda r: r["probability"], reverse=True)
    fallback = apply_normal_fallback(rows, min_nsr_probability=fallback_min_prob) if fallback_min_prob and fallback_min_prob > 0 else {"applied": False}
    rows_sorted = sorted(rows, key=lambda r: r["probability"], reverse=True)
    final = [r for r in rows_sorted if r["prediction"]]
    return rows_sorted, raw, final, fallback


def print_table(title, rows_sorted, max_rows=12):
    print("\n" + title)
    print("-" * len(title))
    print(f"{'rank':>4} {'class':<20} {'prob':>9} {'thr':>9} {'margin':>9} {'pred':>6} {'post':>6}")
    for rank, row in enumerate(rows_sorted[:max_rows], 1):
        prob = float(row["probability"])
        thr = float(row["threshold"])
        print(f"{rank:4d} {row['class']:<20} {prob:9.4f} {thr:9.4f} {prob-thr:9.4f} {int(row['prediction']):6d} {str(bool(row.get('postprocessed', False))):>6}")
