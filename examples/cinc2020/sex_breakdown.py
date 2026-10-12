"""Desglose de F1-macro por sexo (auditoría de sesgo, C10).

Cruza ``predictions_test.csv`` (record_name/true_*/pred_*) con los datasets
``record_names``/``sexes`` del HDF5 de test y reporta F1-macro + soporte por
grupo. No requiere GPU ni torch.

Uso:
    python examples/cinc2020/sex_breakdown.py --predictions <eval>/predictions_test.csv --test-h5 data/cinc2020_12/test.h5 --output sex_breakdown.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CANONICAL = {"M": "Male", "MALE": "Male", "MASCULINO": "Male",
             "F": "Female", "FEMALE": "Female", "FEMENINO": "Female"}


def _decode(values) -> list[str]:
    out = []
    for v in values:
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        out.append(str(v))
    return out


def normalize_sex(raw: str) -> str:
    key = (raw or "").strip().upper()
    if not key or key in {"UNKNOWN", "NAN", "NONE", "?", "U"}:
        return "Unknown"
    return CANONICAL.get(key, key.title())


def main() -> None:
    parser = argparse.ArgumentParser(description="F1-macro por sexo (test).")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--test-h5", required=True)
    parser.add_argument("--output", default="sex_breakdown.json")
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions)
    classes = [c[len("true_"):] for c in pred.columns if c.startswith("true_")]
    if not classes:
        raise SystemExit(f"{args.predictions} no tiene columnas true_<clase>.")
    with h5py.File(args.test_h5, "r") as h5:
        names = _decode(h5["record_names"][:])
        sexes = _decode(h5["sexes"][:])
    meta = pd.DataFrame({"record_name": names, "sex_raw": sexes})
    merged = pred.merge(meta, on="record_name", how="inner")
    print(f"Registros: CSV={len(pred)} H5={len(meta)} cruzados={len(merged)}")
    if not len(merged):
        raise SystemExit("Sin coincidencias por record_name entre CSV y H5.")
    merged["sex"] = merged["sex_raw"].map(normalize_sex)
    print("Distribución cruda:", {k: int(v) for k, v in merged["sex_raw"].value_counts().items()})

    groups = []
    for sex, sub in merged.groupby("sex"):
        yt = sub[[f"true_{c}" for c in classes]].to_numpy(dtype=np.uint8)
        yp = sub[[f"pred_{c}" for c in classes]].to_numpy(dtype=np.uint8)
        f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
        groups.append({"sex": sex, "n": int(len(sub)), "f1_macro": round(f1, 4)})
        flag = " (n<30: no concluyente)" if len(sub) < 30 else ""
        print(f"{sex}: n={len(sub)} F1-macro={f1:.4f}{flag}")

    Path(args.output).write_text(json.dumps({"predictions": str(args.predictions),
                                             "test_h5": str(args.test_h5),
                                             "groups": groups}, indent=2), encoding="utf-8")
    print(f"Guardado en: {args.output}")


if __name__ == "__main__":
    main()
