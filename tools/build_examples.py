"""Selecciona registros de ejemplo del conjunto de test y los copia a ``webapp/examples/``.

Elige, de forma reproducible, un registro por clase objetivo cuya predicción del
ensemble coincide exactamente con la etiqueta, más un caso con diferencias
(aciertos y errores a la vez) para mostrar también los límites del modelo. Solo
considera registros de 10 s a 500 Hz (una ventana). Guarda en el manifiesto las
probabilidades que produjo la evaluación, para verificar la app contra ellas.

    python tools/build_examples.py \\
      --predictions runs/eval/ensemble-f1/predictions_test.csv --dataset dataset2020
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zlib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ecg import schema, signal  # noqa: E402

TARGET_CLASSES = ["NSR", "AF", "STach", "SB", "BBB", "MI"]
FS, SAMPLES = 500, 5000


def find_headers(dataset_dir: Path) -> dict[str, Path]:
    found = {}
    for root, _dirs, names in os.walk(dataset_dir):
        for name in names:
            if name.lower().endswith(".hea"):
                found[name[:-4]] = Path(root) / name
    return found


def _stable_order(name: str, seed: int) -> int:
    return zlib.crc32(f"{seed}:{name}".encode())


def select_records(pred: pd.DataFrame, headers: dict[str, Path], seed: int = 0,
                   targets: list[str] = TARGET_CLASSES) -> list[tuple[str, str]]:
    """Devuelve ``[(registro, tipo)]`` con tipo ``correct`` o ``mixed``."""
    classes = [c[len("true_"):] for c in pred.columns if c.startswith("true_")]
    true = pred[[f"true_{c}" for c in classes]].to_numpy().astype(bool)
    guess = pred[[f"pred_{c}" for c in classes]].to_numpy().astype(bool)

    def usable(name: str) -> bool:
        path = headers.get(name)
        if path is None or not path.with_suffix(".mat").exists():
            return False
        meta = signal.parse_header(path)
        return abs(meta["sampling_freq"] - FS) < 1e-6 and meta["num_samples"] == SAMPLES and meta["num_leads"] == 12

    rows = list(range(len(pred)))
    chosen: list[tuple[str, str]] = []
    for target in targets:
        col = classes.index(target)
        candidates = [i for i in rows if true[i, col] and (true[i] == guess[i]).all()]
        candidates.sort(key=lambda i: (true[i].sum(), _stable_order(pred.record_name.iloc[i], seed)))
        for i in candidates:
            name = pred.record_name.iloc[i]
            if name not in {c[0] for c in chosen} and usable(name):
                chosen.append((name, "correct"))
                break
        else:
            raise RuntimeError(f"No hay un ejemplo utilizable para la clase {target}")
    mixed = [i for i in rows if true[i].sum() >= 2 and (true[i] & guess[i]).any() and (true[i] != guess[i]).any()]
    mixed.sort(key=lambda i: _stable_order(pred.record_name.iloc[i], seed))
    for i in mixed:
        name = pred.record_name.iloc[i]
        if name not in {c[0] for c in chosen} and usable(name):
            chosen.append((name, "mixed"))
            break
    return chosen


def build(predictions: Path, dataset: Path, out_dir: Path, seed: int = 0) -> list[dict]:
    pred = pd.read_csv(predictions)
    headers = find_headers(dataset)
    classes = [c[len("true_"):] for c in pred.columns if c.startswith("true_")]
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():
        if old.is_dir():
            shutil.rmtree(old)
    manifest = []
    for name, kind in select_records(pred, headers, seed):
        hea = headers[name]
        meta = signal.parse_header(hea)
        row = pred[pred.record_name == name].iloc[0]
        target = out_dir / name
        target.mkdir()
        shutil.copyfile(hea, target / f"{name}.hea")
        shutil.copyfile(hea.with_suffix(".mat"), target / f"{name}.mat")
        manifest.append({
            "id": name,
            "record": name,
            "kind": kind,
            "source_db": next((p for p in hea.parts[::-1][1:] if p in {"ptb-xl", "georgia", "cpsc_2018", "cpsc_2018_extra",
                                                                       "ptb", "st_petersburg_incart"}), "unknown"),
            "dx_codes": meta["dx_codes"],
            "true_classes": schema.matched_class_names(meta["dx_codes"]),
            "reference_predicted_classes": [c for c in classes if row[f"pred_{c}"] == 1],
            "reference_probabilities": {c: round(float(row[f"prob_{c}"]), 5) for c in classes},
        })
    (out_dir / "examples.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Genera los registros de ejemplo de la webapp.")
    p.add_argument("--predictions", required=True, help="predictions_test.csv del ensemble")
    p.add_argument("--dataset", required=True, help="Carpeta raíz de CINC2020 con los .hea/.mat")
    p.add_argument("--out", default=str(REPO_ROOT / "webapp" / "examples"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    manifest = build(Path(args.predictions), Path(args.dataset), Path(args.out), args.seed)
    for m in manifest:
        print(f"{m['id']:<10} {m['kind']:<8} {m['source_db']:<14} verdad={m['true_classes']} predicho={m['reference_predicted_classes']}")


if __name__ == "__main__":
    main()
