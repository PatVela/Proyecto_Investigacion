"""Genera un HDF5 sintético pequeño para smoke tests.

Genera un HDF5 sintético mínimo para smoke tests de entreno, evaluación y exportación.
No se debe usar para reportar métricas científicas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecg import load


def write_split(path: Path, n: int, seed: int):
    import h5py

    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Escala tipo mV (coherente con norm_mode=physical): fondo ~N(0, 0.2) mV
    # más "complejos QRS" sintéticos para que haya estructura aprendible.
    signals = rng.normal(0, 0.2, size=(n, load.WINDOW_LENGTH, load.NUM_LEADS)).astype(np.float32)
    for i in range(n):
        for beat in range(6, load.WINDOW_LENGTH - 6, 500):
            amp = rng.uniform(0.5, 1.5) * (1 if rng.random() < 0.8 else -1)
            width = int(rng.integers(8, 20))
            lo, hi = max(0, beat - width), min(load.WINDOW_LENGTH, beat + width)
            bump = np.exp(-0.5 * ((np.arange(lo, hi) - beat) / (width / 2.5)) ** 2)
            lead = int(rng.integers(0, load.NUM_LEADS))
            signals[i, lo:hi, lead] += (amp * bump).astype(np.float32)
    signals = np.clip(signals, -load.PHYSICAL_CLIP_MV, load.PHYSICAL_CLIP_MV)
    labels = np.zeros((n, load.NUM_CLASSES), dtype=np.float32)
    for i in range(n):
        labels[i, i % load.NUM_CLASSES] = 1
        if i % 5 == 0:
            labels[i, (i + 3) % load.NUM_CLASSES] = 1
        if i % 11 == 0:
            labels[i, :] = 0  # algunos all-zero

    # Dx sintéticos: primer código puntuado de cada grupo activo (para challenge_score).
    scored = {c for c, _ in load.SCORED_27_CODES}
    rep_codes = []
    for group in load.CLASS_GROUPS_12:
        hit = next((c for c in group["codes"] if c in scored), group["codes"][0])
        rep_codes.append(hit)
    dx_list, matched_list = [], []
    for i in range(n):
        active = [j for j in range(load.NUM_CLASSES) if labels[i, j] > 0]
        dx_list.append(",".join(rep_codes[j] for j in active))
        matched_list.append(",".join(load.CLASS_NAMES[j] for j in active))

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        h5.create_dataset("signals", data=signals, compression="gzip", compression_opts=2)
        h5.create_dataset("labels", data=labels)
        h5.create_dataset("ages", data=rng.integers(18, 90, size=n).astype(np.float32))
        h5.create_dataset("sexes", data=np.asarray(["Unknown"] * n, dtype=object), dtype=string_dtype)
        h5.create_dataset("source_dbs", data=np.asarray(["synthetic"] * n, dtype=object), dtype=string_dtype)
        h5.create_dataset("record_names", data=np.asarray([f"synth_{i:05d}" for i in range(n)], dtype=object), dtype=string_dtype)
        h5.create_dataset("dx_codes", data=np.asarray(dx_list, dtype=object), dtype=string_dtype)
        h5.create_dataset("matched_classes", data=np.asarray(matched_list, dtype=object), dtype=string_dtype)
        h5.create_dataset("window_positions", data=np.zeros(n, dtype=np.int32))
        h5.create_dataset("window_starts", data=np.zeros(n, dtype=np.int32))
        h5.attrs["dataset"] = "Synthetic smoke-test"
        h5.attrs["label_schema"] = load.LABEL_SCHEMA
        h5.attrs["norm_mode"] = load.DEFAULT_NORM_MODE
        h5.attrs["label_mode"] = "multilabel"
        h5.attrs["sampling_rate"] = load.TARGET_FS
        h5.attrs["window_length"] = load.WINDOW_LENGTH
        h5.attrs["num_leads"] = load.NUM_LEADS
        h5.attrs["num_classes"] = load.NUM_CLASSES
        h5.attrs["lead_order"] = ",".join(load.STANDARD_LEAD_ORDER)
        h5.attrs["normalization"] = "synthetic millivolt-scale with QRS-like bumps, clip ±5 mV"
        h5.attrs.create("classes", np.asarray(load.CLASS_NAMES, dtype=string_dtype))


def main():
    parser = argparse.ArgumentParser(description="Genera dataset sintético mínimo CINC2020-12")
    parser.add_argument("--output-dir", default="data/cinc2020_synth")
    parser.add_argument("--train", type=int, default=48)
    parser.add_argument("--val", type=int, default=24)
    parser.add_argument("--test", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    write_split(out / "train.h5", args.train, args.seed)
    write_split(out / "val.h5", args.val, args.seed + 1)
    write_split(out / "test.h5", args.test, args.seed + 2)
    print(f"Dataset sintético escrito en: {out}")


if __name__ == "__main__":
    main()
