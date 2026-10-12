"""Diagnóstico integral de predicción CINC2020-12.

Verifica, en orden, las causas más frecuentes de "predice mal":

1. checkpoint: metadatos, esquema (v1/v2), clases, candidatos en saved/;
2. thresholds: existencia, cobertura de clases, rango;
3. HDF5: esquemas de train/val/test y compatibilidad con el checkpoint;
4. calibración: media de P por clase vs prevalencia en val, tasa all-negativo;
5. registro puntual (--record/--hea/--mat): pipeline webapp vs HDF5.

Ejemplo::

    python examples/cinc2020/diagnose_prediction.py \\
      --checkpoint saved/cinc2020/.../best.pt \\
      --config examples/cinc2020/config.json \\
      --thresholds results/.../thresholds_validation.csv \\
      --record E00001 --hea dataset2020/.../E00001.hea --mat dataset2020/.../E00001.mat
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecg import load, predict, util  # noqa: E402
from webapp.prediction import (  # noqa: E402
    apply_normal_fallback,
    build_label_comparison,
    threshold_values_for_class_names,
)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def check_checkpoint(path: str) -> tuple[dict, list[str]]:
    ckpt_path = util.resolve_path(path)
    ckpt = util.load_checkpoint(ckpt_path, map_location="cpu")
    class_names = list(ckpt.get("class_names", []))
    print(f"Checkpoint : {ckpt_path}")
    print(f"Época      : {ckpt.get('epoch')}  val_loss={ckpt.get('val_loss')}")
    print(f"Esquema    : {ckpt.get('label_schema')}")
    print(f"Clases     : {class_names}")

    siblings = util.list_checkpoints_info(ckpt_path.parent.parent)
    print(f"\nCheckpoints hermanos bajo {ckpt_path.parent.parent} (rank 0 = auto-seleccionado):")
    for row in siblings[:8]:
        mark = " <-- ESTE" if str(ckpt_path) == row.get("path") else ""
        print(f"  rank={row.get('rank')} val_loss={row.get('val_loss')} schema={row.get('label_schema')} :: {row.get('path')}{mark}")
    if siblings and str(ckpt_path) != siblings[0].get("path"):
        print("AVISO: la webapp en modo auto NO elegiría este checkpoint sino el rank 0.")
    return ckpt, class_names


def check_thresholds(path: str | None, class_names: list[str]) -> np.ndarray:
    if not path:
        print("Thresholds : (no indicados) se usará fallback global 0.5")
        return np.full(len(class_names), 0.5, dtype=np.float32)
    p = util.resolve_path(path)
    df = pd.read_csv(p)
    mapping = {str(r["class"]): float(r["threshold"]) for _, r in df.iterrows()}
    missing = [c for c in class_names if c not in mapping]
    print(f"Thresholds : {p}")
    print(df.to_string(index=False))
    if missing:
        print(f"AVISO: faltan thresholds para {missing}; esas clases usarán 0.5")
    bad = {k: v for k, v in mapping.items() if not (0 < v < 1)}
    if bad:
        print(f"AVISO: thresholds fuera de (0,1): {bad}")
    return threshold_values_for_class_names(class_names, threshold=0.5, thresholds=mapping)


def check_hdf5(h5_path: str, label: str) -> dict | None:
    import h5py
    p = util.resolve_path(h5_path)
    if not p.exists():
        print(f"{label}: NO EXISTE ({p})")
        return None
    with h5py.File(p, "r") as h5:
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in h5.attrs.items()}
        n = int(h5["signals"].shape[0])
        labels = np.asarray(h5["labels"][:], dtype=np.float32)
    prev = labels.mean(axis=0)
    print(f"{label}: {p} N={n} schema={attrs.get('label_schema')} norm={attrs.get('norm_mode', attrs.get('normalization'))}")
    print(f"  clases={list(attrs.get('classes', []))}")
    print(f"  all-zero={(labels.sum(axis=1) == 0).mean() * 100:.1f}%  prevalencias={np.round(prev, 4).tolist()}")
    return {"path": str(p), "n": n, "attrs": attrs, "prevalence": prev}


def calibration_spotcheck(checkpoint_path: str, h5_path: str, thresholds: np.ndarray, device: str, max_n: int = 800) -> None:
    if int(max_n) <= 0:
        print("Spot-check de calibración omitido (--max-calibration 0).")
        return
    import h5py
    import torch
    torch_device = predict.get_device(device)
    model, ckpt, class_names = predict.load_model(checkpoint_path, torch_device)
    with h5py.File(util.resolve_path(h5_path), "r") as h5:
        n = min(int(h5["signals"].shape[0]), max_n)
        x = np.asarray(h5["signals"][:n], dtype=np.float32).transpose(0, 2, 1)
        y = np.asarray(h5["labels"][:n], dtype=np.float32)
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, n, 32):
            t = torch.from_numpy(x[s:s + 32]).to(torch_device)
            probs.append(torch.sigmoid(model(t)).float().cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    pred = (probs >= thresholds[None, :]).astype(np.uint8)
    print(f"Muestras: {n} | P media global={probs.mean():.4f} | all-negativo predicho={(pred.sum(axis=1) == 0).mean() * 100:.1f}%")
    print(f"{'clase':<22} {'Pmedia':>8} {'Pmax':>8} {'pos_rate':>8} {'preval':>8} {'thr':>7}")
    for i, name in enumerate(class_names):
        print(f"{name:<22} {probs[:, i].mean():8.4f} {probs[:, i].max():8.4f} "
              f"{pred[:, i].mean():8.4f} {y[:, i].mean():8.4f} {float(thresholds[i]):7.3f}")
    flat = probs.ravel()
    print(f"\nHistograma de P: <0.1={(flat < 0.1).mean() * 100:.1f}%  0.1-0.5={( (flat >= 0.1) & (flat < 0.5)).mean() * 100:.1f}% "
          f"0.5-0.9={((flat >= 0.5) & (flat < 0.9)).mean() * 100:.1f}%  >0.9={(flat >= 0.9).mean() * 100:.1f}%")


def debug_record(args, class_names: list[str], thresholds: np.ndarray, pre: dict) -> None:
    from examples.cinc2020._record_diag import find_record_in_hdf5, rows_from_probabilities, print_table
    config = json.loads(util.resolve_path(args.config).read_text(encoding="utf-8"))
    device = predict.get_device(args.device)
    model, _ckpt, _names = predict.load_model(args.checkpoint, device=device)
    if args.hea and args.mat:
        signal, meta = load.load_wfdb_record(args.mat, args.hea, norm_mode=pre["norm_mode"], bandpass=pre["bandpass"])
        print(f"\nHeader Dx={meta.get('dx_codes')} fs={meta.get('sampling_freq')} leads={meta.get('lead_names')}")
        print(f"Señal procesada: shape={signal.shape} media={signal.mean():.4f} std={signal.std():.4f} "
              f"min={signal.min():.3f} max={signal.max():.3f}")
        true_classes = load.matched_class_names(meta.get("dx_codes", []))
        if "LAD" in class_names and "AxisDev" not in class_names:
            true_classes = [{"AxisDev": "LAD", "BBB": "RBBB"}.get(c, c) for c in true_classes]
        prob = predict.predict_array(model, signal, device=device)
        rows, raw_pos, final_pos, fallback = rows_from_probabilities(prob, class_names, thresholds, args.normal_fallback_min_prob)
        print_table("WEBAPP/WFDB", rows)
        cmp = build_label_comparison(true_classes, rows, "wfdb_header_dx", dx_codes=meta.get("dx_codes", []))
        print("\nTrue:", true_classes or "ninguna", "| Raw:", [r["class"] for r in raw_pos] or "ninguna",
              "| Final:", [r["class"] for r in final_pos] or "ninguna")
        print("Fallback:", fallback, "| exact:", cmp.get("exact_match"), "F1:", cmp.get("f1"))
    if args.record:
        item = find_record_in_hdf5(config, args.record)
        if item is None:
            print(f"\nRegistro {args.record} no encontrado en HDF5.")
        else:
            print(f"\nHDF5 {item['split']} idx={item['index']} clases_h5={item.get('h5_classes')}")
            prob = predict.predict_array(model, item["signal"], device=device)
            rows, raw_pos, final_pos, fallback = rows_from_probabilities(prob, class_names, thresholds, args.normal_fallback_min_prob)
            print_table(f"HDF5 {item['split']}", rows)
            cmp = build_label_comparison(item["true_classes"], rows, "hdf5_label")
            print("\nTrue:", item["true_classes"] or "ninguna", "| Final:", [r["class"] for r in final_pos] or "ninguna",
                  "| exact:", cmp.get("exact_match"), "F1:", cmp.get("f1"))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico integral de predicción CINC2020-12")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--record", default=None)
    parser.add_argument("--hea", default=None)
    parser.add_argument("--mat", default=None)
    parser.add_argument("--normal-fallback-min-prob", type=float, default=0.40)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-calibration", type=int, default=800)
    args = parser.parse_args(argv)

    section("1) CHECKPOINT")
    ckpt, class_names = check_checkpoint(args.checkpoint)
    pre = predict.preprocessing_for_checkpoint(ckpt)
    print(f"Preprocesamiento compatible: norm={pre['norm_mode']} bandpass={pre['bandpass']}")

    section("2) THRESHOLDS")
    thresholds = check_thresholds(args.thresholds, class_names)

    section("3) HDF5 (config)")
    config = json.loads(util.resolve_path(args.config).read_text(encoding="utf-8"))
    infos = {}
    for key, label in (("train", "TRAIN"), ("dev", "VAL"), ("test", "TEST")):
        if config.get(key):
            infos[key] = check_hdf5(config[key], label)
    for key, info in infos.items():
        if info and info["attrs"].get("label_schema") not in (None, ckpt.get("label_schema")):
            print(f"AVISO: {key}.h5 schema={info['attrs'].get('label_schema')} != checkpoint {ckpt.get('label_schema')}")

    section("4) CALIBRACIÓN EN VAL (spot-check)")
    if config.get("dev") and util.resolve_path(config["dev"]).exists():
        calibration_spotcheck(args.checkpoint, config["dev"], thresholds, args.device, args.max_calibration)
    else:
        print("Sin val.h5; se omite.")

    if args.record or (args.hea and args.mat):
        section("5) REGISTRO PUNTUAL")
        debug_record(args, class_names, thresholds, pre)

    _ = apply_normal_fallback  # re-exportado para uso programático
    print("\nDiagnóstico terminado.")


if __name__ == "__main__":
    main()
