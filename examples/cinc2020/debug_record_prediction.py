"""Diagnóstico de una predicción individual CINC2020-12.

Sirve para responder si una mala predicción proviene de:
  1) el modelo/checkpoint;
  2) una diferencia entre preprocesamiento HDF5 y preprocesamiento de la webapp;
  3) umbrales/postprocesamiento.

Ejemplo:

python examples/cinc2020/debug_record_prediction.py \
  --config examples/cinc2020/config.json \
  --checkpoint saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  --record E00001 \
  --hea dataset2020/training/georgia/g1/E00001.hea \
  --mat dataset2020/training/georgia/g1/E00001.mat \
  --thresholds results/cinc2020_12_resnet/thresholds_validation.csv \
  --normal-fallback-min-prob 0.40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecg import load, predict, util  # noqa: E402
from examples.cinc2020._record_diag import (
    find_record_in_hdf5,
    print_table,
    read_thresholds,
    rows_from_probabilities,
)
from webapp.prediction import (
    build_label_comparison,
    threshold_values_for_class_names,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Depura una predicción CINC2020-12 por registro")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--record", required=True, help="record_name, por ejemplo E00001")
    parser.add_argument("--hea", default=None)
    parser.add_argument("--mat", default=None)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--normal-fallback-min-prob", type=float, default=0.40)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args(argv)

    config = json.loads(util.resolve_path(args.config).read_text(encoding="utf-8"))
    device = predict.get_device(args.device)
    model, checkpoint, class_names = predict.load_model(args.checkpoint, device=device)
    pre = predict.preprocessing_for_checkpoint(checkpoint)
    thresholds_dict = read_thresholds(args.thresholds, class_names)
    thresholds = threshold_values_for_class_names(class_names, threshold=0.5, thresholds=thresholds_dict)

    print("=" * 78)
    print("DEBUG PREDICCIÓN CINC2020-12")
    print("=" * 78)
    print("Registro     :", args.record)
    print("Checkpoint   :", util.resolve_path(args.checkpoint))
    print("Esquema ckpt :", checkpoint.get("label_schema"), "| norm:", pre["norm_mode"], "| bandpass:", pre["bandpass"])
    print("Época/val_loss:", checkpoint.get("epoch"), checkpoint.get("val_loss"))
    print("Device       :", device)
    print("Thresholds   :", args.thresholds or "fallback global 0.5")
    print("Fallback NSR :", args.normal_fallback_min_prob)

    h5_item = find_record_in_hdf5(config, args.record)
    wfdb_signal = None

    if args.hea and args.mat:
        wfdb_signal, meta = load.load_wfdb_record(
            args.mat, args.hea, norm_mode=pre["norm_mode"], bandpass=pre["bandpass"])
        true_classes = load.matched_class_names(meta.get("dx_codes", []))
        if "LAD" in class_names and "AxisDev" not in class_names:
            true_classes = [{"AxisDev": "LAD", "BBB": "RBBB"}.get(c, c) for c in true_classes]
        prob = predict.predict_array(model, wfdb_signal, device=device)
        rows, raw_pos, final_pos, fallback = rows_from_probabilities(prob, class_names, thresholds, args.normal_fallback_min_prob)
        print_table("WEBAPP/WFDB preprocesado", rows)
        cmp = build_label_comparison(true_classes, rows, "wfdb_header_dx", dx_codes=meta.get("dx_codes", []))
        print("\nEtiqueta header:", true_classes or "Ninguna de las 12")
        print("Raw positives :", [r["class"] for r in raw_pos] or "Ninguna")
        print("Final positives:", [r["class"] for r in final_pos] or "Ninguna")
        print("Fallback      :", fallback)
        print("Exact match   :", cmp.get("exact_match"), "F1=", cmp.get("f1"), "Jaccard=", cmp.get("jaccard"))

    if h5_item is not None:
        print(f"\nHDF5 {h5_item['path']} clases={h5_item.get('h5_classes')}")
        prob = predict.predict_array(model, h5_item["signal"], device=device)
        rows, raw_pos, final_pos, fallback = rows_from_probabilities(prob, class_names, thresholds, args.normal_fallback_min_prob)
        print_table(f"HDF5 {h5_item['split']} index={h5_item['index']}", rows)
        cmp = build_label_comparison(h5_item["true_classes"], rows, "hdf5_label")
        print("\nEtiqueta HDF5 :", h5_item["true_classes"] or "Ninguna")
        print("Raw positives :", [r["class"] for r in raw_pos] or "Ninguna")
        print("Final positives:", [r["class"] for r in final_pos] or "Ninguna")
        print("Fallback      :", fallback)
        print("Exact match   :", cmp.get("exact_match"), "F1=", cmp.get("f1"), "Jaccard=", cmp.get("jaccard"))
    else:
        print("\nRegistro no encontrado en HDF5 train/validation/test configurados.")

    if wfdb_signal is not None and h5_item is not None:
        diff = np.asarray(wfdb_signal) - np.asarray(h5_item["signal"])
        print("\nComparación señal webapp vs HDF5")
        print("  max_abs:", float(np.max(np.abs(diff))))
        print("  mean_abs:", float(np.mean(np.abs(diff))))
        print("  shape  :", wfdb_signal.shape, h5_item["signal"].shape)
        if float(np.max(np.abs(diff))) < 1e-5:
            print("  Conclusión: el preprocesamiento coincide; el error viene del modelo/umbral/postproceso.")
        else:
            print("  Conclusión: hay diferencia de preprocesamiento; revisar lectura/remuestreo/normalización.")


if __name__ == "__main__":
    main()
