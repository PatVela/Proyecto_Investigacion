"""Calibra probabilidades multilabel por clase, ajustando solo sobre validación (minimiza la NLL).

``--method temperature`` (defecto): una temperatura T por clase. Guarda ``temperatures_validation.csv`` y
``calibration_report.json`` (ECE/Brier antes/después); se aplican con ``evaluate.py --temperatures`` o en la webapp.

``--method platt``: calibración afín ``sigmoid(a·logit + b)`` por clase. Guarda ``platt_validation.csv`` y
``calibration_report_platt.json`` (no pisa los archivos de la temperatura) y se aplica con ``evaluate.py --platt``,
``ensemble_evaluate.py --platt-a/--platt-b`` o en la webapp. Con ``--checkpoint`` se ajusta sobre los logits crudos
del modelo; con ``--predictions`` solo hay probabilidades y se usa su logit recortado (menos preciso: avisa).

Ejemplos:
    python examples/cinc2020/calibrate.py \\
      --checkpoint <run>/best.pt --val-h5 data/cinc2020_12/val.h5 \\
      --output-dir <dir-calibracion>
    python examples/cinc2020/calibrate.py \\
      --predictions <dir-evaluacion>/predictions_validation.csv \\
      --output-dir <dir-calibracion>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecg import util
from ecg.calibration import (
    apply_platt_logits,
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_platt_per_class,
    fit_temperature_per_class,
    save_platt_csv,
    save_temperatures_csv,
    to_logits,
)


def load_predictions_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Lee predictions_*.csv (columnas true_*/prob_*) y devuelve y_true, y_prob, clases."""
    df = pd.read_csv(path)
    classes = [c[len("true_"):] for c in df.columns if c.startswith("true_")]
    if not classes:
        raise ValueError(f"{path} no tiene columnas true_<clase>.")
    y_true = df[[f"true_{c}" for c in classes]].to_numpy(dtype=np.float64)
    y_prob = df[[f"prob_{c}" for c in classes]].to_numpy(dtype=np.float64)
    return y_true, y_prob, classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibración por clase sobre validación (temperatura o afín).")
    parser.add_argument("--method", choices=["temperature", "platt"], default="temperature",
                        help="temperature: una T por clase (defecto). platt: sigmoid(a·logit + b) por clase")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint .pt (modo HDF5)")
    parser.add_argument("--val-h5", default=None, help="HDF5 de validación (modo HDF5)")
    parser.add_argument("--predictions", default=None, help="predictions_validation.csv (modo CSV)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-preload", action="store_true", help="Desactiva la precarga del HDF5 a RAM (lento)")
    args = parser.parse_args()

    logits = None
    if args.predictions:
        y_true, y_prob, class_names = load_predictions_csv(args.predictions)
        source = str(args.predictions)
        if args.method == "platt":
            print("AVISO: --predictions solo trae probabilidades; se ajusta sobre su logit recortado. "
                  "Para el ajuste definitivo use --checkpoint + --val-h5 (logits crudos).")
    else:
        if not args.checkpoint or not args.val_h5:
            parser.error("Indique --checkpoint + --val-h5 o bien --predictions.")
        from ecg import predict as predict_mod

        result = predict_mod.predict_hdf5(
            args.checkpoint, args.val_h5, batch_size=args.batch_size,
            num_workers=args.num_workers, device=args.device, amp=not args.no_amp,
            preload=not args.no_preload,
        )
        y_true = np.asarray(result["labels"], dtype=np.float64)
        y_prob = np.asarray(result["probabilities"], dtype=np.float64)
        class_names = list(result["class_names"])
        source = str(args.val_h5)
        if "logits" in result:
            logits = np.asarray(result["logits"], dtype=np.float64)

    out_dir = util.resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "platt":
        if logits is None:
            logits = to_logits(y_prob)
        a, b = fit_platt_per_class(y_true, logits)
        y_cal = apply_platt_logits(logits, a, b)
        report = {
            "method": "platt",
            "source": source,
            "logits_source": "checkpoint" if args.checkpoint and not args.predictions else "logit recortado de predictions",
            "n_samples": int(y_true.shape[0]),
            "n_classes": int(y_true.shape[1]),
            "ece_before": expected_calibration_error(y_true, y_prob),
            "ece_after": expected_calibration_error(y_true, y_cal),
            "brier_before": brier_score(y_true, y_prob),
            "brier_after": brier_score(y_true, y_cal),
            "platt": {name: {"a": float(ai), "b": float(bi)} for name, ai, bi in zip(class_names, a, b)},
        }
        save_platt_csv(out_dir / "platt_validation.csv", class_names, a, b)
        util.save_json(out_dir / "calibration_report_platt.json", report)
        table = pd.DataFrame({"class": class_names, "a": np.round(a, 4), "b": np.round(b, 4)})
        print(table.to_string(index=False))
        print(f"\nECE: {report['ece_before']:.4f} -> {report['ece_after']:.4f} | "
              f"Brier: {report['brier_before']:.4f} -> {report['brier_after']:.4f}")
        print(f"Resultados guardados en: {out_dir}")
        return

    temps = fit_temperature_per_class(y_true, y_prob)
    y_cal = apply_temperature(y_prob, temps)
    report = {
        "source": source,
        "n_samples": int(y_true.shape[0]),
        "n_classes": int(y_true.shape[1]),
        "ece_before": expected_calibration_error(y_true, y_prob),
        "ece_after": expected_calibration_error(y_true, y_cal),
        "brier_before": brier_score(y_true, y_prob),
        "brier_after": brier_score(y_true, y_cal),
        "temperatures": {name: float(t) for name, t in zip(class_names, temps)},
    }

    save_temperatures_csv(out_dir / "temperatures_validation.csv", class_names, temps)
    util.save_json(out_dir / "calibration_report.json", report)

    table = pd.DataFrame({"class": class_names, "temperature": np.round(temps, 4)})
    print(table.to_string(index=False))
    print(f"\nECE: {report['ece_before']:.4f} -> {report['ece_after']:.4f} | "
          f"Brier: {report['brier_before']:.4f} -> {report['brier_after']:.4f}")
    print(f"Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()
