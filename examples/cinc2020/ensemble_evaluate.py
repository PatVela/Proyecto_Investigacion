"""Ensemble de dos checkpoints CINC2020-12 (promedio ponderado de probabilidades).

Tunea umbrales sobre el promedio en validation y evalúa test con esos
umbrales. Acepta las mismas opciones de post-proceso que evaluate.py
(temperaturas por modelo, F-beta, fallback/exclusividad NSR) y escribe el
mismo layout de salida, así que la webapp y compare_models.py lo consumen
sin cambios.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecg import load, predict, util
from ecg.calibration import apply_temperature, load_temperatures_csv
from examples.cinc2020 import evaluate as eval_single


def ensemble_probabilities(prob_a, prob_b, alpha: float = 0.5) -> np.ndarray:
    """Promedio ponderado: alpha * A + (1 - alpha) * B."""
    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"alpha debe estar en [0, 1], recibido {alpha}")
    return a * np.asarray(prob_a, dtype=np.float64) + (1.0 - a) * np.asarray(prob_b, dtype=np.float64)


def _predict_pair(checkpoint_a, checkpoint_b, h5_path, batch_size, num_workers, device, no_amp, preload):
    out_a = predict.predict_hdf5(
        checkpoint_a, h5_path, batch_size=batch_size, num_workers=num_workers,
        device=device, amp=not no_amp, preload=preload,
    )
    out_b = predict.predict_hdf5(
        checkpoint_b, h5_path, batch_size=batch_size, num_workers=num_workers,
        device=device, amp=not no_amp, preload=preload,
    )
    if list(out_a["class_names"]) != list(out_b["class_names"]):
        raise ValueError("Los checkpoints usan distinto orden de clases; no se puede promediar")
    if list(out_a["record_names"]) != list(out_b["record_names"]):
        raise ValueError("Los HDF5 recorridos difieren entre modelos; no se puede promediar")
    return out_a, out_b


def run_ensemble_evaluation(
    config,
    checkpoint_a,
    checkpoint_b,
    output_dir,
    alpha: float = 0.5,
    batch_size=None,
    num_workers=None,
    device: str = "auto",
    no_amp: bool = False,
    normal_fallback_min_prob=None,
    nsr_exclusive_min_prob=None,
    preload: bool = True,
    temperatures_a=None,
    temperatures_b=None,
    threshold_beta: float = 1.0,
) -> dict:
    output_dir = util.resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(batch_size or config.get("batch_size", 8))
    num_workers = int(num_workers if num_workers is not None else config.get("num_workers", 2))

    val_a, val_b = _predict_pair(
        checkpoint_a, checkpoint_b, config["dev"], batch_size, num_workers, device, no_amp, preload
    )
    temp_a = load_temperatures_csv(temperatures_a, val_a["class_names"]) if temperatures_a else None
    temp_b = load_temperatures_csv(temperatures_b, val_b["class_names"]) if temperatures_b else None
    if temp_a is not None:
        print(f"Aplicando temperature scaling (A) desde {temperatures_a}.")
        val_a["probabilities"] = apply_temperature(val_a["probabilities"], temp_a)
    if temp_b is not None:
        print(f"Aplicando temperature scaling (B) desde {temperatures_b}.")
        val_b["probabilities"] = apply_temperature(val_b["probabilities"], temp_b)
    val_prob = ensemble_probabilities(val_a["probabilities"], val_b["probabilities"], alpha)
    class_names = val_a["class_names"]

    print(f"Optimizando umbrales del ensemble (alpha={alpha}) con F-beta (beta={threshold_beta}).")
    thresholds, threshold_df = eval_single.optimize_thresholds(val_a["labels"], val_prob, class_names, beta=threshold_beta)
    threshold_df.to_csv(output_dir / "thresholds_validation.csv", index=False)

    val_class_df, val_global, val_pred = eval_single.evaluate_predictions(
        val_a["labels"], val_prob, thresholds, class_names, "validation",
        normal_fallback_min_prob=normal_fallback_min_prob,
        nsr_exclusive_min_prob=nsr_exclusive_min_prob,
    )

    test_a, test_b = _predict_pair(
        checkpoint_a, checkpoint_b, config["test"], batch_size, num_workers, device, no_amp, preload
    )
    if temp_a is not None:
        test_a["probabilities"] = apply_temperature(test_a["probabilities"], temp_a)
    if temp_b is not None:
        test_b["probabilities"] = apply_temperature(test_b["probabilities"], temp_b)
    test_prob = ensemble_probabilities(test_a["probabilities"], test_b["probabilities"], alpha)
    test_class_df, test_global, test_pred = eval_single.evaluate_predictions(
        test_a["labels"], test_prob, thresholds, class_names, "test",
        normal_fallback_min_prob=normal_fallback_min_prob,
        nsr_exclusive_min_prob=nsr_exclusive_min_prob,
    )

    pd.concat([val_class_df, test_class_df], ignore_index=True).to_csv(output_dir / "metrics_per_class.csv", index=False)
    pd.DataFrame([val_global, test_global]).to_csv(output_dir / "metrics_global.csv", index=False)
    eval_single.save_predictions(output_dir / "predictions_validation.csv", val_a["record_names"], val_a["labels"], val_prob, val_pred, class_names)
    eval_single.save_predictions(output_dir / "predictions_test.csv", test_a["record_names"], test_a["labels"], test_prob, test_pred, class_names)
    eval_single.save_confusion_matrices(output_dir, val_class_df, "validation")
    eval_single.save_confusion_matrices(output_dir, test_class_df, "test")
    eval_single.save_roc_pr_curves(output_dir, val_a["labels"], val_prob, class_names, "validation")
    eval_single.save_roc_pr_curves(output_dir, test_a["labels"], test_prob, class_names, "test")

    summary = {
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "ensemble_alpha": float(alpha),
        "temperatures_a": str(temperatures_a) if temperatures_a else None,
        "temperatures_b": str(temperatures_b) if temperatures_b else None,
        "label_schema": load.LABEL_SCHEMA,
        "problem_type": "multilabel sigmoid + BCEWithLogitsLoss (ensemble promedio ponderado)",
        "class_names": class_names,
        "thresholds": thresholds,
        "threshold_beta": float(threshold_beta),
        "normal_fallback_min_prob": None if normal_fallback_min_prob is None else float(normal_fallback_min_prob),
        "nsr_exclusive_min_prob": None if nsr_exclusive_min_prob is None else float(nsr_exclusive_min_prob),
        "validation": val_global,
        "test": test_global,
    }
    util.save_json(output_dir / "evaluation_summary.json", summary)
    print("\nTEST (ensemble)")
    print(pd.DataFrame([test_global]).to_string(index=False))
    print("\nMétricas por clase")
    print(test_class_df[["index", "class", "support_positive", "sensitivity_recall", "specificity", "f1", "auroc", "auprc"]].to_string(index=False))
    print(f"\nResultados guardados en: {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Ensemble promedio de dos checkpoints CINC2020-12")
    parser.add_argument("config", help="JSON de configuración")
    parser.add_argument("checkpoint_a", help="Checkpoint A (.pt)")
    parser.add_argument("checkpoint_b", help="Checkpoint B (.pt)")
    parser.add_argument("--output-dir", default="ensemble")
    parser.add_argument("--alpha", type=float, default=0.5, help="Peso del modelo A en [0, 1] (B recibe 1-alpha)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-preload", action="store_true", help="Desactiva la precarga del HDF5 a RAM (lectura por disco, lento)")
    parser.add_argument("--normal-fallback-min-prob", type=float, default=None)
    parser.add_argument("--nsr-exclusive-min-prob", type=float, default=None, help="Si P(NSR) >= valor, predecir SOLO NSR. AVISO: en ablación redujo F1-macro sin mejorar precisión; no recomendado.")
    parser.add_argument("--temperatures-a", default=None, help="CSV class,temperature de calibrate.py para el modelo A")
    parser.add_argument("--temperatures-b", default=None, help="CSV class,temperature de calibrate.py para el modelo B")
    parser.add_argument("--threshold-beta", type=float, default=1.0, help="Beta del F-beta para optimizar umbrales: 1.0=F1, 0.5=menos FPs, 2.0=menos FNs")
    args = parser.parse_args()

    config = util.load_json(args.config)
    run_ensemble_evaluation(
        config,
        args.checkpoint_a,
        args.checkpoint_b,
        args.output_dir,
        alpha=args.alpha,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        no_amp=args.no_amp,
        normal_fallback_min_prob=args.normal_fallback_min_prob,
        nsr_exclusive_min_prob=args.nsr_exclusive_min_prob,
        preload=not args.no_preload,
        temperatures_a=args.temperatures_a,
        temperatures_b=args.temperatures_b,
        threshold_beta=args.threshold_beta,
    )


if __name__ == "__main__":
    main()
