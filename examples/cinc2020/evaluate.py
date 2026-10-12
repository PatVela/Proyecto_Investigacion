"""Evaluación multilabel de un checkpoint CINC2020-12.

Optimiza thresholds por clase en validation y evalúa test con esos thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecg import load, predict, util
from ecg.calibration import apply_temperature, load_temperatures_csv


def safe_metric(fn, y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(fn(y_true, y_score))
    except Exception:
        return np.nan


def safe_macro(y_true, y_prob, fn):
    values = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        try:
            values.append(float(fn(y_true[:, i], y_prob[:, i])))
        except Exception:
            pass
    return float(np.mean(values)) if values else np.nan


def best_threshold_fbeta(y_true, y_prob, beta=1.0):
    """Umbral que maximiza F-beta en validación.

    beta=1.0 reproduce F1 (equilibrio); beta<1.0 (p. ej. 0.5) pesa más la
    precisión → menos falsos positivos; beta>1.0 pesa más el recall.
    """
    if y_true.sum() == 0:
        return 0.5, 0.0
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        score = fbeta_score(y_true, (y_prob >= t).astype(np.uint8), beta=float(beta), zero_division=0)
        if score > best_score:
            best_t, best_score = float(t), float(score)
    return best_t, best_score


def best_threshold_f1(y_true, y_prob):
    return best_threshold_fbeta(y_true, y_prob, beta=1.0)


def optimize_thresholds(y_true, y_prob, class_names, beta=1.0):
    thresholds = np.zeros(y_true.shape[1], dtype=np.float32)
    rows = []
    for i, name in enumerate(class_names):
        yt = y_true[:, i].astype(np.uint8)
        t, score = best_threshold_fbeta(yt, y_prob[:, i], beta=beta)
        thresholds[i] = t
        rows.append({
            "index": i,
            "class": name,
            "threshold": t,
            "validation_f1": float(f1_score(yt, (y_prob[:, i] >= t).astype(np.uint8), zero_division=0)),
            "validation_objective": score,
            "validation_positives": int(yt.sum()),
        })
    return thresholds, pd.DataFrame(rows)


def fixed_thresholds(y_true, y_prob, class_names, fixed=0.5):
    """Umbral fijo para todas las clases (ablaciones; desactiva la optimización)."""
    thresholds = np.full(y_true.shape[1], float(fixed), dtype=np.float32)
    rows = []
    for i, name in enumerate(class_names):
        yt = y_true[:, i].astype(np.uint8)
        pred = (y_prob[:, i] >= float(fixed)).astype(np.uint8)
        f1 = float(f1_score(yt, pred, zero_division=0))
        rows.append({
            "index": i,
            "class": name,
            "threshold": float(fixed),
            "validation_f1": f1,
            "validation_objective": f1,
            "validation_positives": int(yt.sum()),
        })
    return thresholds, pd.DataFrame(rows)


def apply_normal_fallback(y_pred, y_prob, class_names, min_prob=None):
    """Añade NSR cuando ninguna clase supera umbral y P(NSR) es razonable."""
    if min_prob is None or float(min_prob) <= 0 or "NSR" not in class_names:
        return y_pred, 0
    y_pred = y_pred.copy()
    nsr_idx = list(class_names).index("NSR")
    empty = y_pred.sum(axis=1) == 0
    mask = empty & (y_prob[:, nsr_idx] >= float(min_prob))
    y_pred[mask, nsr_idx] = 1
    return y_pred, int(mask.sum())


def apply_nsr_exclusive(y_pred, y_prob, class_names, min_prob=None):
    """Si P(NSR) es muy alta, predice SOLO NSR (suprime otros positivos).

    Reduce falsos positivos en trazados normales: un NSR con probabilidad
    alta es incompatible con patologías simultáneas.
    """
    if min_prob is None or float(min_prob) <= 0 or "NSR" not in class_names:
        return y_pred, 0
    y_pred = y_pred.copy()
    nsr_idx = list(class_names).index("NSR")
    mask = y_prob[:, nsr_idx] >= float(min_prob)
    y_pred[mask, :] = 0
    y_pred[mask, nsr_idx] = 1
    return y_pred, int(mask.sum())


def evaluate_predictions(y_true, y_prob, thresholds, class_names, split, normal_fallback_min_prob=None, nsr_exclusive_min_prob=None):
    y_true = y_true.astype(np.uint8)
    y_pred = (y_prob >= thresholds[None, :]).astype(np.uint8)
    y_pred, normal_fallback_count = apply_normal_fallback(y_pred, y_prob, class_names, normal_fallback_min_prob)
    y_pred, nsr_exclusive_count = apply_nsr_exclusive(y_pred, y_prob, class_names, nsr_exclusive_min_prob)
    rows = []
    for i, name in enumerate(class_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
        rows.append({
            "split": split,
            "index": i,
            "class": name,
            "threshold": float(thresholds[i]),
            "support_positive": int(y_true[:, i].sum()),
            "support_negative": int(y_true.shape[0] - y_true[:, i].sum()),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "sensitivity_recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "specificity": specificity,
            "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "auroc": safe_metric(roc_auc_score, y_true[:, i], y_prob[:, i]),
            "auprc": safe_metric(average_precision_score, y_true[:, i], y_prob[:, i]),
        })
    class_df = pd.DataFrame(rows)
    global_metrics = {
        "split": split,
        "samples": int(y_true.shape[0]),
        "classes": int(y_true.shape[1]),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "sensitivity_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "specificity_macro": float(class_df["specificity"].mean()),
        "auroc_macro": safe_macro(y_true, y_prob, roc_auc_score),
        "auprc_macro": safe_macro(y_true, y_prob, average_precision_score),
        "exact_match_ratio": float((y_true == y_pred).all(axis=1).mean()),
        "normal_fallback_min_prob": None if normal_fallback_min_prob is None else float(normal_fallback_min_prob),
        "normal_fallback_count": int(normal_fallback_count),
        "nsr_exclusive_min_prob": None if nsr_exclusive_min_prob is None else float(nsr_exclusive_min_prob),
        "nsr_exclusive_count": int(nsr_exclusive_count),
    }
    return class_df, global_metrics, y_pred


def save_predictions(path, record_names, y_true, y_prob, y_pred, class_names):
    payload = {"record_name": record_names}
    for i, name in enumerate(class_names):
        payload[f"true_{name}"] = y_true[:, i].astype(np.uint8)
        payload[f"prob_{name}"] = y_prob[:, i].astype(np.float32)
        payload[f"pred_{name}"] = y_pred[:, i].astype(np.uint8)
    pd.DataFrame(payload).to_csv(path, index=False)


def save_confusion_matrices(out_dir, class_df, split):
    cm_dir = Path(out_dir) / "confusion_matrices" / split
    cm_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in class_df.iterrows():
        matrix = pd.DataFrame(
            [[int(row["TN"]), int(row["FP"])], [int(row["FN"]), int(row["TP"])]],
            index=["true_0", "true_1"],
            columns=["pred_0", "pred_1"],
        )
        safe = str(row["class"]).replace("/", "_").replace(" ", "_")
        matrix.to_csv(cm_dir / f"{int(row['index']):02d}_{safe}.csv")
        rows.append({"split": split, "class": row["class"], "TN": int(row["TN"]), "FP": int(row["FP"]), "FN": int(row["FN"]), "TP": int(row["TP"])})
    pd.DataFrame(rows).to_csv(Path(out_dir) / f"confusion_matrices_{split}.csv", index=False)


def save_roc_pr_curves(out_dir, y_true, y_prob, class_names, split, max_points=200):
    """Guarda curvas ROC y PR por clase (submuestreadas a ``max_points``).

    Escribe ``roc_curves_<split>.csv`` (split,class,fpr,tpr,threshold) y
    ``pr_curves_<split>.csv`` (split,class,precision,recall,threshold).
    Las clases degeneradas (sin positivos o sin negativos) se omiten.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    roc_rows, pr_rows = [], []
    for i, name in enumerate(class_names):
        col, prob = y_true[:, i], y_prob[:, i]
        if col.min() == col.max():
            continue
        try:
            fpr, tpr, thr = roc_curve(col, prob)
            precision, recall, thr_pr = precision_recall_curve(col, prob)
        except Exception:
            continue
        idx = np.linspace(0, len(fpr) - 1, min(len(fpr), max_points)).round().astype(int)
        for j in idx:
            roc_rows.append({"split": split, "class": name, "fpr": float(fpr[j]),
                             "tpr": float(tpr[j]), "threshold": float(thr[min(j, len(thr) - 1)])})
        idx = np.linspace(0, len(precision) - 1, min(len(precision), max_points)).round().astype(int)
        for j in idx:
            pr_rows.append({"split": split, "class": name, "precision": float(precision[j]),
                            "recall": float(recall[j]),
                            "threshold": float(thr_pr[min(j, len(thr_pr) - 1)])})
    out_dir = Path(out_dir)
    pd.DataFrame(roc_rows).to_csv(out_dir / f"roc_curves_{split}.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(out_dir / f"pr_curves_{split}.csv", index=False)
    return len(roc_rows), len(pr_rows)


def run_evaluation(config, checkpoint, output_dir, batch_size=None, num_workers=None, device="auto", no_amp=False, normal_fallback_min_prob=None, preload=True, temperatures=None, threshold_beta=1.0, nsr_exclusive_min_prob=None, fixed_threshold=None):
    output_dir = util.resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_h5 = config["dev"]
    test_h5 = config["test"]
    batch_size = int(batch_size or config.get("batch_size", 8))
    num_workers = int(num_workers if num_workers is not None else config.get("num_workers", 2))

    val = predict.predict_hdf5(checkpoint, val_h5, batch_size=batch_size, num_workers=num_workers, device=device, amp=not no_amp, preload=preload)
    temp_vec = load_temperatures_csv(temperatures, val["class_names"]) if temperatures else None
    if temp_vec is not None:
        print(f"Aplicando temperature scaling desde {temperatures}.")
        val["probabilities"] = apply_temperature(val["probabilities"], temp_vec)
    if fixed_threshold is not None:
        print(f"Usando umbral fijo={fixed_threshold} (sin optimización).")
        thresholds, threshold_df = fixed_thresholds(val["labels"], val["probabilities"], val["class_names"], fixed=float(fixed_threshold))
    else:
        print(f"Optimizando umbrales por clase con F-beta (beta={threshold_beta}).")
        thresholds, threshold_df = optimize_thresholds(val["labels"], val["probabilities"], val["class_names"], beta=threshold_beta)
    threshold_df.to_csv(output_dir / "thresholds_validation.csv", index=False)

    val_class_df, val_global, val_pred = evaluate_predictions(
        val["labels"], val["probabilities"], thresholds, val["class_names"], "validation",
        normal_fallback_min_prob=normal_fallback_min_prob,
        nsr_exclusive_min_prob=nsr_exclusive_min_prob,
    )

    test = predict.predict_hdf5(checkpoint, test_h5, batch_size=batch_size, num_workers=num_workers, device=device, amp=not no_amp, preload=preload)
    if temp_vec is not None:
        test["probabilities"] = apply_temperature(test["probabilities"], temp_vec)
    test_class_df, test_global, test_pred = evaluate_predictions(
        test["labels"], test["probabilities"], thresholds, test["class_names"], "test",
        normal_fallback_min_prob=normal_fallback_min_prob,
        nsr_exclusive_min_prob=nsr_exclusive_min_prob,
    )

    pd.concat([val_class_df, test_class_df], ignore_index=True).to_csv(output_dir / "metrics_per_class.csv", index=False)
    pd.DataFrame([val_global, test_global]).to_csv(output_dir / "metrics_global.csv", index=False)
    save_predictions(output_dir / "predictions_validation.csv", val["record_names"], val["labels"], val["probabilities"], val_pred, val["class_names"])
    save_predictions(output_dir / "predictions_test.csv", test["record_names"], test["labels"], test["probabilities"], test_pred, test["class_names"])
    save_confusion_matrices(output_dir, val_class_df, "validation")
    save_confusion_matrices(output_dir, test_class_df, "test")
    save_roc_pr_curves(output_dir, val["labels"], val["probabilities"], val["class_names"], "validation")
    save_roc_pr_curves(output_dir, test["labels"], test["probabilities"], test["class_names"], "test")

    summary = {
        "checkpoint": str(checkpoint),
        "temperatures": str(temperatures) if temperatures else None,
        "label_schema": load.LABEL_SCHEMA,
        "problem_type": "multilabel sigmoid + BCEWithLogitsLoss",
        "class_names": val["class_names"],
        "thresholds": thresholds,
        "threshold_beta": float(threshold_beta),
        "fixed_threshold": None if fixed_threshold is None else float(fixed_threshold),
        "normal_fallback_min_prob": None if normal_fallback_min_prob is None else float(normal_fallback_min_prob),
        "nsr_exclusive_min_prob": None if nsr_exclusive_min_prob is None else float(nsr_exclusive_min_prob),
        "validation": val_global,
        "test": test_global,
    }
    util.save_json(output_dir / "evaluation_summary.json", summary)
    print("\nTEST")
    print(pd.DataFrame([test_global]).to_string(index=False))
    print("\nMétricas por clase")
    print(test_class_df[["index", "class", "support_positive", "sensitivity_recall", "specificity", "f1", "auroc", "auprc"]].to_string(index=False))
    print(f"\nResultados guardados en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evalúa un checkpoint CINC2020-12")
    parser.add_argument("config", help="JSON de configuración")
    parser.add_argument("checkpoint", nargs="?", default=None, help="Checkpoint .pt; si se omite busca best.pt en save_dir")
    parser.add_argument("--output-dir", default="results/cinc2020_12")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-preload", action="store_true", help="Desactiva la precarga del HDF5 a RAM (lectura por disco, lento)")
    parser.add_argument("--normal-fallback-min-prob", type=float, default=None, help="Postprocesamiento opcional: si no hay positivos, añadir NSR cuando P(NSR) >= valor. Ejemplo: 0.40")
    parser.add_argument("--temperatures", default=None, help="CSV class,temperature de calibrate.py; se aplica antes de optimizar umbrales y métricas")
    parser.add_argument("--threshold-beta", type=float, default=1.0, help="Beta del F-beta para optimizar umbrales: 1.0=F1, 0.5=menos FPs, 2.0=menos FNs")
    parser.add_argument("--fixed-threshold", type=float, default=None, help="Umbral fijo para todas las clases (desactiva la optimización; para ablaciones)")
    parser.add_argument("--nsr-exclusive-min-prob", type=float, default=None, help="Si P(NSR) >= valor, predecir SOLO NSR. AVISO: en ablación redujo F1-macro sin mejorar precisión; no recomendado.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    checkpoint = args.checkpoint
    if checkpoint is None:
        best = util.best_checkpoint(util.resolve_path(config.get("save_dir", "saved/cinc2020")))
        if best is None:
            raise FileNotFoundError("No se indicó checkpoint y no se encontró ninguno en save_dir")
        checkpoint = best

    run_evaluation(
        config,
        checkpoint,
        args.output_dir,
        args.batch_size,
        args.num_workers,
        args.device,
        args.no_amp,
        normal_fallback_min_prob=args.normal_fallback_min_prob,
        preload=not args.no_preload,
        temperatures=args.temperatures,
        threshold_beta=args.threshold_beta,
        fixed_threshold=args.fixed_threshold,
        nsr_exclusive_min_prob=args.nsr_exclusive_min_prob,
    )


if __name__ == "__main__":
    main()
