"""Métrica oficial estilo Challenge 2020 para el modelo de 12 grupos.

Expande las 12 probabilidades agrupadas a los 27 códigos SNOMED puntuados
oficiales (``official/dx_mapping_scored.csv``): cada código hereda la
probabilidad de su grupo. Luego calcula la métrica ponderada oficial
(``official/weights.csv``) con la implementación de referencia
(``official/evaluate_12ECG_score.py``):

- matriz de confusión modificada multilabel ``A[j, k]`` (filas=true j,
  columnas=pred k, normalizada por registro);
- score observado ``sum(W * A)`` normalizado entre el clasificador inactivo
  (siempre NSR) y el clasificador perfecto.

Uso::

    python examples/cinc2020/challenge_score.py \\
      --checkpoint saved/cinc2020/.../best.pt \\
      --test-h5 data/cinc2020_12/test.h5 \\
      --thresholds results/.../thresholds_validation.csv \\
      --output-dir results/.../challenge_metric

Nota: con checkpoints legacy v1 (que no cubren las 27 clases puntuadas), los
códigos no cubiertos reciben probabilidad 0 y la métrica sale penalizada:
es el comportamiento correcto para evidenciar la brecha.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecg import load, predict, util  # noqa: E402

OFFICIAL_DIR = Path(__file__).resolve().parent / "official"


def load_official_tables():
    """Carga (scored_codes, weights 27x27) en el orden de weights.csv."""
    weights_path = OFFICIAL_DIR / "weights.csv"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No existe {weights_path}. Descárguela de "
            "https://github.com/physionetchallenges/evaluation-2020"
        )
    df = pd.read_csv(weights_path, index_col=0)
    codes = [str(c).strip() for c in df.columns]
    weights = df.values.astype(np.float64)
    if weights.shape[0] != weights.shape[1] or weights.shape[0] != len(codes):
        raise ValueError(f"weights.csv inválida: shape={weights.shape}")
    return codes, weights


def group_index_for_code(code: str, class_names: list[str]) -> int | None:
    """Índice de grupo (según nombres del checkpoint) que contiene un código."""
    for i, name in enumerate(class_names):
        codes: list[str] = []
        for group in load.CLASS_GROUPS_12:
            if group["name"] == name:
                codes = group["codes"]
                break
        else:
            for group in load.CLASS_GROUPS_12_V1_LEGACY:
                if group["name"] == name:
                    codes = group["codes"]
                    break
        if str(code) in codes:
            return i
    return None


def expand_to_scored(
    group_prob: np.ndarray,
    group_pred: np.ndarray,
    class_names: list[str],
    scored_codes: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Expande (N,12) -> (N,27). Códigos sin grupo reciben 0."""
    n = group_prob.shape[0]
    n_scored = len(scored_codes)
    code_prob = np.zeros((n, n_scored), dtype=np.float32)
    code_pred = np.zeros((n, n_scored), dtype=np.uint8)
    uncovered: list[str] = []
    for j, code in enumerate(scored_codes):
        idx = group_index_for_code(code, class_names)
        if idx is None:
            uncovered.append(code)
            continue
        code_prob[:, j] = group_prob[:, idx]
        code_pred[:, j] = group_pred[:, idx]
    return code_prob, code_pred, sorted(set(uncovered))


def true_scored_matrix(dx_list: list[str], scored_codes: list[str]) -> np.ndarray:
    index = {code: j for j, code in enumerate(scored_codes)}
    y = np.zeros((len(dx_list), len(scored_codes)), dtype=np.uint8)
    for i, dx_text in enumerate(dx_list):
        for code in load.parse_dx_field(dx_text):
            j = index.get(code)
            if j is not None:
                y[i, j] = 1
    return y


def modified_confusion_matrix(labels: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """Réplica exacta de compute_modified_confusion_matrix oficial."""
    num_recordings, num_classes = labels.shape
    a = np.zeros((num_classes, num_classes))
    for i in range(num_recordings):
        union = np.logical_or(labels[i, :].astype(bool), outputs[i, :].astype(bool)).sum()
        normalization = float(max(union, 1))
        for j in range(num_classes):
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        a[j, k] += 1.0 / normalization
    return a


def challenge_metric(weights: np.ndarray, labels: np.ndarray, outputs: np.ndarray, classes: list[str]) -> dict:
    normal_class = "426783006"
    normal_index = classes.index(normal_class)
    observed = float(np.nansum(weights * modified_confusion_matrix(labels, outputs)))
    correct = float(np.nansum(weights * modified_confusion_matrix(labels, labels)))
    inactive = np.zeros_like(labels)
    inactive[:, normal_index] = 1
    inactive_score = float(np.nansum(weights * modified_confusion_matrix(labels, inactive)))
    normalized = (observed - inactive_score) / (correct - inactive_score) if correct != inactive_score else 0.0
    return {
        "challenge_metric": float(normalized),
        "observed_score": observed,
        "correct_score": correct,
        "inactive_score": inactive_score,
    }


def per_code_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, codes: list[str]) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    rows = []
    for j, code in enumerate(codes):
        yt, yp, yb = y_true[:, j], y_prob[:, j], y_pred[:, j]
        try:
            auroc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) == 2 else np.nan
        except Exception:
            auroc = np.nan
        try:
            auprc = float(average_precision_score(yt, yp)) if len(np.unique(yt)) == 2 else np.nan
        except Exception:
            auprc = np.nan
        rows.append({
            "code": code,
            "support_positive": int(yt.sum()),
            "precision": float(precision_score(yt, yb, zero_division=0)),
            "recall": float(recall_score(yt, yb, zero_division=0)),
            "f1": float(f1_score(yt, yb, zero_division=0)),
            "auroc": auroc,
            "auprc": auprc,
        })
    return pd.DataFrame(rows)


def load_thresholds_for_classes(path: str | None, class_names: list[str]) -> np.ndarray:
    values = np.full(len(class_names), 0.5, dtype=np.float32)
    if not path:
        return values
    df = pd.read_csv(util.resolve_path(path))
    mapping = {str(r["class"]): float(r["threshold"]) for _, r in df.iterrows()}
    for i, name in enumerate(class_names):
        if name in mapping:
            values[i] = mapping[name]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Métrica oficial Challenge 2020 desde 12 grupos")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-h5", required=True)
    parser.add_argument("--thresholds", default=None, help="thresholds_validation.csv (defecto: 0.5 global)")
    parser.add_argument("--output-dir", default="results/challenge_metric")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Workers del DataLoader (0 = recomendado con preload en RAM)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-preload", action="store_true", help="Desactiva la precarga del HDF5 a RAM (lento)")
    args = parser.parse_args()

    out_dir = util.resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_codes, weights = load_official_tables()
    print(f"Clases puntuadas oficiales: {len(scored_codes)}")

    result = predict.predict_hdf5(
        args.checkpoint, args.test_h5, batch_size=args.batch_size,
        num_workers=args.num_workers, device=args.device, amp=not args.no_amp,
        preload=not args.no_preload,
    )
    group_prob = np.asarray(result["probabilities"], dtype=np.float32)
    class_names = list(result["class_names"])
    thresholds = load_thresholds_for_classes(args.thresholds, class_names)
    group_pred = (group_prob >= thresholds[None, :]).astype(np.uint8)

    import h5py
    with h5py.File(util.resolve_path(args.test_h5), "r") as h5:
        dx_raw = h5["dx_codes"][:] if "dx_codes" in h5 else [""] * group_prob.shape[0]
        dx_list = [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in dx_raw]

    code_prob, code_pred, uncovered = expand_to_scored(group_prob, group_pred, class_names, scored_codes)
    if uncovered:
        print(f"AVISO: {len(uncovered)} códigos puntuados sin grupo (prob 0): {uncovered}")
    y_true = true_scored_matrix(dx_list, scored_codes)

    from sklearn.metrics import f1_score
    scores = challenge_metric(weights, y_true, code_pred, scored_codes)
    scores.update({
        "samples": int(y_true.shape[0]),
        "accuracy_exact_match": float((y_true == code_pred).all(axis=1).mean()),
        "f1_macro_27": float(f1_score(y_true, code_pred, average="macro", zero_division=0)),
        "f1_micro_27": float(f1_score(y_true, code_pred, average="micro", zero_division=0)),
        "uncovered_codes": uncovered,
        "thresholds_source": args.thresholds or "global_0.5",
        "checkpoint": str(args.checkpoint),
        "test_h5": str(args.test_h5),
    })
    per_code = per_code_metrics(y_true, code_prob, code_pred, scored_codes)
    per_code.to_csv(out_dir / "metrics_27_scored.csv", index=False)
    util.save_json(out_dir / "challenge_metric.json", scores)

    print("\n" + "=" * 72)
    print("MÉTRICA OFICIAL ESTILO CHALLENGE 2020 (27 códigos puntuados)")
    print("=" * 72)
    print(f"Challenge metric : {scores['challenge_metric']:.4f}")
    print(f"F1 macro-27      : {scores['f1_macro_27']:.4f}")
    print(f"F1 micro-27      : {scores['f1_micro_27']:.4f}")
    print(f"Exact match      : {scores['accuracy_exact_match']:.4f}")
    print(f"Resultados en    : {out_dir}")


if __name__ == "__main__":
    main()
