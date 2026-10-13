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

Igual que el evaluador oficial, las clases equivalentes (CRBBB/RBBB, PAC/SVPB, PVC/VPB) se fusionan y se
puntúan 24 clases en vez de 27 (``--no-merge-equivalent`` conserva las 27 columnas).

Dos formas de uso. Puntuar las decisiones ya guardadas de una evaluación (la única forma de puntuar un ensemble y la
que garantiza que probabilidades, calibración y umbrales son los de la evaluación)::

    python examples/cinc2020/challenge_score.py \\
      --predictions runs/eval/<eval>/predictions_test.csv \\
      --test-h5 data/cinc2020_12/test.h5 \\
      --output-dir runs/eval/challenge_metric

O recalcular desde un checkpoint. Los umbrales de ``--thresholds`` se ajustaron sobre probabilidades calibradas,
así que pase también las temperaturas con las que se ajustaron; sin ellas se aplican a probabilidades crudas y la
métrica sale artificialmente baja::

    python examples/cinc2020/challenge_score.py \\
      --checkpoint saved/cinc2020/.../best.pt \\
      --test-h5 data/cinc2020_12/test.h5 \\
      --thresholds runs/eval/<eval>/thresholds_validation.csv \\
      --temperatures runs/eval/<eval>/temperatures_validation.csv \\
      --output-dir runs/eval/challenge_metric

Nota: con checkpoints legacy v1 (que no cubren las 27 clases puntuadas), los
códigos no cubiertos reciben probabilidad 0 y la métrica sale penalizada:
es el comportamiento correcto para evidenciar la brecha.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecg import load, predict, util  # noqa: E402

OFFICIAL_DIR = Path(__file__).resolve().parent / "official"
_OFFICIAL = None

# Copia de ``equivalent_classes`` de official/evaluate_12ECG_score.py (allí es una variable local de la función
# principal, así que no se puede importar); tests/test_challenge_score.py comprueba que siguen iguales.
EQUIVALENT_CLASSES = [["713427006", "59118001"], ["284470004", "63593006"], ["427172004", "17338001"]]


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


def _official_module():
    """Carga (una vez) el evaluador oficial vendored en ``official/``."""
    global _OFFICIAL
    if _OFFICIAL is None:
        spec = importlib.util.spec_from_file_location(
            "official_evaluate_12ECG_score", OFFICIAL_DIR / "evaluate_12ECG_score.py")
        _OFFICIAL = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_OFFICIAL)
    return _OFFICIAL


def modified_confusion_matrix(labels: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """Matriz de confusión modificada multilabel: función oficial, sin reimplementar."""
    return _official_module().compute_modified_confusion_matrix(labels, outputs)


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


def _representatives(codes: list[str], equivalent=None) -> list[str]:
    equivalent = EQUIVALENT_CLASSES if equivalent is None else equivalent
    return [next((group[0] for group in equivalent if code in group), code) for code in codes]


def merge_columns(matrix: np.ndarray, codes: list[str], equivalent=None, how: str = "max") -> tuple[np.ndarray, list[str]]:
    """Fusiona las columnas de códigos equivalentes en su representante (como el evaluador oficial).

    Binarios: el representante es positivo si lo es cualquiera (``how="max"``). Puntuaciones continuas: media
    (``how="mean"``). El orden de salida es el de primera aparición, igual que ``load_weights`` oficial.
    """
    reps = _representatives(codes, equivalent)
    merged_codes = [code for j, code in enumerate(reps) if code not in reps[:j]]
    out = np.zeros((matrix.shape[0], len(merged_codes)), dtype=np.float64 if how == "mean" else matrix.dtype)
    for k, code in enumerate(merged_codes):
        block = matrix[:, [j for j, r in enumerate(reps) if r == code]]
        out[:, k] = block.mean(axis=1) if how == "mean" else block.max(axis=1)
    return out, merged_codes


def merge_weights(codes: list[str], weights: np.ndarray, equivalent=None) -> tuple[list[str], np.ndarray]:
    """Matriz de pesos sobre los representantes; exige pesos idénticos entre equivalentes (como el oficial)."""
    reps = _representatives(codes, equivalent)
    for j in range(len(reps)):
        for k in range(j + 1, len(reps)):
            if reps[j] == reps[k]:
                if not (np.array_equal(weights[j, :], weights[k, :]) and np.array_equal(weights[:, j], weights[:, k])):
                    raise ValueError(f"Los códigos equivalentes {codes[j]} y {codes[k]} tienen pesos distintos.")
    merged_codes = [code for j, code in enumerate(reps) if code not in reps[:j]]
    indices = [reps.index(code) for code in merged_codes]
    return merged_codes, weights[np.ix_(indices, indices)]


def load_group_predictions(path: str | Path) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Lee predictions_test.csv de evaluate.py / ensemble_evaluate.py: (clases, record_name, prob, pred)."""
    df = pd.read_csv(util.resolve_path(path))
    class_names = [c[len("true_"):] for c in df.columns if c.startswith("true_")]
    if not class_names:
        raise ValueError(f"{path} no tiene columnas true_<clase>.")
    prob = df[[f"prob_{c}" for c in class_names]].to_numpy(dtype=np.float32)
    pred = df[[f"pred_{c}" for c in class_names]].to_numpy(dtype=np.uint8)
    return class_names, [str(x) for x in df["record_name"]], prob, pred


def load_dx_list(test_h5: str | Path, n_expected: int, record_names: list[str] | None = None) -> list[str]:
    """Códigos SNOMED verdaderos (``dx_codes``) del HDF5; comprueba orden y tamaño frente a las predicciones."""
    import h5py

    with h5py.File(util.resolve_path(test_h5), "r") as h5:
        dx_raw = h5["dx_codes"][:] if "dx_codes" in h5 else [""] * n_expected
        names = util.decode_h5_strings(h5["record_names"][:]) if "record_names" in h5 else None
    dx_list = [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in dx_raw]
    if len(dx_list) != n_expected:
        raise ValueError(f"{test_h5} tiene {len(dx_list)} registros y las predicciones {n_expected}.")
    if record_names is not None and names is not None and list(names) != list(record_names):
        raise ValueError(f"El orden de record_name de {test_h5} no coincide con el de las predicciones.")
    return dx_list


def score_groups(
    group_prob: np.ndarray,
    group_pred: np.ndarray,
    class_names: list[str],
    dx_list: list[str],
    scored_codes: list[str],
    weights: np.ndarray,
    merge_equivalent: bool = True,
) -> dict:
    """Métrica Challenge y F1 sobre los códigos puntuados a partir de decisiones por grupo."""
    code_prob, code_pred, uncovered = expand_to_scored(group_prob, group_pred, class_names, scored_codes)
    y_true = true_scored_matrix(dx_list, scored_codes)
    codes, w = list(scored_codes), weights
    if merge_equivalent:
        y_true, _ = merge_columns(y_true, scored_codes)
        code_pred, _ = merge_columns(code_pred, scored_codes)
        code_prob, _ = merge_columns(code_prob, scored_codes, how="mean")
        codes, w = merge_weights(scored_codes, weights)
    from sklearn.metrics import f1_score

    scores = challenge_metric(w, y_true, code_pred, codes)
    scores.update({
        "samples": int(y_true.shape[0]),
        "n_scored_classes": len(codes),
        "merge_equivalent": bool(merge_equivalent),
        "accuracy_exact_match": float((y_true == code_pred).all(axis=1).mean()),
        "f1_macro": float(f1_score(y_true, code_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, code_pred, average="micro", zero_division=0)),
        "uncovered_codes": uncovered,
    })
    scores["per_code"] = per_code_metrics(y_true, code_prob, code_pred, codes)
    return scores


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
    parser.add_argument("--predictions", default=None,
                        help="predictions_test.csv de evaluate.py / ensemble_evaluate.py: se puntúan sus decisiones "
                             "tal cual (sirve también para ensembles; no recalcula nada)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint .pt (alternativa a --predictions)")
    parser.add_argument("--test-h5", required=True)
    parser.add_argument("--thresholds", default=None, help="thresholds_validation.csv (defecto: 0.5 global)")
    parser.add_argument("--temperatures", default=None,
                        help="temperatures_validation.csv con el que se ajustaron los umbrales (solo con --checkpoint)")
    parser.add_argument("--no-merge-equivalent", action="store_true",
                        help="Puntúa 27 columnas en vez de fusionar las 3 parejas equivalentes como el evaluador oficial")
    parser.add_argument("--output-dir", default="runs/eval/challenge_metric")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Workers del DataLoader (0 = recomendado con preload en RAM)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-preload", action="store_true", help="Desactiva la precarga del HDF5 a RAM (lento)")
    args = parser.parse_args()
    if bool(args.predictions) == bool(args.checkpoint):
        parser.error("Indique exactamente una de --predictions o --checkpoint.")
    if args.predictions and (args.thresholds or args.temperatures):
        parser.error("--thresholds y --temperatures solo aplican con --checkpoint; con --predictions las decisiones ya "
                     "están tomadas.")

    out_dir = util.resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_codes, weights = load_official_tables()
    print(f"Clases puntuadas oficiales: {len(scored_codes)}")

    if args.predictions:
        class_names, record_names, group_prob, group_pred = load_group_predictions(args.predictions)
        dx_list = load_dx_list(args.test_h5, group_prob.shape[0], record_names)
        source = {"predictions": str(args.predictions)}
    else:
        result = predict.predict_hdf5(
            args.checkpoint, args.test_h5, batch_size=args.batch_size,
            num_workers=args.num_workers, device=args.device, amp=not args.no_amp,
            preload=not args.no_preload,
        )
        group_prob = np.asarray(result["probabilities"], dtype=np.float32)
        class_names = list(result["class_names"])
        if args.temperatures:
            from ecg.calibration import apply_temperature, load_temperatures_csv

            print(f"Aplicando temperature scaling desde {args.temperatures}.")
            group_prob = apply_temperature(group_prob, load_temperatures_csv(args.temperatures, class_names))
        elif args.thresholds:
            print("AVISO: umbrales sin --temperatures: si se ajustaron sobre probabilidades calibradas, "
                  "aquí se aplican a probabilidades crudas y la métrica sale subestimada.")
        thresholds = load_thresholds_for_classes(args.thresholds, class_names)
        group_pred = (group_prob >= thresholds[None, :]).astype(np.uint8)
        dx_list = load_dx_list(args.test_h5, group_prob.shape[0])
        source = {"checkpoint": str(args.checkpoint), "thresholds_source": args.thresholds or "global_0.5",
                  "temperatures_source": args.temperatures}

    scores = score_groups(group_prob, group_pred, class_names, dx_list, scored_codes, weights,
                          merge_equivalent=not args.no_merge_equivalent)
    if scores["uncovered_codes"]:
        print(f"AVISO: {len(scores['uncovered_codes'])} códigos puntuados sin grupo (prob 0): {scores['uncovered_codes']}")
    per_code = scores.pop("per_code")
    scores.update(source)
    scores["test_h5"] = str(args.test_h5)
    per_code.to_csv(out_dir / "metrics_scored_codes.csv", index=False)
    util.save_json(out_dir / "challenge_metric.json", scores)

    n = scores["n_scored_classes"]
    print("\n" + "=" * 72)
    print(f"MÉTRICA OFICIAL ESTILO CHALLENGE 2020 ({n} clases puntuadas"
          f"{', equivalentes fusionadas' if scores['merge_equivalent'] else ''})")
    print("=" * 72)
    print(f"Challenge metric : {scores['challenge_metric']:.4f}")
    print(f"F1 macro         : {scores['f1_macro']:.4f}")
    print(f"F1 micro         : {scores['f1_micro']:.4f}")
    print(f"Exact match      : {scores['accuracy_exact_match']:.4f}")
    print(f"Resultados en    : {out_dir}")


if __name__ == "__main__":
    main()
