"""Bootstrap por registro: IC 95 % de F1-macro y de la diferencia entre modelos.

Lee ``predictions_test.csv`` (columnas ``record_name``/``true_*``/``pred_*``) de
``evaluate.py`` y remuestrea registros con reemplazo (sistema fijo: solo mide
variabilidad muestral del test, no de particiones ni semillas).

Con ``--predictions-baseline`` alinea ambos CSV por ``record_name`` y estima el
IC de la diferencia pareada (mismos remuestreos) + P(diff > 0).

Uso:
    python examples/cinc2020/bootstrap_ci.py --predictions <eval>/predictions_test.csv --output bootstrap_ci.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def load_predictions(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    classes = [c[len("true_"):] for c in df.columns if c.startswith("true_")]
    if not classes:
        raise ValueError(f"{path} no tiene columnas true_<clase>.")
    y_true = df[[f"true_{c}" for c in classes]].to_numpy(dtype=np.uint8)
    y_pred = df[[f"pred_{c}" for c in classes]].to_numpy(dtype=np.uint8)
    return df, y_true, y_pred, classes


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_f1(y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int, seed: int,
                 idx: np.ndarray | None = None) -> tuple[float, tuple[float, float], np.ndarray]:
    """F1-macro puntual + IC percentil. Si se da ``idx``, reutiliza esos remuestreos."""
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    if idx is None:
        idx = rng.integers(0, n, size=(n_bootstrap, n))
    stats = np.array([f1_score(y_true[b], y_pred[b], average="macro", zero_division=0) for b in idx])
    point = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return point, percentile_ci(stats), stats


def bootstrap_per_class(y_true: np.ndarray, y_pred: np.ndarray, idx: np.ndarray) -> list[dict]:
    rows = []
    for i in range(y_true.shape[1]):
        vals = np.array([f1_score(y_true[b, i], y_pred[b, i], zero_division=0) for b in idx])
        lo, hi = percentile_ci(vals)
        rows.append({"f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                     "ci95": [lo, hi]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="IC bootstrap de F1-macro y diferencia pareada.")
    parser.add_argument("--predictions", required=True, help="predictions_test.csv del modelo")
    parser.add_argument("--predictions-baseline", default=None, help="predictions_test.csv del baseline")
    parser.add_argument("--output", default="bootstrap_ci.json")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df, y_true, y_pred, classes = load_predictions(args.predictions)
    n = y_true.shape[0]
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.n_bootstrap, n))

    point, ci, _ = bootstrap_f1(y_true, y_pred, args.n_bootstrap, args.seed, idx=idx)
    result: dict = {
        "model": str(args.predictions),
        "n_records": n,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "f1_macro": point,
        "f1_macro_ci95": list(ci),
        "per_class": [dict({"class": c}, **r) for c, r in zip(classes, bootstrap_per_class(y_true, y_pred, idx))],
    }
    print(f"F1-macro: {point:.4f} IC95% [{ci[0]:.4f}, {ci[1]:.4f}] (n={n}, B={args.n_bootstrap})")

    if args.predictions_baseline:
        df_b, yt_b, yp_b, cb = load_predictions(args.predictions_baseline)
        if cb != classes:
            raise SystemExit("Las clases del baseline difieren del modelo.")
        order = df_b.set_index("record_name").index.get_indexer(df["record_name"])
        if (order < 0).any():
            missing = int((order < 0).sum())
            raise SystemExit(f"{missing} registros del modelo no están en el baseline.")
        yt_b, yp_b = yt_b[order], yp_b[order]
        point_b, ci_b, stats_b = bootstrap_f1(yt_b, yp_b, args.n_bootstrap, args.seed, idx=idx)
        _, _, stats_m = bootstrap_f1(y_true, y_pred, args.n_bootstrap, args.seed, idx=idx)
        diff = stats_m - stats_b
        ci_d = percentile_ci(diff)
        result["baseline"] = {"model": str(args.predictions_baseline), "f1_macro": point_b,
                              "f1_macro_ci95": list(ci_b)}
        result["diff"] = {"mean": float(point - point_b), "ci95": list(ci_d),
                          "p_positive": float((diff > 0).mean())}
        print(f"Baseline F1-macro: {point_b:.4f} IC95% [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
        print(f"Diferencia: {point - point_b:+.4f} IC95% [{ci_d[0]:+.4f}, {ci_d[1]:+.4f}] "
              f"P(diff>0)={float((diff > 0).mean()):.3f}")

    out = Path(args.output)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado en: {out}")


if __name__ == "__main__":
    main()
