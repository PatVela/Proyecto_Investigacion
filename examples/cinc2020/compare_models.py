"""Compara métricas test de dos evaluaciones CINC2020-12.

Solo compara evaluaciones con el mismo punto de operación (``threshold_beta`` de
``evaluation_summary.json``): umbrales F0.5 frente a F1 mueven el F1-macro varios puntos y esa
diferencia no es mérito del modelo. Use ``--allow-different-beta`` para compararlas a sabiendas;
la columna ``threshold_beta`` queda en el CSV de salida.

Ejemplo:
    python examples/cinc2020/compare_models.py \\
      --eval-a runs/eval/eval-resnet \\
      --eval-b runs/eval/eval-resnet-v2 \\
      --label-a "ResNet v1" --label-b "ResNet v2" \\
      --output comparacion.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def read_threshold_beta(results_dir: Path) -> float | None:
    """``threshold_beta`` guardado por evaluate.py / ensemble_evaluate.py; None si no se puede saber."""
    summary_path = Path(results_dir) / "evaluation_summary.json"
    if not summary_path.exists():
        return None
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8")).get("threshold_beta")
    except (OSError, ValueError):
        return None
    return None if value is None else float(value)


def load_global_metrics(results_dir: Path, model_name: str) -> dict:
    metrics_path = results_dir / "metrics_global.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No existe {metrics_path}")
    df = pd.read_csv(metrics_path)
    test = df[df["split"] == "test"]
    if test.empty:
        test = df.tail(1)
    row = test.iloc[0].to_dict()
    row["model"] = model_name
    row["results_dir"] = str(results_dir)
    row["threshold_beta"] = read_threshold_beta(results_dir)
    return row


def compare(eval_a, eval_b, label_a=None, label_b=None, output="runs/reports/model_comparison.csv",
            allow_different_beta: bool = False) -> pd.DataFrame:
    eval_a, eval_b = Path(eval_a), Path(eval_b)
    rows = [
        load_global_metrics(eval_a, label_a or eval_a.name),
        load_global_metrics(eval_b, label_b or eval_b.name),
    ]
    beta_a, beta_b = rows[0]["threshold_beta"], rows[1]["threshold_beta"]
    if beta_a is not None and beta_b is not None and beta_a != beta_b and not allow_different_beta:
        raise ValueError(
            f"threshold_beta distinto ({rows[0]['model']}: {beta_a}, {rows[1]['model']}: {beta_b}): la diferencia "
            "de F1-macro mezclaría modelo y punto de operación. Reevalúe ambos con el mismo --threshold-beta "
            "o use --allow-different-beta."
        )
    if beta_a is None or beta_b is None:
        print("AVISO: no se pudo leer threshold_beta de alguna evaluación (falta evaluation_summary.json); "
              "no se verificó que el punto de operación coincida.", file=sys.stderr)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Compara métricas test de dos evaluaciones")
    parser.add_argument("--eval-a", required=True, help="Directorio de resultados de la evaluación A")
    parser.add_argument("--eval-b", required=True, help="Directorio de resultados de la evaluación B")
    parser.add_argument("--label-a", default=None, help="Etiqueta de A (por defecto, el nombre del directorio)")
    parser.add_argument("--label-b", default=None, help="Etiqueta de B (por defecto, el nombre del directorio)")
    parser.add_argument("--output", default="runs/reports/model_comparison.csv")
    parser.add_argument("--allow-different-beta", action="store_true",
                        help="Permite comparar evaluaciones con distinto threshold_beta (queda registrado en el CSV)")
    args = parser.parse_args()

    try:
        df = compare(args.eval_a, args.eval_b, args.label_a, args.label_b, args.output, args.allow_different_beta)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(df.to_string(index=False))
    print(f"\nComparación guardada en: {args.output}")


if __name__ == "__main__":
    main()
