"""Compara métricas test de dos evaluaciones CINC2020-12.

Ejemplo:
    python examples/cinc2020/compare_models.py \\
      --eval-a eval-resnet \\
      --eval-b eval-resnet-v2 \\
      --label-a "ResNet v1" --label-b "ResNet v2" \\
      --output comparacion.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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
    return row


def main():
    parser = argparse.ArgumentParser(description="Compara métricas test de dos evaluaciones")
    parser.add_argument("--eval-a", required=True, help="Directorio de resultados de la evaluación A")
    parser.add_argument("--eval-b", required=True, help="Directorio de resultados de la evaluación B")
    parser.add_argument("--label-a", default=None, help="Etiqueta de A (por defecto, el nombre del directorio)")
    parser.add_argument("--label-b", default=None, help="Etiqueta de B (por defecto, el nombre del directorio)")
    parser.add_argument("--output", default="results/cinc2020_12/model_comparison.csv")
    args = parser.parse_args()

    rows = [
        load_global_metrics(Path(args.eval_a), args.label_a or Path(args.eval_a).name),
        load_global_metrics(Path(args.eval_b), args.label_b or Path(args.eval_b).name),
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nComparación guardada en: {out}")


if __name__ == "__main__":
    main()
