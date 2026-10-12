"""Genera las figuras empíricas del informe desde una evaluación final.

Lee los CSV de ``evaluate.py`` (``roc_curves_test.csv``, ``pr_curves_test.csv``,
``thresholds_validation.csv``, ``metrics_per_class.csv``, ``metrics_global.csv``)
y traza figuras con puntos 100 % empíricos (sklearn sobre y_true/y_prob):

  1. ``fig_roc_clases``: ROC por clase (retícula 3×4) con punto de operación.
  2. ``fig_pr_clases``: precisión-recall por clase (3×4) con punto de operación.
  3. ``fig_curvas_destacadas``: ROC+PR de 3 clases (``--classes``).
  4. ``fig_robustez``: barras de todos los niveles (``--robustness``).
  5. ``fig_aprendizaje``: train/val loss + val F1 v1 vs v2 (``--history-v1/--history-v2``).

Los puntos de operación se calculan desde TP/FP/TN/FN del umbral final, por
construcción idénticos a la tabla de F1 por clase. Además imprime una tabla de
verificación con P/R/F1/umbral por clase para cruzar con el texto.

Uso:
    python examples/cinc2020/make_figures.py --eval-dir <eval-final> --outdir figs_eval --robustness <robustness.csv>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PERT_LABELS = {
    "clean": "base (limpio)",
    "gaussian_noise": "ruido",
    "baseline_wander": "deriva",
    "amplitude_scale": "escala",
    "lead_dropout": "derivs. fuera",
}


def load_eval(eval_dir: Path, split: str):
    roc = pd.read_csv(eval_dir / f"roc_curves_{split}.csv")
    pr = pd.read_csv(eval_dir / f"pr_curves_{split}.csv")
    metrics = pd.read_csv(eval_dir / "metrics_per_class.csv")
    metrics = metrics[metrics["split"] == split].reset_index(drop=True)
    glob = pd.read_csv(eval_dir / "metrics_global.csv")
    glob = glob[glob["split"] == split].iloc[0]
    if metrics.empty:
        raise ValueError(f"Sin filas split={split} en metrics_per_class.csv")
    return roc, pr, metrics, glob


def op_points(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    denom = (df["FP"] + df["TN"]).replace(0, pd.NA)
    df["fpr_op"] = (df["FP"] / denom).fillna(0.0)
    df["tpr_op"] = df["sensitivity_recall"]
    return df


def plot_grid(roc, pr, metrics, samples, outpath, kind):
    classes = list(metrics["class"])
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), sharex=False, sharey=False)
    for ax, (_, row) in zip(axes.flat, metrics.iterrows()):
        cls = row["class"]
        if kind == "roc":
            sub = roc[roc["class"] == cls].sort_values("fpr")
            ax.plot(sub["fpr"], sub["tpr"], color="tab:blue", lw=1.5)
            ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
            ax.plot(row["fpr_op"], row["tpr_op"], "o", color="tab:red", ms=5)
            ax.set_title(f"{cls} (AUROC {row['auroc']:.3f})", fontsize=9)
            ax.set_xlabel("Tasa de falsos positivos", fontsize=8)
            ax.set_ylabel("Sensibilidad", fontsize=8)
        else:
            sub = pr[pr["class"] == cls].sort_values("recall")
            ax.plot(sub["recall"], sub["precision"], color="tab:blue", lw=1.5)
            ax.axhline(row["support_positive"] / samples, color="gray", ls="--", lw=1)
            ax.plot(row["tpr_op"], row["precision"], "o", color="tab:red", ms=5)
            ax.set_title(f"{cls} (AUPRC {row['auprc']:.3f})", fontsize=9)
            ax.set_xlabel("Recall (sensibilidad)", fontsize=8)
            ax.set_ylabel("Precisión", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(classes):]:
        ax.axis("off")
    fig.suptitle(
        "Curvas ROC empíricas por clase (punto rojo = umbral de operación)"
        if kind == "roc"
        else "Curvas precisión-recall empíricas por clase (punto rojo = umbral de operación)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_featured(roc, pr, metrics, classes, outpath):
    rows = metrics[metrics["class"].isin(classes)]
    missing = [c for c in classes if c not in set(rows["class"])]
    if missing:
        raise ValueError(f"Clases sin métricas: {missing}")
    fig, axes = plt.subplots(2, len(rows), figsize=(4 * len(rows), 7), squeeze=False)
    for j, (_, row) in enumerate(rows.iterrows()):
        cls = row["class"]
        ax = axes[0][j]
        sub = roc[roc["class"] == cls].sort_values("fpr")
        ax.plot(sub["fpr"], sub["tpr"], color="tab:blue", lw=2, label="ROC empírica")
        ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
        ax.plot(row["fpr_op"], row["tpr_op"], "o", color="tab:red", ms=7,
                label=f"Operación (thr={row['threshold']:.2f})")
        ax.set_title(f"{cls} — ROC (AUROC {row['auroc']:.3f})")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Sensibilidad")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax = axes[1][j]
        sub = pr[pr["class"] == cls].sort_values("recall")
        ax.plot(sub["recall"], sub["precision"], color="tab:blue", lw=2, label="PR empírica")
        ax.plot(row["tpr_op"], row["precision"], "o", color="tab:red", ms=7, label="Operación")
        ax.set_title(f"{cls} — PR (AUPRC {row['auprc']:.3f})")
        ax.set_xlabel("Recall (sensibilidad)")
        ax.set_ylabel("Precisión")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Clases destacadas: curvas empíricas y punto de operación", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_robustness(path: Path, outpath: Path):
    df = pd.read_csv(path)
    labels = [f"{PERT_LABELS.get(p, p)} {lvl}".strip() for p, lvl in zip(df["perturbation"], df["level"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["tab:green" if p == "clean" else "tab:blue" for p in df["perturbation"]]
    bars = ax.barh(labels, df["f1_macro"], color=colors)
    ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xlabel("F1-macro")
    ax.set_title("Robustez: F1-macro bajo perturbaciones controladas (todos los niveles)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_learning(hist_paths: dict[str, Path], outpath: Path):
    hists = {tag: pd.read_csv(p) for tag, p in hist_paths.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for tag, h in hists.items():
        axes[0].plot(h["epoch"], h["train_loss"], label=f"{tag} train", ls="--")
        axes[0].plot(h["epoch"], h["val_loss"], label=f"{tag} val")
        axes[1].plot(h["epoch"], h["val_f1_macro"], label=tag)
        best = h.loc[h["val_f1_macro"].idxmax()]
        axes[1].plot(best["epoch"], best["val_f1_macro"], "o", ms=6)
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Pérdida (BCE)")
    axes[0].set_title("Pérdida de entrenamiento vs validación")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("F1-macro (val)")
    axes[1].set_title("F1-macro en validación (punto = mejor época)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Curvas de aprendizaje v1 vs v2", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera figuras empíricas desde una evaluación final.")
    parser.add_argument("--eval-dir", required=True, help="Directorio con los CSV de evaluate.py")
    parser.add_argument("--outdir", default="figs_eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--classes", default="AF,AVBlock,TAb", help="Clases destacadas (Fig. 3)")
    parser.add_argument("--robustness", default=None, help="robustness.csv con todos los niveles")
    parser.add_argument("--history-v1", default=None, help="history.csv de v1")
    parser.add_argument("--history-v2", default=None, help="history.csv de v2")
    parser.add_argument("--formats", default="png,pdf", help="Formatos de salida separados por coma")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    for name in (f"roc_curves_{args.split}.csv", f"pr_curves_{args.split}.csv",
                 "metrics_per_class.csv", "metrics_global.csv"):
        if not (eval_dir / name).exists():
            raise SystemExit(f"Falta {name} en {eval_dir}. Re-ejecute evaluate.py (solo inferencia).")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip().lstrip(".") for f in args.formats.split(",") if f.strip()]

    roc, pr, metrics, glob = load_eval(eval_dir, args.split)
    metrics = op_points(metrics)
    samples = int(glob["samples"])
    featured = [c.strip() for c in args.classes.split(",") if c.strip()]

    jobs = [
        ("fig_roc_clases", lambda p: plot_grid(roc, pr, metrics, samples, p, "roc")),
        ("fig_pr_clases", lambda p: plot_grid(roc, pr, metrics, samples, p, "pr")),
        ("fig_curvas_destacadas", lambda p: plot_featured(roc, pr, metrics, featured, p)),
    ]
    if args.robustness:
        rob_path = Path(args.robustness)
        jobs.append(("fig_robustez", lambda p: plot_robustness(rob_path, p)))
    hists = {}
    if args.history_v1:
        hists["v1"] = Path(args.history_v1)
    if args.history_v2:
        hists["v2"] = Path(args.history_v2)
    if hists:
        jobs.append(("fig_aprendizaje", lambda p: plot_learning(hists, p)))

    for stem, fn in jobs:
        for fmt in formats:
            fn(outdir / f"{stem}.{fmt}")
    print(f"Figuras guardadas en: {outdir}")

    print("\nVerificación (idéntico a la tabla F1 por clase por construcción):")
    cols = ["class", "threshold", "precision", "sensitivity_recall", "f1",
            "support_positive", "auroc", "auprc"]
    print(metrics[cols].round(4).to_string(index=False))
    print("\nLíneas para el texto (clases destacadas):")
    for _, row in metrics[metrics["class"].isin(featured)].iterrows():
        print(f"{row['class']}: P={row['precision']:.3f} R={row['sensitivity_recall']:.3f} "
              f"F1={row['f1']:.3f} thr={row['threshold']:.2f} AUROC={row['auroc']:.3f}")


if __name__ == "__main__":
    main()
