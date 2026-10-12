"""Tests del generador de figuras empíricas (make_figures.py)."""

import numpy as np
import pandas as pd

from make_figures import op_points, plot_featured, plot_grid, plot_robustness

CLASSES = ["NSR", "AxisDev", "MI", "TAb", "AF", "LVH",
           "VEctopy", "AVBlock", "STach", "BBB", "SB", "AEctopy"]


def _metrics():
    rows = []
    for i, c in enumerate(CLASSES):
        rows.append({"class": c, "threshold": 0.4, "support_positive": 500,
                     "TP": 400, "TN": 5500, "FP": 300, "FN": 100,
                     "precision": 0.57, "sensitivity_recall": 0.8,
                     "specificity": 0.95, "f1": 0.66, "auroc": 0.95, "auprc": 0.8})
    return op_points(pd.DataFrame(rows))


def _curves():
    fpr = np.linspace(0, 1, 20)
    roc_rows, pr_rows = [], []
    for c in CLASSES:
        roc_rows += [{"class": c, "fpr": float(a), "tpr": float(a ** 0.3)} for a in fpr]
        pr_rows += [{"class": c, "recall": float(a), "precision": float(0.9 - 0.4 * a)} for a in fpr]
    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)


def test_op_points_y_grids(tmp_path):
    metrics = _metrics()
    assert abs(metrics.loc[0, "fpr_op"] - 300 / 5800) < 1e-9
    roc, pr = _curves()
    for kind in ("roc", "pr"):
        out = tmp_path / f"g_{kind}.png"
        plot_grid(roc, pr, metrics, 6421, out, kind)
        assert out.exists() and out.stat().st_size > 0


def test_destacadas_y_robustez(tmp_path):
    metrics = _metrics()
    roc, pr = _curves()
    out = tmp_path / "dest.png"
    plot_featured(roc, pr, metrics, ["AF", "AVBlock"], out)
    assert out.exists() and out.stat().st_size > 0
    rob = pd.DataFrame([
        {"perturbation": "clean", "level": 0.0, "f1_macro": 0.67},
        {"perturbation": "gaussian_noise", "level": 0.05, "f1_macro": 0.60},
        {"perturbation": "amplitude_scale", "level": 0.5, "f1_macro": 0.53},
    ])
    rob_path = tmp_path / "robustness.csv"
    rob.to_csv(rob_path, index=False)
    out = tmp_path / "rob.png"
    plot_robustness(rob_path, out)
    assert out.exists() and out.stat().st_size > 0
