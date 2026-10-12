"""Calibración de probabilidades multilabel por *temperature scaling*.

Ajusta una temperatura por clase sobre validación minimizando NLL:

    p_cal = sigmoid(logit(p) / T)

Operar sobre probabilidades equivale a escalar los logits, así que no se
requiere acceso al forward del modelo. Usado por ``calibrate.py`` (ajuste),
``evaluate.py`` (aplicación con ``--temperatures``) y la webapp.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pc = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(pc / (1.0 - pc))


def apply_temperature(y_prob: np.ndarray, temperatures) -> np.ndarray:
    """Aplica temperature scaling por clase. ``temperatures``: (C,) o dict {clase: T}."""
    proba = np.asarray(y_prob, dtype=np.float64)
    if isinstance(temperatures, dict):
        raise ValueError("Pase un arreglo (C,) ordenado por clase, no un dict.")
    temps = np.asarray(list(temperatures), dtype=np.float64).reshape(-1)
    if temps.size != proba.shape[1]:
        raise ValueError(f"Se esperaban {proba.shape[1]} temperaturas; llegaron {temps.size}.")
    temps = np.clip(temps, 1e-3, 100.0)
    return _sigmoid(_logit(proba) / temps[None, :]).astype(np.float32)


def _nll_1d(temp: float, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = _sigmoid(_logit(y_prob) / float(temp))
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(-(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)).mean())


def fit_temperature_per_class(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Ajusta T por clase minimizando NLL (búsqueda acotada en [0.05, 10]).

    Si una clase no tiene positivos y negativos, devuelve T=1.0 (sin cambio).
    """
    from scipy.optimize import minimize_scalar

    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    temps = np.ones(y_true.shape[1], dtype=np.float64)
    for i in range(y_true.shape[1]):
        col = y_true[:, i]
        if col.min() == col.max():
            continue
        res = minimize_scalar(_nll_1d, bounds=(0.05, 10.0), method="bounded",
                              args=(col, y_prob[:, i]), options={"xatol": 1e-4})
        temps[i] = float(res.x) if res.success else 1.0
    return temps


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """ECE promedio sobre clases (bins uniformes en probabilidad)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    eces = []
    for i in range(y_true.shape[1]):
        col, prob = y_true[:, i], y_prob[:, i]
        idx = np.clip(np.digitize(prob, edges) - 1, 0, n_bins - 1)
        ece = 0.0
        for b in range(n_bins):
            mask = idx == b
            if not mask.any():
                continue
            ece += mask.mean() * abs(col[mask].mean() - prob[mask].mean())
        eces.append(ece)
    return float(np.mean(eces)) if eces else 0.0


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score promedio (menor es mejor)."""
    return float(np.mean((np.asarray(y_prob, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)) ** 2))


def load_temperatures_csv(path: str | Path, class_names: list[str]) -> np.ndarray:
    """Lee CSV ``class,temperature`` y devuelve vector (C,) ordenado por ``class_names``."""
    df = pd.read_csv(path)
    if "class" not in df.columns or "temperature" not in df.columns:
        raise ValueError(f"{path} debe tener columnas class,temperature.")
    values = {str(r["class"]): float(r["temperature"]) for _, r in df.iterrows()}
    missing = [c for c in class_names if c not in values]
    if missing:
        raise ValueError(f"Temperaturas faltantes para: {missing}.")
    return np.asarray([values[c] for c in class_names], dtype=np.float64)


def save_temperatures_csv(path: str | Path, class_names: list[str], temperatures) -> None:
    """Guarda vector (C,) como CSV ``class,temperature``."""
    temps = np.asarray(list(temperatures), dtype=np.float64).reshape(-1)
    pd.DataFrame({"class": list(class_names), "temperature": temps}).to_csv(path, index=False)
