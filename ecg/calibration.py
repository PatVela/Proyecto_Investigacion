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


def ensemble_probabilities(prob_a, prob_b, alpha: float = 0.5) -> np.ndarray:
    """Promedio ponderado: alpha * A + (1 - alpha) * B (lo usan ``ensemble_evaluate.py`` y la webapp)."""
    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"alpha debe estar en [0, 1], recibido {alpha}")
    return a * np.asarray(prob_a, dtype=np.float64) + (1.0 - a) * np.asarray(prob_b, dtype=np.float64)


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


def load_thresholds_csv(path: str | Path, class_names: list[str]) -> np.ndarray:
    """Lee ``thresholds_validation.csv`` (``class,threshold,...``) y devuelve el vector (C,) ordenado por ``class_names``.

    Exige una fila por clase: un umbral ausente NO se sustituye por 0.5 (con un checkpoint v1, ``LAD``/``RBBB`` frente a
    ``AxisDev``/``BBB``, eso evaluaba dos clases con un umbral que nadie eligió y sin avisar).
    """
    df = pd.read_csv(path)
    if "class" not in df.columns or "threshold" not in df.columns:
        raise ValueError(f"{path} debe tener columnas class,threshold.")
    values = {str(r["class"]): float(r["threshold"]) for _, r in df.iterrows()}
    missing = [c for c in class_names if c not in values]
    if missing:
        raise ValueError(f"Umbrales faltantes en {path} para: {missing}.")
    return np.asarray([values[c] for c in class_names], dtype=np.float32)


def thresholds_or_default(path: str | Path | None, class_names: list[str], default: float = 0.5) -> np.ndarray:
    """Umbrales por clase de ``path`` (:func:`load_thresholds_csv`); sin archivo, ``default`` para todas."""
    return load_thresholds_csv(path, class_names) if path else np.full(len(class_names), default, dtype=np.float32)


def save_temperatures_csv(path: str | Path, class_names: list[str], temperatures) -> None:
    """Guarda vector (C,) como CSV ``class,temperature``."""
    temps = np.asarray(list(temperatures), dtype=np.float64).reshape(-1)
    pd.DataFrame({"class": list(class_names), "temperature": temps}).to_csv(path, index=False)


# --- Calibración afín por clase (Platt): p_cal = sigmoid(a · logit + b) ----------------------------------------------
#
# La temperatura es el caso b = 0, a = 1/T. Entrenar con ``pos_weight = neg/pos`` (y label smoothing) sube la
# probabilidad base de todos los registros, y eso es un desplazamiento (b), que una temperatura no corrige: con ella el
# ECE de v2 queda en 0.114 y con a, b en 0.014 (ver tasks/analisis_mejoras.md). a > 0 siempre, así que la calibración
# no cambia el orden de las puntuaciones: AUROC y AUPRC no se tocan y solo se mueve la escala en la que se fijan los
# umbrales.

LOGIT_EPS = 1e-6
PLATT_SLOPE_BOUNDS = (0.05, 10.0)
PLATT_BIAS_BOUNDS = (-20.0, 20.0)


def to_logits(y_prob: np.ndarray, eps: float = LOGIT_EPS) -> np.ndarray:
    """logit(p) recortado a [eps, 1-eps] en float64 (una probabilidad exacta de 0 o 1 no da ±inf)."""
    return _logit(y_prob, eps=eps)


def fit_platt_per_class(y_true: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta ``(a, b)`` por clase minimizando la NLL sobre validación. Devuelve dos vectores (C,).

    Ajustar sobre logits crudos del modelo (no sobre probabilidades ya escaladas): una temperatura menor que 1
    empuja muchas probabilidades a 1.0 exacto en float32 y el recorte del logit decide entonces la pendiente. Una
    clase sin positivos o sin negativos devuelve la identidad ``(1, 0)``.
    """
    from scipy.optimize import minimize

    y_true = np.asarray(y_true, dtype=np.float64)
    z_all = np.asarray(logits, dtype=np.float64)
    n_classes = y_true.shape[1]
    a = np.ones(n_classes, dtype=np.float64)
    b = np.zeros(n_classes, dtype=np.float64)
    for i in range(n_classes):
        y, z = y_true[:, i], z_all[:, i]
        if y.min() == y.max():
            continue

        def nll(params, y=y, z=z):
            u = params[0] * z + params[1]
            grad_u = 1.0 / (1.0 + np.exp(-u)) - y
            return float(np.mean(np.logaddexp(0.0, u) - y * u)), np.array([np.mean(grad_u * z), np.mean(grad_u)])

        res = minimize(nll, x0=[1.0, 0.0], jac=True, method="L-BFGS-B",
                       bounds=[PLATT_SLOPE_BOUNDS, PLATT_BIAS_BOUNDS])
        if res.success or np.all(np.isfinite(res.x)):
            a[i], b[i] = float(res.x[0]), float(res.x[1])
    return a, b


def apply_platt_logits(logits: np.ndarray, a, b) -> np.ndarray:
    """``sigmoid(a·z + b)`` por clase en float64 (sin recortar a float32: no crea empates en 1.0)."""
    z = np.asarray(logits, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).reshape(1, -1)
    b = np.asarray(b, dtype=np.float64).reshape(1, -1)
    if a.shape[1] != z.shape[1] or b.shape[1] != z.shape[1]:
        raise ValueError(f"Se esperaban {z.shape[1]} parámetros a y b; llegaron {a.shape[1]} y {b.shape[1]}.")
    return _sigmoid(a * z + b)


def apply_platt(y_prob: np.ndarray, a, b, eps: float = LOGIT_EPS) -> np.ndarray:
    """Igual que :func:`apply_platt_logits` partiendo de probabilidades (webapp, ensembles)."""
    return apply_platt_logits(to_logits(y_prob, eps=eps), a, b)


def load_platt_csv(path: str | Path, class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Lee CSV ``class,a,b`` y devuelve dos vectores (C,) ordenados por ``class_names``."""
    df = pd.read_csv(path)
    if not {"class", "a", "b"} <= set(df.columns):
        raise ValueError(f"{path} debe tener columnas class,a,b.")
    values = {str(r["class"]): (float(r["a"]), float(r["b"])) for _, r in df.iterrows()}
    missing = [c for c in class_names if c not in values]
    if missing:
        raise ValueError(f"Parámetros faltantes para: {missing}.")
    return (np.asarray([values[c][0] for c in class_names], dtype=np.float64),
            np.asarray([values[c][1] for c in class_names], dtype=np.float64))


def save_platt_csv(path: str | Path, class_names: list[str], a, b) -> None:
    """Guarda ``(a, b)`` como CSV ``class,a,b``."""
    pd.DataFrame({"class": list(class_names), "a": np.asarray(a, dtype=np.float64).reshape(-1),
                  "b": np.asarray(b, dtype=np.float64).reshape(-1)}).to_csv(path, index=False)
