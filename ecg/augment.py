"""Aumentación de señal ECG para entrenamiento (preserva etiquetas).

Transformaciones fisiológicamente plausibles aplicadas solo al split de
entrenamiento: ruido gaussiano (EMG), deriva de línea base (respiración),
escala de amplitud, desplazamiento temporal y dropout de derivación.

Todo opera en numpy sobre ventanas (12, 5000) en mV físicos y devuelve
arrays nuevos: nunca muta la entrada (el Dataset precargado en RAM entrega
vistas que no deben modificarse).
"""

from __future__ import annotations

import numpy as np

FS_HZ = 500  # 5000 muestras = 10 s.

DEFAULTS = {
    "aug_prob": 0.5,
    "aug_noise_std": 0.01,      # mV (≈10 µV, ruido muscular realista)
    "aug_baseline_amp": 0.1,    # mV, amplitud máxima de deriva respiratoria
    "aug_scale_range": 0.1,     # ±10 % de amplitud
    "aug_shift_max": 250,       # ±0.5 s de desplazamiento
    "aug_lead_drop_prob": 0.05,
    "aug_seed": None,
}

# Probabilidades internas de cada transformación dentro del pipeline.
P_NOISE = 0.6
P_WANDER = 0.4
P_SCALE = 0.6
P_SHIFT = 0.4


def add_gaussian_noise(x, rng, std):
    """Suma ruido blanco gaussiano de desviación ``std`` (mV)."""
    return (np.asarray(x, dtype=np.float32) + rng.normal(0.0, float(std), size=x.shape)).astype(np.float32)


def add_baseline_wander(x, rng, amp):
    """Suma deriva sinusoidal lenta (0.1–0.35 Hz, respiración)."""
    x = np.asarray(x, dtype=np.float32)
    t = np.arange(x.shape[1], dtype=np.float64) / FS_HZ
    freq = float(rng.uniform(0.1, 0.35))
    phase = float(rng.uniform(0, 2 * np.pi))
    strength = float(rng.uniform(0.5, 1.0)) * float(amp)
    wander = (strength * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
    return (x + wander[None, :]).astype(np.float32)


def random_scale(x, rng, scale_range):
    """Multiplica la amplitud por un factor uniforme en ±``scale_range``."""
    factor = float(rng.uniform(1.0 - float(scale_range), 1.0 + float(scale_range)))
    return (np.asarray(x, dtype=np.float32) * factor).astype(np.float32)


def time_shift(x, rng, max_shift):
    """Desplaza en tiempo ±``max_shift`` muestras, rellenando con ceros."""
    x = np.asarray(x, dtype=np.float32)
    limit = int(max_shift)
    k = int(rng.integers(-limit, limit + 1)) if limit > 0 else 0
    out = np.zeros_like(x)
    n = x.shape[1]
    if k > 0:
        out[:, k:] = x[:, : n - k]
    elif k < 0:
        out[:, : n + k] = x[:, -k:]
    else:
        out[:] = x
    return out.astype(np.float32, copy=False)


def lead_dropout(x, rng, prob):
    """Con prob. ``prob``, pone una derivación aleatoria a cero."""
    if float(prob) <= 0 or rng.random() >= float(prob):
        return np.array(x, dtype=np.float32, copy=True)
    out = np.array(x, dtype=np.float32, copy=True)
    out[int(rng.integers(0, x.shape[0])), :] = 0.0
    return out


def augment_signal(x, rng=None, cfg=None):
    """Aplica el pipeline con prob. maestra; siempre devuelve array nuevo."""
    params = dict(DEFAULTS)
    if cfg:
        params.update(cfg)
    if rng is None:
        rng = np.random.default_rng(params.get("aug_seed"))
    x = np.asarray(x, dtype=np.float32)
    if rng.random() >= float(params["aug_prob"]):
        return np.array(x, dtype=np.float32, copy=True)
    out = x
    if rng.random() < P_NOISE:
        out = add_gaussian_noise(out, rng, params["aug_noise_std"])
    if rng.random() < P_WANDER:
        out = add_baseline_wander(out, rng, params["aug_baseline_amp"])
    if rng.random() < P_SCALE:
        out = random_scale(out, rng, params["aug_scale_range"])
    if rng.random() < P_SHIFT:
        out = time_shift(out, rng, params["aug_shift_max"])
    out = lead_dropout(out, rng, params["aug_lead_drop_prob"])
    return np.asarray(out, dtype=np.float32)
