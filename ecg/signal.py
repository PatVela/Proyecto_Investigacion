"""Lectura WFDB/CSV y preprocesamiento de señal ECG (500 Hz, 12 derivaciones).

Extraído de ``ecg/load.py`` (split de depuración). Sin dependencias
de PyTorch.
"""

from __future__ import annotations
from ecg.schema import BANDPASS_HI_HZ, BANDPASS_LO_HZ, DEFAULT_ADC_GAIN, DEFAULT_NORM_MODE, PHYSICAL_CLIP_MV, parse_dx_field

import re
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt


TARGET_FS = 500
WINDOW_SECONDS = 10
WINDOW_LENGTH = TARGET_FS * WINDOW_SECONDS
STANDARD_LEAD_ORDER = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]
NUM_LEADS = len(STANDARD_LEAD_ORDER)


def normalize_lead_name(name) -> str:
    raw = str(name).strip()
    mapping = {
        "I": "I", "II": "II", "III": "III",
        "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
        "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
    }
    return mapping.get(raw.upper(), raw)


def parse_header(hea_path: str | Path) -> dict:
    hea_path = Path(hea_path)
    with hea_path.open("r", encoding="utf-8-sig") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if not lines:
        raise ValueError("Header vacío")

    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"Primera línea inválida: {lines[0]}")

    record_name = first[0]
    num_leads = int(first[1])
    sampling_freq = float(first[2])
    num_samples = int(first[3])

    if len(lines) < num_leads + 1:
        raise ValueError(f"Header incompleto: se esperaban {num_leads} derivaciones")

    lead_names = [normalize_lead_name(lines[i].split()[-1]) for i in range(1, num_leads + 1)]

    adc_gains: list[float] = []
    adc_baselines: list[float] = []
    adc_units: list[str] = []
    for i in range(1, num_leads + 1):
        gain, baseline, units = parse_adc_spec(lines[i])
        adc_gains.append(gain)
        adc_baselines.append(baseline)
        adc_units.append(units)

    age = np.nan
    sex = "Unknown"
    dx_codes: list[str] = []
    for line in lines[num_leads + 1:]:
        if not line.startswith("#"):
            continue
        comment = line[1:].strip()
        lower = comment.lower()
        if lower.startswith("age:"):
            raw = comment.split(":", 1)[1].strip().replace(">", "").replace("<", "")
            try:
                value = float(raw)
                age = value if 0 <= value <= 120 else np.nan
            except ValueError:
                age = np.nan
        elif lower.startswith("sex:"):
            value = comment.split(":", 1)[1].strip()
            sex = value if value else "Unknown"
        elif lower.startswith("dx:"):
            dx_codes = parse_dx_field(comment.split(":", 1)[1].strip())

    return {
        "record_name": record_name,
        "num_leads": num_leads,
        "sampling_freq": sampling_freq,
        "num_samples": num_samples,
        "lead_names": lead_names,
        "adc_gains": adc_gains,
        "adc_baselines": adc_baselines,
        "adc_units": adc_units,
        "age": age,
        "sex": sex,
        "dx_codes": dx_codes,
    }


def parse_adc_spec(lead_line: str) -> tuple[float, float, str]:
    """Extrae (ganancia, baseline, unidades) de una línea de derivación WFDB.

    Ejemplo: ``A0001.mat 16+24 1000/mV 16 0 28 -1716 0 I`` -> (1000.0, 0.0, 'mV').
    Acepta variantes como ``1000.0(0)/mV``. Si algo falla, usa ganancia 1000 y
    baseline 0 (valores uniformes observados en los 6 subconjuntos CINC2020).
    """
    tokens = lead_line.split()
    gain = DEFAULT_ADC_GAIN
    baseline = 0.0
    units = "mV"
    try:
        if len(tokens) >= 3:
            match = re.match(r"([0-9.eE+-]+)(?:\\(([^)]*)\\))?(?:/(.*))?", tokens[2])
            if match:
                gain = float(match.group(1))
                if match.group(2) not in (None, ""):
                    try:
                        baseline = float(match.group(2))
                    except ValueError:
                        pass
                if match.group(3):
                    units = match.group(3)
        if len(tokens) >= 5:
            # Quinto token = adc_zero (baseline digital). Prevalece si es numérico.
            try:
                baseline = float(tokens[4])
            except ValueError:
                pass
    except (ValueError, IndexError):
        pass
    if not np.isfinite(gain) or gain == 0:
        gain = DEFAULT_ADC_GAIN
    if not np.isfinite(baseline):
        baseline = 0.0
    return float(gain), float(baseline), str(units)


def reorder_leads(signal: np.ndarray, lead_names: Iterable[str]) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"La señal debe ser 2D; shape={signal.shape}")
    normalized = [normalize_lead_name(name) for name in lead_names]
    missing = [lead for lead in STANDARD_LEAD_ORDER if lead not in normalized]
    if missing:
        raise ValueError(f"Faltan derivaciones estándar: {missing}; encontradas={normalized}")
    indices = [normalized.index(lead) for lead in STANDARD_LEAD_ORDER]
    return signal[:, indices]


def resample_signal(signal: np.ndarray, orig_fs: float, target_fs: int = TARGET_FS) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"La señal debe ser 2D; shape={signal.shape}")
    orig_fs = float(orig_fs)
    target_fs = float(target_fs)
    if orig_fs <= 0:
        raise ValueError(f"Frecuencia inválida: {orig_fs}")
    if np.isclose(orig_fs, target_fs):
        return signal
    ratio = Fraction(target_fs / orig_fs).limit_denominator(10000)
    return resample_poly(signal, ratio.numerator, ratio.denominator, axis=0)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Z-score por derivación, robusto frente a derivaciones constantes.

    Modo legacy (``per_lead_zscore``): elimina offsets por derivación pero
    también destruye amplitudes absolutas (criterios de voltaje de LVH/LQRSV)
    y relativas entre derivaciones (eje eléctrico). Se conserva solo para
    compatibilidad con checkpoints entrenados con el esquema v1.
    """
    signal = np.asarray(signal, dtype=np.float64)
    mean = np.nanmean(signal, axis=0, keepdims=True)
    std = np.nanstd(signal, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    signal = (signal - mean) / std
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal.astype(np.float32, copy=False)


def normalize_signal_global(signal: np.ndarray) -> np.ndarray:
    """Z-score global del registro (media/std sobre las 12 derivaciones).

    Preserva amplitudes *relativas* entre derivaciones (eje eléctrico),
    pero no absolutas. Intermedio entre ``per_lead_zscore`` y ``physical``.
    """
    signal = np.asarray(signal, dtype=np.float64)
    mean = float(np.nanmean(signal))
    std = float(np.nanstd(signal))
    if not np.isfinite(mean):
        mean = 0.0
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    signal = (signal - mean) / std
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal.astype(np.float32, copy=False)


def to_physical_units(
    signal: np.ndarray,
    adc_gains: Iterable[float] | None = None,
    adc_baselines: Iterable[float] | None = None,
) -> np.ndarray:
    """Convierte valores digitales WFDB a mV: ``(digital - baseline) / gain``.

    En los 6 subconjuntos CINC2020 la ganancia es uniformemente 1000/mV con
    baseline 0, pero se lee del header por robustez.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if adc_gains is None:
        gains = np.full(signal.shape[1], DEFAULT_ADC_GAIN)
    else:
        gains = np.asarray(list(adc_gains), dtype=np.float64)
    if adc_baselines is None:
        baselines = np.zeros(signal.shape[1])
    else:
        baselines = np.asarray(list(adc_baselines), dtype=np.float64)
    if gains.shape[0] != signal.shape[1] or baselines.shape[0] != signal.shape[1]:
        raise ValueError(
            f"Gains/baselines ({gains.shape[0]}) no coinciden con derivaciones ({signal.shape[1]})"
        )
    gains = np.where((~np.isfinite(gains)) | (gains == 0), DEFAULT_ADC_GAIN, gains)
    baselines = np.where(~np.isfinite(baselines), 0.0, baselines)
    physical = (signal - baselines[None, :]) / gains[None, :]
    return np.nan_to_num(physical, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float = TARGET_FS,
    lo_hz: float = BANDPASS_LO_HZ,
    hi_hz: float = BANDPASS_HI_HZ,
    order: int = 3,
) -> np.ndarray:
    """Pasa-banda Butterworth de fase cero (SOS + sosfiltfilt).

    Atenúa deriva de línea base (<0.5 Hz) y ruido muscular/red (>50 Hz).
    Requiere señal de al menos ~2 s; si es más corta, la devuelve intacta.
    """
    signal = np.asarray(signal, dtype=np.float64)
    nyquist = float(sampling_rate) / 2.0
    if signal.shape[0] < int(2 * sampling_rate):
        return signal.astype(np.float32, copy=False)
    lo = min(max(float(lo_hz) / nyquist, 1e-4), 0.99)
    hi = min(max(float(hi_hz) / nyquist, 1e-4), 0.99)
    if lo >= hi:
        raise ValueError(f"Banda inválida: lo={lo_hz} hi={hi_hz} fs={sampling_rate}")
    sos = butter(int(order), [lo, hi], btype="band", output="sos")
    filtered = sosfiltfilt(sos, signal, axis=0)
    return np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def compute_window_starts(
    n_samples: int,
    target_length: int = WINDOW_LENGTH,
    windows_max: int = 1,
) -> list[int]:
    """Inicios de ventana (en muestras) equiespaciados sobre la señal.

    - Señal corta (``n <= target``): un único inicio 0 (luego se aplica pad).
    - ``windows_max <= 1``: ventana centrada (comportamiento legacy).
    - Señal larga: hasta ``windows_max`` ventanas equiespaciadas que cubren
      de inicio a fin (incluye primera y última muestra).
    """
    n_samples = int(n_samples)
    target_length = int(target_length)
    if n_samples <= target_length or int(windows_max) <= 1:
        if n_samples <= target_length:
            return [0]
        return [(n_samples - target_length) // 2]
    windows_max = int(windows_max)
    span = n_samples - target_length
    if windows_max == 1 or span <= 0:
        return [span // 2]
    # Ventanas no solapadas si caben; si no, solapadas equiespaciadas.
    non_overlap = span // target_length + 1
    k = min(windows_max, max(1, non_overlap))
    if k == 1:
        return [span // 2]
    return [int(round(span * i / (k - 1))) for i in range(k)]


def fix_length(signal: np.ndarray, target_len: int = WINDOW_LENGTH, mode: str = "center") -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"La señal debe ser 2D; shape={signal.shape}")
    n = signal.shape[0]
    if n == target_len:
        return signal
    if n > target_len:
        if mode == "center":
            start = (n - target_len) // 2
        elif mode == "random":
            start = np.random.randint(0, n - target_len + 1)
        else:
            raise ValueError(f"Modo de recorte inválido: {mode}")
        return signal[start:start + target_len, :]
    return np.pad(signal, ((0, target_len - n), (0, 0)), mode="constant")


def preprocess_ecg_array(
    signal: np.ndarray,
    sampling_rate: float = TARGET_FS,
    lead_names: Iterable[str] | None = None,
    target_fs: int = TARGET_FS,
    target_length: int | None = WINDOW_LENGTH,
    normalize: bool = True,
    norm_mode: str = DEFAULT_NORM_MODE,
    adc_gains: Iterable[float] | None = None,
    adc_baselines: Iterable[float] | None = None,
    units: str = "digital",
    bandpass: bool = True,
    clip_mv: float = PHYSICAL_CLIP_MV,
    crop_mode: str = "center",
) -> np.ndarray:
    """Convierte una señal 12-derivaciones a lista para el modelo.

    Modos de normalización (``norm_mode``):

    - ``\"physical\"`` (defecto v2): convierte a mV con ganancia/baseline del
      header, aplica pasa-banda 0.5–50 Hz y clip ±``clip_mv``. Preserva
      amplitudes absolutas y relativas (voltaje y eje eléctrico).
    - ``\"global_zscore\"``: z-score global del registro (preserva eje).
    - ``\"per_lead_zscore\"``: legacy v1 (requiere ``normalize=True``);
      imprescindible para checkpoints entrenados con el esquema v1.

    Si ``target_length`` es ``None`` no se recorta/rellena (útil para extraer
    ventanas deslizantes sobre el registro completo).
    """
    data = np.asarray(signal, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if data.ndim != 2:
        raise ValueError(f"Se esperaba ECG 2D; shape={data.shape}")

    # Acepta tanto (samples, leads) como (leads, samples).
    if data.shape[0] == NUM_LEADS and data.shape[1] != NUM_LEADS:
        data = data.T

    if lead_names is not None:
        # Reordena gains/baselines junto con la señal si vienen en orden del header.
        names = [normalize_lead_name(n) for n in lead_names]
        if adc_gains is not None and adc_baselines is not None and len(names) == len(list(adc_gains)):
            order = [names.index(lead) for lead in STANDARD_LEAD_ORDER]
            adc_gains = [list(adc_gains)[k] for k in order]
            adc_baselines = [list(adc_baselines)[k] for k in order]
        data = reorder_leads(data, lead_names)
    elif data.shape[1] != NUM_LEADS:
        raise ValueError(f"Se requieren {NUM_LEADS} derivaciones; shape={data.shape}")

    norm_mode = str(norm_mode or DEFAULT_NORM_MODE).lower()
    if norm_mode not in {"physical", "global_zscore", "per_lead_zscore"}:
        raise ValueError(f"norm_mode inválido: {norm_mode}")

    if normalize and norm_mode == "physical" and str(units).lower() != "mv":
        data = to_physical_units(data, adc_gains=adc_gains, adc_baselines=adc_baselines)

    data = resample_signal(data, sampling_rate, target_fs)

    if normalize:
        if bandpass and norm_mode == "physical":
            data = bandpass_filter(data, sampling_rate=target_fs)
        if norm_mode == "physical":
            data = np.clip(np.asarray(data, dtype=np.float32), -float(clip_mv), float(clip_mv))
        elif norm_mode == "global_zscore":
            data = normalize_signal_global(data)
        else:
            data = normalize_signal(data)

    if target_length is not None:
        data = fix_length(data, int(target_length), mode=crop_mode)
    return np.asarray(data, dtype=np.float32)


def preprocess_to_windows(
    signal: np.ndarray,
    sampling_rate: float = TARGET_FS,
    lead_names: Iterable[str] | None = None,
    target_fs: int = TARGET_FS,
    target_length: int = WINDOW_LENGTH,
    windows_max: int = 1,
    **preprocess_kwargs,
) -> tuple[np.ndarray, list[int]]:
    """Preprocesa el registro completo y extrae hasta ``windows_max`` ventanas.

    Devuelve ``(ventanas, inicios)`` con ``ventanas`` de forma
    ``(K, target_length, 12)``. Con ``windows_max=1`` equivale al recorte
    centrado legacy.
    """
    full = preprocess_ecg_array(
        signal,
        sampling_rate=sampling_rate,
        lead_names=lead_names,
        target_fs=target_fs,
        target_length=None,
        **preprocess_kwargs,
    )
    starts = compute_window_starts(full.shape[0], target_length, windows_max)
    windows = []
    for start in starts:
        piece = full[start:start + target_length, :]
        if piece.shape[0] < target_length:
            piece = np.pad(piece, ((0, target_length - piece.shape[0]), (0, 0)), mode="constant")
        windows.append(piece.astype(np.float32, copy=False))
    return np.stack(windows).astype(np.float32, copy=False), [int(s) for s in starts]


def load_mat_signal(mat_path: str | Path, num_leads: int = NUM_LEADS) -> np.ndarray:
    mat = loadmat(mat_path)
    if "val" not in mat:
        raise KeyError("No existe variable 'val' en el .mat")
    raw = np.asarray(mat["val"], dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Shape inesperado en .mat: {raw.shape}")
    if raw.shape[0] == num_leads:
        return raw.T
    if raw.shape[1] == num_leads:
        return raw
    # Fallback para el formato habitual WFDB challenge: (leads, samples).
    return raw.T


def load_wfdb_record(
    mat_path: str | Path,
    hea_path: str | Path | None = None,
    norm_mode: str = DEFAULT_NORM_MODE,
    bandpass: bool = True,
    target_length: int | None = WINDOW_LENGTH,
    crop_mode: str = "center",
) -> tuple[np.ndarray, dict]:
    """Carga un registro WFDB challenge desde par ``.mat``/``.hea``."""
    mat_path = Path(mat_path)
    hea_path = Path(hea_path) if hea_path is not None else mat_path.with_suffix(".hea")
    meta = parse_header(hea_path)
    raw = load_mat_signal(mat_path, meta["num_leads"])
    processed = preprocess_ecg_array(
        raw,
        sampling_rate=meta["sampling_freq"],
        lead_names=meta["lead_names"],
        target_length=target_length,
        norm_mode=norm_mode,
        adc_gains=meta.get("adc_gains"),
        adc_baselines=meta.get("adc_baselines"),
        units="digital",
        bandpass=bandpass,
        crop_mode=crop_mode,
    )
    meta["norm_mode"] = norm_mode
    meta["bandpass"] = bool(bandpass)
    return processed, meta


def load_wfdb_record_windows(
    mat_path: str | Path,
    hea_path: str | Path | None = None,
    windows_max: int = 4,
    norm_mode: str = DEFAULT_NORM_MODE,
    bandpass: bool = True,
) -> tuple[np.ndarray, dict]:
    """Carga un registro WFDB y devuelve ``(ventanas, meta)``.

    ``ventanas`` tiene forma ``(K, 5000, 12)`` con K<=``windows_max``.
    Para registros de 10 s, K=1 (idéntico a :func:`load_wfdb_record`).
    """
    mat_path = Path(mat_path)
    hea_path = Path(hea_path) if hea_path is not None else mat_path.with_suffix(".hea")
    meta = parse_header(hea_path)
    raw = load_mat_signal(mat_path, meta["num_leads"])
    windows, starts = preprocess_to_windows(
        raw,
        sampling_rate=meta["sampling_freq"],
        lead_names=meta["lead_names"],
        windows_max=windows_max,
        norm_mode=norm_mode,
        adc_gains=meta.get("adc_gains"),
        adc_baselines=meta.get("adc_baselines"),
        units="digital",
        bandpass=bandpass,
    )
    meta["norm_mode"] = norm_mode
    meta["bandpass"] = bool(bandpass)
    meta["window_starts"] = starts
    meta["num_windows"] = int(windows.shape[0])
    return windows, meta


def read_csv_ecg(csv_path: str | Path) -> tuple[np.ndarray, float, list[str], dict]:
    """Lee CSV para inferencia.

    Líneas iniciales ``#`` (todas las que haya):

    - ``# Sampling Rate: 500 Hz`` -> frecuencia de muestreo;
    - ``# Preprocessed: <norm_mode>`` -> marcador escrito por la webapp al
      convertir WFDB->CSV; si está presente, la señal ya está en 500 Hz /
      5000 muestras / 12 derivaciones y NO debe reprocesarse.

    Devuelve ``(values, sampling_rate, lead_names, csv_meta)``.
    """
    csv_path = Path(csv_path)
    sampling_rate: float = TARGET_FS
    skiprows = 0
    comment_lines: list[str] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text.startswith("#"):
                break
            skiprows += 1
            comment_lines.append(text)
            lower = text.lower()
            if "sampling rate" in lower:
                try:
                    sampling_rate = float(text.split(":", 1)[1].strip().split()[0])
                except Exception:
                    sampling_rate = TARGET_FS

    preprocessed_mode: str | None = None
    for text in comment_lines:
        if text.lower().startswith("# preprocessed:"):
            preprocessed_mode = text.split(":", 1)[1].strip().lower() or "unknown"
            # Formato "physical, bandpass=true, units=mV" -> modo es primer token.
            preprocessed_mode = preprocessed_mode.split(",")[0].strip().split()[0]

    df = pd.read_csv(csv_path, skiprows=skiprows)
    csv_meta = {
        "sampling_rate": float(sampling_rate),
        "preprocessed_mode": preprocessed_mode,
        "comment_lines": comment_lines,
    }
    return df.values.astype(np.float32), float(sampling_rate), list(df.columns), csv_meta


def infer_csv_units(values: np.ndarray) -> str:
    """Heurística de unidades para CSV crudos de usuario.

    - Si la amplitud máxima supera 20 -> valores digitales/ADC (requieren
      conversión con ganancia 1000) -> ``\"digital\"``.
    - En otro caso se asumen milivoltios -> ``\"mv\"``.
    """
    try:
        peak = float(np.nanmax(np.abs(np.asarray(values, dtype=np.float64))))
    except Exception:
        return "mv"
    return "digital" if peak > 20 else "mv"


# ---------------------------------------------------------------------------
# Construcción de dataset HDF5.
# ---------------------------------------------------------------------------
