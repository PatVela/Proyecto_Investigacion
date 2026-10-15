"""Servicio de inferencia y conversión WFDB->CSV para la webapp (el trazado está en ``plotting.py`` y la comparación
con etiquetas reales en ``labels.py``).

Entradas soportadas por la app:
    - CSV de 12 derivaciones.
    - Par WFDB del Challenge: archivo .hea + archivo .mat.

Si se sube .hea + .mat, la señal se lee, se preprocesa a 500 Hz / 5000 muestras
/ 12 derivaciones y se guarda además un CSV convertido para descarga.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

from ecg import load, util
from ecg.calibration import apply_platt, apply_temperature, ensemble_probabilities
from ecg.predict import load_model, predict_array, predict_windows, prepare_csv_windows, preprocessing_for_checkpoint
from webapp.labels import build_label_comparison, group_for_class, split_reference_label_text, translate_classes_to_checkpoint
from webapp.plotting import plot_ecg_array


REPO_ROOT = Path(__file__).resolve().parent.parent
LEADS = load.STANDARD_LEAD_ORDER

# Texto mostrado cuando el checkpoint no declara esquema (no se presenta como si lo declarara).
UNDECLARED_SCHEMA_TEXT = "sin esquema declarado (se asumió el actual)"
# Un CSV no declara sus unidades: se infieren (ecg.signal.infer_csv_units) y se muestran para que se pueda ver si se acertó.
CSV_UNITS_TEXT = {"mv": " · unidades del CSV asumidas: mV", "digital": " · unidades del CSV asumidas: cuentas ADC (÷ 1000)"}
SLIDING_WINDOWS_MAX = 4
# Registro de más de ~20 s: se analizan hasta 4 ventanas y la probabilidad final es la MEDIA de las calibradas.
# Los umbrales y la calibración se ajustaron con una sola ventana; el máximo de varias ventanas correlacionadas
# queda sesgado hacia arriba y en 1019 registros largos de test añadía 85 falsos positivos (426 -> 511, +20 %)
# sin ganar F1 (0,6108 -> 0,6052). La media de calibradas empata con la ventana única (0,6109; 433 FP).
SLIDING_AGGREGATE = "mean"


# ---------------------------------------------------------------------------
# Rutas / modelo
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Modelos en memoria y concurrencia
# ---------------------------------------------------------------------------
# Cada petición recargaba los checkpoints desde disco. Ahora se cargan una vez y se
# reutilizan (se invalidan si el archivo cambia); se descartan los pesos duplicados
# del checkpoint y el estado del optimizador, que no se usan al inferir.
_MODEL_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_MODEL_CACHE_LOCK = threading.Lock()
MODEL_CACHE_SIZE = max(1, int(os.environ.get("ECG_MODEL_CACHE_SIZE", "4")))
_HEAVY_CHECKPOINT_KEYS = ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "pos_weight")

# Máximo de inferencias simultáneas (acota CPU y memoria; el resto espera su turno).
_INFERENCE_SLOTS = threading.BoundedSemaphore(max(1, int(os.environ.get("ECG_MAX_CONCURRENT_INFERENCE", "2"))))


SERVING_DEVICES = ("auto", "cpu", "cuda")


def serving_device():
    """``torch.device`` de inferencia según ``ECG_DEVICE`` (``auto`` = GPU si existe, ``cpu`` o ``cuda``).

    Un ``cuda`` explícito sin GPU es un error: no se sustituye en silencio por la CPU.
    """
    import torch

    requested = os.environ.get("ECG_DEVICE", "auto").strip().lower() or "auto"
    if requested not in SERVING_DEVICES:
        raise ValueError(f"ECG_DEVICE debe ser uno de {', '.join(SERVING_DEVICES)}; recibido {requested!r}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ECG_DEVICE=cuda pero CUDA no está disponible en este equipo")
    return util.get_device(requested)


def get_cached_model(path: str | Path, device):
    """``(model, checkpoint_ligero, class_names)`` con caché por ruta, versión del archivo y dispositivo."""
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, str(device))
    with _MODEL_CACHE_LOCK:
        hit = _MODEL_CACHE.get(key)
        if hit is not None:
            _MODEL_CACHE.move_to_end(key)
            return hit
        model, checkpoint, class_names = load_model(path, device=device)
        light = {k: v for k, v in checkpoint.items() if k not in _HEAVY_CHECKPOINT_KEYS}
        _MODEL_CACHE[key] = (model, light, class_names)
        for stale in [k for k in _MODEL_CACHE if k[0] == key[0] and k != key]:     # versión anterior del mismo archivo
            del _MODEL_CACHE[stale]
        while len(_MODEL_CACHE) > MODEL_CACHE_SIZE:
            _MODEL_CACHE.popitem(last=False)
        return _MODEL_CACHE[key]


def clear_model_cache() -> None:
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


# Enlace markdown pegado junto a una ruta, p. ej. [best.pt](http://...). Con cuantificadores acotados y sin "[" dentro del
# texto del enlace, el costo es lineal: la versión anterior (``\[([^\]]+)\]\([^\)]+\)``) era cuadrática y una sola
# petición con "[" * 16000 mantenía el GIL más de un segundo (minutos con 500 KB).
_MARKDOWN_LINK = re.compile(r"\[([^\[\]]{1,512})\]\(([^)]{1,2048})\)")
MAX_MARKDOWN_SCAN_CHARS = 8192


def clean_path_text(value: str | os.PathLike | None) -> str:
    """Limpia rutas pegadas desde markdown, por ejemplo [best.pt](http://...)."""
    if value is None:
        return ""
    text = str(value).strip().strip('"').strip("'")
    if len(text) <= MAX_MARKDOWN_SCAN_CHARS:             # un enlace pegado nunca es tan largo: lo demás no se escanea
        text = _MARKDOWN_LINK.sub(r"\1", text)
    return text.strip()


def clean_field_text(value: str | os.PathLike | None, max_chars: int) -> str:
    """:func:`clean_path_text` con tope de longitud, para campos de formulario que llegan del cliente."""
    return clean_path_text(value)[:max_chars]


def resolve_existing_path(path: str | os.PathLike | None, base: Path = REPO_ROOT) -> Path | None:
    text = clean_path_text(path)
    if not text:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def resolve_model_path(model_path: str | None = None, saved_dir: str | None = None) -> Path:
    """Resuelve el checkpoint indicado o, si no se indica ninguno, busca el mejor en saved_dir.

    Una ruta explícita que no existe es un error: no se sustituye en silencio por otro
    modelo (en un ensemble eso promediaría el modelo equivocado sin avisar).
    """
    explicit = resolve_existing_path(model_path)
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"El checkpoint indicado no existe: {explicit}")

    saved_text = clean_path_text(saved_dir) or os.environ.get("ECG_SAVED", "saved")
    saved = Path(saved_text)
    if not saved.is_absolute():
        saved = REPO_ROOT / saved

    best = util.best_checkpoint(saved)
    if best is not None and best.exists():
        return best.resolve()

    message = "No se encontró un checkpoint .pt válido."
    message += f" Directorio de búsqueda: {saved}."
    raise FileNotFoundError(message)


# ---------------------------------------------------------------------------
# Preparación de entradas
# ---------------------------------------------------------------------------
ALLOWED_UPLOAD_SUFFIXES = {".csv", ".hea", ".mat"}
MAX_UPLOAD_FILES = 2


def save_uploaded_files(file_storages: Iterable, upload_dir: str | Path) -> list[Path]:
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for storage in file_storages:
        if storage is None or not getattr(storage, "filename", ""):
            continue
        filename = secure_filename(storage.filename)
        if not filename:
            continue
        if Path(filename).suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
            raise ValueError(
                f"Extensión no permitida: {filename!r}. Suba un .csv o el par .hea + .mat."
            )
        if len(saved_paths) >= MAX_UPLOAD_FILES:
            raise ValueError(f"Máximo {MAX_UPLOAD_FILES} archivos por solicitud.")
        path = upload_dir / filename
        storage.save(path)
        saved_paths.append(path)

    if not saved_paths:
        raise ValueError("No se recibió ningún archivo válido.")
    return saved_paths


def _find_file_by_suffix(paths: Iterable[Path], suffix: str) -> Path | None:
    suffix = suffix.lower()
    matches = [path for path in paths if path.suffix.lower() == suffix]
    if not matches:
        return None
    return matches[0]


def write_processed_csv(signal: np.ndarray, output_path: str | Path, norm_mode: str | None = None) -> Path:
    """Guarda señal preprocesada (5000,12) como CSV descargable.

    Incluye marcador ``# Preprocessed: <norm_mode>`` para que una
    re-subida del CSV convertido NO se reprocese por segunda vez.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(np.asarray(signal), columns=LEADS)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# Sampling Rate: {load.TARGET_FS} Hz\n")
        if norm_mode:
            handle.write(f"# Preprocessed: {norm_mode}\n")
        df.to_csv(handle, index=False)
    return output_path


def prepare_saved_input(
    saved_paths: list[Path],
    result_dir: str | Path,
    norm_mode: str | None = None,
    bandpass: bool | None = None,
) -> dict:
    """Devuelve ventanas preprocesadas y metadata desde CSV o par .hea+.mat.

    ``processed_windows`` tiene forma ``(K, 5000, 12)``; ``processed_signal``
    es la ventana central (gráfico/CSV). El modo de normalización debe venir
    del checkpoint (ver :func:`ecg.predict.preprocessing_for_checkpoint`).
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    norm_mode = norm_mode or load.DEFAULT_NORM_MODE
    bandpass = True if bandpass is None else bool(bandpass)

    csv_path = _find_file_by_suffix(saved_paths, ".csv")
    hea_path = _find_file_by_suffix(saved_paths, ".hea")
    mat_path = _find_file_by_suffix(saved_paths, ".mat")

    if csv_path is not None and len(saved_paths) == 1:
        windows, csv_info = prepare_csv_windows(
            csv_path, norm_mode=norm_mode, bandpass=bandpass, windows_max=SLIDING_WINDOWS_MAX
        )
        if csv_info["reused"]:
            windows = windows.astype(np.float32, copy=False)
        sampling_rate, lead_names = csv_info["sampling_rate"], csv_info["lead_names"]
        marker, reused = csv_info["marker"], csv_info["reused"]
        center = windows.shape[0] // 2
        return {
            "input_type": "csv",
            "record_name": csv_path.stem,
            "source_files": [str(csv_path)],
            "csv_path": csv_path,
            "converted_csv_path": None,
            "processed_signal": windows[center],
            "processed_windows": windows,
            "csv_reused_without_reprocessing": reused,
            "csv_preprocessed_marker": marker or None,
            "csv_units": csv_info.get("units"),
            "original_sampling_rate": sampling_rate,
            "lead_names": lead_names,
            "dx_codes": [],
            "matched_classes": [],
            "true_label_source": None,
        }

    if hea_path is not None and mat_path is not None and len(saved_paths) == 2:
        if hea_path.stem != mat_path.stem:
            raise ValueError(
                "El archivo .hea y el archivo .mat deben pertenecer al mismo registro "
                f"(recibidos: {hea_path.name} y {mat_path.name})."
            )
        windows, meta = load.load_wfdb_record_windows(
            mat_path, hea_path, windows_max=SLIDING_WINDOWS_MAX,
            norm_mode=norm_mode, bandpass=bandpass,
        )
        record_name = meta.get("record_name") or hea_path.stem or mat_path.stem
        center = windows.shape[0] // 2
        converted_csv = write_processed_csv(
            windows[center], result_dir / f"{secure_filename(record_name)}_convertido.csv",
            norm_mode=norm_mode,
        )
        dx_codes = meta.get("dx_codes", [])
        return {
            "input_type": "wfdb_hea_mat",
            "record_name": record_name,
            "source_files": [str(hea_path), str(mat_path)],
            "csv_path": converted_csv,
            "converted_csv_path": converted_csv,
            "processed_signal": windows[center],
            "processed_windows": windows,
            "window_starts": meta.get("window_starts", [0]),
            "original_sampling_rate": meta.get("sampling_freq", load.TARGET_FS),
            "lead_names": meta.get("lead_names", LEADS),
            "dx_codes": dx_codes,
            "matched_classes": load.matched_class_names(dx_codes),
            "true_label_source": "wfdb_header_dx",
            "age": meta.get("age"),
            "sex": meta.get("sex"),
        }

    suffixes = ", ".join(sorted({path.suffix.lower() for path in saved_paths}))
    raise ValueError(
        "Formato no soportado. Suba un CSV de 12 derivaciones o el par .hea + .mat del mismo registro. "
        f"Extensiones recibidas: {suffixes or 'ninguna'}."
    )


def threshold_values_for_class_names(
    class_names: list[str],
    threshold: float = 0.5,
    thresholds: dict | list | np.ndarray | None = None,
) -> np.ndarray:
    """Devuelve umbral por clase; si no hay thresholds calibrados usa fallback global."""
    values = np.full(len(class_names), float(threshold), dtype=np.float32)
    if thresholds is None:
        return values
    if isinstance(thresholds, dict):
        for i, name in enumerate(class_names):
            if name in thresholds:
                values[i] = float(thresholds[name])
            elif str(i) in thresholds:
                values[i] = float(thresholds[str(i)])
        return values
    arr = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    if arr.size != len(class_names):
        raise ValueError(f"thresholds tiene {arr.size} valores; se esperaban {len(class_names)}")
    return arr


def temperatures_for_class_names(
    class_names: list[str],
    temperatures: dict | list | np.ndarray | None = None,
) -> np.ndarray | None:
    """Devuelve vector (C,) de temperaturas o None si no hay calibración."""
    if temperatures is None:
        return None
    if isinstance(temperatures, dict):
        values = np.ones(len(class_names), dtype=np.float64)
        for i, name in enumerate(class_names):
            if name in temperatures:
                values[i] = float(temperatures[name])
            elif str(i) in temperatures:
                values[i] = float(temperatures[str(i)])
        return values
    arr = np.asarray(list(temperatures), dtype=np.float64).reshape(-1)
    if arr.size != len(class_names):
        raise ValueError(f"temperatures tiene {arr.size} valores; se esperaban {len(class_names)}")
    return arr


def platt_for_class_names(
    class_names: list[str],
    platt: dict | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Vectores ``(a, b)`` (C,) de una calibración afín ``{clase: {"a": ..., "b": ...}}``; None si no hay.

    Una clase ausente queda en la identidad (a=1, b=0), igual que la temperatura ausente queda en 1.
    """
    if platt is None:
        return None
    a = np.ones(len(class_names), dtype=np.float64)
    b = np.zeros(len(class_names), dtype=np.float64)
    for i, name in enumerate(class_names):
        entry = platt.get(name)
        if entry is not None:
            a[i], b[i] = float(entry["a"]), float(entry["b"])
    return a, b


def _prediction_rows(probabilities: np.ndarray, class_names: list[str], thresholds: np.ndarray) -> list[dict]:
    rows = []
    for i, prob in enumerate(probabilities):
        group = group_for_class(class_names[i])
        thr = float(thresholds[i])
        prob_f = float(prob)
        rows.append({
            "index": i,
            "class": class_names[i],
            "display_name": group["display_name"],
            "description": group["description"],
            "snomed_codes": group["codes"],
            "probability": prob_f,
            "probability_pct": round(prob_f * 100, 2),
            "threshold": thr,
            "threshold_pct": round(thr * 100, 2),
            "margin": float(prob_f - thr),
            "near_threshold": bool(abs(prob_f - thr) < 0.05),
            "prediction": int(prob_f >= thr),
        })
    return rows


def apply_normal_fallback(
    rows: list[dict],
    min_nsr_probability: float = 0.40,
) -> dict:
    """Postprocesamiento opcional para evitar salida multilabel vacía.

    En inferencia multilabel puede ocurrir que ninguna clase supere su umbral
    calibrado. Para la demo clínica/académica es más claro declarar una salida
    final cuando la clase normal tiene probabilidad razonable y todas las clases
    patológicas quedaron bajo su propio umbral. La regla NO modifica las
    probabilidades del modelo; solo añade NSR como decisión final postprocesada.
    """
    positives = [row for row in rows if int(row.get("prediction", 0)) == 1]
    info = {
        "enabled": True,
        "applied": False,
        "rule": "if_no_positive_and_NSR_probability_ge_min",
        "min_nsr_probability": float(min_nsr_probability),
        "added_class": None,
        "reason": "",
    }
    if positives:
        info["reason"] = "already_has_positive_predictions"
        return info
    nsr_row = next((row for row in rows if row.get("class") == "NSR"), None)
    if nsr_row is None:
        info["reason"] = "NSR_not_present_in_class_names"
        return info
    nsr_prob = float(nsr_row.get("probability", 0.0))
    if nsr_prob >= float(min_nsr_probability):
        nsr_row["prediction"] = 1
        nsr_row["postprocessed"] = True
        nsr_row["postprocess_reason"] = "normal_fallback_no_positive"
        info.update({
            "applied": True,
            "added_class": "NSR",
            "reason": "no_class_exceeded_threshold_and_NSR_probability_is_reasonable",
        })
    else:
        info["reason"] = "NSR_probability_below_min"
    return info


def _infer_windows(model, windows: np.ndarray, device) -> np.ndarray:
    """Probabilidades crudas por ventana, forma ``(K, 12)``; la agregación se hace después de calibrar."""
    with _INFERENCE_SLOTS:
        return _infer_windows_unlocked(model, windows, device)


def _infer_windows_unlocked(model, windows: np.ndarray, device) -> np.ndarray:
    if windows.shape[0] == 1:
        return predict_array(model, windows[0], device=device)[None, :]
    return predict_windows(model, windows, device=device, aggregate=SLIDING_AGGREGATE)[1]


def aggregate_window_probabilities(per_window: np.ndarray) -> tuple[np.ndarray, str]:
    """Probabilidad final de un registro a partir de las de sus ventanas YA calibradas, y cómo se obtuvo.

    Una ventana: se devuelve tal cual (``"single"``, igual que la evaluación). Varias: su media (``SLIDING_AGGREGATE``).
    """
    per_window = np.asarray(per_window)
    if per_window.ndim != 2 or per_window.shape[0] < 1:
        raise ValueError(f"per_window debe tener forma (K, C) con K >= 1, recibido {per_window.shape}")
    if per_window.shape[0] == 1:
        return per_window[0], "single"
    return per_window.mean(axis=0).astype(np.float32), SLIDING_AGGREGATE


def _calibrate_model_windows(per_window, class_names, temperatures, platt):
    """Calibra las ventanas ``(K, C)`` de un modelo: la afín gana a la temperatura (son alternativas).

    Devuelve ``(ventanas_calibradas, temperaturas_usadas | None, platt_usada | None)``.
    """
    platt_values = platt_for_class_names(class_names, platt)
    temp_values = None if platt_values is not None else temperatures_for_class_names(class_names, temperatures=temperatures)
    if platt_values is not None:
        per_window = apply_platt(np.asarray(per_window), *platt_values)
    elif temp_values is not None:
        per_window = apply_temperature(np.asarray(per_window), temp_values)
    return per_window, temp_values, platt_values


def run_prediction_from_saved_paths(
    saved_paths: list[Path],
    model_path: str | None,
    saved_dir: str | None,
    result_dir: str | Path,
    threshold: float = 0.5,
    reference_labels: str | Iterable[str] | None = None,
    thresholds: dict | list | np.ndarray | None = None,
    threshold_source: str | None = None,
    temperatures: dict | list | np.ndarray | None = None,
    temperature_source: str | None = None,
    normal_fallback_min_prob: float | None = None,
    model_path_b: str | None = None,
    temperatures_b: dict | list | np.ndarray | None = None,
    temperature_source_b: str | None = None,
    alpha: float = 0.5,
    platt: dict | None = None,
    platt_source: str | None = None,
    platt_b: dict | None = None,
    platt_source_b: str | None = None,
    platt_ensemble: dict | None = None,
    platt_source_ensemble: str | None = None,
) -> dict:
    """Predice a partir de archivos ya guardados en disco.

    Calibración: por modelo, temperatura (``temperatures``) o calibración afín (``platt``; si llegan las dos para el
    mismo modelo gana la afín) y, en un ensemble, una calibración afín del promedio (``platt_ensemble``). Cada ventana
    recorre toda la cadena (modelo -> calibración -> ensemble -> calibración del promedio) y solo al final se promedian
    las ventanas (:func:`aggregate_window_probabilities`): así los umbrales, ajustados con una ventana, se aplican a una
    probabilidad con el mismo significado que en la evaluación.
    """
    manual_true, unrecognized = split_reference_label_text(reference_labels)
    if unrecognized:                                  # antes de cargar el modelo: es una entrada inválida (400), no un fallo
        raise ValueError(
            f"Etiquetas reales no reconocidas: {', '.join(unrecognized)}. Use clases ({', '.join(load.CLASS_NAMES)}) "
            "o códigos SNOMED de esas clases, separados por comas."
        )
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    resolved_model = resolve_model_path(model_path=model_path, saved_dir=saved_dir)

    device = serving_device()
    model, checkpoint, class_names = get_cached_model(resolved_model, device)
    pre = preprocessing_for_checkpoint(checkpoint)
    ckpt_schema = checkpoint.get("label_schema")
    prepared = prepare_saved_input(saved_paths, result_dir, norm_mode=pre["norm_mode"], bandpass=pre["bandpass"])

    windows = np.asarray(prepared.get("processed_windows", prepared["processed_signal"][None, ...]), dtype=np.float32)
    per_window, temp_values, platt_values = _calibrate_model_windows(
        _infer_windows(model, windows, device), class_names, temperatures, platt)
    calibrated_a = temp_values is not None or platt_values is not None

    is_ensemble = model_path_b is not None and str(model_path_b).strip() != ""
    resolved_model_b = None
    temp_values_b = None
    platt_values_b = None
    platt_values_e = None
    calibrated_b = False
    if is_ensemble:
        resolved_model_b = resolve_model_path(model_path=model_path_b, saved_dir=saved_dir)
        _model_b, _checkpoint_b, class_names_b = get_cached_model(resolved_model_b, device)
        if list(class_names_b) != list(class_names):
            raise ValueError("Los checkpoints del ensemble usan distinto orden de clases.")
        per_window_b, temp_values_b, platt_values_b = _calibrate_model_windows(
            _infer_windows(_model_b, windows, device), class_names, temperatures_b, platt_b)
        calibrated_b = temp_values_b is not None or platt_values_b is not None
        per_window = ensemble_probabilities(np.asarray(per_window), np.asarray(per_window_b), alpha)
        platt_values_e = platt_for_class_names(class_names, platt_ensemble)
        if platt_values_e is not None:
            per_window = apply_platt(per_window, *platt_values_e)
        else:
            per_window = per_window.astype(np.float32)
    probabilities, aggregate_used = aggregate_window_probabilities(per_window)
    calibrated = calibrated_a or calibrated_b or platt_values_e is not None
    if platt_values_e is not None:
        calibration_method = "platt+ensemble"
    elif platt_values is not None or platt_values_b is not None:
        calibration_method = "platt" if (temp_values is None and temp_values_b is None) else "mixed"
    elif calibrated:
        calibration_method = "temperature"
    else:
        calibration_method = None
    calibration_text = {
        None: "sin calibrar (probabilidades directas del modelo)",
        "temperature": "temperature scaling por clase",
        "platt": "calibración afín por clase (a·logit + b) en cada modelo",
        "platt+ensemble": "calibración afín por clase en cada modelo y sobre el promedio del ensemble",
        "mixed": "calibración por clase (temperatura y/o afín según el modelo)",
    }[calibration_method]
    threshold_values = threshold_values_for_class_names(class_names, threshold=threshold, thresholds=thresholds)
    rows = _prediction_rows(probabilities, class_names, threshold_values)
    raw_sorted_rows = sorted(rows, key=lambda item: item["probability"], reverse=True)
    raw_positives = [dict(row) for row in raw_sorted_rows if row["prediction"] == 1]
    normal_fallback = {"enabled": False, "applied": False}
    if normal_fallback_min_prob is not None and float(normal_fallback_min_prob) > 0:
        normal_fallback = apply_normal_fallback(rows, min_nsr_probability=float(normal_fallback_min_prob))
    sorted_rows = sorted(rows, key=lambda item: item["probability"], reverse=True)
    positives = [row for row in sorted_rows if row["prediction"] == 1]
    using_class_thresholds = thresholds is not None

    manual_raw = clean_path_text(reference_labels) if isinstance(reference_labels, (str, os.PathLike)) else ""
    ckpt_matched = translate_classes_to_checkpoint(prepared.get("matched_classes", []), class_names)
    if manual_raw and manual_true:                    # solo separadores (",,") no cuenta como etiqueta manual
        label_comparison = build_label_comparison(
            translate_classes_to_checkpoint(manual_true, class_names), sorted_rows, "manual", manual_raw=manual_raw)
    elif prepared.get("true_label_source") == "wfdb_header_dx":
        label_comparison = build_label_comparison(
            ckpt_matched,
            sorted_rows,
            "wfdb_header_dx",
            dx_codes=prepared.get("dx_codes", []),
        )
    else:
        label_comparison = build_label_comparison(None, sorted_rows, None)

    plot_path = plot_ecg_array(
        prepared["processed_signal"],
        result_dir,
        prepared["record_name"],
    )

    age_value = prepared.get("age")
    try:
        age_clean = None if age_value is None or not np.isfinite(float(age_value)) else str(age_value)
    except Exception:
        age_clean = None if age_value is None else str(age_value)

    return {
        "status": "ok",
        "record_name": prepared["record_name"],
        "input_type": prepared["input_type"],
        "source_files": prepared["source_files"],
        "model_path": str(resolved_model),
        "model_path_b": str(resolved_model_b) if resolved_model_b is not None else None,
        "ensemble": bool(is_ensemble),
        "ensemble_alpha": float(alpha) if is_ensemble else None,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "model_type": f"Ensemble ResNet-34 ×2 (α={float(alpha):.2f})" if is_ensemble else "ResNet-34 1D tipo Hannun",
        "threshold": float(threshold),
        "thresholds": threshold_values.tolist(),
        "threshold_source": threshold_source or "fallback_global_0.5",
        "using_class_thresholds": bool(using_class_thresholds),
        "temperatures": temp_values.tolist() if temp_values is not None else None,
        "temperature_source": temperature_source if temp_values is not None else None,
        "temperatures_b": temp_values_b.tolist() if temp_values_b is not None else None,
        "temperature_source_b": temperature_source_b if temp_values_b is not None else None,
        "platt": {"a": platt_values[0].tolist(), "b": platt_values[1].tolist()} if platt_values is not None else None,
        "platt_source": platt_source if platt_values is not None else None,
        "platt_b": {"a": platt_values_b[0].tolist(), "b": platt_values_b[1].tolist()} if platt_values_b is not None else None,
        "platt_source_b": platt_source_b if platt_values_b is not None else None,
        "platt_ensemble": ({"a": platt_values_e[0].tolist(), "b": platt_values_e[1].tolist()}
                           if platt_values_e is not None else None),
        "platt_source_ensemble": platt_source_ensemble if platt_values_e is not None else None,
        "calibration_method": calibration_method,
        "calibrated": bool(calibrated),
        "num_classes": load.NUM_CLASSES,
        "num_leads": load.NUM_LEADS,
        "target_sampling_rate": load.TARGET_FS,
        "input_length": load.WINDOW_LENGTH,
        "window_seconds": load.WINDOW_SECONDS,
        "class_names": class_names,
        "predictions": sorted_rows,
        "positive_predictions": positives,
        "raw_positive_predictions": raw_positives,
        "normal_fallback": normal_fallback,
        "top_predictions": raw_sorted_rows[:3],
        "plot_path": str(plot_path),
        "plot_url": None,
        "plot_style": "ecg_paper_grid",
        "converted_csv_path": str(prepared["converted_csv_path"]) if prepared.get("converted_csv_path") else None,
        "converted_csv_url": None,
        "original_sampling_rate": prepared.get("original_sampling_rate"),
        "lead_names": prepared.get("lead_names"),
        "processed_shape": list(prepared["processed_signal"].shape),
        "num_windows": int(windows.shape[0]),
        "window_aggregation": aggregate_used,
        "window_starts": [int(s) for s in prepared.get("window_starts", [0])],
        "per_window_probabilities": per_window.tolist(),
        "norm_mode": pre["norm_mode"],
        "bandpass": bool(pre["bandpass"]),
        "label_schema": ckpt_schema or UNDECLARED_SCHEMA_TEXT,
        "dx_codes": prepared.get("dx_codes", []),
        "matched_classes": ckpt_matched,
        "label_comparison": label_comparison,
        "patient_age": age_clean,
        "patient_sex": prepared.get("sex", "Unknown"),
        "technical_details": {
            "preprocessing": (
                f"12 derivaciones -> reorden estándar -> 500 Hz -> {pre['norm_mode']}"
                f"{' + bandpass 0.5-50Hz' if pre['bandpass'] else ''} -> "
                f"{windows.shape[0]} ventana(s) de 5000 ({aggregate_used})"
                f"{CSV_UNITS_TEXT.get(prepared.get('csv_units'), '')}"
            ),
            "label_schema": ckpt_schema or UNDECLARED_SCHEMA_TEXT,
            "norm_mode": pre["norm_mode"],
            "activation": "sigmoid por clase",
            "loss": "BCEWithLogitsLoss durante entrenamiento",
            "decision_rule": (
                "probabilidad >= threshold calibrado por clase"
                if using_class_thresholds else f"probabilidad >= {threshold}"
            ),
            "threshold_source": threshold_source or "fallback_global_0.5",
            "calibration": calibration_text,
            "ensemble": (
                f"promedio ponderado α={float(alpha):.2f} de 2 checkpoints"
                if is_ensemble else "no (checkpoint único)"
            ),
            "device": str(device),
            "plot_style": "papel milimetrado ECG",
        },
        "warning": "Uso académico. Estas predicciones no constituyen diagnóstico médico.",
    }


def run_prediction_from_uploads(
    file_storages: Iterable,
    model_path: str | None,
    saved_dir: str | None,
    upload_root: str | Path,
    results_root: str | Path,
    threshold: float = 0.5,
    reference_labels: str | Iterable[str] | None = None,
    thresholds: dict | list | np.ndarray | None = None,
    threshold_source: str | None = None,
    temperatures: dict | list | np.ndarray | None = None,
    temperature_source: str | None = None,
    normal_fallback_min_prob: float | None = None,
    model_path_b: str | None = None,
    temperatures_b: dict | list | np.ndarray | None = None,
    temperature_source_b: str | None = None,
    alpha: float = 0.5,
    platt: dict | None = None,
    platt_source: str | None = None,
    platt_b: dict | None = None,
    platt_source_b: str | None = None,
    platt_ensemble: dict | None = None,
    platt_source_ensemble: str | None = None,
) -> dict:
    """Guarda archivos subidos, convierte si aplica y ejecuta inferencia."""
    job_id = uuid.uuid4().hex[:12]
    upload_dir = Path(upload_root) / job_id
    result_dir = Path(results_root) / job_id
    try:
        saved_paths = save_uploaded_files(file_storages, upload_dir)
        result = run_prediction_from_saved_paths(
            saved_paths=saved_paths,
            model_path=model_path,
            saved_dir=saved_dir,
            result_dir=result_dir,
            threshold=threshold,
            reference_labels=reference_labels,
            thresholds=thresholds,
            threshold_source=threshold_source,
            temperatures=temperatures,
            temperature_source=temperature_source,
            normal_fallback_min_prob=normal_fallback_min_prob,
            model_path_b=model_path_b,
            temperatures_b=temperatures_b,
            temperature_source_b=temperature_source_b,
            alpha=alpha,
            platt=platt,
            platt_source=platt_source,
            platt_b=platt_b,
            platt_source_b=platt_source_b,
            platt_ensemble=platt_ensemble,
            platt_source_ensemble=platt_source_ensemble,
        )
    finally:
        # El ECG subido no se conserva: solo quedan los informes generados (que caducan).
        if os.environ.get("ECG_KEEP_UPLOADS") != "1":
            shutil.rmtree(upload_dir, ignore_errors=True)
    result["job_id"] = job_id
    return result
