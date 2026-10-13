"""Servicio de inferencia, conversión WFDB->CSV y gráficos para la webapp.

Entradas soportadas por la app:
    - CSV de 12 derivaciones.
    - Par WFDB del Challenge: archivo .hea + archivo .mat.

Si se sube .hea + .mat, la señal se lee, se preprocesa a 500 Hz / 5000 muestras
/ 12 derivaciones y se guarda además un CSV convertido para descarga.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

# matplotlib necesita un directorio de configuración; en contenedores mínimos puede no
# existir un "home" y la importación fallaría (RuntimeError) antes de arrancar la app.
if not os.environ.get("MPLCONFIGDIR"):
    try:
        Path.home()
    except RuntimeError:
        os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "ecg-mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

from ecg import load, util
from ecg.calibration import apply_platt, apply_temperature
from ecg.predict import load_model, predict_array, predict_windows, prepare_csv_windows, preprocessing_for_checkpoint


REPO_ROOT = Path(__file__).resolve().parent.parent
LEADS = load.STANDARD_LEAD_ORDER

# Traducción de nombres v2 -> v1 para comparar contra checkpoints legacy.
V2_TO_V1_CLASS = {"AxisDev": "LAD", "BBB": "RBBB"}
# Texto mostrado cuando el checkpoint no declara esquema (no se presenta como si lo declarara).
UNDECLARED_SCHEMA_TEXT = "sin esquema declarado (se asumió el actual)"
SLIDING_WINDOWS_MAX = 4
SLIDING_AGGREGATE = "max"


def _group_for_class(name: str) -> dict:
    for group in load.CLASS_GROUPS_12:
        if group["name"] == name:
            return group
    for group in load.CLASS_GROUPS_12_V1_LEGACY:
        if group["name"] == name:
            return {"name": name, "display_name": name, "description": "", "codes": group["codes"]}
    return {"name": name, "display_name": name, "description": "", "codes": []}


def translate_classes_to_checkpoint(names: Iterable[str], class_names: list[str]) -> list[str]:
    """Traduce clases v2 a nombres v1 si el checkpoint es legacy."""
    names = list(names)
    if "AxisDev" in class_names or "BBB" in class_names or "LAD" not in class_names:
        return names
    return [V2_TO_V1_CLASS.get(n, n) for n in names]


# ---------------------------------------------------------------------------
# Rutas / modelo
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Modelos en memoria, concurrencia y matplotlib
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
# pyplot mantiene estado global y no es seguro entre hilos (waitress usa varios).
_PLOT_LOCK = threading.Lock()


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


# ---------------------------------------------------------------------------
# Etiquetas reales / comparación multilabel
# ---------------------------------------------------------------------------
def parse_reference_label_text(value: str | Iterable[str] | None) -> list[str]:
    """Convierte texto manual con clases o SNOMED a nombres del esquema 12-clases."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,;\n]+", value)
    else:
        parts = list(value)

    upper_to_name = {name.upper(): name for name in load.CLASS_NAMES}
    upper_to_name.update({"LAD": "AxisDev", "RBBB": "BBB"})  # alias legacy v1
    display_to_name = {group["display_name"].upper(): group["name"] for group in load.CLASS_GROUPS_12}
    found: list[str] = []
    seen = set()
    for part in parts:
        raw = str(part).strip()
        if not raw:
            continue
        code = load.normalize_snomed_code(raw)
        name = None
        if code in load.CODE_TO_CLASS_INDEX:
            name = load.CLASS_NAMES[load.CODE_TO_CLASS_INDEX[code]]
        else:
            key = raw.upper().replace(" ", "_")
            name = upper_to_name.get(key) or display_to_name.get(raw.upper())
        if name and name not in seen:
            found.append(name)
            seen.add(name)
    return found


def _class_order(names: Iterable[str]) -> list[str]:
    order = {name: i for i, name in enumerate(load.CLASS_NAMES)}
    return sorted(set(names), key=lambda n: order.get(n, 10_000))


def build_label_comparison(
    true_classes: Iterable[str] | None,
    predicted_rows: list[dict],
    source: str | None,
    dx_codes: Iterable[str] | None = None,
    manual_raw: str | None = None,
) -> dict:
    """Compara predicciones finales positivas vs etiquetas reales dentro del esquema."""
    predicted = _class_order(row["class"] for row in predicted_rows if int(row.get("prediction", 0)) == 1)
    true = _class_order(true_classes or [])
    has_truth = source is not None
    if not has_truth:
        return {
            "available": False,
            "source": None,
            "reason": "No hay etiquetas reales disponibles. En .hea se leen desde el campo Dx; para CSV puede escribir clases o SNOMED manualmente.",
            "true_classes": [],
            "predicted_classes": predicted,
        }

    true_set = set(true)
    pred_set = set(predicted)
    tp = _class_order(true_set & pred_set)
    fp = _class_order(pred_set - true_set)
    fn = _class_order(true_set - pred_set)
    union = true_set | pred_set

    if not true_set and not pred_set:
        precision = recall = f1 = jaccard = 1.0
    elif not pred_set:
        precision = 0.0
        recall = 0.0 if true_set else 1.0
        f1 = 0.0
        jaccard = 0.0 if union else 1.0
    elif not true_set:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        jaccard = 0.0
    else:
        precision = len(tp) / len(pred_set)
        recall = len(tp) / len(true_set)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        jaccard = len(tp) / len(union) if union else 1.0

    return {
        "available": True,
        "source": source,
        "manual_raw": manual_raw or "",
        "dx_codes": list(dx_codes or []),
        "true_classes": true,
        "predicted_classes": predicted,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "exact_match": pred_set == true_set,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
    }


# ---------------------------------------------------------------------------
# Gráficos / resultados
# ---------------------------------------------------------------------------
def _robust_ylim(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(values, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        lo, hi = -2.5, 2.5
    pad = max(0.5, 0.12 * float(hi - lo))
    lo = float(np.floor((lo - pad) * 2) / 2)
    hi = float(np.ceil((hi + pad) * 2) / 2)
    return lo, hi


_MINOR_GRID_ZORDER = 1.6


def _grid_positions(lo: float, hi: float, step: float, major: float | None = None) -> np.ndarray:
    """Múltiplos de ``step`` dentro de [lo, hi] (equivale a MultipleLocator).

    Con ``major`` se omiten los que caen sobre una línea mayor, igual que hace
    Matplotlib con los ticks menores que coinciden con los mayores.
    """
    first = int(np.ceil(lo / step - 1e-9))
    last = int(np.floor(hi / step + 1e-9))
    idx = np.arange(first, last + 1)
    if major is not None:
        ratio = round(major / step)
        idx = idx[idx % ratio != 0]
    return idx * step


def plot_ecg_array(signal: np.ndarray, output_dir: str | Path, stem: str) -> Path:
    """Grafica las 12 derivaciones con estilo de papel milimetrado ECG."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape != (load.WINDOW_LENGTH, load.NUM_LEADS):
        raise ValueError(f"Shape inválida para gráfico ECG: {signal.shape}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(signal.shape[0]) / load.TARGET_FS

    with _PLOT_LOCK:
        return _plot_ecg_figure(signal, t, stem, output_dir)


def _plot_ecg_figure(signal: np.ndarray, t: np.ndarray, stem: str, output_dir: Path) -> Path:
    from matplotlib.ticker import MultipleLocator

    fig, axes = plt.subplots(6, 2, figsize=(16, 11.5), sharex=True)
    fig.patch.set_facecolor("#fffafa")
    fig.suptitle(
        f"ECG de 12 derivaciones — {stem} · papel milimetrado",
        fontsize=15,
        fontweight="bold",
        color="#7f1d1d",
    )

    for i, ax in enumerate(axes.flat):
        ax.set_facecolor("#fffafa")
        ax.plot(t, signal[:, i], linewidth=0.75, color="#111827", zorder=3)
        ax.set_title(LEADS[i], fontsize=10, loc="left", color="#7f1d1d", fontweight="bold")
        ax.set_xlim(0, load.WINDOW_SECONDS)
        y0, y1 = _robust_ylim(signal[:, i])
        ax.set_ylim(y0, y1)
        ax.xaxis.set_major_locator(MultipleLocator(0.20))
        ax.yaxis.set_major_locator(MultipleLocator(0.50))
        ax.grid(which="major", color="#e48b8b", linewidth=0.75)
        # Cuadrícula fina como líneas directas: 250 ticks menores por eje hacían
        # que la figura tardara ~5 s; así es el mismo dibujo en una fracción.
        ax.vlines(_grid_positions(0.0, load.WINDOW_SECONDS, 0.04, major=0.20), y0, y1,
                  colors="#f7c8c8", linewidths=0.35, zorder=_MINOR_GRID_ZORDER)
        ax.hlines(_grid_positions(y0, y1, 0.10, major=0.50), 0.0, load.WINDOW_SECONDS,
                  colors="#f7c8c8", linewidths=0.35, zorder=_MINOR_GRID_ZORDER)
        ax.tick_params(axis="both", which="major", labelsize=7, colors="#7f1d1d", length=0)
        for spine in ax.spines.values():
            spine.set_color("#e48b8b")
            spine.set_linewidth(0.6)
        ax.set_ylabel("z", fontsize=8, color="#7f1d1d")
    axes[-1, 0].set_xlabel("Tiempo (s)", fontsize=9, color="#7f1d1d")
    axes[-1, 1].set_xlabel("Tiempo (s)", fontsize=9, color="#7f1d1d")
    fig.text(
        0.5,
        0.008,
        "Cuadrícula ECG: 0.04 s por cuadro pequeño y 0.20 s por cuadro grande en el eje temporal. Amplitud mostrada normalizada por derivación.",
        ha="center",
        fontsize=8,
        color="#7f1d1d",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_path = output_dir / f"{secure_filename(stem)}_ecg_papel_milimetrado.png"
    _save_figure_png(fig, out_path)
    plt.close(fig)
    return out_path


PLOT_DPI = 160
PNG_PALETTE_COLORS = 256


def _save_figure_png(fig, path: Path) -> None:
    """Guarda la figura como PNG de paleta (≤ 256 colores) sin canal alfa.

    El trazado solo tiene cuadrícula rosa, texto y un trazo negro con suavizado; con 256 colores el error máximo
    es 14/255 en < 0,01 % de los píxeles (PSNR ≈ 59 dB) y el archivo pasa de ≈ 762 KB a ≈ 306 KB, que es lo
    que se descarga y lo que se incrusta en el PDF. La resolución (2545 × 1824) no cambia.
    """
    from PIL import Image

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=PLOT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    buffer.seek(0)
    with Image.open(buffer) as raw:
        rgb = raw.convert("RGB")                       # el fondo es opaco: el canal alfa solo estorba
    palette = rgb.quantize(colors=PNG_PALETTE_COLORS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    palette.save(path, format="PNG", optimize=True, compress_level=9, dpi=(PLOT_DPI, PLOT_DPI))


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
        group = _group_for_class(class_names[i])
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


def _infer_windows(model, windows: np.ndarray, device) -> tuple[np.ndarray, np.ndarray, str]:
    """Inferencia por ventanas; devuelve (probabilidades, por_ventana, agregación)."""
    with _INFERENCE_SLOTS:
        return _infer_windows_unlocked(model, windows, device)


def _infer_windows_unlocked(model, windows: np.ndarray, device) -> tuple[np.ndarray, np.ndarray, str]:
    if windows.shape[0] == 1:
        probabilities = predict_array(model, windows[0], device=device)
        return probabilities, probabilities[None, :], "single"
    probabilities, per_window = predict_windows(model, windows, device=device, aggregate=SLIDING_AGGREGATE)
    return probabilities, per_window, SLIDING_AGGREGATE


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
    normal_fallback_min_prob: float | None = 0.40,
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
    mismo modelo gana la afín) y, en un ensemble, una calibración afín del promedio (``platt_ensemble``). Todas son
    monótonas, así que calibrar la probabilidad agregada con el máximo de ventanas equivale a agregar las calibradas.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    resolved_model = resolve_model_path(model_path=model_path, saved_dir=saved_dir)

    device = serving_device()
    model, checkpoint, class_names = get_cached_model(resolved_model, device)
    pre = preprocessing_for_checkpoint(checkpoint)
    ckpt_schema = checkpoint.get("label_schema")
    prepared = prepare_saved_input(saved_paths, result_dir, norm_mode=pre["norm_mode"], bandpass=pre["bandpass"])

    windows = np.asarray(prepared.get("processed_windows", prepared["processed_signal"][None, ...]), dtype=np.float32)
    probabilities, per_window, aggregate_used = _infer_windows(model, windows, device)
    temp_values = temperatures_for_class_names(class_names, temperatures=temperatures)
    platt_values = platt_for_class_names(class_names, platt)
    if platt_values is not None:
        temp_values = None                      # la calibración afín sustituye a la temperatura del mismo modelo
        probabilities = apply_platt(np.asarray(probabilities)[None, :], *platt_values)[0]
        per_window = apply_platt(np.asarray(per_window), *platt_values)
    elif temp_values is not None:
        probabilities = apply_temperature(np.asarray(probabilities)[None, :], temp_values)[0]
        per_window = apply_temperature(np.asarray(per_window), temp_values)
    calibrated_a = temp_values is not None or platt_values is not None

    is_ensemble = model_path_b is not None and str(model_path_b).strip() != ""
    resolved_model_b = None
    temp_values_b = None
    platt_values_b = None
    platt_values_e = None
    calibrated_b = False
    if is_ensemble:
        from examples.cinc2020.ensemble_evaluate import ensemble_probabilities
        resolved_model_b = resolve_model_path(model_path=model_path_b, saved_dir=saved_dir)
        _model_b, _checkpoint_b, class_names_b = get_cached_model(resolved_model_b, device)
        if list(class_names_b) != list(class_names):
            raise ValueError("Los checkpoints del ensemble usan distinto orden de clases.")
        probabilities_b, per_window_b, _ = _infer_windows(_model_b, windows, device)
        temp_values_b = temperatures_for_class_names(class_names, temperatures=temperatures_b)
        platt_values_b = platt_for_class_names(class_names, platt_b)
        if platt_values_b is not None:
            temp_values_b = None
            probabilities_b = apply_platt(np.asarray(probabilities_b)[None, :], *platt_values_b)[0]
            per_window_b = apply_platt(np.asarray(per_window_b), *platt_values_b)
        elif temp_values_b is not None:
            probabilities_b = apply_temperature(np.asarray(probabilities_b)[None, :], temp_values_b)[0]
            per_window_b = apply_temperature(np.asarray(per_window_b), temp_values_b)
        calibrated_b = temp_values_b is not None or platt_values_b is not None
        probabilities = ensemble_probabilities(probabilities, probabilities_b, alpha)
        per_window = ensemble_probabilities(np.asarray(per_window), np.asarray(per_window_b), alpha)
        platt_values_e = platt_for_class_names(class_names, platt_ensemble)
        if platt_values_e is not None:
            probabilities = apply_platt(probabilities[None, :], *platt_values_e)[0]
            per_window = apply_platt(per_window, *platt_values_e)
        else:
            probabilities = probabilities.astype(np.float32)
            per_window = per_window.astype(np.float32)
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
    manual_true = parse_reference_label_text(reference_labels)
    ckpt_matched = translate_classes_to_checkpoint(prepared.get("matched_classes", []), class_names)
    if manual_raw:
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
    normal_fallback_min_prob: float | None = 0.40,
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
