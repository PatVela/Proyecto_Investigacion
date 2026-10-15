"""Webapp Flask para ECG CINC2020-12.

La interfaz conserva una vista tipo dashboard: carga, resultado, detalle técnico,
PDF y experimentos. El checkpoint se resuelve automáticamente buscando un
best.pt dentro de saved/; también puede fijarse por CLI para despliegues.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from ecg import load, util
from webapp import example_records as example_store
from webapp.compression import compress_response
from webapp import project_info as proj
from webapp.limits import SlidingWindowLimiter, cleanup_old_jobs, int_env
from webapp.prediction import clean_field_text, clean_path_text, resolve_existing_path, resolve_model_path, run_prediction_from_uploads, serving_device
from webapp.report_pdf import build_pdf_report


DEFAULT_UPLOADS = REPO_ROOT / "webapp" / "uploads"
DEFAULT_RESULTS = REPO_ROOT / "webapp" / "results"
DEFAULT_HOST = "127.0.0.1"          # el servidor de desarrollo solo escucha en esta máquina salvo que se pida otra cosa
MAX_FORM_TEXT_CHARS = 1024          # tope de los textos de formulario que llegan del cliente
MAX_PATIENT_NAME_CHARS = 120        # con 480 000 caracteres reportlab tardaba 9.5 s y caía al PDF mínimo
MAX_PATIENT_AGE_CHARS = 20
DEFAULT_EXAMPLES = REPO_ROOT / "webapp" / "examples"
DEFAULT_SAVED = REPO_ROOT / "saved"


def _resolve_dir_env(name: str, default: Path) -> Path:
    value = clean_path_text(os.environ.get(name, ""))
    path = Path(value) if value else default
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _jsonable(value: Any):
    try:
        import numpy as np
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if np.isnan(value) else float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, float) and (value != value or abs(value) == float("inf")):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _public_path(value):
    """Lo que ve el visitante de una ruta del servidor: relativa al repositorio o, si queda fuera, solo el nombre.

    ``/health``, ``/models``, la portada y la respuesta de ``/predict`` devolvían rutas absolutas: con ``-Tunnel`` cualquiera
    con la URL veía cómo está organizado el disco. Un texto sin separadores (p. ej. ``fallback_global_0.5``) no se toca.
    """
    text = str(value) if value else ""
    if "/" not in text and "\\" not in text:
        return value
    path = Path(text)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return path.name


PUBLIC_PATH_FIELDS = ("model_path", "model_path_b", "plot_path", "converted_csv_path", "pdf_path", "threshold_source",
                      "temperature_source", "temperature_source_b", "platt_source", "platt_source_b", "platt_source_ensemble")


def _public_result(result: dict) -> dict:
    """Copia de la respuesta de ``/predict`` con las rutas del servidor en su forma pública."""
    public = dict(result)
    for key in PUBLIC_PATH_FIELDS:
        if public.get(key):
            public[key] = _public_path(public[key])
    public["source_files"] = [_public_path(p) for p in public.get("source_files") or []]
    details = dict(public.get("technical_details") or {})
    if details.get("threshold_source"):
        details["threshold_source"] = _public_path(details["threshold_source"])
    public["technical_details"] = details
    return public


def _short_model_id(model_path: str | Path | None) -> str:
    if not model_path:
        return ""
    path = Path(model_path)
    if not path.exists():
        return ""
    h = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()[:10]
    except OSError:
        return ""


def _model_train_date(path: str | Path | None) -> str:
    if not path:
        return ""
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return ""


def _list_checkpoints(saved_dir: str | Path) -> list[Path]:
    saved = Path(clean_path_text(saved_dir) or DEFAULT_SAVED)
    if not saved.is_absolute():
        saved = REPO_ROOT / saved
    checkpoints = util.find_checkpoints(saved)

    def score(path: Path):
        val = util.checkpoint_val_loss(path)
        if val is not None:
            return (0, val, str(path))
        return (1, path.stat().st_mtime if path.exists() else 0, str(path))

    return sorted(checkpoints, key=score)


BATCHNORM_BUFFER_SUFFIXES = (".running_mean", ".running_var", ".num_batches_tracked")


def _checkpoint_info(model_path: str | None, saved_dir: str | None) -> dict | None:
    try:
        resolved = resolve_model_path(model_path=model_path, saved_dir=saved_dir)
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "model_path": "",
            "model_type": "No cargado",
            "classes": load.CLASS_NAMES,
            "config": {},
            "meta": {},
            "num_parameters": None,
        }

    try:
        ckpt = util.load_checkpoint(resolved, map_location="cpu")
        config = ckpt.get("config", {}) or {}
        classes = ckpt.get("class_names") or load.CLASS_NAMES.copy()
        state = ckpt.get("model_state_dict") or {}
        n_params = None
        if state:
            try:                                  # parámetros, sin los buffers de BatchNorm (media, varianza y contador)
                n_params = {"total": int(sum(v.numel() for k, v in state.items() if not k.endswith(BATCHNORM_BUFFER_SUFFIXES)))}
            except Exception:
                n_params = None
        return {
            "available": True,
            "model_path": str(resolved),
            "model_type": "ResNet-34 1D tipo Hannun",
            "classes": classes,
            "device": "auto",
            "config": config,
            "meta": {
                "epoch": ckpt.get("epoch"),
                "val_loss": ckpt.get("val_loss"),
                "label_schema": ckpt.get("label_schema", load.LABEL_SCHEMA),
                "problem_type": ckpt.get("problem_type", "multilabel_sigmoid_bce"),
            },
            "num_parameters": n_params,
        }
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "model_path": str(resolved),
            "model_type": "Checkpoint no legible",
            "classes": load.CLASS_NAMES,
            "config": {},
            "meta": {},
            "num_parameters": None,
        }


def _resolve_model_for_display(model_path: str | None, saved_dir: str | None) -> str:
    try:
        return str(resolve_model_path(model_path=model_path, saved_dir=saved_dir))
    except Exception:
        return "No configurado o no encontrado"


def _relative_to_results(app: Flask, path_text: str | None) -> str | None:
    if not path_text:
        return None
    path = Path(path_text).resolve()
    root = Path(app.config["ECG_RESULTS_DIR"]).resolve()
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return path.name


def _read_csv_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    import pandas as pd
    df = pd.read_csv(path)
    return _jsonable(df.to_dict(orient="records"))


def _candidate_result_dirs() -> list[Path]:
    """Carpetas con salidas de evaluate.py (--eval-dir va primero si se indicó)."""
    dirs: list[Path] = []
    extra = clean_path_text(os.environ.get("ECG_EVAL_DIR", ""))
    if extra:
        p = Path(extra)
        dirs.append(p if p.is_absolute() else REPO_ROOT / p)
    return dirs + [
        REPO_ROOT / "runs" / "eval" / "cinc2020_12",
    ]


def _find_class_values_csv(
    filename: str,
    value_column: str,
    valid_name_sets: list[list[str]],
    model_path: str | None,
    saved_dir: str | None,
    env_var: str,
) -> tuple[dict | None, str | None]:
    """Busca un CSV ``class,<value_column>`` y devuelve ``({clase: valor}, ruta)``.

    Orden de búsqueda: ruta explícita (``env_var``), carpeta del checkpoint (y su
    padre) y las carpetas de evaluación. Se usa el primer archivo válido, es
    decir, con las columnas esperadas y con todos los nombres de alguno de
    ``valid_name_sets``; los inválidos se ignoran.
    """
    candidates: list[Path] = []
    try:
        resolved = resolve_model_path(model_path=model_path, saved_dir=saved_dir)
        candidates.extend([resolved.parent / filename, resolved.parent.parent / filename])
    except Exception:
        pass
    explicit = clean_path_text(os.environ.get(env_var, ""))
    if explicit:
        p = Path(explicit)
        candidates.insert(0, p if p.is_absolute() else REPO_ROOT / p)
    candidates.extend(folder / filename for folder in _candidate_result_dirs())

    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            import pandas as pd
            df = pd.read_csv(path)
            if "class" not in df.columns or value_column not in df.columns:
                continue
            values = {str(row["class"]): float(row[value_column]) for _, row in df.iterrows()}
            if any(all(name in values for name in names) for names in valid_name_sets):
                return values, str(path)
        except Exception:
            continue
    return None, None


def _load_thresholds_for_app(model_path: str | None, saved_dir: str | None) -> tuple[dict | None, str | None]:
    """Busca thresholds_validation.csv generado por evaluate.py (``ECG_THRESHOLDS`` tiene prioridad).

    Si existe, la webapp usa umbrales por clase en lugar del fallback global 0.5.
    Esto reduce falsos positivos por clases desbalanceadas y hace la demo más
    consistente con la evaluación final. Acepta también los nombres de clase v1.
    """
    return _find_class_values_csv(
        "thresholds_validation.csv", "threshold", [load.CLASS_NAMES, load.LEGACY_CLASS_NAMES],
        model_path, saved_dir, "ECG_THRESHOLDS")


def _load_temperatures_for_app(model_path: str | None, saved_dir: str | None, env_var: str = "ECG_TEMPERATURES") -> tuple[dict | None, str | None]:
    """Busca temperatures_validation.csv generado por calibrate.py.

    Si existe, la webapp aplica temperature scaling por clase antes de los
    umbrales, igual que ``evaluate.py --temperatures``. Con ``env_var`` se
    distingue el modelo A del B en modo ensemble.
    """
    return _find_class_values_csv(
        "temperatures_validation.csv", "temperature", [load.CLASS_NAMES],
        model_path, saved_dir, env_var)


def _load_platt_for_app(
    model_path: str | None,
    saved_dir: str | None,
    env_var: str = "ECG_PLATT",
    filename: str = "platt_validation.csv",
) -> tuple[dict | None, str | None]:
    """Busca la calibración afín ``class,a,b`` de ``calibrate.py --method platt``.

    ``ECG_PLATT`` / ``ECG_PLATT_B`` son la de cada modelo y ``ECG_PLATT_ENSEMBLE`` (archivo
    ``platt_ensemble_validation.csv`` de ``ensemble_evaluate.py --recalibrate-ensemble``) la del promedio.
    Devuelve ``({clase: {"a": ..., "b": ...}}, ruta)``; un archivo sin las dos columnas o sin todas las clases
    se ignora.
    """
    a_values, source = _find_class_values_csv(filename, "a", [load.CLASS_NAMES], model_path, saved_dir, env_var)
    if a_values is None:
        return None, None
    b_values, source_b = _find_class_values_csv(filename, "b", [load.CLASS_NAMES], model_path, saved_dir, env_var)
    if b_values is None or source_b != source:
        return None, None
    return {name: {"a": a_values[name], "b": b_values[name]} for name in a_values}, source


def _load_platt_set_for_app(app) -> dict:
    """Calibración afín de A, B y del promedio (solo hay B y promedio en modo ensemble)."""
    out = {"platt": None, "platt_source": None, "platt_b": None, "platt_source_b": None,
           "platt_ensemble": None, "platt_source_ensemble": None}
    out["platt"], out["platt_source"] = _load_platt_for_app(app.config["ECG_MODEL"], app.config["ECG_SAVED_DIR"])
    if app.config["ECG_MODEL_B"]:
        out["platt_b"], out["platt_source_b"] = _load_platt_for_app(
            app.config["ECG_MODEL_B"], app.config["ECG_SAVED_DIR"], env_var="ECG_PLATT_B")
        out["platt_ensemble"], out["platt_source_ensemble"] = _load_platt_for_app(
            app.config["ECG_MODEL"], app.config["ECG_SAVED_DIR"], env_var="ECG_PLATT_ENSEMBLE",
            filename="platt_ensemble_validation.csv")
    return out


PERTURBATION_LABELS = {
    "clean": ("Señal limpia", "Clean signal"),
    "gaussian_noise": ("Ruido gaussiano", "Gaussian noise"),
    "baseline_wander": ("Deriva de línea base", "Baseline wander"),
    "amplitude_scale": ("Escalado de amplitud", "Amplitude scaling"),
    "lead_dropout": ("Apagado de derivaciones", "Lead dropout"),
}


def _short_display_path(path) -> str:
    """Últimos 3 segmentos de una ruta (la completa se muestra en title)."""
    text = str(path or "").replace("\\", "/")
    parts = [p for p in text.split("/") if p]
    return "/".join(parts[-3:]) if parts else "—"


def _best_worst(rows: list[dict], key: str, label: str) -> tuple[dict | None, dict | None]:
    """Mejor/peor fila por una métrica numérica (tolerante a valores ausentes)."""
    best = worst = None
    for record in rows:
        try:
            value = float(record.get(key))
        except (TypeError, ValueError):
            continue
        if best is None or value > best["value"]:
            best = {"name": str(record.get(label, "—")), "value": value}
        if worst is None or value < worst["value"]:
            worst = {"name": str(record.get(label, "—")), "value": value}
    return best, worst


def _load_metrics() -> dict:
    candidates = _candidate_result_dirs()
    for folder in candidates:
        global_csv = folder / "metrics_global.csv"
        per_class_csv = folder / "metrics_per_class.csv"
        summary_json = folder / "evaluation_summary.json"
        if not (global_csv.exists() or per_class_csv.exists() or summary_json.exists()):
            continue
        summary = None
        if summary_json.exists():
            try:
                summary = json.loads(summary_json.read_text(encoding="utf-8"))
            except Exception:
                summary = None
        per_class = _read_csv_records(per_class_csv)
        per_class_test = [r for r in per_class if r.get("split") == "test"] or per_class
        best_class, worst_class = _best_worst(per_class_test, key="f1", label="class")
        return {
            "found": True,
            "path": str(folder),
            "global": _read_csv_records(global_csv),
            "per_class": per_class,
            "per_class_test": per_class_test,
            "best_class": best_class,
            "worst_class": worst_class,
            "summary": _jsonable(summary),
        }
    return {"found": False, "path": "", "global": [], "per_class": [], "per_class_test": [],
            "best_class": None, "worst_class": None, "summary": None}


def _load_experiment_results() -> dict:
    roots: list[Path] = []
    extra = clean_path_text(os.environ.get("ECG_EXP_DIR", ""))
    if extra:
        p = Path(extra)
        roots.append(p if p.is_absolute() else REPO_ROOT / p)
    roots.append(REPO_ROOT / "runs" / "reports")
    comparison: list = []
    comparison_root = roots[-1]
    for root in roots:
        comparison = _read_csv_records(root / "model_comparison.csv")
        if comparison:
            comparison_root = root
            break
    robustness_paths = [r / "robustness.csv" for r in roots]
    robustness = []
    robustness_path = ""
    for path in robustness_paths:
        rows = _read_csv_records(path)
        if rows:
            robustness = rows
            robustness_path = str(path)
            break
    comparison_test = [r for r in comparison if str(r.get("split", "")).lower() == "test"] or comparison
    _winner, _ = _best_worst(comparison_test, key="f1_macro", label="model")
    comparison_winner = _winner["name"] if _winner else None
    clean_f1 = None
    for record in robustness:
        if str(record.get("perturbation", "")).lower() == "clean":
            try:
                clean_f1 = float(record.get("f1_macro"))
            except (TypeError, ValueError):
                clean_f1 = None
            break
    biggest_drop = None
    for record in robustness:
        try:
            delta = float(record.get("f1_macro")) - clean_f1 if clean_f1 is not None else None
        except (TypeError, ValueError):
            delta = None
        record["delta_f1_macro"] = delta
        if delta is not None and str(record.get("perturbation", "")).lower() != "clean":
            if biggest_drop is None or delta < biggest_drop["delta"]:
                biggest_drop = {
                    "name": str(record.get("perturbation", "—")),
                    "level": record.get("level", "—"),
                    "delta": delta,
                }
    for record in robustness:
        code = str(record.get("perturbation", ""))
        label_es, label_en = PERTURBATION_LABELS.get(code.lower(), (code, code))
        record["perturbation_label_es"] = label_es
        record["perturbation_label_en"] = label_en
    if biggest_drop is not None:
        b_es, b_en = PERTURBATION_LABELS.get(
            str(biggest_drop["name"]).lower(), (biggest_drop["name"], biggest_drop["name"]))
        biggest_drop["label_es"] = b_es
        biggest_drop["label_en"] = b_en
    comparison_verdict = None
    scored = []
    for record in comparison_test:
        try:
            scored.append((float(record.get("f1_macro")), str(record.get("model", "—"))))
        except (TypeError, ValueError):
            continue
    if len(scored) >= 2:
        scored.sort(reverse=True)
        comparison_verdict = {
            "winner": scored[0][1],
            "runner_up": scored[1][1],
            "delta": scored[0][0] - scored[1][0],
        }
    return {
        "comparison_found": bool(comparison),
        "comparison_path": str(comparison_root / "model_comparison.csv"),
        "comparison": comparison,
        "comparison_winner": comparison_winner,
        "comparison_verdict": comparison_verdict,
        "robustness_found": bool(robustness),
        "robustness_path": robustness_path,
        "robustness": robustness,
        "biggest_drop": biggest_drop,
    }


def _curves_available(split: str = "test") -> bool:
    for folder in _candidate_result_dirs():
        if (folder / f"roc_curves_{split}.csv").exists() or (folder / f"pr_curves_{split}.csv").exists():
            return True
    return False


def _load_curves(split: str = "test") -> dict:
    """Lee curvas ROC/PR por clase para el endpoint /curves (split test o validation)."""
    import pandas as pd
    for folder in _candidate_result_dirs():
        roc_csv = folder / f"roc_curves_{split}.csv"
        pr_csv = folder / f"pr_curves_{split}.csv"
        if not (roc_csv.exists() or pr_csv.exists()):
            continue
        curves: dict[str, dict[str, list]] = {}
        try:
            if roc_csv.exists():
                for name, group in pd.read_csv(roc_csv).groupby("class"):
                    key = str(name)
                    curves.setdefault(key, {})["fpr"] = [round(float(v), 4) for v in group["fpr"]]
                    curves[key]["tpr"] = [round(float(v), 4) for v in group["tpr"]]
            if pr_csv.exists():
                for name, group in pd.read_csv(pr_csv).groupby("class"):
                    key = str(name)
                    curves.setdefault(key, {})["precision"] = [round(float(v), 4) for v in group["precision"]]
                    curves[key]["recall"] = [round(float(v), 4) for v in group["recall"]]
        except Exception:
            continue
        if curves:
            return {"found": True, "split": split, "classes": sorted(curves), "curves": curves}
    return {"found": False, "split": split, "classes": [], "curves": {}}


def create_app() -> Flask:
    app = Flask(__name__)
    max_upload_mb = max(1, int_env("ECG_MAX_UPLOAD_MB", 5))      # un ECG de 10 s pesa < 1 MB
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024
    app.config["ECG_MAX_UPLOAD_MB"] = max_upload_mb
    serving_device()                                              # falla al arrancar si ECG_DEVICE es inválido
    trusted_proxies = int_env("ECG_TRUST_PROXY", 0)               # nº de proxies de confianza delante (p. ej. un túnel)
    if trusted_proxies > 0:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=trusted_proxies, x_proto=1, x_host=1)
    limiter = SlidingWindowLimiter(int_env("ECG_RATE_LIMIT_MAX", 20), int_env("ECG_RATE_LIMIT_WINDOW", 60))
    result_ttl = int_env("ECG_RESULT_TTL_SECONDS", 3600)          # 0 = no borrar

    app.config["ECG_MODEL"] = clean_path_text(os.environ.get("ECG_MODEL", ""))
    app.config["ECG_MODEL_B"] = clean_path_text(os.environ.get("ECG_MODEL_B", ""))
    for key in ("ECG_MODEL", "ECG_MODEL_B"):                 # falla al arrancar, no en la primera predicción
        if app.config.get(key) and not resolve_existing_path(app.config[key]).exists():
            raise RuntimeError(f"{key} apunta a un checkpoint que no existe: {app.config[key]}")
    alpha = float(os.environ.get("ECG_ALPHA", "0.5"))
    if not 0.0 <= alpha <= 1.0:                              # falla al arrancar: si no, cada /predict del ensemble daría 400
        raise RuntimeError(f"ECG_ALPHA debe estar en [0, 1]; recibido {os.environ.get('ECG_ALPHA')!r}")
    app.config["ECG_ALPHA"] = alpha
    app.config["ECG_SAVED_DIR"] = str(_resolve_dir_env("ECG_SAVED", DEFAULT_SAVED))
    app.config["ECG_UPLOADS_DIR"] = str(_resolve_dir_env("ECG_UPLOADS", DEFAULT_UPLOADS))
    app.config["ECG_RESULTS_DIR"] = str(_resolve_dir_env("ECG_RESULTS", DEFAULT_RESULTS))
    app.config["ECG_REFERENCE"] = clean_path_text(os.environ.get("ECG_REFERENCE", ""))
    app.config["ECG_EXAMPLES_DIR"] = str(Path(clean_path_text(os.environ.get("ECG_EXAMPLES_DIR", "")) or DEFAULT_EXAMPLES))
    examples_manifest = example_store.load_manifest(app.config["ECG_EXAMPLES_DIR"])
    app.config["ECG_THRESHOLDS"] = clean_path_text(os.environ.get("ECG_THRESHOLDS", ""))
    app.config["ECG_NORMAL_FALLBACK_MIN_PROB"] = float(os.environ.get("ECG_NORMAL_FALLBACK_MIN_PROB", "0"))      # 0 = apagado: las métricas publicadas se calcularon sin esta regla

    def _purge_old_jobs() -> None:
        for key in ("ECG_RESULTS_DIR", "ECG_UPLOADS_DIR"):
            cleanup_old_jobs(app.config[key], result_ttl)

    _purge_old_jobs()

    @app.before_request
    def rate_limit_predictions():
        if request.method == "POST" and request.path == "/predict":
            allowed, retry_after = limiter.check(request.remote_addr or "unknown")
            if not allowed:
                resp = jsonify({"ok": False, "status": "error",
                                "error": f"Demasiadas solicitudes. Reintente en {int(retry_after) + 1} s."})
                resp.status_code = 429
                resp.headers["Retry-After"] = str(int(retry_after) + 1)
                return resp
            _purge_old_jobs()

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"ok": False, "status": "error",
                        "error": f"El archivo supera el máximo de {max_upload_mb} MB. Reduzca el tamaño o suba solo un registro de 10 s."}), 413

    static_versions: dict[tuple, str] = {}

    def static_v(filename: str) -> str:
        """URL de un estático con ``?v=<huella>``: cambia solo si cambia el archivo, así puede cachearse un año."""
        path = Path(app.static_folder) / filename
        stat = path.stat()
        key = (filename, stat.st_mtime_ns, stat.st_size)
        if key not in static_versions:
            static_versions[key] = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
        return url_for("static", filename=filename, v=static_versions[key])

    app.jinja_env.globals["static_v"] = static_v

    static_gzip_cache: dict = {}          # (ruta, ETag) -> cuerpo gzip de los estáticos, comprimidos una sola vez

    @app.after_request
    def add_security_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        resp.headers["Content-Security-Policy"] = "frame-ancestors 'none'"   # nadie puede incrustar la demo (clickjacking); la app no usa marcos
        resp.headers["X-Frame-Options"] = "DENY"                              # lo mismo para navegadores que no entienden CSP
        if request.path.startswith("/static/"):
            # Estáticos versionados (?v=): caché de un año; sin versión: se revalidan con ETag (304).
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable" if "v" in request.args else "no-cache"
        else:
            resp.headers["Cache-Control"] = "no-store"          # páginas, API, informes y subidas nunca se guardan
        is_static = request.path.startswith("/static/")
        etag = resp.get_etag()[0] if is_static else None
        return compress_response(resp, request.headers.get("Accept-Encoding", ""),
                                 static_gzip_cache if is_static else None, (request.path, etag) if etag else None)

    @app.get("/")
    def index():
        saved_dir = app.config["ECG_SAVED_DIR"]
        model_path = app.config["ECG_MODEL"]
        resolved_model = _resolve_model_for_display(model_path, saved_dir)
        checkpoints = _list_checkpoints(saved_dir)
        model_info = _checkpoint_info(model_path, saved_dir)
        if app.config["ECG_MODEL_B"] and model_info:
            model_info["model_type"] = f"Ensemble ResNet-34 ×2 (α={float(app.config['ECG_ALPHA']):.2f})"
            model_info["model_b"] = _public_path(app.config["ECG_MODEL_B"])
            model_info["model_b_short"] = _short_display_path(app.config["ECG_MODEL_B"])
        threshold_values, _ = _load_thresholds_for_app(model_path, saved_dir)
        thresholds_info = {
            "found": bool(threshold_values),
            "num_classes": len(threshold_values) if threshold_values else 0,
        }
        return render_template(
            "index.html",
            class_groups=load.CLASS_GROUPS_12,
            model_info=model_info,
            checkpoints=[str(p) for p in checkpoints],
            checkpoint_options=[str(p.relative_to(REPO_ROOT)) if str(p).startswith(str(REPO_ROOT)) else str(p) for p in checkpoints],
            model_path=model_path or "",
            resolved_model=_public_path(resolved_model),
            resolved_model_short=_short_display_path(_public_path(resolved_model)),
            saved_dir=saved_dir,
            uploads_dir=app.config["ECG_UPLOADS_DIR"],
            results_dir=app.config["ECG_RESULTS_DIR"],
            reference_path=app.config["ECG_REFERENCE"],
            num_classes=load.NUM_CLASSES,
            num_leads=load.NUM_LEADS,
            sampling_rate=load.TARGET_FS,
            input_length=load.WINDOW_LENGTH,
            window_seconds=load.WINDOW_SECONDS,
            project=proj.PROJECT_INFO,
            reference=proj.REFERENCE,
            model_train_date=_model_train_date(resolved_model) if resolved_model != "No configurado o no encontrado" else "",
            model_hash=_short_model_id(resolved_model) if resolved_model != "No configurado o no encontrado" else "",
            metrics=_load_metrics(),
            experiments=_load_experiment_results(),
            thresholds_info=thresholds_info,
            dev_help=os.environ.get("ECG_DEV_HELP") == "1",
            curves_found=_curves_available(),
        )

    @app.get("/models")
    def models_route():
        # El directorio lo fija el servidor; nunca el cliente (evita escanear/cargar rutas arbitrarias).
        saved_dir = app.config["ECG_SAVED_DIR"]
        checkpoints = _list_checkpoints(saved_dir)
        return jsonify({
            "ok": True,
            "saved_dir": _public_path(saved_dir),
            "models": [_public_path(p) for p in checkpoints],
            "current": _public_path(app.config["ECG_MODEL"]) or None,
        })

    @app.post("/use_model")
    def use_model_route():
        # Cambia el modelo de todos los visitantes y la interfaz no lo usa: apagado salvo que se active a propósito.
        if os.environ.get("ECG_ENABLE_USE_MODEL") != "1":
            return jsonify({"ok": False, "error": "Cambio de modelo desactivado. Defina ECG_ENABLE_USE_MODEL=1 para permitirlo."}), 403
        data = request.get_json(silent=True) or {}
        model = clean_field_text(data.get("model", ""), MAX_FORM_TEXT_CHARS)
        if not model:
            app.config["ECG_MODEL"] = ""
            return jsonify({"ok": True, "model": "", "message": "Selección automática activada."})
        try:
            resolved = resolve_model_path(model_path=model, saved_dir=app.config["ECG_SAVED_DIR"])
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        saved_root = Path(app.config["ECG_SAVED_DIR"]).resolve()
        if resolved.suffix.lower() != ".pt" or not resolved.is_relative_to(saved_root):
            return jsonify({"ok": False, "error": "El checkpoint debe ser un .pt dentro del directorio de modelos."}), 400
        app.config["ECG_MODEL"] = str(resolved)
        return jsonify({"ok": True, "model": str(resolved), "info": _checkpoint_info(str(resolved), app.config["ECG_SAVED_DIR"])})

    @app.get("/examples")
    def examples_route():
        return jsonify({"ok": True, "examples": example_store.public_listing(examples_manifest)})

    @app.post("/predict")
    def predict_route():
        example_id = clean_field_text(request.form.get("example", ""), MAX_FORM_TEXT_CHARS)
        if example_id:                                   # registro de ejemplo incluido: mismo flujo que una subida
            try:
                stored = example_store.example_files(examples_manifest, app.config["ECG_EXAMPLES_DIR"], example_id)
            except KeyError:
                return jsonify({"ok": False, "status": "error", "error": f"Ejemplo desconocido: {example_id[:40]}"}), 400
            files = [FileStorage(stream=io.BytesIO(data), filename=name) for name, data in stored]
        else:
            files = request.files.getlist("files")
            if not files:
                one = request.files.get("file")
                files = [one] if one is not None else []
        if not files or all(not getattr(f, "filename", "") for f in files):
            return jsonify({"ok": False, "status": "error", "error": "Suba un CSV o el par .hea + .mat."}), 400

        try:
            threshold = float(request.form.get("threshold", 0.5))
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("El threshold debe estar entre 0 y 1.")

            class_thresholds, threshold_source = _load_thresholds_for_app(
                app.config["ECG_MODEL"],
                app.config["ECG_SAVED_DIR"],
            )
            class_temperatures, temperature_source = _load_temperatures_for_app(
                app.config["ECG_MODEL"],
                app.config["ECG_SAVED_DIR"],
            )
            class_temperatures_b, temperature_source_b = None, None
            if app.config["ECG_MODEL_B"]:
                class_temperatures_b, temperature_source_b = _load_temperatures_for_app(
                    app.config["ECG_MODEL_B"],
                    app.config["ECG_SAVED_DIR"],
                    env_var="ECG_TEMPERATURES_B",
                )

            result = run_prediction_from_uploads(
                file_storages=files,
                model_path=app.config["ECG_MODEL"],
                saved_dir=app.config["ECG_SAVED_DIR"],
                upload_root=app.config["ECG_UPLOADS_DIR"],
                results_root=app.config["ECG_RESULTS_DIR"],
                threshold=threshold,
                reference_labels=clean_field_text(request.form.get("true_labels", ""), MAX_FORM_TEXT_CHARS),
                thresholds=class_thresholds,
                threshold_source=threshold_source,
                temperatures=class_temperatures,
                temperature_source=temperature_source,
                normal_fallback_min_prob=app.config["ECG_NORMAL_FALLBACK_MIN_PROB"],
                model_path_b=app.config["ECG_MODEL_B"] or None,
                temperatures_b=class_temperatures_b,
                temperature_source_b=temperature_source_b,
                alpha=app.config["ECG_ALPHA"],
                **_load_platt_set_for_app(app),
            )
            result["patient_name"] = clean_field_text(request.form.get("patient_name", ""), MAX_PATIENT_NAME_CHARS)
            result["patient_age"] = clean_field_text(request.form.get("patient_age", ""), MAX_PATIENT_AGE_CHARS) or result.get("patient_age")
            result["ok"] = True

            if result.get("plot_path"):
                result["plot_url"] = url_for(
                    "result_file",
                    filename=_relative_to_results(app, result["plot_path"]),
                )
            if result.get("converted_csv_path"):
                result["converted_csv_url"] = url_for(
                    "result_file",
                    filename=_relative_to_results(app, result["converted_csv_path"]),
                )

            results_root = Path(app.config["ECG_RESULTS_DIR"]).resolve()
            safe_name = secure_filename(str(result["record_name"])) or "ecg"
            pdf_path = (results_root / result["job_id"] / f"{safe_name}_reporte.pdf").resolve()
            if not pdf_path.is_relative_to(results_root):
                raise ValueError("Ruta de informe fuera del directorio de resultados.")
            build_pdf_report(result, pdf_path)
            result["pdf_path"] = str(pdf_path)
            result["pdf_url"] = url_for(
                "result_file",
                filename=_relative_to_results(app, str(pdf_path)),
            )
            return jsonify(_jsonable(_public_result(result)))
        except ValueError as exc:                           # entrada inválida (formato, rango, extensión, etc.)
            return jsonify({"ok": False, "status": "error", "error": str(exc)}), 400
        except Exception as exc:                            # fallo interno: detalle solo en el registro del servidor
            app.logger.exception("Fallo interno en /predict")
            detail = str(exc) if os.environ.get("ECG_DEBUG_ERRORS") == "1" else                 "Error interno al procesar el ECG. Inténtelo de nuevo o pruebe con un ejemplo."
            return jsonify({"ok": False, "status": "error", "error": detail}), 500

    @app.get("/results/<path:filename>")
    def result_file(filename):
        # La caducidad prometida (1 h) debe valer también para quien descarga: la limpieza solo corría al arrancar y en cada
        # POST /predict, y un PDF con el nombre del paciente seguía respondiendo 200 horas después.
        # shortcut: al apagar el servidor los informes recientes quedan en disco hasta el siguiente arranque.
        _purge_old_jobs()
        return send_from_directory(app.config["ECG_RESULTS_DIR"], filename, as_attachment=False)

    @app.get("/metrics")
    def metrics_status():
        metrics = _load_metrics()
        return jsonify({"ok": True, **metrics})

    @app.get("/experiments")
    def experiments_status():
        experiments = _load_experiment_results()
        return jsonify({"ok": True, **experiments})

    @app.get("/curves")
    def curves_status():
        split = request.args.get("split", "test")
        if split not in ("test", "validation"):
            split = "test"
        return jsonify({"ok": True, **_load_curves(split)})

    @app.get("/health")
    def health():
        resolved = _resolve_model_for_display(app.config["ECG_MODEL"], app.config["ECG_SAVED_DIR"])
        ths, ths_source = _load_thresholds_for_app(app.config["ECG_MODEL"], app.config["ECG_SAVED_DIR"])
        tmps, tmps_source = _load_temperatures_for_app(app.config["ECG_MODEL"], app.config["ECG_SAVED_DIR"])
        tmps_b, tmps_b_source = None, None
        if app.config["ECG_MODEL_B"]:
            tmps_b, tmps_b_source = _load_temperatures_for_app(
                app.config["ECG_MODEL_B"], app.config["ECG_SAVED_DIR"], env_var="ECG_TEMPERATURES_B")
        platt = _load_platt_set_for_app(app)
        return jsonify({
            "status": "ok",
            "ECG_MODEL": _public_path(app.config["ECG_MODEL"]),
            "resolved_model": _public_path(resolved),
            "ECG_SAVED": _public_path(app.config["ECG_SAVED_DIR"]),
            "ECG_UPLOADS": _public_path(app.config["ECG_UPLOADS_DIR"]),
            "ECG_RESULTS": _public_path(app.config["ECG_RESULTS_DIR"]),
            "thresholds_found": ths is not None,
            "thresholds_source": _public_path(ths_source),
            "thresholds": ths,
            "temperatures_found": tmps is not None,
            "temperature_source": _public_path(tmps_source),
            "ensemble": bool(app.config["ECG_MODEL_B"]),
            "ECG_MODEL_B": _public_path(app.config["ECG_MODEL_B"]),
            "ensemble_alpha": app.config["ECG_ALPHA"],
            "temperatures_b_found": tmps_b is not None,
            "temperature_b_source": _public_path(tmps_b_source),
            "platt_found": platt["platt"] is not None,
            "platt_source": _public_path(platt["platt_source"]),
            "platt_b_found": platt["platt_b"] is not None,
            "platt_b_source": _public_path(platt["platt_source_b"]),
            "platt_ensemble_found": platt["platt_ensemble"] is not None,
            "platt_ensemble_source": _public_path(platt["platt_source_ensemble"]),
            "normal_fallback_min_probability": app.config["ECG_NORMAL_FALLBACK_MIN_PROB"],
            "num_classes": load.NUM_CLASSES,
            "accepted_uploads": [".csv", ".hea + .mat"],
        })

    return app


CLI_TO_ENV_PATHS = {
    "model": "ECG_MODEL", "model_b": "ECG_MODEL_B", "saved": "ECG_SAVED", "uploads": "ECG_UPLOADS", "results": "ECG_RESULTS",
    "reference": "ECG_REFERENCE", "thresholds": "ECG_THRESHOLDS", "temperatures": "ECG_TEMPERATURES",
    "temperatures_b": "ECG_TEMPERATURES_B", "platt": "ECG_PLATT", "platt_b": "ECG_PLATT_B", "platt_ensemble": "ECG_PLATT_ENSEMBLE",
    "eval_dir": "ECG_EVAL_DIR", "exp_dir": "ECG_EXP_DIR",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Webapp CINC2020-12")
    parser.add_argument("--model", default=None, help="Checkpoint .pt. Si se omite, se busca automáticamente en --saved.")
    parser.add_argument("--model-b", default=None, help="Segundo checkpoint para modo ensemble (promedio ponderado con --alpha).")
    parser.add_argument("--alpha", type=float, default=None, help="Peso del modelo A en el ensemble [0, 1] (B recibe 1-alpha). Por defecto 0.5.")
    parser.add_argument("--saved", default=None, help="Directorio de checkpoints para búsqueda automática.")
    parser.add_argument("--uploads", default=None, help="Directorio de archivos subidos.")
    parser.add_argument("--results", default=None, help="Directorio de resultados generados.")
    parser.add_argument("--reference", default=None, help="Referencia opcional para documentación interna.")
    parser.add_argument("--thresholds", default=None, help="thresholds_validation.csv generado por evaluate.py. Si se omite, se busca automáticamente.")
    parser.add_argument("--temperatures", default=None, help="temperatures_validation.csv de calibrate.py para el modelo A. Si se omite, se busca automáticamente.")
    parser.add_argument("--temperatures-b", default=None, help="temperatures_validation.csv de calibrate.py para el modelo B (ensemble).")
    parser.add_argument("--platt", default=None, help="platt_validation.csv de calibrate.py --method platt para el modelo A (sustituye a su temperatura).")
    parser.add_argument("--platt-b", default=None, help="platt_validation.csv para el modelo B (ensemble).")
    parser.add_argument("--platt-ensemble", default=None, help="platt_ensemble_validation.csv de ensemble_evaluate.py --recalibrate-ensemble (calibra el promedio).")
    parser.add_argument("--eval-dir", default=None, help="Carpeta con salidas de evaluate.py (métricas, umbrales, temperaturas, curvas). Si se omite, se busca automáticamente.")
    parser.add_argument("--exp-dir", default=None, help="Carpeta con model_comparison.csv y robustness.csv. Si se omite, se busca automáticamente.")
    parser.add_argument("--normal-fallback-min-prob", type=float, default=None, help="Si ninguna clase supera umbral, añadir NSR cuando P(NSR) sea al menos este valor. Use 0 para desactivar.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Interfaz de escucha. Por defecto solo esta máquina; 0.0.0.0 abre la app a la red local.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5002")))
    args = parser.parse_args(argv)

    if args.alpha is not None and not 0.0 <= float(args.alpha) <= 1.0:
        parser.error("--alpha debe estar en [0, 1]")
    for option, variable in CLI_TO_ENV_PATHS.items():             # cada opción de ruta equivale a su variable de entorno
        value = getattr(args, option)
        if value is not None:
            os.environ[variable] = clean_path_text(value)
    if args.alpha is not None:
        os.environ["ECG_ALPHA"] = str(float(args.alpha))
    if args.normal_fallback_min_prob is not None:
        os.environ["ECG_NORMAL_FALLBACK_MIN_PROB"] = str(args.normal_fallback_min_prob)

    runtime_app = create_app()
    print("=" * 72)
    print("WEBAPP CINC2020-12")
    print("=" * 72)
    print("Modelo CLI :", runtime_app.config["ECG_MODEL"] or "(auto)")
    if runtime_app.config["ECG_MODEL_B"]:
        print("Modelo B   :", runtime_app.config["ECG_MODEL_B"])
        print("Ensemble alpha:", runtime_app.config["ECG_ALPHA"])
    print("Resuelto   :", _resolve_model_for_display(runtime_app.config["ECG_MODEL"], runtime_app.config["ECG_SAVED_DIR"]))
    _ths, _ths_source = _load_thresholds_for_app(runtime_app.config["ECG_MODEL"], runtime_app.config["ECG_SAVED_DIR"])
    print("Búsqueda   :", runtime_app.config["ECG_SAVED_DIR"])
    print("Thresholds :", _ths_source or "no encontrados; fallback global 0.5")
    print("Eval dir   :", os.environ.get("ECG_EVAL_DIR", "") or "(búsqueda automática)")
    print("Exp dir    :", os.environ.get("ECG_EXP_DIR", "") or "(búsqueda automática)")
    if _ths:
        print("  LVH      :", _ths.get("LVH", "—"), "| NSR:", _ths.get("NSR", "—"))
    print("Fallback N :", runtime_app.config["ECG_NORMAL_FALLBACK_MIN_PROB"], "(0 desactiva)")
    print("Subidas    :", runtime_app.config["ECG_UPLOADS_DIR"])
    print("Resultados :", runtime_app.config["ECG_RESULTS_DIR"])
    print("URL        :", f"http://{args.host}:{args.port}")
    print("=" * 72)
    runtime_app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
