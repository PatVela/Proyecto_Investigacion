"""Webapp Flask para ECG CINC2020-12.

La interfaz conserva una vista tipo dashboard: carga, resultado, detalle técnico,
PDF y experimentos. El checkpoint se resuelve automáticamente buscando un
best.pt dentro de saved/; también puede fijarse por CLI para despliegues.
"""

from __future__ import annotations

import argparse
import hashlib
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

from ecg import load, util
from webapp import project_info as proj
from webapp.prediction import clean_path_text, resolve_model_path, run_prediction_from_uploads
from webapp.report_pdf import build_pdf_report


DEFAULT_UPLOADS = REPO_ROOT / "webapp" / "uploads"
DEFAULT_RESULTS = REPO_ROOT / "webapp" / "results"
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
        try:
            ckpt = util.load_checkpoint(path, map_location="cpu")
            val = ckpt.get("val_loss")
            if val is not None:
                return (0, float(val), str(path))
        except Exception:
            pass
        return (1, path.stat().st_mtime if path.exists() else 0, str(path))

    return sorted(checkpoints, key=score)


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
            try:
                n_params = {
                    "total": int(sum(v.numel() for v in state.values())),
                    "trainable": int(sum(v.numel() for v in state.values())),
                    "non_trainable": 0,
                }
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
        REPO_ROOT / "results" / "cinc2020_12_resnet",
        REPO_ROOT / "results" / "cinc2020_12",
        REPO_ROOT / "results" / "cinc2020_12_eval",
    ]


def _load_thresholds_for_app(model_path: str | None, saved_dir: str | None) -> tuple[dict | None, str | None]:
    """Busca thresholds_validation.csv generado por evaluate.py.

    Si existe, la webapp usa umbrales por clase en lugar del fallback global 0.5.
    Esto reduce falsos positivos por clases desbalanceadas y hace la demo más
    consistente con la evaluación final.
    """
    candidates: list[Path] = []
    try:
        resolved = resolve_model_path(model_path=model_path, saved_dir=saved_dir)
        candidates.extend([
            resolved.parent / "thresholds_validation.csv",
            resolved.parent.parent / "thresholds_validation.csv",
        ])
    except Exception:
        pass
    explicit_thresholds = clean_path_text(os.environ.get("ECG_THRESHOLDS", ""))
    if explicit_thresholds:
        p = Path(explicit_thresholds)
        candidates.insert(0, p if p.is_absolute() else REPO_ROOT / p)

    for folder in _candidate_result_dirs():
        candidates.append(folder / "thresholds_validation.csv")

    # Último respaldo robusto: buscar en cualquier subcarpeta de results/.
    results_root = REPO_ROOT / "results"
    if results_root.exists():
        candidates.extend(sorted(results_root.rglob("thresholds_validation.csv"), key=lambda p: p.stat().st_mtime, reverse=True))

    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            import pandas as pd
            df = pd.read_csv(path)
            if "class" not in df.columns or "threshold" not in df.columns:
                continue
            values = {str(row["class"]): float(row["threshold"]) for _, row in df.iterrows()}
            if all(name in values for name in load.CLASS_NAMES):
                return values, str(path)
            if all(name in values for name in load.LEGACY_CLASS_NAMES):
                return values, str(path)
        except Exception:
            continue
    return None, None


def _load_temperatures_for_app(model_path: str | None, saved_dir: str | None, env_var: str = "ECG_TEMPERATURES") -> tuple[dict | None, str | None]:
    """Busca temperatures_validation.csv generado por calibrate.py.

    Si existe, la webapp aplica temperature scaling por clase antes de los
    umbrales, igual que ``evaluate.py --temperatures``. Con ``env_var`` se
    distingue el modelo A del B en modo ensemble.
    """
    candidates: list[Path] = []
    try:
        resolved = resolve_model_path(model_path=model_path, saved_dir=saved_dir)
        candidates.extend([
            resolved.parent / "temperatures_validation.csv",
            resolved.parent.parent / "temperatures_validation.csv",
        ])
    except Exception:
        pass
    explicit = clean_path_text(os.environ.get(env_var, ""))
    if explicit:
        p = Path(explicit)
        candidates.insert(0, p if p.is_absolute() else REPO_ROOT / p)

    for folder in _candidate_result_dirs():
        candidates.append(folder / "temperatures_validation.csv")

    results_root = REPO_ROOT / "results"
    if results_root.exists():
        candidates.extend(sorted(results_root.rglob("temperatures_validation.csv"), key=lambda p: p.stat().st_mtime, reverse=True))

    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            import pandas as pd
            df = pd.read_csv(path)
            if "class" not in df.columns or "temperature" not in df.columns:
                continue
            values = {str(row["class"]): float(row["temperature"]) for _, row in df.iterrows()}
            if all(name in values for name in load.CLASS_NAMES):
                return values, str(path)
        except Exception:
            continue
    return None, None


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
    roots.append(REPO_ROOT / "results" / "cinc2020_12")
    comparison: list = []
    comparison_root = roots[-1]
    for root in roots:
        comparison = _read_csv_records(root / "model_comparison.csv")
        if comparison:
            comparison_root = root
            break
    robustness_paths = [r / "robustness.csv" for r in roots] + [
        REPO_ROOT / "results" / "cinc2020_12_resnet" / "robustness.csv",

    ]
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
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

    app.config["ECG_MODEL"] = clean_path_text(os.environ.get("ECG_MODEL", ""))
    app.config["ECG_MODEL_B"] = clean_path_text(os.environ.get("ECG_MODEL_B", ""))
    app.config["ECG_ALPHA"] = float(os.environ.get("ECG_ALPHA", "0.5"))
    app.config["ECG_SAVED_DIR"] = str(_resolve_dir_env("ECG_SAVED", DEFAULT_SAVED))
    app.config["ECG_UPLOADS_DIR"] = str(_resolve_dir_env("ECG_UPLOADS", DEFAULT_UPLOADS))
    app.config["ECG_RESULTS_DIR"] = str(_resolve_dir_env("ECG_RESULTS", DEFAULT_RESULTS))
    app.config["ECG_REFERENCE"] = clean_path_text(os.environ.get("ECG_REFERENCE", ""))
    app.config["ECG_THRESHOLDS"] = clean_path_text(os.environ.get("ECG_THRESHOLDS", ""))
    app.config["ECG_NORMAL_FALLBACK_MIN_PROB"] = float(os.environ.get("ECG_NORMAL_FALLBACK_MIN_PROB", "0.40"))

    @app.after_request
    def add_security_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/")
    def index():
        saved_dir = app.config["ECG_SAVED_DIR"]
        model_path = app.config["ECG_MODEL"]
        resolved_model = _resolve_model_for_display(model_path, saved_dir)
        checkpoints = _list_checkpoints(saved_dir)
        model_info = _checkpoint_info(model_path, saved_dir)
        if app.config["ECG_MODEL_B"] and model_info:
            model_info["model_type"] = f"Ensemble ResNet-34 ×2 (α={float(app.config['ECG_ALPHA']):.2f})"
            model_info["model_b"] = app.config["ECG_MODEL_B"]
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
            resolved_model=resolved_model,
            resolved_model_short=_short_display_path(resolved_model),
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
            curves_found=_curves_available(),
        )

    @app.get("/models")
    def models_route():
        saved_dir = request.args.get("saved") or app.config["ECG_SAVED_DIR"]
        checkpoints = _list_checkpoints(saved_dir)
        return jsonify({
            "ok": True,
            "saved_dir": saved_dir,
            "models": [str(p) for p in checkpoints],
            "current": app.config["ECG_MODEL"] or None,
        })

    @app.post("/use_model")
    def use_model_route():
        data = request.get_json(silent=True) or {}
        model = clean_path_text(data.get("model", ""))
        if not model:
            app.config["ECG_MODEL"] = ""
            return jsonify({"ok": True, "model": "", "message": "Selección automática activada."})
        try:
            resolved = resolve_model_path(model_path=model, saved_dir=app.config["ECG_SAVED_DIR"])
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        app.config["ECG_MODEL"] = str(resolved)
        return jsonify({"ok": True, "model": str(resolved), "info": _checkpoint_info(str(resolved), app.config["ECG_SAVED_DIR"])})

    @app.post("/predict")
    def predict_route():
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
                reference_labels=clean_path_text(request.form.get("true_labels", "")),
                thresholds=class_thresholds,
                threshold_source=threshold_source,
                temperatures=class_temperatures,
                temperature_source=temperature_source,
                normal_fallback_min_prob=app.config["ECG_NORMAL_FALLBACK_MIN_PROB"],
                model_path_b=app.config["ECG_MODEL_B"] or None,
                temperatures_b=class_temperatures_b,
                temperature_source_b=temperature_source_b,
                alpha=app.config["ECG_ALPHA"],
            )
            result["patient_name"] = clean_path_text(request.form.get("patient_name", ""))
            result["patient_age"] = clean_path_text(request.form.get("patient_age", "")) or result.get("patient_age")
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

            pdf_path = Path(app.config["ECG_RESULTS_DIR"]) / result["job_id"] / f"{result['record_name']}_reporte.pdf"
            build_pdf_report(result, pdf_path)
            result["pdf_path"] = str(pdf_path)
            result["pdf_url"] = url_for(
                "result_file",
                filename=_relative_to_results(app, str(pdf_path)),
            )
            return jsonify(_jsonable(result))
        except Exception as exc:
            return jsonify({"ok": False, "status": "error", "error": str(exc)}), 500

    @app.get("/results/<path:filename>")
    def result_file(filename):
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
        return jsonify({
            "status": "ok",
            "ECG_MODEL": app.config["ECG_MODEL"],
            "resolved_model": resolved,
            "ECG_SAVED": app.config["ECG_SAVED_DIR"],
            "ECG_UPLOADS": app.config["ECG_UPLOADS_DIR"],
            "ECG_RESULTS": app.config["ECG_RESULTS_DIR"],
            "thresholds_found": ths is not None,
            "thresholds_source": ths_source,
            "thresholds": ths,
            "temperatures_found": tmps is not None,
            "temperature_source": tmps_source,
            "ensemble": bool(app.config["ECG_MODEL_B"]),
            "ECG_MODEL_B": app.config["ECG_MODEL_B"],
            "ensemble_alpha": app.config["ECG_ALPHA"],
            "temperatures_b_found": tmps_b is not None,
            "temperature_b_source": tmps_b_source,
            "normal_fallback_min_probability": app.config["ECG_NORMAL_FALLBACK_MIN_PROB"],
            "num_classes": load.NUM_CLASSES,
            "accepted_uploads": [".csv", ".hea + .mat"],
        })

    return app


app = create_app()


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
    parser.add_argument("--eval-dir", default=None, help="Carpeta con salidas de evaluate.py (métricas, umbrales, temperaturas, curvas). Si se omite, se busca automáticamente.")
    parser.add_argument("--exp-dir", default=None, help="Carpeta con model_comparison.csv y robustness.csv. Si se omite, se busca automáticamente.")
    parser.add_argument("--normal-fallback-min-prob", type=float, default=None, help="Si ninguna clase supera umbral, añadir NSR cuando P(NSR) sea al menos este valor. Use 0 para desactivar.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5002")))
    args = parser.parse_args(argv)

    if args.model is not None:
        os.environ["ECG_MODEL"] = clean_path_text(args.model)
    if args.model_b is not None:
        os.environ["ECG_MODEL_B"] = clean_path_text(args.model_b)
    if args.alpha is not None:
        a = float(args.alpha)
        if not 0.0 <= a <= 1.0:
            parser.error("--alpha debe estar en [0, 1]")
        os.environ["ECG_ALPHA"] = str(a)
    if args.saved is not None:
        os.environ["ECG_SAVED"] = clean_path_text(args.saved)
    if args.uploads is not None:
        os.environ["ECG_UPLOADS"] = clean_path_text(args.uploads)
    if args.results is not None:
        os.environ["ECG_RESULTS"] = clean_path_text(args.results)
    if args.reference is not None:
        os.environ["ECG_REFERENCE"] = clean_path_text(args.reference)
    if args.thresholds is not None:
        os.environ["ECG_THRESHOLDS"] = clean_path_text(args.thresholds)
    if args.temperatures is not None:
        os.environ["ECG_TEMPERATURES"] = clean_path_text(args.temperatures)
    if args.temperatures_b is not None:
        os.environ["ECG_TEMPERATURES_B"] = clean_path_text(args.temperatures_b)
    if args.eval_dir is not None:
        os.environ["ECG_EVAL_DIR"] = clean_path_text(args.eval_dir)
    if args.exp_dir is not None:
        os.environ["ECG_EXP_DIR"] = clean_path_text(args.exp_dir)
    if args.normal_fallback_min_prob is not None:
        os.environ["ECG_NORMAL_FALLBACK_MIN_PROB"] = str(args.normal_fallback_min_prob)

    runtime_app = create_app()
    print("=" * 72)
    print("WEBAPP CINC2020-12")
    print("=" * 72)
    print("Modelo CLI :", runtime_app.config["ECG_MODEL"] or "(auto)")
    if runtime_app.config["ECG_MODEL_B"]:
        print("Modelo B   :", runtime_app.config["ECG_MODEL_B"])
        print("Ensemble α :", runtime_app.config["ECG_ALPHA"])
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
