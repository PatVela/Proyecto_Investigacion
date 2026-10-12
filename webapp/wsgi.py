"""Entrada WSGI para la webapp CINC2020-12.

Variables reconocidas:
    ECG_MODEL      -> checkpoint .pt explícito.
    ECG_SAVED      -> directorio donde buscar checkpoints si ECG_MODEL no existe.
    ECG_UPLOADS    -> carpeta para archivos subidos.
    ECG_RESULTS    -> carpeta para PNG/PDF/CSV convertidos.
    ECG_REFERENCE  -> referencia opcional.
    ECG_THRESHOLDS -> thresholds_validation.csv opcional.
    ECG_NORMAL_FALLBACK_MIN_PROB -> umbral mínimo para fallback NSR.

También puede ejecutarse sin depender de variables de entorno:

    python webapp/wsgi.py --saved saved
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _clean(value):
    if value is None:
        return None
    from webapp.prediction import clean_path_text
    return clean_path_text(value)


def _apply_env(model=None, saved=None, uploads=None, results=None, reference=None, thresholds=None, normal_fallback_min_prob=None):
    if model is not None:
        os.environ["ECG_MODEL"] = _clean(model)
    os.environ.setdefault("ECG_MODEL", "")

    if saved is not None:
        os.environ["ECG_SAVED"] = _clean(saved)
    os.environ.setdefault("ECG_SAVED", str(REPO_ROOT / "saved"))

    if uploads is not None:
        os.environ["ECG_UPLOADS"] = _clean(uploads)
    os.environ.setdefault("ECG_UPLOADS", str(REPO_ROOT / "webapp" / "uploads"))

    if results is not None:
        os.environ["ECG_RESULTS"] = _clean(results)
    os.environ.setdefault("ECG_RESULTS", str(REPO_ROOT / "webapp" / "results"))

    if reference is not None:
        os.environ["ECG_REFERENCE"] = _clean(reference)
    os.environ.setdefault("ECG_REFERENCE", "")

    if thresholds is not None:
        os.environ["ECG_THRESHOLDS"] = _clean(thresholds)
    os.environ.setdefault("ECG_THRESHOLDS", "")

    if normal_fallback_min_prob is not None:
        os.environ["ECG_NORMAL_FALLBACK_MIN_PROB"] = str(normal_fallback_min_prob)
    os.environ.setdefault("ECG_NORMAL_FALLBACK_MIN_PROB", "0.40")


def create_wsgi_app(**kwargs):
    _apply_env(**kwargs)
    from webapp.app import create_app
    return create_app()


# Variable usada por gunicorn/waitress: webapp.wsgi:app
app = create_wsgi_app()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ejecuta la webapp CINC2020-12")
    parser.add_argument("--model", default=None, help="Checkpoint .pt. Equivale a ECG_MODEL.")
    parser.add_argument("--saved", default=None, help="Directorio de checkpoints. Equivale a ECG_SAVED.")
    parser.add_argument("--uploads", default=None, help="Directorio de subida. Equivale a ECG_UPLOADS.")
    parser.add_argument("--results", default=None, help="Directorio de salida. Equivale a ECG_RESULTS.")
    parser.add_argument("--reference", default=None, help="Referencia opcional. Equivale a ECG_REFERENCE.")
    parser.add_argument("--thresholds", default=None, help="thresholds_validation.csv opcional. Equivale a ECG_THRESHOLDS.")
    parser.add_argument("--normal-fallback-min-prob", type=float, default=None, help="Fallback NSR si no hay positivos; 0 desactiva.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5002")))
    args = parser.parse_args(argv)

    runtime_app = create_wsgi_app(
        model=args.model,
        saved=args.saved,
        uploads=args.uploads,
        results=args.results,
        reference=args.reference,
        thresholds=args.thresholds,
        normal_fallback_min_prob=args.normal_fallback_min_prob,
    )

    from webapp.app import _load_thresholds_for_app, _resolve_model_for_display
    print("=" * 72)
    print("WEBAPP CINC2020-12")
    print("=" * 72)
    print("ECG_MODEL   :", runtime_app.config["ECG_MODEL"] or "(auto)")
    print("Resuelto    :", _resolve_model_for_display(runtime_app.config["ECG_MODEL"], runtime_app.config["ECG_SAVED_DIR"]))
    print("ECG_SAVED   :", runtime_app.config["ECG_SAVED_DIR"])
    print("ECG_UPLOADS :", runtime_app.config["ECG_UPLOADS_DIR"])
    _ths, _ths_source = _load_thresholds_for_app(runtime_app.config["ECG_MODEL"], runtime_app.config["ECG_SAVED_DIR"])
    print("ECG_RESULTS :", runtime_app.config["ECG_RESULTS_DIR"])
    print("ECG_REFERENCE:", runtime_app.config["ECG_REFERENCE"] or "(vacío)")
    print("ECG_THRESHOLDS:", _ths_source or "no encontrados; fallback global 0.5")
    print("Fallback N :", runtime_app.config["ECG_NORMAL_FALLBACK_MIN_PROB"], "(0 desactiva)")
    print("URL         :", f"http://{args.host}:{args.port}")
    print("=" * 72)
    runtime_app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
