"""Entrada WSGI para la webapp CINC2020-12.

Variables reconocidas:
    ECG_MODEL      -> checkpoint .pt explícito.
    ECG_SAVED      -> directorio donde buscar checkpoints si ECG_MODEL no existe.
    ECG_UPLOADS    -> carpeta para archivos subidos.
    ECG_RESULTS    -> carpeta para PNG/PDF/CSV convertidos.
    ECG_REFERENCE  -> referencia opcional.
    ECG_THRESHOLDS -> thresholds_validation.csv opcional.
    ECG_NORMAL_FALLBACK_MIN_PROB -> umbral mínimo para fallback NSR.
    ECG_ENV_FILE   -> archivo KEY=VALUE con las variables anteriores (más ECG_MODEL_B,
                      ECG_ALPHA, ECG_TEMPERATURES[_B], ECG_PLATT[_B|_ENSEMBLE], ECG_EVAL_DIR, ECG_EXP_DIR).
                      Por defecto ``deploy/ensemble.env`` si existe; vacío = no cargar.
                      Las variables ya definidas en el entorno tienen prioridad.

Se sirve con ``python -m waitress webapp.wsgi:app`` (``deploy/start_local.ps1``). Para desarrollo con opciones de línea
de comandos use ``python webapp/app.py --help``: es el único servidor de desarrollo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_ENV_FILE = REPO_ROOT / "deploy" / "ensemble.env"


def load_env_file(path) -> dict:
    """Lee ``CLAVE=VALOR`` (ignora comentarios y vacías) y las define si no existen ya en el entorno."""
    loaded = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def _env_file_to_load():
    configured = os.environ.get("ECG_ENV_FILE")
    if configured is not None:                      # definido (aunque vacío) manda sobre el valor por defecto
        return Path(configured) if configured.strip() else None
    return DEFAULT_ENV_FILE if DEFAULT_ENV_FILE.exists() else None


def create_wsgi_app():
    """Carga ``deploy/ensemble.env`` (si procede) y construye la app; lo que no se defina lo decide ``create_app``."""
    env_file = _env_file_to_load()
    if env_file is not None:
        load_env_file(env_file)
    from webapp.app import create_app
    return create_app()


# Variable usada por gunicorn/waitress: webapp.wsgi:app
app = create_wsgi_app()
