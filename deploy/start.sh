#!/usr/bin/env bash
# Arranque de produccion: waitress sirve webapp.wsgi:app.
# Carga deploy/ensemble.env por defecto (ECG_ENV_FILE para otro archivo; vacio = ninguno).
# Variables: PORT (5002), HOST (0.0.0.0), WAITRESS_THREADS (4).
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m waitress --host="${HOST:-0.0.0.0}" --port="${PORT:-5002}" --threads="${WAITRESS_THREADS:-4}" webapp.wsgi:app
