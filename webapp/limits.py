"""Límites para exponer la webapp en público: tasa por cliente y limpieza de archivos temporales.

Sin dependencias externas: un limitador de ventana deslizante en memoria (válido para
un único proceso, que es como se despliega la demo) y una limpieza que solo toca
carpetas con nombre de trabajo (12 caracteres hexadecimales).
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

JOB_DIR_RE = re.compile(r"^[0-9a-f]{12}$")


def int_env(name: str, default: int) -> int:
    """Entero de la variable ``name``; ausente o vacía = ``default``. Un texto que no es entero es un error de configuración:
    antes valía ``default`` en silencio (``ECG_TRUST_PROXY=uno`` quedaba en 0 y todo un túnel compartía un solo cupo)."""
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} debe ser un número entero; recibido {value!r}") from None


def float_env(name: str, default: float) -> float:
    """Decimal de la variable ``name``; ausente o vacía = ``default``. Un texto que no es número (``0,5`` con coma) es un error
    de configuración que nombra la variable, no el ``could not convert string to float`` de ``float()``."""
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name} debe ser un número; recibido {value!r}") from None


class SlidingWindowLimiter:
    """Permite ``max_requests`` por ``window_seconds`` para cada clave (p. ej. una IP).

    ``max_requests <= 0`` desactiva el límite. Seguro entre hilos.
    """

    def __init__(self, max_requests: int, window_seconds: float, clock=time.monotonic):
        self.max_requests = int(max_requests)
        self.window = float(window_seconds)
        self._clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_requests > 0 and self.window > 0

    def check(self, key: str) -> tuple[bool, float]:
        """``(permitido, segundos_hasta_poder_reintentar)``; registra la petición si se permite."""
        if not self.enabled:
            return True, 0.0
        now = self._clock()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False, max(0.0, self.window - (now - hits[0]))
            hits.append(now)
            if len(self._hits) > 10_000:                         # acota la memoria ante muchas IP distintas
                for stale in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                    del self._hits[stale]
            return True, 0.0


def cleanup_old_jobs(root: str | Path, ttl_seconds: float, now: float | None = None) -> int:
    """Borra las carpetas de trabajo de ``root`` más antiguas que ``ttl_seconds``.

    Solo considera subcarpetas con nombre de trabajo (``^[0-9a-f]{12}$``); cualquier otra
    cosa en ``root`` se respeta. ``ttl_seconds <= 0`` desactiva la limpieza. Devuelve cuántas borró.
    """
    root = Path(root)
    if ttl_seconds <= 0 or not root.is_dir():
        return 0
    now = time.time() if now is None else now
    removed = 0
    for child in root.iterdir():
        try:
            if child.is_dir() and JOB_DIR_RE.match(child.name) and now - child.stat().st_mtime > ttl_seconds:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed
