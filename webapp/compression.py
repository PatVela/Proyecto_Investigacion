"""Compresión gzip de las respuestas de texto (HTML, CSS, JS, JSON, CSV) cuando el cliente la acepta.

waitress no comprime y por un túnel de internet desde esta PC Plotly (4,6 MB) no terminaba de bajar. Con gzip pesa
≈ 1,3 MB y el resto de texto baja a una cuarta parte. Las imágenes y el PDF ya van comprimidos y no se tocan.
Los archivos estáticos se comprimen una sola vez (nivel 9) y se guardan en memoria; el contenido es idéntico
después de descomprimir. Un ETag fuerte pasa a débil (``W/``), como hacen nginx y Apache para la variante comprimida.
"""

from __future__ import annotations

import gzip
import re

COMPRESSIBLE_TYPES = {"text/html", "text/css", "text/javascript", "application/javascript", "application/json", "text/csv"}
MIN_COMPRESS_BYTES = 1024
_GZIP_ITEM = re.compile(r"^\s*(?P<coding>[A-Za-z0-9*-]+)\s*(?:;\s*q\s*=\s*(?P<q>[0-9.]+))?\s*$")


def accepts_gzip(accept_encoding: str) -> bool:
    """``True`` si ``Accept-Encoding`` admite gzip (o ``*``) con calidad mayor que 0."""
    for item in (accept_encoding or "").split(","):
        match = _GZIP_ITEM.match(item)
        if not match or match.group("coding").lower() not in {"gzip", "x-gzip", "*"}:
            continue
        try:
            return float(match.group("q") or 1.0) > 0.0
        except ValueError:
            return False
    return False


def compress_response(response, accept_encoding: str, cache: dict | None = None, cache_key=None):
    """Comprime ``response`` con gzip si procede; ``cache``/``cache_key`` reutilizan el resultado de archivos estáticos."""
    if response.mimetype not in COMPRESSIBLE_TYPES:
        return response
    response.headers.add("Vary", "Accept-Encoding")
    if (response.status_code != 200 or "Content-Encoding" in response.headers
            or not accepts_gzip(accept_encoding)):
        return response

    packed = cache.get(cache_key) if cache is not None and cache_key is not None else None
    if packed is None:
        response.direct_passthrough = False            # los archivos estáticos llegan como flujo: se leen enteros
        data = response.get_data()
        if len(data) < MIN_COMPRESS_BYTES:
            response.set_data(data)                    # devuelve el cuerpo que se acaba de leer
            return response
        packed = gzip.compress(data, compresslevel=9 if cache is not None else 6, mtime=0)
        if cache is not None and cache_key is not None:
            cache[cache_key] = packed
    response.direct_passthrough = False
    response.set_data(packed)
    response.headers["Content-Encoding"] = "gzip"
    etag = response.get_etag()[0]
    if etag:
        response.set_etag(etag, weak=True)
    return response
