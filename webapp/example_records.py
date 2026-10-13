"""Registros de ejemplo incluidos con la webapp (``webapp/examples/``).

Permiten probar la demo sin tener un ECG propio. El manifiesto
(``examples.json``) lo genera ``tools/build_examples.py``; aquí solo se lee y se
valida. Un identificador que no esté en el manifiesto nunca se convierte en ruta.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
PUBLIC_FIELDS = ("id", "record", "kind", "source_db", "true_classes", "reference_predicted_classes")


def load_manifest(directory: str | Path) -> list[dict]:
    """Lee ``examples.json``; descarta entradas con nombre inseguro o archivos ausentes."""
    directory = Path(directory)
    manifest_path = directory / "examples.json"
    if not manifest_path.exists():
        return []
    valid = []
    for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
        name = str(entry.get("id", ""))
        if not SAFE_NAME.match(name) or entry.get("record") != name:
            continue
        folder = directory / name
        if (folder / f"{name}.hea").is_file() and (folder / f"{name}.mat").is_file():
            valid.append(entry)
    return valid


def public_listing(manifest: list[dict]) -> list[dict]:
    """Lo que ve el navegador: identificadores y etiquetas, sin rutas ni probabilidades de referencia."""
    return [{key: entry[key] for key in PUBLIC_FIELDS if key in entry} for entry in manifest]


def example_files(manifest: list[dict], directory: str | Path, example_id: str) -> list[tuple[str, bytes]]:
    """``[(nombre, contenido)]`` del par .hea/.mat; ``KeyError`` si el ejemplo no existe."""
    if not any(entry["id"] == example_id for entry in manifest):
        raise KeyError(example_id)
    folder = Path(directory) / example_id
    return [(f"{example_id}{suffix}", (folder / f"{example_id}{suffix}").read_bytes()) for suffix in (".hea", ".mat")]
