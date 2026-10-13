"""Ningún print() del código debe llevar caracteres que una consola de Windows (cp1252) no puede codificar.

Regresión: ``ecg/train.py`` imprimía "✓" al guardar un mejor modelo. Con la salida redirigida (``| Tee-Object``, un archivo
de log) Python usa cp1252, el print lanzaba ``UnicodeEncodeError`` y el entrenamiento moría en la primera época que mejoraba.
"""

import ast
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _tracked_python_files():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO_ROOT, capture_output=True, text=True)
    return [REPO_ROOT / name for name in out.stdout.split()
            if not name.startswith("tests/") and "/official/" not in name]


def _unencodable_prints(path: Path):
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print":
            for part in ast.walk(node):
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    try:
                        part.value.encode("cp1252")
                    except UnicodeEncodeError as exc:
                        found.append((node.lineno, ascii(part.value[exc.start:exc.end])))
                        break
    return found


def test_no_print_uses_characters_outside_cp1252():
    problems = {}
    for path in _tracked_python_files():
        bad = _unencodable_prints(path)
        if bad:
            problems[str(path.relative_to(REPO_ROOT))] = bad
    assert not problems, problems
