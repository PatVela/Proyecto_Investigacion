"""El código propio no debe acumular imports ni variables sin usar (pyflakes)."""

import subprocess
from pathlib import Path

import pytest

pyflakes_api = pytest.importorskip("pyflakes.api")
reporter = pytest.importorskip("pyflakes.reporter")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _own_python_files():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO_ROOT, capture_output=True, text=True)
    return [REPO_ROOT / name for name in out.stdout.split()
            if "/official/" not in name and "/_archive/" not in name]       # vendored y depuración fuera de la secuencia


def test_pyflakes_reports_nothing_in_our_own_code():
    import io

    stream = io.StringIO()
    rep = reporter.Reporter(stream, stream)
    for path in _own_python_files():
        pyflakes_api.checkPath(str(path), rep)
    assert stream.getvalue() == "", stream.getvalue()


def test_serving_code_does_not_import_the_experiment_scripts():
    """webapp/ y ecg/ no dependen de examples/ (la demo importaba ensemble_evaluate.py solo para promediar dos vectores)."""
    import ast

    offenders = []
    for folder in ("webapp", "ecg"):
        for path in (REPO_ROOT / folder).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                        [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
                offenders += [f"{path.name}:{node.lineno} {n}" for n in names if n == "examples" or n.startswith("examples.")]
    assert not offenders, offenders
