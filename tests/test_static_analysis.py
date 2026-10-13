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
