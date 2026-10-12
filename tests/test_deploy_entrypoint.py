"""Arranque de producción: waitress sirve webapp.wsgi:app y respeta PORT."""

import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("waitress")

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PRESENT = (REPO_ROOT / "deploy" / "ensemble.env").exists() and (REPO_ROOT / "models").is_dir()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_start_script_and_procfile_launch_waitress_on_the_wsgi_app():
    start = (REPO_ROOT / "deploy" / "start.sh").read_text(encoding="utf-8")
    procfile = (REPO_ROOT / "Procfile").read_text(encoding="utf-8")
    for text in (start, procfile):
        assert "python -m waitress" in text and "webapp.wsgi:app" in text
        assert "$PORT" in text or "${PORT" in text                    # puerto desde el entorno
        assert "flask" not in text.lower() and "app.py" not in text    # nunca el servidor de desarrollo
    assert start.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in start
    assert re.search(r"--host=.*0\.0\.0\.0", start)


def test_wsgi_module_exposes_the_callable_waitress_expects():
    out = subprocess.run([sys.executable, "-c", "import webapp.wsgi as w; print(callable(w.app))"],
                         cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120,
                         env={**os.environ, "ECG_ENV_FILE": "", "ECG_MODEL": "", "CUDA_VISIBLE_DEVICES": "-1"})
    assert out.returncode == 0 and out.stdout.strip().endswith("True"), out.stderr[-800:]


@pytest.mark.skipif(not BUNDLE_PRESENT, reason="falta el bundle")
def test_real_waitress_server_serves_the_bundle_and_passes_the_smoke_script(tmp_path):
    from tools import smoke_deployment

    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("ECG_")}
    env.update({"ECG_UPLOADS": str(tmp_path / "up"), "ECG_RESULTS": str(tmp_path / "res"), "CUDA_VISIBLE_DEVICES": "-1"})
    proc = subprocess.Popen([sys.executable, "-m", "waitress", f"--listen=127.0.0.1:{port}", "--threads=4", "webapp.wsgi:app"],
                            cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail("waitress terminó antes de tiempo:\n" + proc.stdout.read().decode(errors="replace")[-1500:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                break
            except Exception:
                time.sleep(0.5)
        else:
            pytest.fail("el servidor no respondió en 120 s")
        assert smoke_deployment.main([f"http://127.0.0.1:{port}", "--expect-ensemble"]) == 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
