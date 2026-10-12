"""Arranque de producción: waitress sirve webapp.wsgi:app y respeta PORT."""

import os
import re
import shutil
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


POWERSHELL = shutil.which("powershell")
windows_only = pytest.mark.skipif(sys.platform != "win32" or not POWERSHELL, reason="requiere Windows con PowerShell")
SCRIPTS = [REPO_ROOT / "deploy" / "start_local.ps1"]


def test_local_launcher_runs_waitress_on_the_wsgi_app_bound_to_loopback():
    text = (REPO_ROOT / "deploy" / "start_local.ps1").read_text(encoding="utf-8")
    assert '"-m", "waitress"' in text and "webapp.wsgi:app" in text
    assert "flask run" not in text.lower() and "app.py" not in text        # nunca el servidor de desarrollo
    assert re.search(r'\$bindHost = if \(\$Lan\) \{ "0\.0\.0\.0" \} else \{ "127\.0\.0\.1" \}', text)   # solo local salvo -Lan
    assert re.search(r'elseif \(-not \$env:ECG_DEVICE\) \{ \$env:ECG_DEVICE = "cpu" \}', text)             # CPU por defecto
    assert 'if ($Tunnel) { $env:ECG_TRUST_PROXY = "1" }' in text           # sin esto todos los visitantes comparten el límite
    assert re.search(r"if \(\$Lan -and \$Tunnel\)", text)                     # ECG_TRUST_PROXY solo es seguro en 127.0.0.1


def test_windows_scripts_are_ascii_only_so_powershell_5_does_not_garble_them():
    for script in SCRIPTS:
        script.read_bytes().decode("ascii")


@windows_only
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_windows_scripts_parse_without_syntax_errors(script):
    command = ("$e = $null; [System.Management.Automation.Language.Parser]::ParseFile("
               f"'{script}', [ref]$null, [ref]$e) | Out-Null; $e.Count")
    out = subprocess.run([POWERSHELL, "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "0", out.stdout + out.stderr


@windows_only
def test_launcher_rejects_conflicting_or_impossible_options():
    script = str(REPO_ROOT / "deploy" / "start_local.ps1")
    both = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, "-Lan", "-Tunnel"],
                          capture_output=True, text=True, timeout=60)
    assert both.returncode != 0 and "incompatibles" in both.stderr + both.stdout
    if shutil.which("cloudflared") is None:
        tunnel = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, "-Tunnel"],
                                capture_output=True, text=True, timeout=60)
        assert tunnel.returncode != 0 and "cloudflared" in tunnel.stderr + tunnel.stdout
        assert "ECG webapp:" not in tunnel.stdout                            # falla antes de arrancar nada


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


@windows_only
@pytest.mark.skipif(not BUNDLE_PRESENT, reason="falta el bundle")
def test_launcher_script_serves_the_bundle_on_cpu_and_passes_the_smoke_script():
    from tools import smoke_deployment

    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("ECG_")}
    env.pop("CUDA_VISIBLE_DEVICES", None)                                    # el lanzador decide el dispositivo, no el entorno
    proc = subprocess.Popen([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                             str(REPO_ROOT / "deploy" / "start_local.ps1"), "-Port", str(port)],
                            cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 180
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail("el lanzador terminó antes de tiempo:\n" + proc.stdout.read().decode(errors="replace")[-1500:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                break
            except Exception:
                time.sleep(0.5)
        else:
            pytest.fail("el servidor no respondió en 180 s")
        assert smoke_deployment.main([f"http://127.0.0.1:{port}", "--expect-ensemble"]) == 0
        body = urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/predict", data=b"example=HR10268"), timeout=120).read().decode("utf-8")
        assert '"device":"cpu"' in body.replace(" ", "")                     # sirve en CPU aunque la PC tenga GPU
    finally:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)    # powershell + waitress
