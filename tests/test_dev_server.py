"""`python webapp/app.py` (servidor de desarrollo) debe funcionar igual que waitress.

Regresión: `webapp/examples.py` tapaba al paquete `examples/` de la raíz cuando se ejecutaba `python webapp/app.py`
(en `sys.path` queda la carpeta `webapp/`, y un módulo normal gana a un paquete sin `__init__.py`), de modo que
`/predict` con el ensemble fallaba con "'examples' is not a package". waitress no lo mostraba porque no pone
`webapp/` en la ruta de importación.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PRESENT = (REPO_ROOT / "deploy" / "ensemble.env").exists() and (REPO_ROOT / "models").is_dir()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_no_webapp_module_shares_its_name_with_a_top_level_package():
    """Cualquier `webapp/<nombre>.py` o carpeta con el nombre de una carpeta de la raíz puede tapar su import."""
    top_level = {p.name for p in REPO_ROOT.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))}
    inside_webapp = {p.stem for p in (REPO_ROOT / "webapp").glob("*.py")}
    assert not (top_level & inside_webapp), sorted(top_level & inside_webapp)


def test_repo_packages_import_with_the_webapp_folder_first_on_the_path():
    """Lo que ve el proceso cuando se ejecuta `python webapp/app.py`: `sys.path[0]` es `webapp/`."""
    code = ("import sys; sys.path[:0] = [r'%s', r'%s']\n"
            "from examples.cinc2020.ensemble_evaluate import ensemble_probabilities\n"
            "import ecg, tools\n"
            "print('ok')") % (REPO_ROOT / "webapp", REPO_ROOT)
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-600:]


@pytest.mark.skipif(not BUNDLE_PRESENT, reason="falta el bundle")
def test_dev_server_serves_an_ensemble_prediction(tmp_path):
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("ECG_")}
    env.update({"ECG_UPLOADS": str(tmp_path / "up"), "ECG_RESULTS": str(tmp_path / "res"), "ECG_DEVICE": "cpu",
                "ECG_RATE_LIMIT_MAX": "0", "CUDA_VISIBLE_DEVICES": "-1"})
    command = [sys.executable, "webapp/app.py", "--host", "127.0.0.1", "--port", str(port),
               "--model", "models/resnet34_v1.pt", "--model-b", "models/resnet34_v2.pt", "--alpha", "0.5",
               "--platt", "deploy/platt_a.csv", "--platt-b", "deploy/platt_b.csv", "--platt-ensemble", "deploy/platt_ensemble.csv",
               "--thresholds", "deploy/results/ensemble-f1-platt/thresholds_validation.csv",
               "--eval-dir", "deploy/results/ensemble-f1-platt", "--exp-dir", "deploy/reports"]
    proc = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail("el servidor de desarrollo terminó antes de tiempo")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                break
            except Exception:
                time.sleep(0.5)
        else:
            pytest.fail("el servidor no respondió en 120 s")
        request = urllib.request.Request(f"http://127.0.0.1:{port}/predict", data=b"example=HR10268", method="POST")
        body = json.loads(urllib.request.urlopen(request, timeout=120).read())      # un 500 lanza HTTPError
        assert body["ok"] is True and "ensemble" in json.dumps(body).lower()
    finally:
        proc.kill()
