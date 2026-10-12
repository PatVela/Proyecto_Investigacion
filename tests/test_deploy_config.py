"""Configuración por entorno y arranque desde un clon limpio (sin saved/, runs/ ni data/)."""

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("flask")

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PRESENT = (REPO_ROOT / "deploy" / "ensemble.env").exists() and (REPO_ROOT / "models").is_dir()


@pytest.fixture()
def clean_env():
    saved = dict(os.environ)
    for key in [k for k in os.environ if k.startswith("ECG_")]:
        del os.environ[key]
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture()
def wsgi():
    """Importa ``webapp.wsgi`` sin dejar variables ECG_* definidas.

    El módulo construye la app al importarse (``app = create_wsgi_app()``) y eso carga
    ``deploy/ensemble.env`` en ``os.environ``; sin restaurar, contaminaría otras pruebas.
    """
    saved = dict(os.environ)
    import webapp.wsgi as module
    os.environ.clear()
    os.environ.update(saved)
    return module


# --------------------------------------------------------------------------- #
# lector del archivo de entorno
# --------------------------------------------------------------------------- #
def test_load_env_file_parses_and_respects_existing_environment(tmp_path, clean_env, wsgi):
    load_env_file = wsgi.load_env_file
    for key in [k for k in os.environ if k.startswith("ECG_")]:
        del os.environ[key]
    env = tmp_path / "x.env"
    env.write_text('# comentario\n\nECG_ALPHA=0.7\nECG_MODEL="models/a.pt"\nECG_EXP_DIR = \'deploy/reports\'\nsin_igual\n',
                   encoding="utf-8")
    os.environ["ECG_ALPHA"] = "0.3"                                   # lo ya definido gana
    loaded = load_env_file(env)
    assert os.environ["ECG_ALPHA"] == "0.3" and "ECG_ALPHA" not in loaded
    assert os.environ["ECG_MODEL"] == "models/a.pt" and os.environ["ECG_EXP_DIR"] == "deploy/reports"


def test_env_file_selection_rules(tmp_path, clean_env, wsgi):
    custom = tmp_path / "c.env"
    os.environ["ECG_ENV_FILE"] = str(custom)
    assert wsgi._env_file_to_load() == custom
    os.environ["ECG_ENV_FILE"] = ""                                   # vacío: no cargar nada
    assert wsgi._env_file_to_load() is None
    del os.environ["ECG_ENV_FILE"]
    assert wsgi._env_file_to_load() == (wsgi.DEFAULT_ENV_FILE if wsgi.DEFAULT_ENV_FILE.exists() else None)


# --------------------------------------------------------------------------- #
# clon limpio: solo código versionado + models/ + deploy/
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not BUNDLE_PRESENT, reason="falta el bundle (python tools/export_inference_bundle.py)")
def test_app_starts_from_a_clean_copy_using_only_the_bundle(tmp_path):
    clone = tmp_path / "clone"
    clone.mkdir()
    local_only = {REPO_ROOT / "webapp" / "uploads", REPO_ROOT / "webapp" / "results"}   # datos locales, no versionados

    def ignore(directory, names):
        skipped = {n for n in names if n == "__pycache__" or (Path(directory) / n) in local_only}
        return skipped

    for name in ("ecg", "webapp", "examples", "models", "deploy"):
        shutil.copytree(REPO_ROOT / name, clone / name, ignore=ignore)
    assert not (clone / "saved").exists() and not (clone / "runs").exists() and not (clone / "data").exists()

    env = {k: v for k, v in os.environ.items() if not k.startswith("ECG_")}
    env.update({"PYTHONPATH": str(clone), "ECG_UPLOADS": str(tmp_path / "up"), "ECG_RESULTS": str(tmp_path / "res"),
                "CUDA_VISIBLE_DEVICES": "-1"})
    code = ("import json; from webapp.wsgi import app; c = app.test_client(); "
            "print(json.dumps({'health': c.get('/health').get_json(), 'metrics': c.get('/metrics').get_json()['found'], "
            "'exp': c.get('/experiments').get_json()['comparison_found'], 'models': len(c.get('/models').get_json()['models'])}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=clone, env=env, capture_output=True, text=True,
                         encoding="utf-8", timeout=300)
    assert out.returncode == 0, out.stderr[-1500:]
    data = json.loads(out.stdout.strip().splitlines()[-1])
    health = data["health"]
    assert health["ensemble"] is True and health["ensemble_alpha"] == 0.5
    assert health["thresholds_found"] and health["temperatures_found"] and health["temperatures_b_found"]
    assert str(clone) in health["resolved_model"]                       # usa el bundle, no otra carpeta
    assert data["metrics"] is True and data["exp"] is True and data["models"] == 2
    assert not (clone / "saved").exists() or not any((clone / "saved").iterdir())


# --------------------------------------------------------------------------- #
# prueba de humo completa contra un servidor real
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not BUNDLE_PRESENT, reason="falta el bundle")
def test_smoke_script_passes_against_a_live_server(tmp_path, clean_env, wsgi):
    from werkzeug.serving import make_server

    from tools import smoke_deployment
    create_wsgi_app = wsgi.create_wsgi_app

    os.environ.update({"ECG_UPLOADS": str(tmp_path / "up"), "ECG_RESULTS": str(tmp_path / "res")})
    server = make_server("127.0.0.1", 0, create_wsgi_app(), threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        logs = []
        results = smoke_deployment.run_smoke(f"http://127.0.0.1:{server.server_port}", expect_ensemble=True,
                                             log=logs.append)
        failed = [name for name, ok, detail in results if not ok]
        assert not failed, "\n".join(logs)
        assert len(results) >= 15
    finally:
        server.shutdown()
        thread.join(timeout=10)


def test_smoke_script_fails_against_a_dead_url():
    from tools import smoke_deployment

    with pytest.raises(Exception):
        smoke_deployment.run_smoke("http://127.0.0.1:9", timeout=2, log=lambda *_: None)
