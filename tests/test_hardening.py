"""Endurecimiento de la webapp y del cargador de señales (hallazgos S1–S6 de la auditoría).

- S1: ``clean_path_text`` tenía una regex cuadrática (``"[" * 16000`` tardaba 1.1 s y se cuadruplicaba al duplicar N) que
  congelaba waitress con una sola petición.
- S2: ``loadmat`` leía sin tope un ``.mat`` comprimido (5 MB podían ser ≈ 5 GB en RAM).
- S3: ``/use_model`` cambiaba el modelo global sin autenticación y la interfaz no lo usa.
- S5: los servidores de desarrollo escuchaban en 0.0.0.0 por defecto.
- S6: ``escapeHtml`` no escapaba comillas y se usa en atributos HTML.
"""

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from ecg import signal
from webapp import prediction

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# S1 · clean_path_text
# --------------------------------------------------------------------------- #
def _seconds(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


@pytest.mark.parametrize("hostile", ["[" * 200_000, "[a](" * 50_000, "[" + "a" * 200_000, "](" * 100_000, "[[a](" * 40_000, "[a](" * 2000],
                         ids=["corchetes", "enlaces-abiertos", "un-corchete", "cierres", "anidados", "bajo-el-tope"])   # ids cortos: Windows limita PYTEST_CURRENT_TEST
def test_clean_path_text_is_linear_on_hostile_input(hostile):
    _, elapsed = _seconds(prediction.clean_path_text, hostile)
    assert elapsed < 1.0, f"{elapsed:.2f} s con {len(hostile)} caracteres"


def test_clean_path_text_still_unwraps_markdown_links_and_quotes():
    assert prediction.clean_path_text("[best.pt](http://example.com/best.pt)") == "best.pt"
    assert prediction.clean_path_text('  "runs/eval/x" ') == "runs/eval/x"
    assert prediction.clean_path_text("models/[v1].pt") == "models/[v1].pt"
    assert prediction.clean_path_text(None) == ""


def test_clean_field_text_caps_the_length():
    assert prediction.clean_field_text("x" * 5000, 120) == "x" * 120
    assert prediction.clean_field_text("[ana](http://a.b)", 120) == "ana"


# --------------------------------------------------------------------------- #
# S2 · .mat con tope de tamaño
# --------------------------------------------------------------------------- #
def test_load_mat_signal_refuses_a_matrix_larger_than_the_cap_without_reading_it(tmp_path, monkeypatch):
    path = tmp_path / "big.mat"
    scipy.io.savemat(path, {"val": np.zeros((12, 1000), dtype=np.int16)}, do_compression=True)
    monkeypatch.setattr(signal, "MAX_MAT_ELEMENTS", 5_000)
    monkeypatch.setattr(signal, "loadmat", lambda *a, **k: pytest.fail("no debe leerse el contenido de un .mat sobredimensionado"))
    with pytest.raises(ValueError, match="demasiado grande"):
        signal.load_mat_signal(path)


def test_load_mat_signal_reads_a_normal_record_and_has_a_cap_above_real_records(tmp_path):
    path = tmp_path / "ok.mat"
    scipy.io.savemat(path, {"val": np.arange(12 * 5000, dtype=np.int16).reshape(12, 5000)})
    out = signal.load_mat_signal(path)
    assert out.shape == (5000, 12)
    # el registro más largo de CINC2020 (INCART, 30 min a 257 Hz) cabe holgadamente
    assert signal.MAX_MAT_ELEMENTS > 12 * 1800 * 500


def test_load_mat_signal_rejects_a_mat_without_the_val_variable(tmp_path):
    path = tmp_path / "other.mat"
    scipy.io.savemat(path, {"x": np.zeros((12, 10))})
    with pytest.raises(KeyError, match="val"):
        signal.load_mat_signal(path)


# --------------------------------------------------------------------------- #
# S3 · /use_model desactivado por defecto
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client(tmp_path, monkeypatch):
    from webapp.app import create_app

    saved = tmp_path / "saved"
    saved.mkdir()
    monkeypatch.setenv("ECG_SAVED", str(saved))
    monkeypatch.setenv("ECG_UPLOADS", str(tmp_path / "up"))
    monkeypatch.setenv("ECG_RESULTS", str(tmp_path / "res"))
    monkeypatch.setenv("ECG_MODEL", "")
    monkeypatch.delenv("ECG_ENABLE_USE_MODEL", raising=False)
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_use_model_is_disabled_unless_explicitly_enabled(client):
    resp = client.post("/use_model", json={"model": ""})
    assert resp.status_code == 403 and resp.get_json()["ok"] is False
    assert "ECG_ENABLE_USE_MODEL" in resp.get_json()["error"]
    assert client.application.config["ECG_MODEL"] == ""


def test_use_model_works_when_enabled(client, monkeypatch):
    monkeypatch.setenv("ECG_ENABLE_USE_MODEL", "1")
    resp = client.post("/use_model", json={"model": ""})
    assert resp.status_code == 200 and resp.get_json()["ok"] is True


# --------------------------------------------------------------------------- #
# S5 · los servidores escuchan solo en la máquina local por defecto
# --------------------------------------------------------------------------- #
def test_development_servers_default_to_loopback():
    from webapp import app as app_module
    from webapp import wsgi

    assert app_module.DEFAULT_HOST == "127.0.0.1" and wsgi.DEFAULT_HOST == "127.0.0.1"
    for rel in ("webapp/app.py", "webapp/wsgi.py"):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert 'default="0.0.0.0"' not in text, rel


# --------------------------------------------------------------------------- #
# S4 · waitress limita el cuerpo antes de que lo vea Flask
# --------------------------------------------------------------------------- #
def test_local_launcher_limits_the_request_body_to_the_upload_cap():
    text = (REPO_ROOT / "deploy" / "start_local.ps1").read_text(encoding="utf-8")
    assert re.search(r'"--max-request-body-size=\$maxBody"', text)
    assert "ECG_MAX_UPLOAD_MB" in text                                    # el tope sale del mismo valor que usa Flask
    assert text.index("--max-request-body-size") < text.index("webapp.wsgi:app")


# --------------------------------------------------------------------------- #
# S6 · escapeHtml escapa también comillas (se usa dentro de atributos HTML)
# --------------------------------------------------------------------------- #
def _escape_html_source() -> str:
    js = (REPO_ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
    match = re.search(r"function escapeHtml\(s\) \{.*?\n\}", js, flags=re.S)
    assert match, "no se encontró escapeHtml"
    return match.group(0)


def test_escape_html_escapes_quotes_without_needing_the_dom():
    source = _escape_html_source()
    assert "document." not in source                                     # puro: probable sin navegador
    assert "&quot;" in source and "&#39;" in source
    node = shutil.which("node")
    if not node:
        pytest.skip("node no disponible")
    script = source + '\nconsole.log(JSON.stringify([escapeHtml(\'x" onload="alert(1)\'), escapeHtml("<b>&\'"), escapeHtml(null)]));'
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == ['x&quot; onload=&quot;alert(1)', "&lt;b&gt;&amp;&#39;", ""]
