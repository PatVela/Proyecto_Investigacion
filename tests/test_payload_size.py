"""Tamaño de lo que se descarga: PNG del ECG, PDF y compresión HTTP, sin cambiar los resultados.

Por un túnel de internet desde esta PC las descargas grandes iban a ~10-20 KB/s: el PNG (762 KB), el PDF (952 KB)
y Plotly (4,6 MB) no terminaban. Estas pruebas fijan cuánto se reduce cada uno y que la imagen sigue siendo la misma.
"""

import gzip
import io
import os
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flask")
from PIL import Image

from webapp import prediction as pred

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "webapp" / "examples"
BUNDLE_PRESENT = (REPO_ROOT / "deploy" / "ensemble.env").exists() and (REPO_ROOT / "models").is_dir()


@pytest.fixture(scope="module")
def real_signal(tmp_path_factory):
    """Ventana real preprocesada (10 s, 12 derivaciones) de un registro de ejemplo."""
    work = tmp_path_factory.mktemp("signal")
    prepared = pred.prepare_saved_input([EXAMPLES / "A3763" / "A3763.hea", EXAMPLES / "A3763" / "A3763.mat"], work)
    return np.asarray(prepared["processed_signal"], dtype=np.float32)


# --------------------------------------------------------------------------- #
# PNG del ECG
# --------------------------------------------------------------------------- #
def _reference_png(signal, tmp_path):
    """Lo que se escribía antes: ``savefig`` directo, sin optimizar."""
    original = pred._save_figure_png
    pred._save_figure_png = lambda fig, path: fig.savefig(path, dpi=pred.PLOT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    try:
        return pred.plot_ecg_array(signal, tmp_path / "ref", "rec")
    finally:
        pred._save_figure_png = original


def test_ecg_png_is_much_smaller_and_still_the_same_picture(real_signal, tmp_path):
    reference = _reference_png(real_signal, tmp_path)
    optimized = pred.plot_ecg_array(real_signal, tmp_path / "opt", "rec")
    assert optimized.stat().st_size <= 0.5 * reference.stat().st_size            # antes ≈ 762 KB, ahora ≈ 306 KB

    with Image.open(reference) as a, Image.open(optimized) as b:
        assert a.size == b.size and b.mode == "P"                                  # mismas dimensiones; paleta de ≤ 256 colores
        ref = np.asarray(a.convert("RGB")).astype(int)
        new = np.asarray(b.convert("RGB")).astype(int)
    diff = np.abs(new - ref)
    assert diff.max() <= 20                                                        # imperceptible (≈ 8 % de un escalón de gris)
    assert (diff.max(axis=2) > 8).mean() < 0.001                                   # < 0,1 % de los píxeles cambia de forma apreciable
    trace_ref = (ref < 90).all(axis=2)                                              # píxeles del trazo negro
    trace_new = (new < 110).all(axis=2)
    assert (trace_ref & trace_new).sum() / trace_ref.sum() > 0.999                 # el trazo de la señal se conserva


def test_png_is_written_without_an_alpha_channel_and_with_dpi(real_signal, tmp_path):
    path = pred.plot_ecg_array(real_signal, tmp_path, "rec")
    with Image.open(path) as image:
        assert "A" not in image.getbands() and "transparency" not in image.info     # el fondo es opaco: no hace falta alfa
        assert round(image.info.get("dpi", (0, 0))[0]) == pred.PLOT_DPI


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not BUNDLE_PRESENT, reason="falta el bundle")
def test_pdf_report_shrinks_and_keeps_its_content(tmp_path, monkeypatch):
    from pypdf import PdfReader

    from webapp.app import create_app

    for key in [k for k in os.environ if k.startswith("ECG_")]:
        monkeypatch.delenv(key, raising=False)
    env = dict(line.split("=", 1) for line in (REPO_ROOT / "deploy" / "ensemble.env").read_text(encoding="utf-8").splitlines()
               if line and not line.startswith("#"))
    for key, value in env.items():
        monkeypatch.setenv(key, str(REPO_ROOT / value) if key != "ECG_ALPHA" else value)
    monkeypatch.setenv("ECG_UPLOADS", str(tmp_path / "up"))
    monkeypatch.setenv("ECG_RESULTS", str(tmp_path / "res"))
    monkeypatch.setenv("ECG_RATE_LIMIT_MAX", "0")
    monkeypatch.setenv("ECG_DEVICE", "cpu")
    client = create_app().test_client()
    body = client.post("/predict", data={"example": "A3763"}).get_json()
    pdf_url = re.search(r"/results/[0-9a-f]{12}/[^\"']+\.pdf", str(body)).group(0)
    pdf_bytes = client.get(pdf_url).data
    assert len(pdf_bytes) < 600_000                                                 # antes ≈ 952 KB

    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = " ".join(page.extract_text() for page in reader.pages)
    assert "A3763" in text and "AF" in text                                         # el informe sigue diciendo lo mismo
    big = [im for page in reader.pages for im in page.images if im.image.size[0] > 2000]
    assert big, "el trazado ECG sigue incrustado a su resolución completa"


# --------------------------------------------------------------------------- #
# compresión HTTP
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client(monkeypatch, tmp_path):
    from webapp.app import create_app

    for key in [k for k in os.environ if k.startswith("ECG_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ECG_UPLOADS", str(tmp_path / "up"))
    monkeypatch.setenv("ECG_RESULTS", str(tmp_path / "res"))
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _versioned(client, name):
    html = client.get("/").get_data(as_text=True)
    return re.search(rf'(/static/{re.escape(name)}\?v=[0-9a-f]{{10}})', html).group(1)


def test_plotly_is_served_gzipped_when_the_client_accepts_it(client):
    url = re.search(r'data-plotly-src="([^"]+)"', client.get("/").get_data(as_text=True)).group(1)
    plain = client.get(url)
    packed = client.get(url, headers={"Accept-Encoding": "gzip"})
    assert "Content-Encoding" not in plain.headers
    assert packed.headers["Content-Encoding"] == "gzip" and "Accept-Encoding" in packed.headers["Vary"]
    assert gzip.decompress(packed.data) == plain.data                               # exactamente el mismo archivo
    assert len(packed.data) < 0.4 * len(plain.data)                                 # 4,6 MB -> ≈ 1,3 MB
    assert int(packed.headers["Content-Length"]) == len(packed.data)
    assert packed.headers["Cache-Control"] == "public, max-age=31536000, immutable"


def test_css_js_html_and_json_are_compressed_but_images_and_pdfs_are_not(client):
    accept = {"Accept-Encoding": "gzip, deflate"}
    for url in (_versioned(client, "app.js"), _versioned(client, "style.css"), "/"):
        response = client.get(url, headers=accept)
        assert response.headers.get("Content-Encoding") == "gzip", url
        assert gzip.decompress(response.data) == client.get(url).data, url
    assert "Content-Encoding" not in client.get(_versioned(client, "escudo-unsa.png"), headers=accept).headers   # PNG ya comprimido


def test_a_client_that_does_not_accept_gzip_gets_the_plain_file(client):
    for header in ({}, {"Accept-Encoding": "identity"}, {"Accept-Encoding": "gzip;q=0"}):
        response = client.get(_versioned(client, "app.js"), headers=header)
        assert "Content-Encoding" not in response.headers and response.status_code == 200


def test_compressed_static_files_still_revalidate_with_a_304(client):
    first = client.get("/static/app.js", headers={"Accept-Encoding": "gzip"})
    again = client.get("/static/app.js", headers={"Accept-Encoding": "gzip", "If-None-Match": first.headers["ETag"]})
    assert first.headers["ETag"].startswith("W/")                                   # la variante comprimida lleva ETag débil
    assert again.status_code == 304 and len(again.data) == 0


def test_error_and_empty_responses_are_left_alone(client):
    response = client.post("/predict", data={}, headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 400 and response.get_json()["ok"] is False        # cuerpo pequeño: sin comprimir


@pytest.mark.skipif(not BUNDLE_PRESENT, reason="falta el bundle")
def test_prediction_json_is_identical_after_decompression(tmp_path, monkeypatch):
    from webapp.app import create_app

    for key in [k for k in os.environ if k.startswith("ECG_")]:
        monkeypatch.delenv(key, raising=False)
    env = dict(line.split("=", 1) for line in (REPO_ROOT / "deploy" / "ensemble.env").read_text(encoding="utf-8").splitlines()
               if line and not line.startswith("#"))
    for key, value in env.items():
        monkeypatch.setenv(key, str(REPO_ROOT / value) if key != "ECG_ALPHA" else value)
    monkeypatch.setenv("ECG_UPLOADS", str(tmp_path / "up"))
    monkeypatch.setenv("ECG_RESULTS", str(tmp_path / "res"))
    monkeypatch.setenv("ECG_RATE_LIMIT_MAX", "0")
    monkeypatch.setenv("ECG_DEVICE", "cpu")
    client = create_app().test_client()
    plain = client.post("/predict", data={"example": "HR10268"}).get_json()
    packed = client.post("/predict", data={"example": "HR10268"}, headers={"Accept-Encoding": "gzip"})
    assert packed.headers["Content-Encoding"] == "gzip"
    again = __import__("json").loads(gzip.decompress(packed.data))
    probabilities = lambda body: {p["class"]: p["probability"] for p in body["predictions"]}
    assert probabilities(again) == probabilities(plain)                              # mismas probabilidades
