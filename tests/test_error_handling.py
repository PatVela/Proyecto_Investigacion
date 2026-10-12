"""/predict: los errores de entrada son 400 y los fallos internos no filtran rutas del servidor."""

import io
import os

import pytest
import torch

pytest.importorskip("flask")

from ecg import load, network
from tools.smoke_deployment import synthetic_csv
from webapp import app as app_mod

SECRET = r"C:\srv\secret\models\best.pt"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    for key in [k for k in os.environ if k.startswith("ECG_")]:
        monkeypatch.delenv(key, raising=False)
    torch.save({"model_state_dict": network.build_network(num_leads=12, num_classes=12).state_dict(),
                "class_names": load.CLASS_NAMES, "label_schema": load.LABEL_SCHEMA}, str(tmp_path / "m.pt"))
    for key, value in {"ECG_MODEL": tmp_path / "m.pt", "ECG_SAVED": tmp_path / "s", "ECG_UPLOADS": tmp_path / "u",
                       "ECG_RESULTS": tmp_path / "r", "ECG_RATE_LIMIT_MAX": 0}.items():
        monkeypatch.setenv(key, str(value))
    app = app_mod.create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _csv(name="a.csv", data=None):
    return {"files": (io.BytesIO(data if data is not None else synthetic_csv()), name)}


@pytest.mark.parametrize("payload,fragment", [
    (lambda: {"files": (io.BytesIO(b"x"), "evil.pt")}, "Extensión no permitida"),
    (lambda: {"files": (io.BytesIO(b"x"), "a.hea")}, "Formato no soportado"),
    (lambda: {**_csv(), "threshold": "1.5"}, "threshold"),
    (lambda: _csv("bad.csv", b"# Sampling Rate: 0.01 Hz\n" + synthetic_csv().split(b"\n", 1)[1]), "fuera de rango"),
])
def test_invalid_input_is_a_400_with_a_helpful_message(client, payload, fragment):
    resp = client.post("/predict", data=payload(), content_type="multipart/form-data")
    body = resp.get_json()
    assert resp.status_code == 400 and body["ok"] is False and fragment in body["error"]


def test_internal_failures_are_500_without_leaking_server_paths(client, monkeypatch, caplog):
    def boom(**kwargs):
        raise FileNotFoundError(f"No se pudo abrir {SECRET}")

    monkeypatch.setattr(app_mod, "run_prediction_from_uploads", boom)
    with caplog.at_level("ERROR"):
        resp = client.post("/predict", data=_csv(), content_type="multipart/form-data")
    body = resp.get_json()
    assert resp.status_code == 500 and body["ok"] is False
    assert SECRET not in body["error"] and "secret" not in body["error"] and "Error interno" in body["error"]
    assert SECRET in caplog.text and "Fallo interno en /predict" in caplog.text            # el detalle sí queda en el log del servidor


def test_debug_flag_shows_the_detail_for_local_development(client, monkeypatch):
    monkeypatch.setattr(app_mod, "run_prediction_from_uploads", lambda **kw: (_ for _ in ()).throw(RuntimeError("detalle interno")))
    monkeypatch.setenv("ECG_DEBUG_ERRORS", "1")
    resp = client.post("/predict", data=_csv(), content_type="multipart/form-data")
    assert resp.status_code == 500 and resp.get_json()["error"] == "detalle interno"


def test_a_valid_request_still_succeeds(client):
    assert client.post("/predict", data=_csv(), content_type="multipart/form-data").status_code == 200
