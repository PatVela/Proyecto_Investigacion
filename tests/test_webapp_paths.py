"""La webapp no revela cómo está organizado el disco del servidor.

Medido antes: /health devolvía 8 rutas absolutas, /models las de cada checkpoint y la portada la del modelo en un atributo
``title``; con ``-Tunnel`` cualquiera con la URL las veía. Ahora se muestran relativas al repositorio o solo con su nombre.
"""

import io
import json
import re
from pathlib import Path

import pandas as pd
import pytest
import torch

pytest.importorskip("flask")

from ecg import load, network
from tools.smoke_deployment import synthetic_csv
from webapp.app import _public_path, create_app

REPO_ROOT = Path(__file__).resolve().parent.parent
ABSOLUTE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\\\[A-Za-z]|\"/(?:home|Users|tmp|var|etc|mnt)/")


def test_public_path_is_relative_to_the_repo_or_just_the_name(tmp_path):
    assert _public_path(REPO_ROOT / "models" / "resnet34_v1.pt") == "models/resnet34_v1.pt"
    assert _public_path(str(tmp_path / "secreto" / "a.pt")) == "a.pt"             # fuera del repo: solo el nombre
    assert _public_path("deploy/platt_a.csv") == "deploy/platt_a.csv"
    assert _public_path("fallback_global_0.5") == "fallback_global_0.5"            # no es una ruta
    assert _public_path(None) is None and _public_path("") == ""


@pytest.fixture()
def client(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = network.build_network(num_leads=12, num_classes=12)
    saved = tmp_path / "datos_privados" / "saved"
    saved.mkdir(parents=True)
    ckpt = saved / "a.pt"
    torch.save({"model_state_dict": model.state_dict(), "class_names": load.CLASS_NAMES.copy(),
                "config": {"label_schema": load.LABEL_SCHEMA, "norm_mode": "physical", "bandpass": True},
                "label_schema": load.LABEL_SCHEMA, "epoch": 1, "val_loss": 0.5}, str(ckpt))
    for key in ("ECG_TEMPERATURES", "ECG_TEMPERATURES_B", "ECG_EVAL_DIR", "ECG_PLATT", "ECG_PLATT_B",
                "ECG_PLATT_ENSEMBLE", "ECG_MODEL_B"):
        monkeypatch.delenv(key, raising=False)
    thresholds = tmp_path / "datos_privados" / "thresholds_validation.csv"        # su ruta aparece en technical_details
    pd.DataFrame({"class": load.CLASS_NAMES, "threshold": [0.5] * 12}).to_csv(thresholds, index=False)
    monkeypatch.setenv("ECG_THRESHOLDS", str(thresholds))
    monkeypatch.setenv("ECG_MODEL", str(ckpt))
    monkeypatch.setenv("ECG_SAVED", str(saved))
    monkeypatch.setenv("ECG_UPLOADS", str(tmp_path / "datos_privados" / "up"))
    monkeypatch.setenv("ECG_RESULTS", str(tmp_path / "datos_privados" / "res"))
    monkeypatch.setenv("ECG_DEVICE", "cpu")
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client(), tmp_path


def _assert_no_server_paths(text, tmp_path, where):
    assert str(tmp_path) not in text and "datos_privados" not in text, where
    assert not ABSOLUTE.search(text), (where, ABSOLUTE.search(text).group(0))


def test_health_models_and_home_page_show_no_absolute_paths(client):
    http, tmp_path = client
    _assert_no_server_paths(json.dumps(http.get("/health").get_json()), tmp_path, "/health")
    _assert_no_server_paths(json.dumps(http.get("/models").get_json()), tmp_path, "/models")
    _assert_no_server_paths(http.get("/").get_data(as_text=True), tmp_path, "/")
    assert http.get("/models").get_json()["models"] == ["a.pt"]


def test_predict_response_shows_no_absolute_paths_but_the_links_still_work(client):
    http, tmp_path = client
    r = http.post("/predict", data={"files": (io.BytesIO(synthetic_csv()), "ecg.csv")}, content_type="multipart/form-data")
    body = r.get_json()
    assert r.status_code == 200 and body["ok"] is True
    _assert_no_server_paths(json.dumps(body), tmp_path, "/predict")
    assert body["model_path"] == "a.pt" and body["source_files"] == ["ecg.csv"]
    assert body["technical_details"]["threshold_source"] == "thresholds_validation.csv"
    assert http.get(body["pdf_url"]).status_code == 200 and http.get(body["plot_url"]).status_code == 200


def test_the_real_paths_are_still_used_internally(client):
    """Ocultar la ruta en la respuesta no puede cambiar la que usa el servidor."""
    http, tmp_path = client
    assert Path(http.application.config["ECG_MODEL"]).is_file()
