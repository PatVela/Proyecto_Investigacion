"""Etiquetas reales escritas a mano: lo que no se reconoce se rechaza, no se descarta en silencio.

Regresión medida en la webapp: con el registro A3763 (AF) y la etiqueta ``AFF`` la comparación usaba "ninguna etiqueta
real", marcaba el AF acertado como falso positivo y mostraba F1 0.0; con ``AF, XYZ`` ocultaba el texto inválido y salía
un acierto perfecto. Nada de esto tenía pruebas.
"""

import io

import numpy as np
import pytest
import torch

pytest.importorskip("flask")

from ecg import load, network
from webapp import labels
from webapp import prediction as pred
from webapp.app import create_app

LEADS = load.STANDARD_LEAD_ORDER


# --- el analizador ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,found,unknown", [
    ("AF, MI", ["AF", "MI"], []),
    ("af;mi", ["AF", "MI"], []),                                       # minúsculas y punto y coma
    ("164889003, 164865005", ["AF", "MI"], []),                         # códigos SNOMED
    ("LAD, RBBB", ["AxisDev", "BBB"], []),                              # alias del esquema v1
    ("Atrial arrhythmia group", ["AF"], []),                            # nombre largo de la clase
    ("NSR, NSR", ["NSR"], []),
    ("AFF, MI", ["MI"], ["AFF"]),                                       # errata de AF
    ("AF, XYZ", ["AF"], ["XYZ"]),
    ("AF MI", [], ["AF MI"]),                                           # sin coma es un solo texto desconocido
    ("fibrilacion auricular", [], ["fibrilacion auricular"]),
    ("AFF, AFF", [], ["AFF"]),                                          # sin repetidos
    ("55827005", [], ["55827005"]),                                     # SNOMED válido pero de una clase que no modelamos
    ("", [], []), ("  ,  ;\n", [], []), (None, [], []),
], ids=lambda v: repr(v)[:22])
def test_split_reference_label_text(text, found, unknown):
    assert labels.split_reference_label_text(text) == (found, unknown)


def test_a_list_of_labels_is_also_checked():
    assert labels.split_reference_label_text(["AF", "AFF"]) == (["AF"], ["AFF"])


def test_the_prediction_rejects_unrecognized_labels_before_loading_any_model(tmp_path):
    with pytest.raises(ValueError, match=r"no reconocidas: AFF, XYZ"):
        pred.run_prediction_from_saved_paths([tmp_path / "no_importa.csv"], str(tmp_path / "no_existe.pt"), None,
                                             tmp_path / "r", reference_labels="AF, AFF, XYZ")


# --- por HTTP ---------------------------------------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = network.build_network(num_leads=12, num_classes=12)
    ckpt = tmp_path / "a.pt"
    torch.save({"model_state_dict": model.state_dict(), "class_names": load.CLASS_NAMES.copy(),
                "config": {"label_schema": load.LABEL_SCHEMA, "norm_mode": "physical", "bandpass": True},
                "label_schema": load.LABEL_SCHEMA, "epoch": 1, "val_loss": 0.5}, str(ckpt))
    monkeypatch.setenv("ECG_MODEL", str(ckpt))
    monkeypatch.setenv("ECG_SAVED", str(tmp_path))
    monkeypatch.setenv("ECG_UPLOADS", str(tmp_path / "up"))
    monkeypatch.setenv("ECG_RESULTS", str(tmp_path / "res"))
    monkeypatch.setenv("ECG_DEVICE", "cpu")
    monkeypatch.setenv("ECG_RATE_LIMIT_MAX", "0")
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client(), tmp_path


def _csv_bytes():
    t = np.arange(5000) / 500.0
    wave = 0.1 * np.sin(2 * np.pi * 1.2 * t) + 1.0 * (np.mod(t, 0.8) < 0.02)
    signal = wave[:, None] + 0.02 * np.random.default_rng(0).normal(size=(5000, 12))
    buffer = io.BytesIO()
    np.savetxt(buffer, signal, delimiter=",", header=",".join(LEADS), comments="")
    return buffer.getvalue()


def _post(http, labels):
    return http.post("/predict", data={"files": (io.BytesIO(_csv_bytes()), "ecg.csv"), "true_labels": labels},
                     content_type="multipart/form-data")


@pytest.mark.parametrize("labels,shown", [("AFF", "AFF"), ("AF, XYZ", "XYZ"), ("fibrilacion auricular", "fibrilacion auricular")])
def test_predict_answers_400_naming_the_unrecognized_text(client, labels, shown):
    http, tmp = client
    r = _post(http, labels)
    body = r.get_json()
    assert r.status_code == 400 and body["ok"] is False
    assert f"no reconocidas: {shown}" in body["error"] and "AF" in body["error"]            # dice qué falló y qué se admite
    assert not list((tmp / "res").glob("*/*.pdf"))                                           # y no genera ningún informe


def test_predict_compares_against_the_typed_labels_when_they_are_valid(client):
    http, _ = client
    body = _post(http, "AF, 164865005").get_json()
    comparison = body["label_comparison"]
    assert comparison["source"] == "manual" and sorted(comparison["true_classes"]) == ["AF", "MI"]


def test_only_separators_is_not_a_manual_label(client):
    """",," daba una comparación con "ninguna etiqueta real" (todo falso positivo) en vez de no comparar."""
    http, _ = client
    body = _post(http, ", ;").get_json()
    assert body["ok"] is True and body["label_comparison"]["available"] is False
