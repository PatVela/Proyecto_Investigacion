"""Regresiones de seguridad de la webapp.

Cubren dos fallos críticos de la revisión de código:
  1. Ejecución de código vía pickle: subir un ``.pt`` y listarlo con
     ``/models?saved=...`` ejecutaba ``torch.load(weights_only=False)``.
  2. Escritura fuera de ``ECG_RESULTS`` por un ``record_name`` tipo ``../..``
     tomado del header ``.hea`` subido.

El payload de pickle solo crea un archivo marcador dentro de ``tmp_path``.
"""

import io
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("flask")

from ecg import load, network, util
from webapp.app import create_app


def _evil_pickle_bytes(marker: Path) -> bytes:
    class Evil:
        def __reduce__(self):
            return (Path(str(marker)).write_text, ("ejecutado",))

    return pickle.dumps(Evil())


def _tiny_checkpoint(path: Path) -> Path:
    model = network.build_network(num_leads=12, num_classes=12)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"label_schema": load.LABEL_SCHEMA, "norm_mode": "physical", "bandpass": True},
        "class_names": load.CLASS_NAMES.copy(),
        "label_schema": load.LABEL_SCHEMA,
        "epoch": 1,
        "val_loss": 0.5,
    }, str(path))
    return path


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    saved = tmp_path / "saved"
    saved.mkdir()
    dirs = {"saved": saved, "uploads": tmp_path / "uploads", "results": tmp_path / "results"}
    monkeypatch.setenv("ECG_SAVED", str(dirs["saved"]))
    monkeypatch.setenv("ECG_UPLOADS", str(dirs["uploads"]))
    monkeypatch.setenv("ECG_RESULTS", str(dirs["results"]))
    monkeypatch.setenv("ECG_MODEL", "")
    monkeypatch.setenv("ECG_ENABLE_USE_MODEL", "1")        # /use_model está apagado por defecto (ver tests/test_hardening.py)
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client(), dirs, tmp_path


# --------------------------------------------------------------------------- #
# 1. pickle / carga de checkpoints
# --------------------------------------------------------------------------- #
def test_load_checkpoint_does_not_execute_pickle(tmp_path):
    marker = tmp_path / "PWNED.txt"
    evil = tmp_path / "evil.pt"
    evil.write_bytes(_evil_pickle_bytes(marker))
    with pytest.raises(Exception):
        util.load_checkpoint(evil)
    assert not marker.exists()


def test_load_checkpoint_reads_normal_checkpoints(tmp_path):
    ckpt = util.load_checkpoint(_tiny_checkpoint(tmp_path / "ok.pt"))
    assert ckpt["class_names"] == load.CLASS_NAMES
    assert "model_state_dict" in ckpt


def test_models_route_ignores_client_supplied_directory(sandbox):
    client, dirs, tmp_path = sandbox
    marker = tmp_path / "PWNED.txt"
    attacker_dir = tmp_path / "attacker"
    attacker_dir.mkdir()
    (attacker_dir / "evil.pt").write_bytes(_evil_pickle_bytes(marker))

    resp = client.get("/models", query_string={"saved": str(attacker_dir)})
    assert resp.status_code == 200
    data = resp.get_json()
    assert Path(data["saved_dir"]) == dirs["saved"].resolve()
    assert data["models"] == []
    assert not marker.exists()


def test_predict_rejects_uploads_with_unexpected_extension(sandbox):
    client, dirs, tmp_path = sandbox
    marker = tmp_path / "PWNED.txt"
    resp = client.post(
        "/predict",
        data={"files": (io.BytesIO(_evil_pickle_bytes(marker)), "evil.pt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code >= 400
    assert resp.get_json()["ok"] is False
    assert not list(dirs["uploads"].rglob("*.pt"))
    assert not marker.exists()


def test_use_model_rejects_checkpoints_outside_models_dir(sandbox):
    client, dirs, tmp_path = sandbox
    outside = _tiny_checkpoint(tmp_path / "outside.pt")

    resp = client.post("/use_model", json={"model": str(outside)})
    assert resp.status_code == 400
    assert client.application.config["ECG_MODEL"] == ""


def test_use_model_accepts_checkpoints_inside_models_dir(sandbox):
    client, dirs, _ = sandbox
    inside = _tiny_checkpoint(dirs["saved"] / "best.pt")

    resp = client.post("/use_model", json={"model": str(inside)})
    assert resp.status_code == 200
    assert Path(client.application.config["ECG_MODEL"]) == inside.resolve()


# --------------------------------------------------------------------------- #
# 2. record_name del header .hea -> ruta del PDF
# --------------------------------------------------------------------------- #
def _wfdb_pair(record_name: str):
    from scipy.io import savemat

    header = f"{record_name} 12 500 5000\n" + "".join(
        f"x.mat 16+24 1000/mV 16 0 0 0 0 {lead}\n" for lead in load.STANDARD_LEAD_ORDER
    ) + "#Dx: 426783006\n"
    mat = io.BytesIO()
    savemat(mat, {"val": (np.random.default_rng(0).normal(0, 100, (12, 5000))).astype(np.int16)})
    mat.seek(0)
    return [(io.BytesIO(header.encode()), "rec.hea"), (mat, "rec.mat")]


def test_hea_record_name_cannot_escape_results_dir(sandbox, monkeypatch):
    client, dirs, tmp_path = sandbox
    checkpoint = _tiny_checkpoint(dirs["saved"] / "best.pt")
    client.application.config["ECG_MODEL"] = str(checkpoint)

    # results/<job>/../../ESCAPED_reporte.pdf cae justo en tmp_path (fuera de results).
    resp = client.post("/predict", data={"files": _wfdb_pair("../../ESCAPED")},
                       content_type="multipart/form-data")
    assert resp.status_code == 200, resp.get_json()

    results_root = dirs["results"].resolve()
    created = [p for p in tmp_path.rglob("*ESCAPED*")]
    assert created, "el informe debería existir con nombre saneado"
    assert all(results_root in p.resolve().parents for p in created), created
    assert resp.get_json()["pdf_url"].startswith("/results/")
