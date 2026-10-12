"""Tests del modo ensemble (dos checkpoints) en la webapp."""

import numpy as np
import pytest
import torch

from ecg import load, network
from webapp import prediction as pred
from webapp.app import create_app


@pytest.fixture()
def ensemble_client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _tiny_checkpoint(path, seed: int):
    torch.manual_seed(seed)
    model = network.build_network(num_leads=12, num_classes=12)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"label_schema": load.LABEL_SCHEMA, "norm_mode": "physical", "bandpass": True},
        "class_names": load.CLASS_NAMES.copy(),
        "epoch": 1,
        "val_loss": 0.5,
    }, str(path))
    return str(path)


def _synthetic_csv(path):
    signal = np.random.default_rng(4).normal(0, 0.3, (load.WINDOW_LENGTH, load.NUM_LEADS))
    header = ",".join(load.STANDARD_LEAD_ORDER)
    np.savetxt(str(path), signal, delimiter=",", header=header, comments="")
    return path


def test_ensemble_averages_both_models(tmp_path):
    ckpt_a = _tiny_checkpoint(tmp_path / "a.pt", 1)
    ckpt_b = _tiny_checkpoint(tmp_path / "b.pt", 2)
    csv_path = _synthetic_csv(tmp_path / "rec.csv")

    run_a = pred.run_prediction_from_saved_paths([csv_path], ckpt_a, None, tmp_path / "ra")
    run_b = pred.run_prediction_from_saved_paths([csv_path], ckpt_b, None, tmp_path / "rb")
    run_e = pred.run_prediction_from_saved_paths(
        [csv_path], ckpt_a, None, tmp_path / "re", model_path_b=ckpt_b, alpha=0.6)

    assert run_e["ensemble"] is True
    assert "Ensemble" in run_e["model_type"]
    assert run_e["model_path_b"] == ckpt_b
    assert run_e["ensemble_alpha"] == 0.6
    assert run_a["ensemble"] is False
    expected = (0.6 * np.asarray(run_a["per_window_probabilities"][0])
                + 0.4 * np.asarray(run_b["per_window_probabilities"][0]))
    assert np.allclose(run_e["per_window_probabilities"][0], expected, atol=1e-6)


def test_ensemble_applies_per_model_temperatures(tmp_path):
    ckpt_a = _tiny_checkpoint(tmp_path / "a.pt", 1)
    ckpt_b = _tiny_checkpoint(tmp_path / "b.pt", 2)
    csv_path = _synthetic_csv(tmp_path / "rec.csv")
    temps_b = {name: 0.05 for name in load.CLASS_NAMES}

    plain = pred.run_prediction_from_saved_paths(
        [csv_path], ckpt_a, None, tmp_path / "r0", model_path_b=ckpt_b, alpha=0.5)
    cal = pred.run_prediction_from_saved_paths(
        [csv_path], ckpt_a, None, tmp_path / "r1", model_path_b=ckpt_b, alpha=0.5,
        temperatures_b=temps_b, temperature_source_b="test-b")
    assert cal["calibrated"] is True
    assert cal["temperature_source_b"] == "test-b"
    assert not np.allclose(plain["per_window_probabilities"][0],
                           cal["per_window_probabilities"][0])


def test_health_reports_ensemble(ensemble_client):
    ensemble_client.application.config["ECG_MODEL_B"] = "b.pt"
    ensemble_client.application.config["ECG_ALPHA"] = 0.6
    data = ensemble_client.get("/health").get_json()
    assert data["status"] == "ok"
    assert data["ensemble"] is True
    assert data["ECG_MODEL_B"] == "b.pt"
    assert data["ensemble_alpha"] == 0.6
