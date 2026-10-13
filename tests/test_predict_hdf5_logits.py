"""predict_hdf5 devuelve también los logits crudos (la calibración afín se ajusta sobre ellos)."""

import h5py
import numpy as np
import pytest
import torch

from ecg import load, predict


@pytest.fixture()
def tiny_h5(tmp_path):
    path = tmp_path / "tiny.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as h5:
        h5["signals"] = rng.normal(size=(6, load.WINDOW_LENGTH, load.NUM_LEADS)).astype(np.float32)
        h5["labels"] = (rng.random((6, load.NUM_CLASSES)) < 0.3).astype(np.float32)
        h5["record_names"] = np.array([f"R{i}" for i in range(6)], dtype="S")
    return path


def test_predict_hdf5_returns_logits_consistent_with_probabilities(tiny_h5, monkeypatch):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(load.NUM_LEADS * load.WINDOW_LENGTH, load.NUM_CLASSES))
    model.eval()
    monkeypatch.setattr(predict, "load_model", lambda *a, **k: (model, {}, list(load.CLASS_NAMES)))
    out = predict.predict_hdf5("ignored.pt", tiny_h5, batch_size=4, device="cpu", amp=False)
    assert out["logits"].shape == out["probabilities"].shape == (6, load.NUM_CLASSES)
    assert out["logits"].dtype == np.float32
    np.testing.assert_allclose(1 / (1 + np.exp(-out["logits"].astype(np.float64))), out["probabilities"], atol=1e-6)
