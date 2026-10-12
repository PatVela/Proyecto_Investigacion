"""ECG_DEVICE elige el dispositivo de inferencia de la webapp (en una PC con GPU de entrenamiento se sirve en CPU)."""

import os

import pytest

pytest.importorskip("flask")
torch = pytest.importorskip("torch")


@pytest.fixture()
def prediction(monkeypatch):
    monkeypatch.delenv("ECG_DEVICE", raising=False)
    from webapp import prediction as module

    return module


def test_default_is_auto_and_follows_cuda_availability(prediction):
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert prediction.serving_device().type == expected


@pytest.mark.parametrize("value", ["cpu", "CPU", " cpu "])
def test_cpu_is_honoured_even_when_a_gpu_exists(prediction, monkeypatch, value):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("ECG_DEVICE", value)
    assert prediction.serving_device().type == "cpu"


def test_explicit_cuda_without_a_gpu_is_an_error_not_a_silent_cpu(prediction, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("ECG_DEVICE", "cuda")
    with pytest.raises(RuntimeError, match="CUDA"):
        prediction.serving_device()


def test_unknown_value_is_rejected(prediction, monkeypatch):
    monkeypatch.setenv("ECG_DEVICE", "tpu")
    with pytest.raises(ValueError, match="ECG_DEVICE"):
        prediction.serving_device()


def test_app_refuses_to_start_with_a_bad_device(monkeypatch, tmp_path):
    from webapp.app import create_app

    for key in [k for k in os.environ if k.startswith("ECG_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ECG_UPLOADS", str(tmp_path / "up"))
    monkeypatch.setenv("ECG_RESULTS", str(tmp_path / "res"))
    monkeypatch.setenv("ECG_DEVICE", "tpu")
    with pytest.raises(ValueError, match="ECG_DEVICE"):
        create_app()


def test_prediction_uses_the_configured_device():
    source = open(os.path.join(os.path.dirname(__file__), "..", "webapp", "prediction.py"), encoding="utf-8").read()
    assert "serving_device()" in source
    assert 'torch.device("cuda" if torch.cuda.is_available() else "cpu")' not in source
