"""Tests de arquitecturas: ResNet-34 vs CNN convencional (ecg/network.py)."""

import torch

from ecg import load
from ecg.network import ECGRegularCNN, ECGResNet34, build_network
from ecg.predict import load_model


def _count(model):
    return sum(p.numel() for p in model.parameters())


def test_resnet_param_count_pinned():
    assert _count(ECGResNet34()) == 4806668


def test_plain_param_count_pinned_and_smaller():
    plain = ECGRegularCNN()
    assert _count(plain) == 4586764
    assert _count(plain) < _count(ECGResNet34())


def test_model_types_and_routing():
    assert ECGResNet34().model_type == "ResNet-34 1D tipo Hannun"
    assert ECGRegularCNN().model_type == "CNN convencional equivalente"
    assert build_network().model_type == "ResNet-34 1D tipo Hannun"
    assert build_network(is_regular_conv=True).model_type == "CNN convencional equivalente"


def test_forward_shapes_short_signal():
    x = torch.randn(2, 12, 512)
    assert ECGResNet34()(x).shape == (2, 12)
    assert ECGRegularCNN()(x).shape == (2, 12)


def test_load_model_roundtrip_plain(tmp_path):
    model = ECGRegularCNN()
    ckpt = tmp_path / "plain.pt"
    torch.save({
        "config": {"is_regular_conv": True, "label_schema": load.LABEL_SCHEMA},
        "model_state_dict": model.state_dict(),
        "class_names": load.CLASS_NAMES.copy(),
    }, ckpt)
    loaded, _, names = load_model(ckpt, device=torch.device("cpu"))
    assert loaded.model_type == "CNN convencional equivalente"
    assert names == load.CLASS_NAMES
    model.eval()
    x = torch.randn(1, 12, 512)
    with torch.no_grad():
        assert torch.equal(loaded(x), model(x))
