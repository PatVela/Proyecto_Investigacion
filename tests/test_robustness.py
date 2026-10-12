"""Tests de perturbaciones controladas (examples/cinc2020/robustness.py)."""

import numpy as np
import pytest
import torch

from robustness import apply_perturbation


def _batch():
    return torch.randn(2, 12, 500)


def test_clean_devuelve_identico():
    x = _batch()
    y = apply_perturbation(x, "clean", 0.0, np.random.default_rng(0))
    assert torch.equal(x, y)


def test_kinds_conocidos_cambian_senal():
    x = _batch()
    kinds = [("gaussian_noise", 0.05), ("baseline_wander", 0.1),
             ("amplitude_scale", 0.5), ("lead_dropout", 1)]
    for kind, level in kinds:
        y = apply_perturbation(x, kind, level, np.random.default_rng(42))
        assert y.shape == x.shape
        assert not torch.equal(x, y)


def test_kind_desconocido_lanza():
    with pytest.raises(ValueError):
        apply_perturbation(_batch(), "no_existe", 1.0, np.random.default_rng(0))


def test_evaluate_under_perturbation_con_temperaturas(tmp_path):
    from torch.utils.data import TensorDataset
    from robustness import evaluate_under_perturbation

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.LazyLinear(12))
    xs = torch.randn(8, 12, 500)
    _ = model(xs)
    ds = TensorDataset(xs, torch.randint(0, 2, (8, 12)).float())
    thr = __import__("numpy").full(12, 0.5, dtype="float32")
    base = evaluate_under_perturbation(model, ds, thr, torch.device("cpu"), "clean", 0.0, 4, 0, None, 42)
    assert base["samples"] == 8 and "f1_macro" in base
    temps = __import__("numpy").full(12, 2.0)
    cal = evaluate_under_perturbation(model, ds, thr, torch.device("cpu"), "clean", 0.0, 4, 0, None, 42, temperatures=temps)
    assert cal["samples"] == 8 and "f1_macro" in cal
