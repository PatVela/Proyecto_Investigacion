"""Tests de aumentación de señal y label smoothing."""

import numpy as np

from ecg import augment
from ecg.train import smooth_targets


def _signal(seed=0):
    return np.random.default_rng(seed).normal(0, 0.5, (12, 5000)).astype(np.float32)


def test_noise_changes_signal_with_expected_std():
    x = _signal()
    before = x.copy()
    out = augment.add_gaussian_noise(x, np.random.default_rng(1), 0.01)
    assert out.shape == (12, 5000) and out.dtype == np.float32
    assert np.allclose((out - x).std(), 0.01, atol=0.002)
    assert (x == before).all(), "la entrada no debe mutarse"


def test_baseline_wander_is_bounded():
    x = _signal()
    out = augment.add_baseline_wander(x, np.random.default_rng(2), 0.1)
    assert out.shape == x.shape and out.dtype == np.float32
    assert np.abs(out - x).max() <= 0.1 + 1e-6
    assert not np.allclose(out, x)


def test_scale_stays_in_range():
    x = np.full((12, 5000), 2.0, dtype=np.float32)
    out = augment.random_scale(x, np.random.default_rng(3), 0.1)
    assert 0.9 <= float(out[0, 0] / 2.0) <= 1.1


def test_shift_zero_copy_and_shape():
    x = _signal()
    out = augment.time_shift(x, np.random.default_rng(4), 0)
    assert (out == x).all() and out is not x
    out2 = augment.time_shift(x, np.random.default_rng(5), 250)
    assert out2.shape == x.shape and out2.dtype == np.float32


def test_lead_dropout_exactly_one_lead():
    x = _signal()
    out = augment.lead_dropout(x, np.random.default_rng(6), 1.0)
    zero_leads = [i for i in range(12) if not out[i].any()]
    assert len(zero_leads) == 1
    kept = [i for i in range(12) if i not in zero_leads]
    assert (out[kept] == x[kept]).all()
    out_none = augment.lead_dropout(x, np.random.default_rng(7), 0.0)
    assert (out_none == x).all()


def test_composer_identity_and_determinism():
    x = _signal()
    cfg_off = dict(augment.DEFAULTS, aug_prob=0.0)
    out = augment.augment_signal(x, np.random.default_rng(8), cfg_off)
    assert (out == x).all() and out is not x
    cfg_on = dict(augment.DEFAULTS, aug_prob=1.0)
    first = augment.augment_signal(x, np.random.default_rng(9), cfg_on)
    second = augment.augment_signal(x, np.random.default_rng(9), cfg_on)
    assert (first == second).all(), "misma semilla debe dar idéntico resultado"
    assert any(
        not np.allclose(augment.augment_signal(x, np.random.default_rng(s), cfg_on), x)
        for s in range(5)
    ), "con prob 1.0 alguna semilla debe transformar"
    assert (x == _signal()).all(), "el pipeline no debe mutar la entrada"


def test_smooth_targets():
    y = np.array([[0.0, 1.0]], dtype=np.float32)
    assert np.allclose(smooth_targets(y, 0.1), [[0.05, 0.95]])
    assert np.allclose(smooth_targets(y, 0.0), y)
