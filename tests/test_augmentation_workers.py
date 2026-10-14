"""La aumentación no debe repetirse igual en todos los workers del DataLoader.

Medido antes: con ``num_workers=2`` y señales idénticas, los ítems 0 y 1 (workers distintos) salían exactamente iguales
porque cada worker heredaba una copia del mismo generador; con ``num_workers=0`` salían distintos. Las configuraciones
actuales usan 0 workers, así que el fallo estaba latente.
"""

import copy
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ecg.hdf5_dataset import HDF5ECGDataset

CFG = {"aug_prob": 1.0, "aug_noise_std": 0.05, "aug_baseline_amp": 0.1, "aug_scale_range": 0.1, "aug_shift_max": 0,
       "aug_lead_drop_prob": 0.0, "aug_seed": 123}


@pytest.fixture()
def h5_path(tmp_path):
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f["signals"] = np.zeros((8, 5000, 12), dtype=np.float32)          # señales idénticas: solo la aumentación las distingue
        f["labels"] = np.zeros((8, 12), dtype=np.float32)
    return path


def _as_worker(monkeypatch, dataset, worker_id, seed=777):
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: SimpleNamespace(id=worker_id, seed=seed + worker_id))
    return dataset


def test_without_workers_nothing_changes(h5_path):
    ds = HDF5ECGDataset(h5_path, preload=True, augment_cfg=CFG)
    first, second = ds[0][0].numpy(), ds[1][0].numpy()
    assert not np.array_equal(first, second)
    again = HDF5ECGDataset(h5_path, preload=True, augment_cfg=CFG)
    assert np.array_equal(first, again[0][0].numpy())                      # con aug_seed la secuencia se repite entre ejecuciones
    assert ds._aug_worker is None


def test_copies_of_the_dataset_in_different_workers_draw_different_augmentations(h5_path, monkeypatch):
    base = HDF5ECGDataset(h5_path, preload=True, augment_cfg=CFG)
    worker0, worker1 = copy.deepcopy(base), copy.deepcopy(base)         # lo que el DataLoader entrega a cada worker
    out0 = _as_worker(monkeypatch, worker0, 0)[0][0].numpy()
    out1 = _as_worker(monkeypatch, worker1, 1)[0][0].numpy()
    assert not np.array_equal(out0, out1)


def test_a_worker_is_reproducible_and_keeps_advancing_its_own_stream(h5_path, monkeypatch):
    base = HDF5ECGDataset(h5_path, preload=True, augment_cfg=CFG)
    first, second = copy.deepcopy(base), copy.deepcopy(base)
    a = [_as_worker(monkeypatch, first, 0)[i][0].numpy() for i in range(2)]
    b = [_as_worker(monkeypatch, second, 0)[i][0].numpy() for i in range(2)]
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])    # misma semilla y mismo worker: misma secuencia
    assert not np.array_equal(a[0], a[1])                               # no se reinicia el generador en cada ítem


def test_without_aug_seed_workers_still_differ(h5_path, monkeypatch):
    cfg = {**CFG, "aug_seed": None}
    base = HDF5ECGDataset(h5_path, preload=True, augment_cfg=cfg)
    worker0, worker1 = copy.deepcopy(base), copy.deepcopy(base)
    assert not np.array_equal(_as_worker(monkeypatch, worker0, 0)[0][0].numpy(), _as_worker(monkeypatch, worker1, 1)[0][0].numpy())


def test_real_dataloader_with_two_workers(h5_path):
    ds = HDF5ECGDataset(h5_path, preload=True, augment_cfg=CFG)
    outs = [x[0].numpy() for x, _ in DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)]
    assert not np.array_equal(outs[0], outs[1])                         # ítems 0 y 1 los procesan workers distintos
