"""Pérdida enmascarada de celdas (origen, clase) sin anotar: ``compute_cell_mask``, ``compute_pos_weight``, ``MaskedBCEWithLogits`` y el dataset."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from ecg import train
from ecg.hdf5_dataset import HDF5ECGDataset
from ecg.schema import NUM_CLASSES
from ecg.signal import NUM_LEADS, WINDOW_LENGTH

# origen alfa: clases 0 y 1; origen beta: clases 1 y 2; el resto de las celdas no tiene positivos
FUENTES = ["alfa"] * 4 + ["beta"] * 4
ETIQUETAS = np.zeros((8, NUM_CLASSES), dtype=np.float32)
ETIQUETAS[0, 0] = ETIQUETAS[1, 0] = ETIQUETAS[0, 1] = 1
ETIQUETAS[4, 1] = ETIQUETAS[5, 2] = ETIQUETAS[6, 2] = 1


@pytest.fixture()
def h5_path(tmp_path):
    ruta = tmp_path / "train.h5"
    with h5py.File(ruta, "w") as h5:
        h5.create_dataset("signals", data=np.zeros((8, WINDOW_LENGTH, NUM_LEADS), dtype=np.float32))
        h5.create_dataset("labels", data=ETIQUETAS)
        h5.create_dataset("source_dbs", data=FUENTES, dtype=h5py.string_dtype(encoding="utf-8"))
        h5.create_dataset("record_names", data=[f"r{i}" for i in range(8)], dtype=h5py.string_dtype(encoding="utf-8"))
    return ruta


def test_compute_cell_mask_marca_las_celdas_con_positivos(h5_path):
    nombres, matriz = train.compute_cell_mask(h5_path)
    assert nombres == ["alfa", "beta"]
    esperada = np.zeros((2, NUM_CLASSES), dtype=np.float32)
    esperada[0, [0, 1]] = 1
    esperada[1, [1, 2]] = 1
    assert np.array_equal(matriz, esperada)


def test_min_positives_exige_mas_de_un_positivo(h5_path):
    _, matriz = train.compute_cell_mask(h5_path, min_positives=2)
    assert matriz[0, 0] == 1 and matriz[0, 1] == 0           # alfa: clase 0 con 2 positivos, clase 1 con 1
    assert matriz[1, 2] == 1 and matriz[1, 1] == 0


def test_pos_weight_rechaza_clases_sin_positivos_aunque_haya_mascara(h5_path):
    with pytest.raises(ValueError, match="sin positivos"):
        train.compute_pos_weight(h5_path, train.compute_cell_mask(h5_path))


def test_pos_weight_con_y_sin_mascara(h5_path, tmp_path):
    completa = tmp_path / "completa.h5"
    etiquetas = np.ones((8, NUM_CLASSES), dtype=np.float32)
    etiquetas[:4, :6] = 0
    with h5py.File(completa, "w") as h5:
        h5.create_dataset("signals", data=np.zeros((8, WINDOW_LENGTH, NUM_LEADS), dtype=np.float32))
        h5.create_dataset("labels", data=etiquetas)
        h5.create_dataset("source_dbs", data=FUENTES, dtype=h5py.string_dtype(encoding="utf-8"))
    peso, positivos, negativos = train.compute_pos_weight(completa)
    assert np.allclose(positivos, etiquetas.sum(0)) and np.allclose(negativos, 8 - etiquetas.sum(0))
    mascara = (["alfa", "beta"], np.stack([np.zeros(NUM_CLASSES), np.ones(NUM_CLASSES)]).astype(np.float32))
    peso_m, pos_m, neg_m = train.compute_pos_weight(completa, mascara)   # alfa queda sin anotar: solo cuentan los 4 registros de beta
    assert np.allclose(pos_m, etiquetas[4:].sum(0)) and np.allclose(neg_m, 4 - etiquetas[4:].sum(0))


def test_mascara_de_unos_equivale_a_bce_normal():
    torch.manual_seed(0)
    logits, y = torch.randn(16, NUM_CLASSES), (torch.rand(16, NUM_CLASSES) < 0.3).float()
    peso = torch.rand(NUM_CLASSES) * 5 + 1
    normal = nn.BCEWithLogitsLoss(pos_weight=peso)(logits, y)
    enmascarada = train.MaskedBCEWithLogits(peso)(logits, y, torch.ones_like(y))
    sin_mascara = train.MaskedBCEWithLogits(peso)(logits, y)
    assert torch.allclose(normal, enmascarada) and torch.allclose(normal, sin_mascara)


def test_celda_enmascarada_no_aporta_gradiente():
    logits = torch.zeros(4, NUM_CLASSES, requires_grad=True)
    y = torch.ones(4, NUM_CLASSES)
    mascara = torch.ones(4, NUM_CLASSES)
    mascara[:, 3] = 0
    mascara[2, 5] = 0
    train.MaskedBCEWithLogits(torch.ones(NUM_CLASSES))(logits, y, mascara).backward()
    assert torch.all(logits.grad[:, 3] == 0) and logits.grad[2, 5] == 0
    assert torch.all(logits.grad[:, 0] != 0) and logits.grad[0, 5] != 0


def test_mascara_toda_en_cero_da_perdida_cero_sin_dividir_por_cero():
    valor = train.MaskedBCEWithLogits(torch.ones(NUM_CLASSES))(torch.randn(3, NUM_CLASSES), torch.ones(3, NUM_CLASSES), torch.zeros(3, NUM_CLASSES))
    assert float(valor) == 0.0


def test_dataset_devuelve_la_mascara_segun_el_origen(h5_path):
    mascara = train.compute_cell_mask(h5_path)
    ds = HDF5ECGDataset(h5_path, require_labels=True, cell_mask=mascara)
    assert ds.has_cell_mask
    x, y, m = ds[0]
    assert m.shape == (NUM_CLASSES,) and m[0] == 1 and m[5] == 0
    assert ds[5][2][2] == 1 and ds[5][2][0] == 0
    sin = HDF5ECGDataset(h5_path, require_labels=True)
    assert not sin.has_cell_mask and len(sin[0]) == 2


def test_origen_desconocido_se_trata_como_anotado(h5_path):
    mascara = (["otro"], np.zeros((1, NUM_CLASSES), dtype=np.float32))
    ds = HDF5ECGDataset(h5_path, require_labels=True, cell_mask=mascara)
    assert float(ds[0][2].sum()) == NUM_CLASSES


def test_split_batch_acepta_dos_o_tres_elementos():
    a, b, c = torch.zeros(1), torch.ones(1), torch.full((1,), 2.0)
    assert train.split_batch((a, b)) == (a, b, None)
    assert train.split_batch((a, b, c))[2] is c


def test_las_claves_nuevas_son_conocidas():
    assert train.unknown_config_keys({"mask_unannotated": True, "mask_min_positives": 3}) == []
