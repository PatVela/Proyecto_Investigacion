"""La pérdida de entrenamiento: dirección de ``pos_weight`` y fórmula del suavizado de etiquetas.

Medido por mutación: invertir ``negatives / positives`` en ``compute_pos_weight`` dejaba las 535 pruebas en verde. Ese
cociente fija el desplazamiento de las probabilidades que la calibración afín corrige después, así que una inversión
silenciosa entrenaría otro modelo sin que nada avisara.
"""

import h5py
import numpy as np
import pytest
import torch

from ecg import load
from ecg.train import compute_pos_weight, smooth_targets


def _h5_with_counts(path, positives):
    """HDF5 mínimo con 10 registros y ``positives[j]`` positivos en la clase j."""
    labels = np.zeros((10, load.NUM_CLASSES), dtype=np.float32)
    for j, count in enumerate(positives):
        labels[:count, j] = 1.0
    with h5py.File(path, "w") as f:
        f["signals"] = np.zeros((10, load.WINDOW_LENGTH, load.NUM_LEADS), dtype=np.float32)
        f["labels"] = labels
    return path


def test_pos_weight_is_negatives_over_positives(tmp_path):
    counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 5, 7]                       # positivos por clase, de 10 registros (ninguna con 0 ni con 10)
    weight, positives, negatives = compute_pos_weight(_h5_with_counts(tmp_path / "t.h5", counts))
    assert positives.tolist() == [float(c) for c in counts] and negatives.tolist() == [float(10 - c) for c in counts]
    np.testing.assert_allclose(weight.numpy(), [(10 - c) / c for c in counts], rtol=1e-6)
    assert weight[0] > weight[8] > 0                                   # la clase más rara (1 positivo) pesa más que la más común (9)
    assert weight.dtype == torch.float32


def test_pos_weight_makes_a_missed_positive_cost_more_than_a_false_alarm(tmp_path):
    """Con pos_weight = neg/pos > 1 el positivo no detectado pierde más que una falsa alarma del mismo logit."""
    weight, _, _ = compute_pos_weight(_h5_with_counts(tmp_path / "t.h5", [1] * 12))     # 1 positivo de 10 -> peso 9
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight[:1], reduction="none")
    missed_positive = criterion(torch.tensor([[-2.0]]), torch.tensor([[1.0]]))
    false_alarm = criterion(torch.tensor([[2.0]]), torch.tensor([[0.0]]))
    assert float(missed_positive / false_alarm) == pytest.approx(9.0, rel=1e-5)                 # mismo softplus(2), multiplicado por el peso


def test_a_class_without_positives_is_rejected(tmp_path):
    counts = [1] * 12
    counts[4] = 0
    with pytest.raises(ValueError, match="AF"):
        compute_pos_weight(_h5_with_counts(tmp_path / "t.h5", counts))


def test_label_smoothing_moves_targets_to_s_over_two_and_one_minus_s_over_two():
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])
    smoothed = smooth_targets(y, 0.1)
    np.testing.assert_allclose(smoothed.numpy(), [0.05, 0.95, 0.95, 0.05], atol=1e-7)     # no 0.1 / 0.9
    assert smooth_targets(y, 0.0) is y                                                    # sin suavizado, tal cual
    assert smooth_targets(y, -1.0) is y
