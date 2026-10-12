"""Tests del monitor de early stopping (val_loss vs val_f1_macro)."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from ecg.train import evaluate_val_f1, is_monitor_improvement


def test_is_monitor_improvement():
    assert is_monitor_improvement(0.6, 0.5, "val_f1_macro") is True
    assert is_monitor_improvement(0.4, 0.5, "val_f1_macro") is False
    assert is_monitor_improvement(0.5, 0.5, "val_f1_macro") is False
    assert is_monitor_improvement(0.4, 0.5, "val_loss") is True
    assert is_monitor_improvement(0.6, 0.5, "val_loss") is False


def test_evaluate_val_f1_known_case():
    xs = torch.randn(4, 3)
    ys = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=torch.float32)
    loader = DataLoader(TensorDataset(xs, ys), batch_size=2)

    class Fixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            if self.calls == 1:  # muestras 0-1: clase 0 perfecta, clase 1 positiva
                return torch.tensor([[10.0, 10.0], [10.0, 10.0]])
            return torch.tensor([[-10.0, 10.0], [-10.0, 10.0]])  # muestras 2-3

    f1 = evaluate_val_f1(Fixed(), loader, torch.device("cpu"), amp=False)
    # Clase 0: F1=1.0. Clase 1: predice todo 1 con y=[1,0,1,0] → P=0.5,R=1,F1=2/3.
    assert abs(f1 - (1.0 + 2.0 / 3.0) / 2.0) < 1e-4
