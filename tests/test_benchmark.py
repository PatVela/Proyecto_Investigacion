"""Tests del benchmark de latencia (examples/cinc2020/benchmark_latency.py)."""

import torch

from benchmark_latency import benchmark_forward
from ecg.network import ECGResNet34


def test_benchmark_forward_keys():
    model = ECGResNet34()
    stats = benchmark_forward(model, torch.device("cpu"), batch_size=1,
                              input_length=256, iters=2, warmup=1)
    assert stats["mean_ms"] > 0 and stats["windows_per_s"] > 0
    assert stats["min_ms"] <= stats["mean_ms"]
