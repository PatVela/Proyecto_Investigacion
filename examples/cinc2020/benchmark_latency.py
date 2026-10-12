"""Mide latencia de inferencia (ms/ventana) y parámetros (C11).

Carga un checkpoint con ``predict.load_model`` y cronometra pasadas forward
sobre ventanas aleatorias de 12×5000 (una ventana = un registro de 10 s).

Uso:
    python examples/cinc2020/benchmark_latency.py --checkpoint <best.pt> --device auto --iters 200
"""

from __future__ import annotations

import argparse
import sys
import time

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from ecg import load, predict, util


def benchmark_forward(model, device, batch_size=1, input_length=5000, iters=200, warmup=20) -> dict:
    model.eval()
    x = torch.randn(int(batch_size), load.NUM_LEADS, int(input_length), device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - t0) * 1000.0 / batch_size)
    arr = torch.tensor(times)
    return {"mean_ms": float(arr.mean()), "std_ms": float(arr.std()),
            "min_ms": float(arr.min()), "windows_per_s": float(1000.0 / arr.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Latencia de inferencia + parámetros.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    device = predict.get_device(args.device)
    model, checkpoint, _ = predict.load_model(args.checkpoint, device=device)
    params = util.count_parameters(model)
    print(f"Modelo: {model.model_type} | parámetros: {params['total']:,} "
          f"(entrenables: {params['trainable']:,}) | device: {device}")
    stats = benchmark_forward(model, device, args.batch_size, iters=args.iters, warmup=args.warmup)
    print(f"Latencia/ventana 10 s (batch={args.batch_size}): "
          f"{stats['mean_ms']:.2f}±{stats['std_ms']:.2f} ms "
          f"(min {stats['min_ms']:.2f} ms, {stats['windows_per_s']:.1f} ventanas/s)")


if __name__ == "__main__":
    main()
