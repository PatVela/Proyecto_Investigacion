"""Exporta un checkpoint CINC2020-12 a ONNX (opset 17, batch dinámico).

Verifica el grafo con ``onnx.checker`` y, si ``onnxruntime`` está instalado,
compara numéricamente la salida ONNX vs PyTorch sobre una entrada aleatoria.

Ejemplo:
    python examples/cinc2020/export_onnx.py \\
      --checkpoint <run>/best.pt --output <modelo>.onnx

Requiere: ``pip install onnx onnxruntime``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta checkpoint CINC2020-12 a ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt")
    parser.add_argument("--output", required=True, help="Ruta del .onnx de salida")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-verify", action="store_true", help="Omite verificación con onnxruntime")
    args = parser.parse_args()

    try:
        import onnx
    except ImportError:
        raise SystemExit("Falta el paquete 'onnx'. Instálelo con: pip install onnx onnxruntime")

    import numpy as np
    import torch

    from ecg import load, predict

    device = torch.device("cpu")
    model, checkpoint, class_names = predict.load_model(args.checkpoint, device)
    model.eval()
    print(f"Clases: {len(class_names)} | esquema: {checkpoint.get('label_schema')}")

    dummy = torch.randn(1, load.NUM_LEADS, load.WINDOW_LENGTH, dtype=torch.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["ecg"],
        output_names=["logits"],
        dynamic_axes={"ecg": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
        opset_version=args.opset,
    )
    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX válido: {output} ({output.stat().st_size / 1024:.1f} KB, opset {args.opset})")

    if args.no_verify:
        return
    try:
        import onnxruntime as ort
    except ImportError:
        print("AVISO: onnxruntime no instalado; se omite la verificación numérica.")
        return
    with torch.no_grad():
        expected = model(dummy).cpu().numpy()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    got = session.run(["logits"], {"ecg": dummy.cpu().numpy()})[0]
    diff = float(np.max(np.abs(got - expected)))
    print(f"Diferencia máxima vs PyTorch: {diff:.3e} "
          f"({'OK' if diff < 1e-4 else 'REVISAR: supera 1e-4'})")


if __name__ == "__main__":
    main()
