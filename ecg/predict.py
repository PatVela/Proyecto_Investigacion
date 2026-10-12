"""Inferencia desde checkpoints CINC2020-12.

Uso CSV::

    python -m ecg.predict saved/cinc2020/cinc2020_resnet/<run>/best.pt ecg.csv

El CSV debe tener 12 columnas de derivaciones. Opcionalmente puede comenzar con
una línea ``# Sampling Rate: 500 Hz``. La salida del modelo es multilabel: se
aplica sigmoid a 12 logits independientes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from . import load, network, util
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from ecg import load, network, util


class HDF5PredictionDataset(Dataset):
    """Dataset HDF5 mínimo para inferencia/evaluación.

    Con ``preload=True`` (defecto) carga signals/labels a RAM una sola vez
    (~1.5 GB para val/test CINC2020). Úselo con ``num_workers=0``: en Windows
    los workers con spawn duplicarían esos GB por worker. Si el archivo
    supera ``max_preload_gb``, cae automáticamente a lectura por disco.
    """

    def __init__(self, h5_path: str | Path, preload: bool = True, max_preload_gb: float = 16):
        import h5py

        self.h5_path = str(util.resolve_path(h5_path))
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"No existe HDF5: {self.h5_path}")
        self._signals = None
        self._labels = None
        self._in_ram = False
        with h5py.File(self.h5_path, "r") as h5:
            self.length = int(h5["signals"].shape[0])
            self.signal_shape = tuple(h5["signals"].shape)
            self.label_shape = tuple(h5["labels"].shape) if "labels" in h5 else None
            classes = h5.attrs.get("classes", None)
            self.classes = [_decode(c) for c in classes] if classes is not None else load.CLASS_NAMES.copy()
            if "record_names" in h5:
                self.record_names = [_decode(x) for x in h5["record_names"][:]]
            else:
                self.record_names = [f"record_{i}" for i in range(self.length)]
            nbytes = int(h5["signals"].size * 4 + (h5["labels"].size * 4 if "labels" in h5 else 0))
            if preload and nbytes <= max_preload_gb * 1024 ** 3:
                print(f"Precargando a RAM {self.h5_path} ({nbytes / 1024 ** 3:.1f} GB)...", flush=True)
                self._signals = np.asarray(h5["signals"][:], dtype=np.float32)
                self._labels = np.asarray(h5["labels"][:], dtype=np.float32) if "labels" in h5 else None
                self._in_ram = True
            elif preload:
                print(f"AVISO: {self.h5_path} pesa {nbytes / 1024 ** 3:.1f} GB; lectura por disco (lento).")
        self.h5 = None

    def _open(self):
        if self.h5 is None:
            import h5py
            self.h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self._in_ram:
            x = self._signals[index].T
            y = self._labels[index] if self._labels is not None else np.zeros(load.NUM_CLASSES, dtype=np.float32)
        else:
            self._open()
            x = np.asarray(self.h5["signals"][index], dtype=np.float32).T
            if "labels" in self.h5:
                y = np.asarray(self.h5["labels"][index], dtype=np.float32)
            else:
                y = np.zeros(load.NUM_CLASSES, dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    def __del__(self):
        try:
            if self.h5 is not None:
                self.h5.close()
        except Exception:
            pass


def _loader_kwargs(batch_size: int, num_workers: int, pin_memory: bool, prefetch_factor: int = 4) -> dict:
    """kwargs de DataLoader seguros con num_workers=0 (evita error de persistent/prefetch)."""
    kwargs: dict = dict(batch_size=int(batch_size), shuffle=False,
                        num_workers=int(num_workers), pin_memory=bool(pin_memory))
    if int(num_workers) > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=int(prefetch_factor))
    return kwargs


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def get_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def model_params_from_config(config: dict) -> dict:
    return {
        "num_leads": int(config.get("num_leads", load.NUM_LEADS)),
        "num_classes": int(config.get("num_classes", load.NUM_CLASSES)),
        "input_length": int(config.get("input_length", load.WINDOW_LENGTH)),
        "conv_filter_length": int(config.get("conv_filter_length", config.get("kernel_size", 7))),
        "conv_num_filters_start": int(config.get("conv_num_filters_start", config.get("filter_length", 32))),
        "conv_subsample_lengths": config.get("conv_subsample_lengths", [1, 2] * 8),
        "conv_num_skip": int(config.get("conv_num_skip", 2)),
        "conv_dropout": float(config.get("conv_dropout", config.get("drop_rate", 0.2))),
        "is_regular_conv": bool(config.get("is_regular_conv", False)),
    }


def load_model(checkpoint_path: str | Path, device: torch.device | None = None, strict_classes: bool = False):
    """Carga checkpoint y devuelve ``(model, checkpoint, class_names)``.

    Además de validar el número de clases, compara los *nombres* contra el
    esquema actual (:data:`ecg.load.CLASS_NAMES`). Si difieren (p.ej.
    checkpoint v1 con ``LAD``/``RBBB`` frente a esquema v2 con
    ``AxisDev``/``BBB``), imprime una advertencia con la correspondencia,
    porque las probabilidades deben interpretarse con los nombres del
    checkpoint, no con los actuales. Con ``strict_classes=True`` la
    diferencia de nombres lanza ``ValueError``.
    """
    device = device or get_device("auto")
    checkpoint_path = util.resolve_path(checkpoint_path)
    checkpoint = util.load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    class_names = list(checkpoint.get("class_names", load.CLASS_NAMES.copy()))
    if len(class_names) != load.NUM_CLASSES:
        raise ValueError(f"El checkpoint tiene {len(class_names)} clases; se esperaban {load.NUM_CLASSES}")
    if class_names != load.CLASS_NAMES:
        lines = [f"  [{i}] checkpoint={c!r} actual={n!r}" for i, (c, n) in enumerate(zip(class_names, load.CLASS_NAMES))]
        message = (
            "AVISO: los nombres de clase del checkpoint NO coinciden con el esquema actual.\n"
            + "\n".join(lines)
            + f"\nCheckpoint label_schema={checkpoint.get('label_schema')!r}; actual={load.LABEL_SCHEMA!r}.\n"
            + "Interprete las probabilidades con los nombres del checkpoint. "
            + "Si entrenó con el esquema v1, regenere HDF5 con el esquema v2 y reentrene."
        )
        if strict_classes:
            raise ValueError(message)
        print(message)

    params = model_params_from_config(config)
    params["num_classes"] = len(class_names)
    model = network.build_network(**params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, class_names


def preprocessing_for_checkpoint(checkpoint: dict) -> dict:
    """Parámetros de preprocesamiento compatibles con un checkpoint cargado."""
    return load.preprocessing_for_schema(checkpoint.get("label_schema"))


@torch.no_grad()
def predict_array(model, signal: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """Predice probabilidades para una señal preprocesada ``(5000,12)`` o ``(12,5000)``."""
    device = device or next(model.parameters()).device
    x = np.asarray(signal, dtype=np.float32)
    if x.shape == (load.WINDOW_LENGTH, load.NUM_LEADS):
        x = x.T
    elif x.shape == (load.NUM_LEADS, load.WINDOW_LENGTH):
        pass
    else:
        raise ValueError(f"Forma no válida para inferencia: {x.shape}")
    tensor = torch.from_numpy(x[None, ...]).to(device)
    logits = model(tensor)
    return torch.sigmoid(logits)[0].detach().cpu().numpy()


@torch.no_grad()
def predict_windows(
    model,
    windows: np.ndarray,
    device: torch.device | None = None,
    aggregate: str = "max",
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Predice K ventanas ``(K, 5000, 12)`` y agrega a un vector de 12 probs.

    ``aggregate``: ``"max"`` (recomendado para no perder eventos focales),
    ``"mean"`` o ``"median"``. Devuelve ``(probs_agregadas, probs_por_ventana)``.
    """
    device = device or next(model.parameters()).device
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (load.WINDOW_LENGTH, load.NUM_LEADS):
        raise ValueError(f"Ventanas inválidas: {x.shape}, se esperaba (K, 5000, 12)")
    model.eval()
    per_window = []
    for start in range(0, x.shape[0], max(1, int(batch_size))):
        batch = torch.from_numpy(x[start:start + batch_size].transpose(0, 2, 1)).to(device)
        per_window.append(torch.sigmoid(model(batch)).float().cpu().numpy())
    per_window = np.concatenate(per_window, axis=0)
    aggregate = str(aggregate).lower()
    if aggregate == "max":
        agg = per_window.max(axis=0)
    elif aggregate == "mean":
        agg = per_window.mean(axis=0)
    elif aggregate == "median":
        agg = np.median(per_window, axis=0)
    else:
        raise ValueError(f"aggregate inválido: {aggregate} (use max/mean/median)")
    return agg.astype(np.float32), per_window.astype(np.float32)


def prepare_csv_windows(
    csv_path: str | Path,
    norm_mode: str,
    bandpass: bool = True,
    windows_max: int = 4,
) -> tuple[np.ndarray, dict]:
    """Lee un CSV ECG y devuelve ``(windows, info)`` preprocesados.

    Si el CSV trae el marcador ``# Preprocessed: <norm_mode>`` y ya mide
    ``(5000, 12)``, se reutiliza tal cual (``info["reused"]=True``); si no,
    se infieren unidades y se preprocesa a ``(K, 5000, 12)``. Fuente única
    para :func:`predict_csv` y ``webapp.prediction.prepare_saved_input``.
    """
    raw, sampling_rate, lead_names, csv_meta = load.read_csv_ecg(csv_path)
    marker = (csv_meta.get("preprocessed_mode") or "").lower()
    if marker and marker == str(norm_mode).lower() and raw.shape == (load.WINDOW_LENGTH, load.NUM_LEADS):
        return raw[None, ...], {
            "sampling_rate": sampling_rate, "lead_names": lead_names,
            "marker": marker, "reused": True, "units": None, "starts": [0],
        }
    units = load.infer_csv_units(raw)
    windows, starts = load.preprocess_to_windows(
        raw, sampling_rate=sampling_rate, lead_names=lead_names,
        windows_max=max(1, int(windows_max)), norm_mode=norm_mode,
        units=units, bandpass=bool(bandpass),
    )
    return windows, {
        "sampling_rate": sampling_rate, "lead_names": lead_names,
        "marker": marker, "reused": False, "units": units, "starts": list(starts),
    }

def predict_csv(
    checkpoint_path: str | Path,
    csv_path: str | Path,
    threshold: float = 0.5,
    device: str = "auto",
    windows_max: int = 1,
    aggregate: str = "max",
) -> dict:
    """Lee un CSV ECG, preprocesa y predice 12 probabilidades.

    El preprocesamiento (modo de normalización/filtrado) se elige según el
    ``label_schema`` del checkpoint para no mezclar v1 con v2. Si el CSV fue
    generado por la webapp (marcador ``# Preprocessed:``) y el modo coincide,
    se reutiliza tal cual sin reprocesar.
    """
    torch_device = get_device(device)
    model, checkpoint, class_names = load_model(checkpoint_path, torch_device)
    pre = preprocessing_for_checkpoint(checkpoint)
    windows, csv_info = prepare_csv_windows(
        csv_path, norm_mode=pre["norm_mode"], bandpass=pre["bandpass"], windows_max=windows_max
    )
    sampling_rate, lead_names = csv_info["sampling_rate"], csv_info["lead_names"]
    marker, reused = csv_info["marker"], csv_info["reused"]
    if windows.shape[0] == 1:
        probabilities = predict_array(model, windows[0], torch_device)
        per_window = probabilities[None, :]
    else:
        probabilities, per_window = predict_windows(model, windows, torch_device, aggregate=aggregate)
    predictions = (probabilities >= threshold).astype(np.uint8)
    rows = [
        {
            "index": i,
            "class": class_names[i],
            "probability": float(probabilities[i]),
            "prediction": int(predictions[i]),
        }
        for i in range(len(class_names))
    ]
    return {
        "checkpoint": str(checkpoint_path),
        "csv_path": str(csv_path),
        "threshold": float(threshold),
        "class_names": class_names,
        "probabilities": probabilities,
        "predictions": predictions,
        "rows": rows,
        "sampling_rate": sampling_rate,
        "lead_names": lead_names,
        "label_schema": checkpoint.get("label_schema"),
        "norm_mode": pre["norm_mode"],
        "bandpass": bool(pre["bandpass"]),
        "csv_preprocessed_marker": marker or None,
        "csv_reused_without_reprocessing": bool(reused),
        "num_windows": int(windows.shape[0]),
        "aggregate": str(aggregate),
        "per_window_probabilities": per_window,
        "processed_shape": list(windows.shape),
    }


@torch.no_grad()
def predict_hdf5(
    checkpoint_path: str | Path,
    h5_path: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    device: str = "auto",
    amp: bool = True,
    preload: bool = True,
    max_preload_gb: float = 16,
    prefetch_factor: int = 4,
):
    """Predice todo un HDF5. Devuelve probabilidades, etiquetas y metadata.

    Con ``preload=True`` el HDF5 se carga a RAM una vez (rápido). Combine con
    ``num_workers=0`` en Windows para no duplicar la RAM por worker.
    """
    torch_device = get_device(device)
    model, checkpoint, class_names = load_model(checkpoint_path, torch_device)
    dataset = HDF5PredictionDataset(h5_path, preload=preload, max_preload_gb=max_preload_gb)
    _check_hdf5_checkpoint_compat(h5_path, checkpoint)
    loader = DataLoader(dataset, **_loader_kwargs(
        batch_size, num_workers, pin_memory=(torch_device.type == "cuda"),
        prefetch_factor=prefetch_factor))

    all_prob, all_true = [], []
    use_amp = amp and torch_device.type == "cuda"
    for x, y in tqdm(loader, desc=f"Prediciendo {Path(h5_path).name}", leave=False):
        x = x.to(torch_device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                probs = torch.sigmoid(model(x))
        else:
            probs = torch.sigmoid(model(x))
        all_prob.append(probs.float().cpu().numpy())
        all_true.append(y.cpu().numpy())

    return {
        "probabilities": np.concatenate(all_prob, axis=0),
        "labels": np.concatenate(all_true, axis=0),
        "record_names": dataset.record_names,
        "class_names": class_names,
        "checkpoint": checkpoint,
    }


def _check_hdf5_checkpoint_compat(h5_path: str | Path, checkpoint: dict) -> None:
    """Advierte si el HDF5 y el checkpoint usan esquemas distintos (v1 vs v2)."""
    try:
        import h5py
        with h5py.File(util.resolve_path(h5_path), "r") as h5:
            h5_schema = h5.attrs.get("label_schema", None)
            h5_norm = h5.attrs.get("norm_mode", h5.attrs.get("normalization", "?"))
            if isinstance(h5_schema, bytes):
                h5_schema = h5_schema.decode("utf-8", errors="replace")
            if isinstance(h5_norm, bytes):
                h5_norm = h5_norm.decode("utf-8", errors="replace")
    except Exception:
        return
    ckpt_schema = checkpoint.get("label_schema")
    if h5_schema is not None and ckpt_schema is not None and str(h5_schema) != str(ckpt_schema):
        print(
            "AVISO DE COMPATIBILIDAD: el HDF5 y el checkpoint usan esquemas distintos:\n"
            f"  HDF5 label_schema={h5_schema!r} norm={h5_norm!r}\n"
            f"  checkpoint label_schema={ckpt_schema!r}\n"
            "Las métricas/predicciones resultantes NO son válidas. Regenere el HDF5 "
            "con el esquema del checkpoint (o reentrene) y vuelva a evaluar."
        )


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inferencia CINC2020-12 desde checkpoint")
    parser.add_argument("checkpoint", help="Checkpoint .pt")
    parser.add_argument("input", help="CSV ECG o HDF5")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Workers del DataLoader (0 = recomendado con preload en RAM)")
    parser.add_argument("--no-preload", action="store_true",
                        help="Desactiva la precarga del HDF5 a RAM (lectura por disco, lento)")
    parser.add_argument("--windows-max", type=int, default=4,
                        help="Ventanas de 10 s para CSV largos (1 = recorte centrado)")
    parser.add_argument("--aggregate", choices=["max", "mean", "median"], default="max",
                        help="Agregación multi-ventana")
    parser.add_argument("--output", default=None, help="CSV/JSON de salida")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if input_path.suffix.lower() == ".h5":
        result = predict_hdf5(args.checkpoint, input_path, batch_size=args.batch_size, num_workers=args.num_workers,
                              device=args.device, preload=not args.no_preload)
        probs = result["probabilities"]
        payload = {"record_name": result["record_names"]}
        for i, name in enumerate(result["class_names"]):
            payload[f"prob_{name}"] = probs[:, i]
            payload[f"pred_{name}"] = (probs[:, i] >= args.threshold).astype(np.uint8)
        df = pd.DataFrame(payload)
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Predicciones guardadas en {args.output}")
        else:
            print(df.head().to_string(index=False))
    else:
        result = predict_csv(args.checkpoint, input_path, threshold=args.threshold, device=args.device,
                             windows_max=args.windows_max, aggregate=args.aggregate)
        print(f"Esquema={result.get('label_schema')} norm={result.get('norm_mode')} "
              f"ventanas={result.get('num_windows')} agreg={result.get('aggregate')}")
        df = rows_to_dataframe(result["rows"])
        if args.output:
            out = Path(args.output)
            if out.suffix.lower() == ".json":
                util.save_json(out, {k: v for k, v in result.items() if k not in {"probabilities", "predictions"}})
            else:
                df.to_csv(out, index=False)
            print(f"Predicción guardada en {out}")
        else:
            print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
