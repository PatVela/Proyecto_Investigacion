"""Utilidades compartidas: seeds, paths, checkpoints, conteo de parámetros."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def decode_h5_string(value) -> str:
    """Convierte un valor de HDF5 (bytes o str) a ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def decode_h5_strings(values) -> list[str]:
    return [decode_h5_string(v) for v in values]


def get_device(requested: str | None = "auto"):
    """``torch.device``: ``cpu`` si se pide o no hay CUDA; ``cuda`` con ``auto``/``cuda``/``None``."""
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested in {None, "auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dataloader_kwargs(batch_size: int, num_workers: int = 0, pin_memory: bool = False,
                      prefetch_factor: int = 4, shuffle: bool = False) -> dict:
    """kwargs de DataLoader seguros con ``num_workers=0`` (evita error de persistent/prefetch)."""
    kwargs: dict = dict(batch_size=int(batch_size), shuffle=bool(shuffle),
                        num_workers=int(num_workers), pin_memory=bool(pin_memory))
    if int(num_workers) > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=int(prefetch_factor))
    return kwargs


def set_seed(seed: int | None) -> None:
    """Fija semillas de Python, NumPy y PyTorch si está disponible."""
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
    except Exception:
        pass


def resolve_path(path: str | os.PathLike, base: str | os.PathLike | None = None) -> Path:
    """Resuelve rutas relativas respecto a la raíz del repositorio por defecto."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(base) if base is not None else REPO_ROOT) / path


def timestamped_dir(root: str | os.PathLike, experiment: str) -> Path:
    """Crea directorio reproducible estilo ``save_dir/experiment/timestamp-rand``."""
    root = resolve_path(root)
    dirname = f"{int(time.time())}-{random.randrange(1000):03d}"
    save_dir = root / experiment / dirname
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def count_parameters(model) -> dict:
    """Devuelve conteo total/entrenable/no entrenable de parámetros PyTorch."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "non_trainable": int(total - trainable),
    }


def save_json(path: str | os.PathLike, payload: dict) -> None:
    """Guarda JSON con conversión básica NumPy/Path."""
    def convert(obj: Any):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=convert), encoding="utf-8")


def load_json(path: str | os.PathLike) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(
    path: str | os.PathLike,
    model,
    optimizer=None,
    scheduler=None,
    epoch: int | None = None,
    val_loss: float | None = None,
    config: dict | None = None,
    class_names: list[str] | None = None,
    extra: dict | None = None,
) -> None:
    """Guarda checkpoint PyTorch completo."""
    import torch

    from ecg.schema import LABEL_SCHEMA as _schema

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch) if epoch is not None else None,
        "val_loss": float(val_loss) if val_loss is not None else None,
        "config": config or {},
        "class_names": class_names or [],
        "num_classes": len(class_names or []),
        "label_schema": _schema,
        "problem_type": "multilabel_sigmoid_bce",
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | os.PathLike, map_location="cpu", trusted: bool = False) -> dict:
    """Carga un checkpoint PyTorch con ``weights_only=True`` (sin ejecutar pickle).

    Un ``.pt`` es un pickle: cargarlo con ``weights_only=False`` ejecuta código
    arbitrario. Use ``trusted=True`` solo para archivos propios que no se
    pueden cargar en modo seguro.
    """
    import torch
    return torch.load(path, map_location=map_location, weights_only=not trusted)


def find_checkpoints(models_dir: str | os.PathLike) -> list[Path]:
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    return sorted(models_dir.rglob("*.pt"))


_VAL_LOSS_CACHE: dict[tuple, float | None] = {}


def checkpoint_val_loss(path: str | os.PathLike) -> float | None:
    """``val_loss`` guardada en un checkpoint; None si no la trae o no se puede leer.

    Ordenar los checkpoints cargaba cada ``.pt`` entero (hasta decenas de MB con el estado del optimizador) y la webapp lo
    repetía varias veces por página y por análisis: con 12 checkpoints en ``saved/`` cada ``GET /`` tardaba 3,5-4,7 s.
    Se recuerda por (ruta, fecha, tamaño), así que solo se vuelve a leer un archivo si cambia.
    """
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _VAL_LOSS_CACHE:
        try:
            value = load_checkpoint(path, map_location="cpu").get("val_loss")
            _VAL_LOSS_CACHE[key] = None if value is None else float(value)
        except Exception:
            _VAL_LOSS_CACHE[key] = None
    return _VAL_LOSS_CACHE[key]


def checkpoint_sort_key(path: Path):
    """Clave de ordenamiento: reales antes que sintéticos, best.pt primero, menor val_loss.

    Evita el fallo silencioso de auto-seleccionar un checkpoint de smoke-test
    sintético (val_loss artificialmente baja) en lugar del modelo CINC2020 real.
    """
    text = str(path).lower()
    is_synth = 1 if "synth" in text else 0
    is_best_name = 0 if path.name.lower() == "best.pt" else 1
    loss = checkpoint_val_loss(path)
    if loss is None:
        try:
            loss = float(path.name.split("-")[0])
        except Exception:
            loss = float("inf")
    return (is_synth, is_best_name, loss, str(path))


def best_checkpoint(models_dir: str | os.PathLike) -> Path | None:
    """Busca el mejor checkpoint: reales primero, ``best.pt`` primero, menor ``val_loss``."""
    checkpoints = find_checkpoints(models_dir)
    if not checkpoints:
        return None
    return min(checkpoints, key=checkpoint_sort_key)


def list_checkpoints_info(models_dir: str | os.PathLike) -> list[dict]:
    """Lista checkpoints con metadatos para diagnóstico (ordenados por preferencia)."""
    checkpoints = sorted(find_checkpoints(models_dir), key=checkpoint_sort_key)
    rows = []
    for order, path in enumerate(checkpoints):
        try:
            ckpt = load_checkpoint(path, map_location="cpu")
            rows.append({
                "rank": order,
                "path": str(path),
                "val_loss": ckpt.get("val_loss"),
                "epoch": ckpt.get("epoch"),
                "label_schema": ckpt.get("label_schema"),
                "class_names": ckpt.get("class_names"),
                "is_regular_conv": (ckpt.get("config") or {}).get("is_regular_conv"),
            })
        except Exception as exc:
            rows.append({"rank": order, "path": str(path), "error": str(exc)})
    return rows
