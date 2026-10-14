"""Dataset HDF5 de ECG compartido por entrenamiento, evaluación y robustez.

Con ``preload=True`` (defecto) carga signals/labels a RAM una sola vez
(~7-11 GB para train CINC2020, ~1.5 GB para val/test). Úselo con
``num_workers=0``: en Windows los workers con spawn duplicarían esos GB por
worker. Si el archivo supera ``max_preload_gb``, cae automáticamente a lectura
por disco. La validación de forma y número de clases se hace *antes* de
precargar, para fallar rápido.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg import util
from ecg.schema import CLASS_NAMES, NUM_CLASSES
from ecg.signal import NUM_LEADS, WINDOW_LENGTH


class HDF5ECGDataset(Dataset):
    """Devuelve ``(x, y)`` con ``x`` de forma (12, 5000) y ``y`` de forma (num_classes,).

    Args:
        expected_num_classes: si se indica, valida el número de clases de ``labels``.
        require_labels: exige el dataset ``labels`` (entrenamiento); si es False y
            falta, ``y`` es un vector de ceros (inferencia sobre datos sin etiquetas).
        augment_cfg: configuración de :mod:`ecg.augment` (solo para el split de train).
    """

    def __init__(self, h5_path: str | Path, preload: bool = True, max_preload_gb: float = 16,
                 expected_num_classes: int | None = None, require_labels: bool = False,
                 augment_cfg: dict | None = None):
        import h5py

        self.h5_path = str(util.resolve_path(h5_path))
        self.augment_cfg = dict(augment_cfg) if augment_cfg else None
        self._aug_rng = np.random.default_rng(self.augment_cfg.get("aug_seed")) if self.augment_cfg else None
        self._aug_worker = None
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"No se encontró HDF5: {self.h5_path}")

        self._signals = None
        self._labels = None
        self._in_ram = False
        self.h5 = None
        with h5py.File(self.h5_path, "r") as h5:
            has_labels = "labels" in h5
            self.length = int(h5["signals"].shape[0])
            self.signal_shape = tuple(h5["signals"].shape)
            self.label_shape = tuple(h5["labels"].shape) if has_labels else None
            classes = h5.attrs.get("classes", None)
            self.classes = util.decode_h5_strings(classes) if classes is not None else CLASS_NAMES.copy()
            self.normalization = util.decode_h5_string(h5.attrs.get("normalization", "unknown"))
            self.label_schema = util.decode_h5_string(h5.attrs.get("label_schema", "unknown"))
            self.norm_mode = util.decode_h5_string(h5.attrs.get("norm_mode", "unknown"))
            if "record_names" in h5:
                self.record_names = util.decode_h5_strings(h5["record_names"][:])
            else:
                self.record_names = [f"record_{i}" for i in range(self.length)]

            self._validate(has_labels, expected_num_classes, require_labels)

            nbytes = int(h5["signals"].size * 4 + (h5["labels"].size * 4 if has_labels else 0))
            if preload and nbytes <= max_preload_gb * 1024 ** 3:
                print(f"Precargando a RAM {self.h5_path} ({nbytes / 1024 ** 3:.1f} GB)...", flush=True)
                self._signals = np.asarray(h5["signals"][:], dtype=np.float32)
                self._labels = np.asarray(h5["labels"][:], dtype=np.float32) if has_labels else None
                self._in_ram = True
            elif preload:
                print(f"AVISO: {self.h5_path} pesa {nbytes / 1024 ** 3:.1f} GB; lectura por disco (lento).")

    def _validate(self, has_labels: bool, expected_num_classes: int | None, require_labels: bool) -> None:
        if self.signal_shape[1:] != (WINDOW_LENGTH, NUM_LEADS):
            raise ValueError(f"signals debe ser (N, {WINDOW_LENGTH}, {NUM_LEADS}); actual={self.signal_shape}")
        if require_labels and not has_labels:
            raise ValueError(f"{self.h5_path} no contiene el dataset 'labels'.")
        if expected_num_classes is not None and has_labels and self.label_shape[1] != expected_num_classes:
            raise ValueError(
                f"{self.h5_path} contiene {self.label_shape[1]} clases; se esperaban {expected_num_classes}. "
                "Regenera los HDF5 con examples/cinc2020/build_datasets.py."
            )

    def _open(self):
        if self.h5 is None:
            import h5py
            self.h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self._in_ram:
            # HDF5: (5000,12). PyTorch Conv1d: (12,5000).
            x = self._signals[index].T
            y = self._labels[index] if self._labels is not None else np.zeros(NUM_CLASSES, dtype=np.float32)
        else:
            self._open()
            x = np.asarray(self.h5["signals"][index], dtype=np.float32).T
            if "labels" in self.h5:
                y = np.asarray(self.h5["labels"][index], dtype=np.float32)
            else:
                y = np.zeros(NUM_CLASSES, dtype=np.float32)
        if self.augment_cfg is not None:
            from ecg import augment
            self._reseed_for_worker()
            x = augment.augment_signal(x, rng=self._aug_rng, cfg=self.augment_cfg)
        return torch.from_numpy(x), torch.from_numpy(y)

    def _reseed_for_worker(self) -> None:
        """Cada worker del DataLoader recibe una COPIA del dataset con el mismo generador: sin esto todos aplicaban
        exactamente la misma aumentación (con señales iguales, ítems de workers distintos salían idénticos)."""
        info = torch.utils.data.get_worker_info()
        if info is not None and self._aug_worker != info.id:
            seed = self.augment_cfg.get("aug_seed")
            self._aug_rng = np.random.default_rng([int(seed), info.id] if seed is not None else info.seed)
            self._aug_worker = info.id

    def __del__(self):
        try:
            if self.h5 is not None:
                self.h5.close()
        except Exception:
            pass
