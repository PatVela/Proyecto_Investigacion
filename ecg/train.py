"""Entrenamiento reproducible para CINC2020-12.

Uso recomendado::

    python -m ecg.train examples/cinc2020/config.json --experiment cinc2020_resnet

La formulación es multilabel: el modelo produce logits y se entrena con
``BCEWithLogitsLoss``. No se usa softmax ni argmax.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from . import hdf5_dataset, load, network, util
except ImportError:  # Permite `python ecg/train.py ...`.
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from ecg import hdf5_dataset, load, network, util


class HDF5ECGDataset(hdf5_dataset.HDF5ECGDataset):
    """Dataset de entrenamiento: exige ``labels`` y número de clases; admite aumentación."""

    def __init__(self, h5_path: str | Path, expected_num_classes: int = load.NUM_CLASSES,
                 preload: bool = True, max_preload_gb: float = 16,
                 augment_cfg: dict | None = None, cell_mask: tuple[list[str], np.ndarray] | None = None):
        super().__init__(h5_path, preload=preload, max_preload_gb=max_preload_gb,
                         expected_num_classes=expected_num_classes, require_labels=True,
                         augment_cfg=augment_cfg, cell_mask=cell_mask)


# Helpers compartidos con ecg.predict (definidos una sola vez en util/network).
decode_h5_string = util.decode_h5_string
get_device = util.get_device
model_params = network.params_from_config


def _loader_kwargs(params: dict, shuffle: bool) -> dict:
    return util.dataloader_kwargs(
        batch_size=params.get("batch_size", params.get("batch", 8)),
        num_workers=params.get("num_workers", 2),
        pin_memory=params.get("pin_memory", True),
        prefetch_factor=params.get("prefetch_factor", 4),
        shuffle=shuffle,
    )


# Claves de la configuración que algún código lee (train.py, network.params_from_config, hdf5_dataset/augment, evaluate.py)
# más los metadatos informativos de los config*.json. Cualquier otra se ignora sin error: una errata como "conv_dropuot"
# entrenaba con el valor por defecto (0.2) y config_used.json seguía mostrando la clave mal escrita, como si hubiera
# surtido efecto. tests/test_train_monitor.py comprueba que este conjunto cubre lo que el código lee de verdad.
KNOWN_CONFIG_KEYS = frozenset({
    "amp", "aug_baseline_amp", "aug_lead_drop_prob", "aug_noise_std", "aug_prob", "aug_scale_range", "aug_seed",
    "aug_shift_max", "augment", "batch", "batch_size", "config_path", "conv_dropout", "conv_filter_length",
    "conv_num_filters_start", "conv_num_skip", "conv_subsample_lengths", "dev", "device", "drop_rate",
    "early_stopping_metric", "early_stopping_patience", "epochs", "filter_length", "input_length", "is_regular_conv",
    "kernel_size", "label_smoothing", "learning_rate", "lr", "max_epochs", "min_lr", "num_classes", "num_leads",
    "num_workers", "patience", "pin_memory", "prefetch_factor", "reduce_lr_factor", "reduce_lr_patience", "save_dir",
    "seed", "test", "train", "weight_decay", "mask_unannotated", "mask_min_positives",
    "dataset", "label_schema", "problem_type",                    # metadatos: documentan la receta, nada los lee
})


def unknown_config_keys(params: dict) -> list[str]:
    """Claves de ``params`` que ningún código lee (probable errata)."""
    return sorted(set(params) - KNOWN_CONFIG_KEYS)


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        params = json.load(handle)
    params["config_path"] = str(path)
    return params


def compute_cell_mask(train_h5: str | Path, min_positives: int = 1) -> tuple[list[str], np.ndarray]:
    """Celdas (origen, clase) anotadas: ``1`` si el origen tiene al menos ``min_positives`` positivos de la clase en entrenamiento.

    Devuelve ``(orígenes, matriz (orígenes, clases))``. Es la regla inferida que usa ``examples/cinc2020/error_analysis.py``; la
    cobertura documentada por origen puede contrastarse aparte. Una celda con ``0`` no aporta pérdida: su ausencia de positivos no
    se interpreta como un negativo confirmado."""
    import h5py

    with h5py.File(util.resolve_path(train_h5), "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.float32)
        if "source_dbs" not in h5:
            raise ValueError(f"{train_h5} no trae 'source_dbs': no se puede calcular la máscara de celdas sin anotar")
        sources = np.asarray(util.decode_h5_strings(h5["source_dbs"][:]))
    names = sorted(set(sources.tolist()))
    matrix = np.stack([(labels[sources == n].sum(axis=0) >= int(min_positives)).astype(np.float32) for n in names])
    return names, matrix


def compute_pos_weight(train_h5: str | Path, cell_mask: tuple[list[str], np.ndarray] | None = None
                       ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(util.resolve_path(train_h5), "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.float32)
        sources = util.decode_h5_strings(h5["source_dbs"][:]) if cell_mask is not None else None
    if cell_mask is None:
        positives = labels.sum(axis=0)
        negatives = labels.shape[0] - positives
    else:                                                  # solo cuentan las celdas anotadas
        names, matrix = cell_mask
        row = {n: i for i, n in enumerate(names)}
        annotated = np.stack([matrix[row[s]] for s in sources])
        positives = (labels * annotated).sum(axis=0)
        negatives = annotated.sum(axis=0) - positives
    if np.any(positives <= 0):
        missing = [load.CLASS_NAMES[i] for i in np.where(positives <= 0)[0]]
        raise ValueError("Clases sin positivos en train.h5: " + ", ".join(missing))
    return torch.tensor(negatives / positives, dtype=torch.float32), positives, negatives


def make_dataloaders(params: dict):
    expected_classes = int(params.get("num_classes", load.NUM_CLASSES))
    augment_cfg = None
    if params.get("augment", False):
        augment_cfg = {
            "aug_prob": float(params.get("aug_prob", 0.5)),
            "aug_noise_std": float(params.get("aug_noise_std", 0.01)),
            "aug_baseline_amp": float(params.get("aug_baseline_amp", 0.1)),
            "aug_scale_range": float(params.get("aug_scale_range", 0.1)),
            "aug_shift_max": int(params.get("aug_shift_max", 250)),
            "aug_lead_drop_prob": float(params.get("aug_lead_drop_prob", 0.05)),
            "aug_seed": params.get("aug_seed", None),
        }
        print(f"Aumentación activada (solo train): {augment_cfg}")
    cell_mask = None
    if params.get("mask_unannotated", False):
        cell_mask = compute_cell_mask(params["train"], int(params.get("mask_min_positives", 1)))
        sin_anotar = [(n, load.CLASS_NAMES[j]) for i, n in enumerate(cell_mask[0]) for j in range(cell_mask[1].shape[1]) if cell_mask[1][i, j] == 0]
        print(f"Pérdida enmascarada: {len(sin_anotar)} celdas origen-clase sin anotar no aportan pérdida: {sin_anotar}")
    train_ds = HDF5ECGDataset(params["train"], expected_num_classes=expected_classes, augment_cfg=augment_cfg, cell_mask=cell_mask)
    dev_ds = HDF5ECGDataset(params["dev"], expected_num_classes=expected_classes, cell_mask=cell_mask)
    train_ds.cell_mask = cell_mask
    if train_ds.classes != dev_ds.classes:
        raise ValueError("Las clases de train y dev no coinciden")
    for name, ds in (("train", train_ds), ("dev", dev_ds)):
        if ds.label_schema not in {"unknown", load.LABEL_SCHEMA}:
            print(
                f"AVISO: {name}.h5 usa label_schema={ds.label_schema!r} distinto del actual "
                f"({load.LABEL_SCHEMA!r}). Regenere los HDF5 con examples/cinc2020/build_datasets.py."
            )
    print(f"HDF5 train: schema={train_ds.label_schema} norm={train_ds.norm_mode} clases={train_ds.classes}")

    train_loader = DataLoader(train_ds, **_loader_kwargs(params, shuffle=True))
    dev_loader = DataLoader(dev_ds, **_loader_kwargs(params, shuffle=False))
    return train_loader, dev_loader, train_ds, dev_ds


class MaskedBCEWithLogits(nn.Module):
    """``BCEWithLogitsLoss`` con peso de positivos y una máscara de celdas: promedia solo sobre las celdas con máscara 1.

    Con una máscara de unos equivale a ``nn.BCEWithLogitsLoss(pos_weight=...)`` (promedio sobre todas las celdas)."""

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        if mask is None:
            return loss.mean()
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def split_batch(batch):
    """``(x, y)`` o ``(x, y, máscara)`` -> ``(x, y, máscara o None)``."""
    return (batch[0], batch[1], batch[2]) if len(batch) == 3 else (batch[0], batch[1], None)


def smooth_targets(y, smoothing: float):
    """Label smoothing para BCE multilabel: 0→s/2, 1→1-s/2."""
    s = float(smoothing)
    if s <= 0:
        return y
    return y * (1.0 - s) + 0.5 * s


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, label_smoothing: float = 0.0) -> float:
    model.train()
    total_loss, total = 0.0, 0
    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        x, y, w = split_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True) if w is not None else None
        y = smooth_targets(y, label_smoothing)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss_value = criterion(logits, y) if w is None else criterion(logits, y, w)
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(5.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss_value = criterion(logits, y) if w is None else criterion(logits, y, w)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(5.0))
            optimizer.step()

        total_loss += float(loss_value.item()) * x.shape[0]
        total += x.shape[0]
        progress.set_postfix(loss=f"{loss_value.item():.4f}")
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, amp: bool = False) -> float:
    model.eval()
    total_loss, total = 0.0, 0
    for batch in tqdm(loader, desc="dev", leave=False):
        x, y, w = split_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True) if w is not None else None
        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss_value = criterion(logits, y) if w is None else criterion(logits, y, w)
        else:
            logits = model(x)
            loss_value = criterion(logits, y) if w is None else criterion(logits, y, w)
        total_loss += float(loss_value.item()) * x.shape[0]
        total += x.shape[0]
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate_val_f1(model, loader, device, amp: bool = False) -> float:
    """F1-macro en validación con umbral fijo 0.5 (métrica de selección)."""
    from sklearn.metrics import f1_score
    model.eval()
    all_true, all_pred, all_mask = [], [], []
    for batch in tqdm(loader, desc="dev-f1", leave=False):
        x, y, w = split_batch(batch)
        x = x.to(device, non_blocking=True)
        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
        else:
            logits = model(x)
        all_true.append(y.detach().cpu())
        all_pred.append((torch.sigmoid(logits).float().cpu() >= 0.5).to(torch.uint8))
        if w is not None:
            all_mask.append(w.detach().cpu())
    y_true = torch.cat(all_true, dim=0).numpy()
    y_pred = torch.cat(all_pred, dim=0).numpy()
    if not all_mask:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    keep = torch.cat(all_mask, dim=0).numpy() > 0                   # cada clase se evalúa solo en los registros con su celda anotada
    return float(np.mean([f1_score(y_true[keep[:, j], j], y_pred[keep[:, j], j], zero_division=0) for j in range(y_true.shape[1])]))


def is_monitor_improvement(value: float, best: float, monitor: str) -> bool:
    """Compara según la métrica monitorizada (F1 se maximiza, loss se minimiza)."""
    if monitor == "val_f1_macro":
        return float(value) > float(best)
    return float(value) < float(best)


def write_history(save_dir: Path, history: list[dict]) -> None:
    if not history:
        return
    path = save_dir / "history.csv"
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def train(args, params: dict, epoch_callback=None):
    """Entrena y guarda artefactos en un save_dir timestamped.

    Args:
        epoch_callback: opcional ``f(epoch_1based, row_dict)`` invocado tras
            cada época (row trae train_loss/val_loss/val_f1_macro/...).
            Solo observa: si lanza una excepción (p. ej. poda Optuna),
            el entrenamiento se aborta y la excepción se propaga.
    """
    seed = args.seed if args.seed is not None else params.get("seed", 42)
    util.set_seed(seed)
    params["seed"] = int(seed)

    unknown = unknown_config_keys(params)
    if unknown:
        print(f"AVISO: la configuración trae claves que nada lee y se ignoran: {unknown}. "
              "Si es una errata de un hiperparámetro, esta corrida usa su valor por defecto.")
    params.setdefault("num_classes", load.NUM_CLASSES)
    params.setdefault("num_leads", load.NUM_LEADS)
    params.setdefault("input_length", load.WINDOW_LENGTH)
    if int(params["num_classes"]) != load.NUM_CLASSES:
        raise ValueError("Esta versión espera num_classes=12")

    device = get_device(args.device or params.get("device", "auto"))
    use_amp = bool(params.get("amp", True)) and not args.no_amp and device.type == "cuda"

    train_loader, dev_loader, train_ds, _dev_ds = make_dataloaders(params)
    class_names = train_ds.classes

    save_dir = util.timestamped_dir(params.get("save_dir", "saved/cinc2020"), args.experiment)
    (save_dir / "config_used.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")

    model = network.build_network(**model_params(params)).to(device)
    param_count = util.count_parameters(model)

    print("\n" + "=" * 72)
    print("ENTRENAMIENTO CINC2020-12")
    print("=" * 72)
    print(f"Modelo       : {model.model_type}")
    print(f"Device       : {device}")
    print(f"AMP          : {'sí' if use_amp else 'no'}")
    print(f"Train        : {len(train_ds):,} ECGs")
    print(f"Dev          : {len(_dev_ds):,} ECGs")
    print(f"Clases       : {class_names}")
    print("Parámetros   : total={total:,} | entrenables={trainable:,}".format(**param_count))
    print(f"Save dir     : {save_dir}")
    print("=" * 72)

    pos_weight, positives, negatives = compute_pos_weight(params["train"], train_ds.cell_mask)
    print("\nPositivos por clase:")
    for i, name in enumerate(class_names):
        print(f"  {i:02d} {name:<20} pos={positives[i]:8.0f} neg={negatives[i]:8.0f} weight={pos_weight[i].item():.4f}")

    if train_ds.cell_mask is None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = MaskedBCEWithLogits(pos_weight.to(device)).to(device)
        names, matrix = train_ds.cell_mask
        with (save_dir / "cell_mask.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["source_db", *class_names])
            for n, row in zip(names, matrix):
                writer.writerow([n, *[int(v) for v in row]])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params.get("learning_rate", params.get("lr", 1e-3))),
        weight_decay=float(params.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(params.get("reduce_lr_factor", 0.5)),
        patience=int(params.get("reduce_lr_patience", 3)),
        min_lr=float(params.get("min_lr", 5e-5)),
    )
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except Exception:
            scaler = torch.cuda.amp.GradScaler()

    monitor = str(params.get("early_stopping_metric", "val_loss"))
    if monitor not in ("val_loss", "val_f1_macro"):
        raise ValueError(f"early_stopping_metric debe ser val_loss o val_f1_macro, recibido {monitor}")
    print(f"Monitorizando: {monitor} (el scheduler sigue usando val_loss)")
    start_epoch = 0
    best_monitor = float("-inf") if monitor == "val_f1_macro" else float("inf")
    if args.resume:
        ckpt = util.load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        if monitor == "val_f1_macro":
            best_monitor = evaluate_val_f1(model, dev_loader, device, amp=use_amp)
            print(f"F1-macro inicial del checkpoint reanudado: {best_monitor:.4f}")
        else:
            best_monitor = float(ckpt.get("val_loss", best_monitor))
        print(f"Reanudado desde {args.resume}, época {start_epoch}")

    max_epochs = int(args.epochs or params.get("max_epochs", params.get("epochs", 80)))
    patience = int(params.get("early_stopping_patience", params.get("patience", 10)))
    no_improve = 0
    history: list[dict] = []
    total_start = time.time()

    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()
        print(f"\nÉpoca {epoch + 1}/{max_epochs}")
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler,
                               label_smoothing=float(params.get("label_smoothing", 0.0)))
        val_loss = evaluate_loss(model, dev_loader, criterion, device, amp=use_amp)
        val_f1 = evaluate_val_f1(model, dev_loader, device, amp=use_amp)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - epoch_start

        monitored = val_f1 if monitor == "val_f1_macro" else val_loss
        improved = is_monitor_improvement(monitored, best_monitor, monitor)
        if improved:
            best_monitor = float(monitored)
            no_improve = 0
        else:
            no_improve += 1

        latest_path = save_dir / "latest.pt"
        util.save_checkpoint(
            latest_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            val_loss=val_loss,
            config=params,
            class_names=class_names,
            extra={"pos_weight": pos_weight.cpu(), "input_shape": [load.NUM_LEADS, load.WINDOW_LENGTH]},
        )
        checkpoint_name = "latest.pt"

        if improved:
            best_path = save_dir / "best.pt"
            util.save_checkpoint(
                best_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                val_loss=val_loss,
                config=params,
                class_names=class_names,
                extra={"pos_weight": pos_weight.cpu(), "input_shape": [load.NUM_LEADS, load.WINDOW_LENGTH]},
            )
            checkpoint_name = "best.pt"
            print("-> Nuevo mejor modelo")

        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_f1_macro": float(val_f1),
            "learning_rate": float(lr),
            "epoch_seconds": float(epoch_seconds),
            "improved": bool(improved),
            "checkpoint": checkpoint_name,
        }
        history.append(row)
        write_history(save_dir, history)

        print(f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_f1={val_f1:.4f} | lr={lr:.8f} | {epoch_seconds:.1f}s")
        if epoch_callback is not None:
            epoch_callback(epoch + 1, row)
        if no_improve >= patience:
            print("Early stopping activado")
            break

    summary = {
        "dataset": "PhysioNet/CinC Challenge 2020",
        "label_schema": load.LABEL_SCHEMA,
        "problem_type": "multilabel sigmoid + BCEWithLogitsLoss",
        "model_type": model.model_type,
        "class_names": class_names,
        "num_parameters": param_count,
        "monitor": monitor,
        "best_monitored_value": float(best_monitor),
        "final_val_loss": float(history[-1]["val_loss"]) if history else None,
        "final_val_f1_macro": float(history[-1]["val_f1_macro"]) if history else None,
        "epochs_completed": len(history),
        "total_seconds": float(time.time() - total_start),
        "save_dir": str(save_dir),
        "history": history,
        "config": params,
    }
    util.save_json(save_dir / "training_summary.json", summary)
    print(f"\nEntrenamiento terminado. Artefactos en: {save_dir}")
    return save_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Entrena ResNet-34 CINC2020-12")
    parser.add_argument("config", help="JSON de configuración")
    parser.add_argument("--experiment", "-e", default="cinc2020_resnet", help="Nombre del experimento")
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescribe max_epochs")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--no-amp", action="store_true", help="Desactiva mixed precision")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint para reanudar")
    args = parser.parse_args(argv)
    params = load_config(args.config)
    train(args, params)


if __name__ == "__main__":
    main()
