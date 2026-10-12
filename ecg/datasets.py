"""Escaneo de registros, splits estratificados y construcción de HDF5.

Extraído de ``ecg/load.py`` (split de depuración), incluyendo el CLI
``main()`` (``python -m ecg.load`` / ``build_datasets.py``).
"""

from __future__ import annotations
from ecg.schema import BANDPASS_HI_HZ, BANDPASS_LO_HZ, CLASS_DISPLAY_NAMES, CLASS_NAMES, DEFAULT_NORM_MODE, EXCLUDED_GROUPS, LABEL_SCHEMA, NUM_CLASSES, PHYSICAL_CLIP_MV, class_codes_as_strings, class_groups_as_records, codes_to_vector, matched_class_names
from ecg.signal import NUM_LEADS, STANDARD_LEAD_ORDER, TARGET_FS, WINDOW_LENGTH, WINDOW_SECONDS, compute_window_starts, load_mat_signal, normalize_lead_name, parse_header, preprocess_ecg_array

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "dataset2020"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "cinc2020_12"


def infer_source_db(hea_path: str | Path, data_dir: str | Path) -> str:
    parts = Path(os.path.relpath(hea_path, data_dir)).parts
    lower = [part.lower() for part in parts]
    if "training" in lower:
        idx = lower.index("training")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return parts[0] if len(parts) > 1 else "Unknown"


def scan_records(data_dir: str | Path, drop_no_selected_labels: bool = False) -> tuple[list[dict], dict]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No existe data_dir: {data_dir}")

    hea_files = sorted(Path(p) for p in glob.glob(str(data_dir / "**" / "*.hea"), recursive=True))
    counters = {
        "headers_found": len(hea_files),
        "missing_mat": 0,
        "bad_header": 0,
        "bad_leads": 0,
        "no_selected_labels": 0,
        "kept_records": 0,
    }

    from tqdm import tqdm
    records: list[dict] = []
    for hea_path in tqdm(hea_files, desc="Escaneando headers", unit="reg"):
        mat_path = hea_path.with_suffix(".mat")
        if not mat_path.exists():
            counters["missing_mat"] += 1
            continue
        try:
            meta = parse_header(hea_path)
        except Exception as exc:
            counters["bad_header"] += 1
            tqdm.write(f"AVISO header inválido {hea_path}: {exc}")
            continue

        normalized_leads = [normalize_lead_name(lead) for lead in meta["lead_names"]]
        if meta["num_leads"] != NUM_LEADS or any(lead not in normalized_leads for lead in STANDARD_LEAD_ORDER):
            counters["bad_leads"] += 1
            continue

        label = codes_to_vector(meta["dx_codes"])
        matched = matched_class_names(meta["dx_codes"])
        if label.sum() == 0:
            counters["no_selected_labels"] += 1
            if drop_no_selected_labels:
                continue

        records.append({
            "hea_path": str(hea_path),
            "mat_path": str(mat_path),
            "record_name": meta["record_name"],
            "num_leads": meta["num_leads"],
            "lead_names": meta["lead_names"],
            "sampling_freq": meta["sampling_freq"],
            "num_samples": meta["num_samples"],
            "adc_gains": meta.get("adc_gains"),
            "adc_baselines": meta.get("adc_baselines"),
            "age": meta["age"],
            "sex": meta["sex"],
            "source_db": infer_source_db(hea_path, data_dir),
            "dx_codes": meta["dx_codes"],
            "matched_classes": matched,
            "label": label,
        })

    counters["kept_records"] = len(records)
    return records, counters


def label_distribution(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    labels = np.stack([record["label"] for record in records])
    positives = labels.sum(axis=0).astype(int)
    total = labels.shape[0]
    return pd.DataFrame({
        "index": np.arange(NUM_CLASSES),
        "class": CLASS_NAMES,
        "display_name": CLASS_DISPLAY_NAMES,
        "positives": positives,
        "negatives": total - positives,
        "prevalence": positives / max(total, 1),
        "snomed_codes": class_codes_as_strings(),
    })


def source_distribution(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = [
        {
            "source_db": record["source_db"],
            "num_labels": int(record["label"].sum()),
            "all_zero": bool(record["label"].sum() == 0),
        }
        for record in records
    ]
    return (
        pd.DataFrame(rows)
        .groupby("source_db", dropna=False)
        .agg(records=("source_db", "size"), all_zero=("all_zero", "sum"), avg_labels=("num_labels", "mean"))
        .reset_index()
    )


def stratified_multilabel_split(
    labels: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Split 70/15/15 multilabel. Requiere iterative-stratification."""
    labels = np.asarray(labels, dtype=np.int32)
    if labels.shape[0] < 3:
        raise ValueError("No hay suficientes registros para train/val/test")
    if val_frac <= 0 or test_frac <= 0 or val_frac + test_frac >= 1:
        raise ValueError("Fracciones inválidas")

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as exc:
        raise RuntimeError(
            "Falta iterative-stratification. Instala requirements.txt. "
            "No se usa fallback por fuente/subconjunto para evitar sesgo."
        ) from exc

    n = labels.shape[0]
    split_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
    trainval_idx, test_idx = next(split_test.split(np.zeros((n, 1)), labels))

    val_frac_local = val_frac / (1.0 - test_frac)
    split_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_frac_local, random_state=random_state)
    train_sub, val_sub = next(split_val.split(np.zeros((len(trainval_idx), 1)), labels[trainval_idx]))

    return {
        "train": np.asarray(trainval_idx[train_sub], dtype=int),
        "val": np.asarray(trainval_idx[val_sub], dtype=int),
        "test": np.asarray(test_idx, dtype=int),
    }


def split_summary(records: list[dict], splits: dict[str, np.ndarray]) -> pd.DataFrame:
    labels = np.stack([record["label"] for record in records])
    rows = []
    for split_name, idx in splits.items():
        split_labels = labels[idx]
        positives = split_labels.sum(axis=0).astype(int)
        for i, name in enumerate(CLASS_NAMES):
            rows.append({
                "split": split_name,
                "records": int(len(idx)),
                "class_index": i,
                "class": name,
                "positives": int(positives[i]),
                "prevalence": float(positives[i] / max(len(idx), 1)),
            })
        rows.append({
            "split": split_name,
            "records": int(len(idx)),
            "class_index": -1,
            "class": "ALL_ZERO",
            "positives": int((split_labels.sum(axis=1) == 0).sum()),
            "prevalence": float((split_labels.sum(axis=1) == 0).mean()),
        })
    return pd.DataFrame(rows)


def split_assignments(records: list[dict], splits: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for split_name, indices in splits.items():
        for idx in indices:
            record = records[int(idx)]
            rows.append({
                "split": split_name,
                "index": int(idx),
                "record_name": record["record_name"],
                "source_db": record["source_db"],
                "dx_codes": ",".join(record["dx_codes"]),
                "matched_classes": ",".join(record["matched_classes"]),
                **{CLASS_NAMES[i]: int(record["label"][i]) for i in range(NUM_CLASSES)},
            })
    return pd.DataFrame(rows).sort_values(["split", "record_name"])


def _process_single_record(record: dict) -> dict:
    try:
        norm_mode = record.get("_norm_mode", DEFAULT_NORM_MODE)
        bandpass = record.get("_bandpass", True)
        window_position = int(record.get("_window_position", 0))
        windows_max = int(record.get("_windows_max", 1))
        raw = load_mat_signal(record["mat_path"], record.get("num_leads", NUM_LEADS))
        full = preprocess_ecg_array(
            raw,
            sampling_rate=record["sampling_freq"],
            lead_names=record["lead_names"],
            target_fs=TARGET_FS,
            target_length=None,
            normalize=True,
            norm_mode=norm_mode,
            adc_gains=record.get("adc_gains"),
            adc_baselines=record.get("adc_baselines"),
            units="digital",
            bandpass=bool(bandpass),
        )
        starts = compute_window_starts(full.shape[0], WINDOW_LENGTH, windows_max)
        start = starts[min(window_position, len(starts) - 1)]
        piece = full[start:start + WINDOW_LENGTH, :]
        if piece.shape[0] < WINDOW_LENGTH:
            piece = np.pad(piece, ((0, WINDOW_LENGTH - piece.shape[0]), (0, 0)), mode="constant")
        signal = np.asarray(piece, dtype=np.float32)
        return {
            "ok": True,
            "signal": signal,
            "label": record["label"],
            "age": record["age"],
            "sex": record["sex"],
            "source_db": record["source_db"],
            "record_name": record["record_name"],
            "window_position": window_position,
            "window_start": int(start),
            "full_length": int(full.shape[0]),
            "dx_codes": ",".join(record["dx_codes"]),
            "matched_classes": ",".join(record["matched_classes"]),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "record_name": record.get("record_name", "Unknown"),
            "error": str(exc),
        }


def _expand_windows_for_record(record: dict, windows_max: int) -> list[dict]:
    """Expande un registro en 1..K specs (una por ventana de entrenamiento).

    El número de ventanas se estima desde el header (muestras * 500/fs).
    Registros de 10 s siempre producen 1 spec (ventana centrada).
    """
    windows_max = int(windows_max)
    try:
        n_target = int(round(float(record["num_samples"]) * TARGET_FS / float(record["sampling_freq"])))
    except Exception:
        n_target = WINDOW_LENGTH
    n_windows = len(compute_window_starts(n_target, WINDOW_LENGTH, windows_max))
    specs = []
    for position in range(n_windows):
        spec = dict(record)
        spec["_window_position"] = position
        spec["_windows_max"] = windows_max
        specs.append(spec)
    return specs


def _write_string_attr(h5_file, name: str, values: list[str]) -> None:
    import h5py
    dtype = h5py.string_dtype(encoding="utf-8")
    h5_file.attrs.create(name, np.asarray(values, dtype=dtype))


def _write_hdf5_attributes(
    h5_file,
    split_name: str,
    n_requested: int,
    norm_mode: str = DEFAULT_NORM_MODE,
    bandpass: bool = True,
    windows_max: int = 1,
) -> None:
    h5_file.attrs["dataset"] = "PhysioNet/CinC Challenge 2020"
    h5_file.attrs["label_schema"] = LABEL_SCHEMA
    h5_file.attrs["label_mode"] = "multilabel"
    h5_file.attrs["problem_type"] = "multilabel_sigmoid_bce"
    h5_file.attrs["split"] = split_name
    h5_file.attrs["requested_records_before_signal_errors"] = int(n_requested)
    h5_file.attrs["sampling_rate"] = TARGET_FS
    h5_file.attrs["target_fs"] = TARGET_FS
    h5_file.attrs["window_seconds"] = WINDOW_SECONDS
    h5_file.attrs["window_length"] = WINDOW_LENGTH
    h5_file.attrs["num_leads"] = NUM_LEADS
    h5_file.attrs["num_classes"] = NUM_CLASSES
    h5_file.attrs["lead_order"] = ",".join(STANDARD_LEAD_ORDER)
    h5_file.attrs["norm_mode"] = str(norm_mode)
    h5_file.attrs["bandpass_filter"] = bool(bandpass)
    h5_file.attrs["windows_max"] = int(windows_max)
    if norm_mode == "physical":
        h5_file.attrs["normalization"] = (
            f"physical millivolts via header gain/baseline + bandpass {BANDPASS_LO_HZ}-{BANDPASS_HI_HZ} Hz + clip ±{PHYSICAL_CLIP_MV} mV"
            if bandpass else f"physical millivolts via header gain/baseline + clip ±{PHYSICAL_CLIP_MV} mV"
        )
    elif norm_mode == "global_zscore":
        h5_file.attrs["normalization"] = "per-record global z-score (all leads jointly) before padding"
    else:
        h5_file.attrs["normalization"] = "per-lead z-score before padding"
    h5_file.attrs["class_metadata_json"] = json.dumps(class_groups_as_records(), ensure_ascii=False)
    h5_file.attrs["excluded_groups_json"] = json.dumps(EXCLUDED_GROUPS, ensure_ascii=False)
    _write_string_attr(h5_file, "classes", CLASS_NAMES)
    _write_string_attr(h5_file, "class_display_names", CLASS_DISPLAY_NAMES)
    _write_string_attr(h5_file, "class_snomed_codes", class_codes_as_strings())


def process_split_to_hdf5(
    records: list[dict],
    indices: Iterable[int],
    output_path: str | Path,
    num_workers: int = 6,
    chunksize: int = 8,
    write_batch_size: int = 32,
    norm_mode: str = DEFAULT_NORM_MODE,
    bandpass: bool = True,
    windows_max: int = 1,
) -> dict:
    import h5py
    from tqdm import tqdm

    indices = np.asarray(list(indices), dtype=int)
    selected = [records[int(i)] for i in indices]
    # Expansión multi-ventana (train): registros largos aportan hasta
    # windows_max muestras de 10 s equiespaciadas con la misma etiqueta.
    expanded: list[dict] = []
    for record in selected:
        for spec in _expand_windows_for_record(record, windows_max):
            spec["_norm_mode"] = norm_mode
            spec["_bandpass"] = bool(bandpass)
            expanded.append(spec)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as h5:
        chunk_n = min(16, max(1, len(expanded)))
        signals_ds = h5.create_dataset(
            "signals",
            shape=(0, WINDOW_LENGTH, NUM_LEADS),
            maxshape=(None, WINDOW_LENGTH, NUM_LEADS),
            dtype="float32",
            chunks=(chunk_n, WINDOW_LENGTH, NUM_LEADS),
            compression="gzip",
            compression_opts=4,
        )
        labels_ds = h5.create_dataset(
            "labels",
            shape=(0, NUM_CLASSES),
            maxshape=(None, NUM_CLASSES),
            dtype="float32",
            chunks=(min(256, max(1, len(expanded))), NUM_CLASSES),
        )
        ages_ds = h5.create_dataset("ages", shape=(0,), maxshape=(None,), dtype="float32")
        sexes_ds = h5.create_dataset("sexes", shape=(0,), maxshape=(None,), dtype=string_dtype)
        sources_ds = h5.create_dataset("source_dbs", shape=(0,), maxshape=(None,), dtype=string_dtype)
        names_ds = h5.create_dataset("record_names", shape=(0,), maxshape=(None,), dtype=string_dtype)
        dx_ds = h5.create_dataset("dx_codes", shape=(0,), maxshape=(None,), dtype=string_dtype)
        matched_ds = h5.create_dataset("matched_classes", shape=(0,), maxshape=(None,), dtype=string_dtype)
        winpos_ds = h5.create_dataset("window_positions", shape=(0,), maxshape=(None,), dtype="int32")
        winstart_ds = h5.create_dataset("window_starts", shape=(0,), maxshape=(None,), dtype="int32")

        _write_hdf5_attributes(h5, output_path.stem, len(selected), norm_mode, bandpass, windows_max)

        buffers = {
            "signals": [], "labels": [], "ages": [], "sexes": [],
            "source_dbs": [], "record_names": [], "dx_codes": [], "matched_classes": [],
            "window_positions": [], "window_starts": [],
        }
        n_ok, n_error = 0, 0
        error_examples = []

        def flush() -> None:
            nonlocal n_ok
            if not buffers["signals"]:
                return
            batch_size = len(buffers["signals"])
            old, new = n_ok, n_ok + batch_size
            for dataset in [signals_ds, labels_ds, ages_ds, sexes_ds, sources_ds, names_ds, dx_ds, matched_ds, winpos_ds, winstart_ds]:
                dataset.resize(new, axis=0)
            signals_ds[old:new] = np.stack(buffers["signals"]).astype(np.float32, copy=False)
            labels_ds[old:new] = np.stack(buffers["labels"]).astype(np.float32, copy=False)
            ages_ds[old:new] = np.asarray(buffers["ages"], dtype=np.float32)
            sexes_ds[old:new] = buffers["sexes"]
            sources_ds[old:new] = buffers["source_dbs"]
            names_ds[old:new] = buffers["record_names"]
            dx_ds[old:new] = buffers["dx_codes"]
            matched_ds[old:new] = buffers["matched_classes"]
            winpos_ds[old:new] = np.asarray(buffers["window_positions"], dtype=np.int32)
            winstart_ds[old:new] = np.asarray(buffers["window_starts"], dtype=np.int32)
            n_ok = new
            for values in buffers.values():
                values.clear()

        if num_workers == 1:
            iterator = map(_process_single_record, expanded)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=num_workers)
            iterator = executor.map(_process_single_record, expanded, chunksize=chunksize)

        try:
            for result in tqdm(iterator, total=len(expanded), desc=f"Procesando {output_path.name}", unit="muestra"):
                if not result["ok"]:
                    n_error += 1
                    if len(error_examples) < 50:
                        error_examples.append({"record_name": result["record_name"], "error": result["error"]})
                    tqdm.write(f"AVISO {result['record_name']}: {result['error']}")
                    continue
                buffers["signals"].append(result["signal"])
                buffers["labels"].append(result["label"])
                buffers["ages"].append(result["age"])
                buffers["sexes"].append(result["sex"])
                buffers["source_dbs"].append(result["source_db"])
                buffers["record_names"].append(result["record_name"])
                buffers["dx_codes"].append(result["dx_codes"])
                buffers["matched_classes"].append(result["matched_classes"])
                buffers["window_positions"].append(result.get("window_position", 0))
                buffers["window_starts"].append(result.get("window_start", 0))
                if len(buffers["signals"]) >= write_batch_size:
                    flush()
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        flush()
        h5.attrs["written_records"] = int(n_ok)
        h5.attrs["signal_processing_errors"] = int(n_error)

    print(f"Guardado {output_path}: {n_ok:,} OK, {n_error:,} errores")
    return {"path": str(output_path), "ok": n_ok, "errors": n_error, "error_examples": error_examples}


def write_reports(
    output_dir: str | Path,
    records: list[dict],
    counters: dict,
    splits: dict[str, np.ndarray] | None = None,
    drop_no_selected_labels: bool = False,
    norm_mode: str = DEFAULT_NORM_MODE,
    bandpass: bool = True,
    train_windows_max: int = 4,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(class_groups_as_records()).to_csv(output_dir / "class_mapping_12.csv", index=False)
    label_distribution(records).to_csv(output_dir / "label_distribution.csv", index=False)
    source_distribution(records).to_csv(output_dir / "source_distribution.csv", index=False)
    if splits is not None:
        split_summary(records, splits).to_csv(output_dir / "split_summary.csv", index=False)
        split_assignments(records, splits).to_csv(output_dir / "split_assignments.csv", index=False)

    summary = {
        "dataset": "PhysioNet/CinC Challenge 2020",
        "label_schema": LABEL_SCHEMA,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "classes": class_groups_as_records(),
        "excluded_groups": EXCLUDED_GROUPS,
        "target_fs": TARGET_FS,
        "window_seconds": WINDOW_SECONDS,
        "window_length": WINDOW_LENGTH,
        "num_leads": NUM_LEADS,
        "lead_order": STANDARD_LEAD_ORDER,
        "norm_mode": str(norm_mode),
        "bandpass_filter": bool(bandpass),
        "bandpass_hz": [BANDPASS_LO_HZ, BANDPASS_HI_HZ],
        "physical_clip_mv": PHYSICAL_CLIP_MV,
        "train_windows_max": int(train_windows_max),
        "normalization": (
            f"physical millivolts + bandpass + clip ±{PHYSICAL_CLIP_MV} mV" if norm_mode == "physical"
            else "per-record global z-score" if norm_mode == "global_zscore"
            else "per-lead z-score (legacy v1)"
        ),
        "split_method": "MultilabelStratifiedShuffleSplit by label vector, not by source",
        "drop_no_selected_labels": bool(drop_no_selected_labels),
        "scan_counters": counters,
        "splits": {name: int(len(idx)) for name, idx in splits.items()} if splits is not None else None,
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_datasets(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    drop_no_selected_labels: bool = False,
    scan_only: bool = False,
    workers: int = 6,
    chunksize: int = 8,
    write_batch_size: int = 32,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
    norm_mode: str = DEFAULT_NORM_MODE,
    bandpass: bool = True,
    train_windows_max: int = 4,
) -> dict:
    """Función programática usada por ``examples/cinc2020/build_datasets.py``."""
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()

    print("\n" + "=" * 80)
    print("PREPROCESAMIENTO CINC2020 — 12 CLASES AGRUPADAS")
    print("=" * 80)
    print(f"Esquema     : {LABEL_SCHEMA}")
    print(f"Dataset     : {data_dir}")
    print(f"Salida      : {output_dir}")
    print(f"Señal       : {NUM_LEADS} leads, {TARGET_FS} Hz, {WINDOW_LENGTH} muestras")
    print(f"Norm        : {norm_mode} (bandpass={bandpass})")
    print(f"Ventanas    : train x{train_windows_max} / val-test x1")
    print(f"Clases      : {NUM_CLASSES} -> {CLASS_NAMES}")
    print(f"All-zero    : {'excluir' if drop_no_selected_labels else 'conservar'}")

    records, counters = scan_records(data_dir, drop_no_selected_labels=drop_no_selected_labels)
    print("\nEscaneo:")
    for key, value in counters.items():
        print(f"  {key:<24}: {value:,}")
    if not records:
        raise RuntimeError("No quedaron registros válidos tras el escaneo")

    dist = label_distribution(records)
    print("\nDistribución por clase:")
    print(dist[["index", "class", "positives", "prevalence"]].to_string(index=False))

    labels = np.stack([record["label"] for record in records])
    empty = [CLASS_NAMES[i] for i, count in enumerate(labels.sum(axis=0)) if count <= 0]
    if empty:
        raise RuntimeError("Clases sin positivos: " + ", ".join(empty))

    if scan_only:
        write_reports(output_dir, records, counters, splits=None, drop_no_selected_labels=drop_no_selected_labels,
                      norm_mode=norm_mode, bandpass=bandpass, train_windows_max=train_windows_max)
        return {"records": len(records), "counters": counters, "splits": None}

    splits = stratified_multilabel_split(labels, val_frac=val_frac, test_frac=test_frac, random_state=random_state)
    print("\nSplits:")
    for split_name, idx in splits.items():
        print(f"  {split_name:<5}: {len(idx):,}")

    write_reports(output_dir, records, counters, splits=splits, drop_no_selected_labels=drop_no_selected_labels,
                  norm_mode=norm_mode, bandpass=bandpass, train_windows_max=train_windows_max)

    processing = []
    for split_name in ["train", "val", "test"]:
        processing.append(process_split_to_hdf5(
            records=records,
            indices=splits[split_name],
            output_path=output_dir / f"{split_name}.h5",
            num_workers=workers,
            chunksize=chunksize,
            write_batch_size=write_batch_size,
            norm_mode=norm_mode,
            bandpass=bandpass,
            windows_max=train_windows_max if split_name == "train" else 1,
        ))

    (output_dir / "signal_processing_summary.json").write_text(
        json.dumps(processing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"records": len(records), "counters": counters, "splits": {k: len(v) for k, v in splits.items()}, "processing": processing}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Construye HDF5 CINC2020-12 desde archivos .hea/.mat")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Raíz local con CINC2020 training/")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directorio de salida")
    parser.add_argument("--drop-no-selected-labels", action="store_true", help="Excluir ECGs sin ninguna de las 12 clases")
    parser.add_argument("--scan-only", action="store_true", help="Solo escanear headers y escribir reportes")
    parser.add_argument("--workers", type=int, default=6, help="Procesos de preprocesamiento")
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--write-batch-size", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--norm-mode", choices=["physical", "global_zscore", "per_lead_zscore"],
                        default=DEFAULT_NORM_MODE,
                        help="Normalización: physical=mV+clip (v2), global_zscore, per_lead_zscore (legacy v1)")
    parser.add_argument("--bandpass", dest="bandpass", action="store_true", default=True,
                        help="Aplica pasa-banda 0.5-50 Hz en modo physical (defecto: activado)")
    parser.add_argument("--no-bandpass", dest="bandpass", action="store_false",
                        help="Desactiva el pasa-banda")
    parser.add_argument("--train-windows-max", type=int, default=4,
                        help="Ventanas de 10 s por registro largo en train (val/test siempre 1 centrada)")
    args = parser.parse_args(argv)

    build_datasets(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        drop_no_selected_labels=args.drop_no_selected_labels,
        scan_only=args.scan_only,
        workers=args.workers,
        chunksize=args.chunksize,
        write_batch_size=args.write_batch_size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_state=args.random_state,
        norm_mode=args.norm_mode,
        bandpass=args.bandpass,
        train_windows_max=args.train_windows_max,
    )


if __name__ == "__main__":
    freeze_support()
    main()
