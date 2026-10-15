"""ecg/datasets.py: lectura de registros, reparto 70/15/15 estratificado y HDF5, sobre un mini-Challenge sintético.

La construcción de los HDF5 solo tenía una prueba de ``--help``: un cambio en la proporción del reparto (la fracción de
validación sin reescalar daba 12.75 % en vez de 15 %) pasaba toda la suite. Aquí se comprueba lo que de verdad importa para
que la evaluación sea válida: ningún registro en dos particiones, proporciones, una ventana por registro en val/test,
ventanas extra solo en train, etiquetas que salen del encabezado y señal en milivoltios.
"""

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.io

from ecg import datasets, load
from ecg.util import decode_h5_strings

LEADS = load.STANDARD_LEAD_ORDER
N_RECORDS = 120                  # 12 clases x 10 registros
N_LONG = 4                       # los primeros 4 duran 40 s: 4 ventanas de 10 s en train
SECONDS = {True: 40, False: 10}


def _write_record(folder, name, codes, seconds=10, n_leads=12, with_mat=True):
    folder.mkdir(parents=True, exist_ok=True)
    n = seconds * 500
    t = np.arange(n)
    sig = (np.sin(t / 30.0) * 600 + (t % 250 < 5) * 1500).astype(np.int16)                  # ≈ 0.6 mV + picos de 1.5 mV
    lead_names = (LEADS + ["X"])[:n_leads]                                                # 13: las 12 estándar y una extra
    header = f"{name} {n_leads} 500 {n}\n" + "".join(f"{name}.mat 16 1000/mV 16 0 0 0 0 {lead}\n" for lead in lead_names)
    (folder / f"{name}.hea").write_text(header + f"# Age: 50\n# Sex: Male\n# Dx: {','.join(codes)}\n", encoding="utf-8")
    if with_mat:
        scipy.io.savemat(folder / f"{name}.mat", {"val": np.tile(sig, (n_leads, 1))})


def _classes_of(i):
    classes = {i % 12}
    if i % 5 == 0:
        classes.add((i + 3) % 12)
    return classes


def _write_challenge(root):
    """120 registros con la estructura del Challenge (training/<origen>/g1/); devuelve {nombre: (clases, segundos)}."""
    info = {}
    for i in range(N_RECORDS):
        origin, name = ("cpsc_2018", f"A{i:04d}") if i % 2 == 0 else ("ptb-xl", f"HR{i:05d}")
        seconds = SECONDS[i < N_LONG]
        classes = _classes_of(i)
        _write_record(root / "training" / origin / "g1", name, [load.CLASS_GROUPS_12[c]["codes"][0] for c in sorted(classes)], seconds)
        info[name] = ({load.CLASS_NAMES[c] for c in classes}, seconds)
    return info


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    base = tmp_path_factory.mktemp("challenge")
    info = _write_challenge(base / "dataset")
    summary = datasets.build_datasets(base / "dataset", base / "out", workers=1)
    return {"info": info, "out": base / "out", "root": base / "dataset", "summary": summary}


def _read(out, split):
    with h5py.File(out / f"{split}.h5", "r") as f:
        return {"names": decode_h5_strings(f["record_names"][:]), "labels": f["labels"][:], "signals": f["signals"][:],
                "matched": decode_h5_strings(f["matched_classes"][:])}


def test_the_split_proportions_are_70_15_15():
    """Con pocos registros el redondeo de la estratificación iterativa tapa un error de proporción (a n = 120 el defecto daba el
    mismo tamaño de validación): se prueba con 600 etiquetas, donde la fracción de val sin reescalar daba 12.8 % en vez de 15 %."""
    n = 600
    labels = np.zeros((n, 12), dtype=int)
    for i in range(n):
        labels[i, i % 12] = 1
        if i % 5 == 0:
            labels[i, (i + 3) % 12] = 1
    splits = datasets.stratified_multilabel_split(labels, random_state=42)
    fractions = {k: len(v) / n for k, v in splits.items()}
    assert abs(fractions["val"] - 0.15) < 0.02 and abs(fractions["test"] - 0.15) < 0.02 and abs(fractions["train"] - 0.70) < 0.03, fractions
    assert sorted(np.concatenate(list(splits.values()))) == list(range(n))              # cada registro en una sola partición


def test_the_built_split_covers_every_record(built):
    sizes = built["summary"]["splits"]
    assert sum(sizes.values()) == N_RECORDS and min(sizes.values()) > 0


def test_no_record_is_in_two_splits_and_the_csv_matches_the_hdf5(built):
    assignments = pd.read_csv(built["out"] / "split_assignments.csv")
    assert not assignments.record_name.duplicated().any() and set(assignments.record_name) == set(built["info"])
    for split in ("train", "val", "test"):
        assert set(_read(built["out"], split)["names"]) == set(assignments[assignments.split == split].record_name), split


def test_val_and_test_keep_one_window_per_record_and_train_expands_the_long_ones(built):
    assignments = pd.read_csv(built["out"] / "split_assignments.csv")
    for split in ("val", "test"):
        assert len(_read(built["out"], split)["names"]) == (assignments.split == split).sum(), split
    train_records = assignments[assignments.split == "train"].record_name
    expected = sum(len(load.compute_window_starts(built["info"][name][1] * 500, load.WINDOW_LENGTH, 4)) for name in train_records)
    names = _read(built["out"], "train")["names"]
    assert len(names) == expected and len(names) > len(train_records)               # hay al menos un registro largo en train


def test_every_class_has_positives_in_each_split(built):
    for split in ("train", "val", "test"):
        assert (_read(built["out"], split)["labels"].sum(axis=0) > 0).all(), split


def test_the_labels_in_the_hdf5_come_from_the_header_codes(built):
    data = _read(built["out"], "test")
    for name, row, matched in zip(data["names"], data["labels"], data["matched"]):
        expected = built["info"][name][0]
        assert {load.CLASS_NAMES[j] for j in np.flatnonzero(row)} == expected, name
        assert set(matched.split(",")) == expected, name


def test_signals_are_in_millivolts_inside_the_clip(built):
    signals = _read(built["out"], "test")["signals"]
    assert signals.shape[1:] == (load.WINDOW_LENGTH, load.NUM_LEADS)
    assert 0.5 < np.abs(signals).max() < load.PHYSICAL_CLIP_MV                       # ×1000 en vez de ÷1000 saturaría en ±5 mV


def test_the_same_seed_gives_the_same_split_and_another_seed_a_different_one(built):
    records, _ = datasets.scan_records(built["root"])
    labels = np.stack([r["label"] for r in records])
    first = datasets.stratified_multilabel_split(labels, random_state=42)
    again = datasets.stratified_multilabel_split(labels, random_state=42)
    other = datasets.stratified_multilabel_split(labels, random_state=7)
    assert all(np.array_equal(first[k], again[k]) for k in first)
    assert not np.array_equal(first["test"], other["test"])


def test_invalid_fractions_are_rejected():
    with pytest.raises(ValueError, match="Fracciones inválidas"):
        datasets.stratified_multilabel_split(np.eye(12, dtype=int), val_frac=0.6, test_frac=0.6)


def test_scan_records_counts_what_it_discards_and_infers_the_origin(tmp_path):
    folder = tmp_path / "training" / "cpsc_2018" / "g1"
    nsr, af = (load.CLASS_GROUPS_12[0]["codes"][0], load.CLASS_GROUPS_12[4]["codes"][0])
    _write_record(folder, "A0001", [nsr])
    _write_record(tmp_path / "training" / "ptb-xl" / "g1", "HR00001", [af])
    _write_record(folder, "A9001", [nsr], with_mat=False)                           # sin .mat
    _write_record(folder, "A9002", [nsr], n_leads=8)                                # faltan derivaciones estándar
    _write_record(folder, "A9004", [nsr], n_leads=13)                               # están las 12 estándar y sobra una
    _write_record(folder, "A9003", ["999999"])                                      # Dx fuera del esquema: all-zero
    records, counters = datasets.scan_records(tmp_path)
    assert (counters["headers_found"], counters["missing_mat"], counters["bad_leads"], counters["no_selected_labels"]) == (6, 1, 2, 1)
    assert counters["kept_records"] == len(records) == 3
    assert {r["record_name"]: r["source_db"] for r in records} == {"A0001": "cpsc_2018", "HR00001": "ptb-xl", "A9003": "cpsc_2018"}
    _, dropped = datasets.scan_records(tmp_path, drop_no_selected_labels=True)
    assert dropped["kept_records"] == 2
