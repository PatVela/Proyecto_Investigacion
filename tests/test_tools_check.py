"""Validación estructural tools/check_dataset2020 con HDF5 sintéticos."""

import sys
from pathlib import Path

import h5py
import numpy as np

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import check_dataset2020 as api

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASSES = [f"C{i:02d}" for i in range(12)]


def make_h5(path, n=4, drop=None, bad_signal_shape=False, bad_labels=False, seed=7):
    rng = np.random.default_rng(seed)
    shape = (n, 5000, 11) if bad_signal_shape else (n, 5000, 12)
    labels = (rng.random((n, 12)) < 0.3).astype(np.uint8)
    for j in range(12):
        labels[j % n, j] = 1
    if bad_labels:
        labels[0, 0] = 2
    with h5py.File(path, "w") as h5:
        if drop != "signals":
            h5.create_dataset("signals", data=rng.normal(size=shape).astype(np.float32))
        if drop != "labels":
            h5.create_dataset("labels", data=labels)
        h5.create_dataset("ages", data=np.full(n, 55.0, dtype=np.float32))
        h5.create_dataset("sexes", data=np.array([b"Male", b"Female"] * ((n + 1) // 2))[:n])
        h5.create_dataset("source_dbs", data=np.array([b"CPSC", b"PTB-XL"] * ((n + 1) // 2))[:n])
        h5.create_dataset("record_names", data=np.array([f"R{i:04d}".encode() for i in range(n)]))
        h5.attrs["classes"] = np.array([c.encode() for c in CLASSES])
        h5.attrs["sampling_rate"] = 500
        h5.attrs["window_seconds"] = 10
        h5.attrs["window_length"] = 5000
        h5.attrs["num_leads"] = 12
        h5.attrs["lead_order"] = ",".join(LEADS)
    return str(path)


def test_valid_h5_passes(tmp_path, capsys):
    assert api.validate_h5(make_h5(tmp_path / "ok.h5")) is True
    out = capsys.readouterr().out
    assert "RESULTADO: OK" in out


def test_missing_file_fails(tmp_path, capsys):
    assert api.validate_h5(str(tmp_path / "nope.h5")) is False
    assert "no encontrado" in capsys.readouterr().out


def test_missing_dataset_fails(tmp_path, capsys):
    assert api.validate_h5(make_h5(tmp_path / "nodata.h5", drop="labels")) is False
    assert "falta" in capsys.readouterr().out


def test_bad_signal_shape_fails(tmp_path, capsys):
    assert api.validate_h5(make_h5(tmp_path / "badshape.h5", bad_signal_shape=True)) is False
    assert "RESULTADO" in capsys.readouterr().out


def test_invalid_label_values_fail(tmp_path, capsys):
    assert api.validate_h5(make_h5(tmp_path / "badlab.h5", bad_labels=True)) is False
    assert "distintos de 0/1" in capsys.readouterr().out
