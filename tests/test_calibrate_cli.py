"""calibrate.py --method platt: ajusta (a, b) por clase sin tocar los archivos de la calibración por temperatura."""

import json

import numpy as np
import pandas as pd
import pytest

from ecg.calibration import expected_calibration_error, load_platt_csv
from examples.cinc2020 import calibrate

CLASSES = ["NSR", "AF", "MI"]


def _predictions_csv(path, n=20000, a_true=(0.8, 1.2, 1.0), b_true=(-1.5, 0.0, -2.0), seed=4):
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, size=(n, 3))
    y = (rng.random(z.shape) < 1 / (1 + np.exp(-(z * np.asarray(a_true) + np.asarray(b_true))))).astype(int)
    payload = {"record_name": [f"R{i}" for i in range(n)]}
    for j, c in enumerate(CLASSES):
        payload[f"true_{c}"] = y[:, j]
        payload[f"prob_{c}"] = 1 / (1 + np.exp(-z[:, j]))
    pd.DataFrame(payload).to_csv(path, index=False)
    return y, 1 / (1 + np.exp(-z))


def test_platt_method_writes_its_own_files_and_improves_calibration(tmp_path, monkeypatch):
    y, p = _predictions_csv(tmp_path / "predictions_validation.csv")
    out = tmp_path / "cal"
    monkeypatch.setattr("sys.argv", ["calibrate", "--predictions", str(tmp_path / "predictions_validation.csv"),
                                     "--output-dir", str(out), "--method", "platt"])
    calibrate.main()
    a, b = load_platt_csv(out / "platt_validation.csv", CLASSES)
    np.testing.assert_allclose(a, [0.8, 1.2, 1.0], atol=0.06)
    np.testing.assert_allclose(b, [-1.5, 0.0, -2.0], atol=0.1)
    report = json.loads((out / "calibration_report_platt.json").read_text(encoding="utf-8"))
    assert report["method"] == "platt" and report["ece_after"] < report["ece_before"] / 4
    assert report["ece_before"] == pytest.approx(expected_calibration_error(y.astype(float), p), abs=1e-9)
    assert not (out / "temperatures_validation.csv").exists() and not (out / "calibration_report.json").exists()


def test_default_method_is_still_temperature_scaling(tmp_path, monkeypatch):
    _predictions_csv(tmp_path / "predictions_validation.csv", n=2000)
    out = tmp_path / "cal"
    monkeypatch.setattr("sys.argv", ["calibrate", "--predictions", str(tmp_path / "predictions_validation.csv"),
                                     "--output-dir", str(out)])
    calibrate.main()
    assert (out / "temperatures_validation.csv").exists() and (out / "calibration_report.json").exists()
    assert not (out / "platt_validation.csv").exists()
