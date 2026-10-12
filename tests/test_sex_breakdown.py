"""Tests del desglose por sexo (examples/cinc2020/sex_breakdown.py)."""

import json

import h5py
import pandas as pd

from sex_breakdown import main as sex_main, normalize_sex


def test_normalize_sex():
    assert normalize_sex("Male") == "Male"
    assert normalize_sex("f") == "Female"
    assert normalize_sex("") == "Unknown"
    assert normalize_sex("Unknown") == "Unknown"


def test_main_groups(tmp_path, monkeypatch):
    names = [f"R{i}" for i in range(10)]
    sexes = ["Male"] * 6 + ["Female"] * 3 + ["Unknown"]
    with h5py.File(tmp_path / "t.h5", "w") as h5:
        h5.create_dataset("record_names", data=[n.encode() for n in names])
        h5.create_dataset("sexes", data=[s.encode() for s in sexes])
    pred = pd.DataFrame({"record_name": names, "true_A": [1] * 10,
                         "prob_A": [0.9] * 10, "pred_A": [1] * 10})
    csv = tmp_path / "p.csv"
    pred.to_csv(csv, index=False)
    out = tmp_path / "s.json"
    monkeypatch.setattr("sys.argv", ["sex_breakdown.py", "--predictions", str(csv),
                                     "--test-h5", str(tmp_path / "t.h5"),
                                     "--output", str(out)])
    sex_main()
    result = json.loads(out.read_text(encoding="utf-8"))
    by_sex = {g["sex"]: g for g in result["groups"]}
    assert by_sex["Male"]["n"] == 6 and by_sex["Female"]["n"] == 3
    assert by_sex["Male"]["f1_macro"] == 1.0
