"""Fases tools/audit_signal_scaling con registros WFDB sintéticos."""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import audit_signal_scaling as api

HEA = """R001 2 500 100
R001.mat 16 1000(0)/mV 16 0 0 0 V1
R001.mat 16 1000(0)/mV 16 0 0 0 V2
"""


def make_record(root, name="R001", mat_dict="default"):
    d = Path(root) / "training" / "cpsc"
    d.mkdir(parents=True, exist_ok=True)
    hea = d / f"{name}.hea"
    hea.write_text(HEA)
    if mat_dict == "default":
        val = np.random.default_rng(7).integers(-500, 500, (2, 100))
        scipy.io.savemat(d / f"{name}.mat", {"val": val})
    elif mat_dict is not None:
        scipy.io.savemat(d / f"{name}.mat", mat_dict)
    return hea


def test_valid_record_ok(tmp_path):
    res = api.analyze_record(make_record(tmp_path))
    assert res["ok"] is True
    assert res["db"] == "cpsc"
    assert res["n_sig"] == 2
    assert res["fs"] == 500.0
    assert len(res["leads"]) == 2
    assert res["leads"][0]["name"] == "V1"
    assert res["leads"][0]["normalized_name"] == "V1"
    assert res["error"] is None


def test_missing_mat_reports_error(tmp_path):
    res = api.analyze_record(make_record(tmp_path, mat_dict=None))
    assert res["ok"] is False
    assert "No existe" in res["error"]


def test_dim_mismatch_reports_error(tmp_path):
    val = np.zeros((3, 100), dtype=np.int16)
    res = api.analyze_record(make_record(tmp_path, mat_dict={"val": val}))
    assert res["ok"] is False
    assert "leads" in res["error"]


def test_missing_val_reports_error(tmp_path):
    res = api.analyze_record(make_record(tmp_path, mat_dict={"foo": 1}))
    assert res["ok"] is False
    assert "val" in res["error"]


def test_validate_phase_raises_directly(tmp_path):
    analyzer = api._RecordAnalyzer(make_record(tmp_path))
    analyzer.load()
    analyzer.parse_header()
    analyzer.digital = np.zeros((5, 100))
    with pytest.raises(ValueError, match="leads"):
        analyzer.validate_dimensions()
