"""Entradas que no son un ECG utilizable: señal plana/corta (se rechaza) y unidades de un CSV con un pico suelto.

Regresiones medidas en la webapp con el bundle real:
  * un CSV todo 0, todo NaN o de 1,2 s recibía diagnósticos (SB 65 %, MI 59 %, TAb 48 %) con un PDF "hallazgos compatibles";
  * un CSV en mV con UNA muestra de 25 mV se leía como cuentas ADC (÷ 1000): el QRS llegaba al modelo con 0,001 mV.
"""

import io

import numpy as np
import pytest
import torch

pytest.importorskip("flask")

from ecg import load, network
from ecg.predict import prepare_csv_windows
from webapp import prediction as pred
from webapp.app import create_app

LEADS = load.STANDARD_LEAD_ORDER


def _ecg(seconds=10, seed=0):
    """ECG sintético en mV: QRS de 1 mV cada 0,8 s, onda lenta y algo de ruido."""
    rng = np.random.default_rng(seed)
    n = load.TARGET_FS * seconds
    t = np.arange(n) / load.TARGET_FS
    wave = 0.1 * np.sin(2 * np.pi * 1.2 * t) + 1.0 * (np.mod(t, 0.8) < 0.02)
    return wave[:, None] + 0.02 * rng.normal(size=(n, 12))


def _csv(path, values):
    np.savetxt(str(path), values, delimiter=",", header=",".join(LEADS), comments="")
    return path


# --- señal inutilizable -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("flat_leads,ok", [(0, True), (6, True), (7, False), (12, False)])
def test_more_than_half_flat_leads_is_rejected(flat_leads, ok):
    signal = _ecg()
    signal[:, :flat_leads] = 0.0
    if ok:
        load.require_usable_signal(signal)
    else:
        with pytest.raises(ValueError, match="plana.*%d de 12" % flat_leads):
            load.require_usable_signal(signal)


def test_too_short_signal_is_rejected_with_its_duration():
    with pytest.raises(ValueError, match=r"dura 1\.2 s"):
        load.require_usable_signal(_ecg(seconds=2)[:600])
    load.require_usable_signal(_ecg(seconds=2))                               # justo el mínimo


def test_nan_signal_counts_as_flat_after_preprocessing():
    with pytest.raises(ValueError, match="plana"):
        load.preprocess_to_windows(np.full((5000, 12), np.nan), 500, load.STANDARD_LEAD_ORDER, units="mv")


def test_a_single_spike_in_a_flat_line_is_still_flat():
    signal = np.zeros((5000, 12))
    signal[2500, :] = 3.0
    with pytest.raises(ValueError, match="plana"):
        load.require_usable_signal(signal)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = network.build_network(num_leads=12, num_classes=12)
    ckpt = tmp_path / "a.pt"
    torch.save({"model_state_dict": model.state_dict(), "class_names": load.CLASS_NAMES.copy(),
                "config": {"label_schema": load.LABEL_SCHEMA, "norm_mode": "physical", "bandpass": True},
                "label_schema": load.LABEL_SCHEMA, "epoch": 1, "val_loss": 0.5}, str(ckpt))
    monkeypatch.setenv("ECG_MODEL", str(ckpt))
    monkeypatch.setenv("ECG_SAVED", str(tmp_path))
    monkeypatch.setenv("ECG_UPLOADS", str(tmp_path / "up"))
    monkeypatch.setenv("ECG_RESULTS", str(tmp_path / "res"))
    monkeypatch.setenv("ECG_DEVICE", "cpu")
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client(), tmp_path


@pytest.mark.parametrize("name,values", [
    ("ceros", np.zeros((5000, 12))),
    ("nan", np.full((5000, 12), np.nan)),
    ("corta", _ecg(seconds=1)[:600]),
], ids=["ceros", "nan", "corta"])
def test_predict_answers_400_instead_of_a_diagnosis(client, name, values):
    http, tmp = client
    buffer = io.BytesIO()
    np.savetxt(buffer, values, delimiter=",", header=",".join(LEADS), comments="")
    r = http.post("/predict", data={"files": (io.BytesIO(buffer.getvalue()), "x.csv")}, content_type="multipart/form-data")
    body = r.get_json()
    assert r.status_code == 400 and body["ok"] is False and ("plana" in body["error"] or "dura" in body["error"])
    assert not list((tmp / "res").glob("*/*.pdf"))                                      # no se generó ningún informe


def test_predict_still_accepts_a_real_looking_csv(client):
    http, _ = client
    buffer = io.BytesIO()
    np.savetxt(buffer, _ecg(), delimiter=",", header=",".join(LEADS), comments="")
    r = http.post("/predict", data={"files": (io.BytesIO(buffer.getvalue()), "x.csv")}, content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["ok"] is True


def test_a_preprocessed_csv_marker_does_not_skip_the_check(tmp_path):
    path = tmp_path / "plano.csv"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Sampling Rate: 500 Hz\n# Preprocessed: physical\n")
        np.savetxt(handle, np.zeros((5000, 12)), delimiter=",", header=",".join(LEADS), comments="")
    with pytest.raises(ValueError, match="plana"):
        prepare_csv_windows(path, norm_mode="physical")


def _marked_csv(path, values):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Sampling Rate: 500 Hz\n# Preprocessed: physical\n")
        np.savetxt(handle, values, delimiter=",", header=",".join(LEADS), comments="")
    return path


def test_nan_in_a_preprocessed_csv_never_reaches_the_model(tmp_path):
    values = _ecg()
    values[100:200, 2] = np.nan
    windows, info = prepare_csv_windows(_marked_csv(tmp_path / "con_nan.csv", values), norm_mode="physical")
    assert info["reused"] is True and np.isfinite(windows).all()
    with pytest.raises(ValueError, match="plana"):
        prepare_csv_windows(_marked_csv(tmp_path / "todo_nan.csv", np.full((5000, 12), np.nan)), norm_mode="physical")


# --- unidades de un CSV ------------------------------------------------------------------------------------------

def test_one_saturation_spike_does_not_turn_millivolts_into_adc_counts():
    clean, spiky = _ecg(), _ecg()
    spiky[1234, 3] = 25.0
    assert load.infer_csv_units(clean) == "mv" and load.infer_csv_units(spiky) == "mv"
    processed = load.preprocess_ecg_array(spiky, 500, LEADS, units=load.infer_csv_units(spiky))
    assert processed[:, 1].max() > 0.8                                            # antes: 0,0012 mV


def test_real_adc_counts_are_still_detected():
    counts = np.round(_ecg() * 1000)                                              # 1 mV = 1000 cuentas
    assert load.infer_csv_units(counts) == "digital"
    assert load.preprocess_ecg_array(counts, 500, LEADS, units="digital")[:, 1].max() > 0.8


def test_units_without_finite_values_default_to_mv():
    assert load.infer_csv_units(np.full((10, 12), np.nan)) == "mv"


def test_assumed_units_are_shown_in_the_technical_details(tmp_path):
    torch.manual_seed(0)
    model = network.build_network(num_leads=12, num_classes=12)
    ckpt = tmp_path / "a.pt"
    torch.save({"model_state_dict": model.state_dict(), "class_names": load.CLASS_NAMES.copy(),
                "config": {"label_schema": load.LABEL_SCHEMA, "norm_mode": "physical", "bandpass": True},
                "label_schema": load.LABEL_SCHEMA, "epoch": 1, "val_loss": 0.5}, str(ckpt))
    spiky = _ecg()
    spiky[10, 0] = 25.0
    run = pred.run_prediction_from_saved_paths([_csv(tmp_path / "s.csv", spiky)], str(ckpt), None, tmp_path / "r")
    assert "unidades del CSV asumidas: mV" in run["technical_details"]["preprocessing"]
