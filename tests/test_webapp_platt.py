"""Calibración afín (Platt) en la webapp: aritmética exacta, ensemble con recalibración y cargadores."""

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("flask")

from ecg import load, network
from ecg.calibration import apply_platt
from webapp import app as app_mod
from webapp import prediction as pred

NAMES = load.CLASS_NAMES


def _tiny_checkpoint(path, seed):
    torch.manual_seed(seed)
    model = network.build_network(num_leads=12, num_classes=12)
    torch.save({"model_state_dict": model.state_dict(),
                "config": {"label_schema": load.LABEL_SCHEMA, "norm_mode": "physical", "bandpass": True},
                "class_names": NAMES.copy(), "label_schema": load.LABEL_SCHEMA, "epoch": 1, "val_loss": 0.5}, str(path))
    return str(path)


def _csv_signal(path):
    signal = np.random.default_rng(4).normal(0, 0.3, (load.WINDOW_LENGTH, load.NUM_LEADS))
    np.savetxt(str(path), signal, delimiter=",", header=",".join(load.STANDARD_LEAD_ORDER), comments="")
    return path


def _platt(a, b):
    return {name: {"a": a + 0.01 * i, "b": b - 0.01 * i} for i, name in enumerate(NAMES)}


def _vectors(platt):
    return (np.array([platt[n]["a"] for n in NAMES]), np.array([platt[n]["b"] for n in NAMES]))


@pytest.fixture()
def setup(tmp_path):
    a = _tiny_checkpoint(tmp_path / "a.pt", 1)
    b = _tiny_checkpoint(tmp_path / "b.pt", 2)
    csv = _csv_signal(tmp_path / "rec.csv")
    return a, b, csv, tmp_path


def test_single_model_platt_is_sigmoid_a_logit_plus_b_of_the_raw_probability(setup):
    ckpt_a, _, csv, tmp = setup
    plain = pred.run_prediction_from_saved_paths([csv], ckpt_a, None, tmp / "r0")
    cal = pred.run_prediction_from_saved_paths([csv], ckpt_a, None, tmp / "r1", platt=_platt(0.7, -1.2), platt_source="pa")
    a, b = _vectors(_platt(0.7, -1.2))
    np.testing.assert_allclose(cal["per_window_probabilities"][0],
                               apply_platt(np.asarray(plain["per_window_probabilities"][0])[None, :], a, b)[0], atol=1e-6)
    assert cal["calibrated"] is True and cal["calibration_method"] == "platt" and cal["platt_source"] == "pa"
    assert plain["calibration_method"] is None and plain["platt"] is None


def test_platt_wins_over_temperatures_for_the_same_model(setup):
    ckpt_a, _, csv, tmp = setup
    only_platt = pred.run_prediction_from_saved_paths([csv], ckpt_a, None, tmp / "r0", platt=_platt(0.8, -0.5))
    both = pred.run_prediction_from_saved_paths([csv], ckpt_a, None, tmp / "r1", platt=_platt(0.8, -0.5),
                                                temperatures={n: 0.05 for n in NAMES})
    np.testing.assert_allclose(only_platt["per_window_probabilities"], both["per_window_probabilities"])
    assert both["calibration_method"] == "platt" and both["temperatures"] is None


def test_ensemble_platt_per_model_then_recalibrated_average(setup):
    ckpt_a, ckpt_b, csv, tmp = setup
    plain_a = pred.run_prediction_from_saved_paths([csv], ckpt_a, None, tmp / "ra")
    plain_b = pred.run_prediction_from_saved_paths([csv], ckpt_b, None, tmp / "rb")
    pa, pb, pe = _platt(0.9, -0.6), _platt(1.1, -0.9), _platt(1.05, -0.3)
    run = pred.run_prediction_from_saved_paths(
        [csv], ckpt_a, None, tmp / "re", model_path_b=ckpt_b, alpha=0.6,
        platt=pa, platt_b=pb, platt_ensemble=pe, platt_source_ensemble="pe")
    cal_a = apply_platt(np.asarray(plain_a["per_window_probabilities"][0])[None, :], *_vectors(pa))[0]
    cal_b = apply_platt(np.asarray(plain_b["per_window_probabilities"][0])[None, :], *_vectors(pb))[0]
    expected = apply_platt((0.6 * cal_a + 0.4 * cal_b)[None, :], *_vectors(pe))[0]
    np.testing.assert_allclose(run["per_window_probabilities"][0], expected, atol=1e-5)
    np.testing.assert_allclose([r["probability"] for r in sorted(run["predictions"], key=lambda r: NAMES.index(r["class"]))],
                               expected, atol=1e-5)
    assert run["calibration_method"] == "platt+ensemble" and run["platt_source_ensemble"] == "pe"
    assert "calibración afín" in run["technical_details"]["calibration"]


def test_platt_commutes_with_the_max_over_windows():
    """Por qué la webapp puede calibrar el vector ya agregado y el de cada ventana por separado."""
    rng = np.random.default_rng(0)
    p = rng.random((4, 12))
    a, b = np.full(12, 0.7), np.full(12, -0.8)
    np.testing.assert_allclose(apply_platt(p.max(axis=0)[None, :], a, b)[0], apply_platt(p, a, b).max(axis=0))


# --- cargadores y /health --------------------------------------------------------------------------------------------

def _platt_csv(path, names=NAMES, a=0.9, b=-1.0):
    pd.DataFrame({"class": names, "a": [a + 0.01 * i for i, _ in enumerate(names)],
                  "b": [b] * len(names)}).to_csv(path, index=False)
    return path


@pytest.fixture()
def env(monkeypatch, tmp_path):
    for var in ("ECG_THRESHOLDS", "ECG_TEMPERATURES", "ECG_TEMPERATURES_B", "ECG_EVAL_DIR", "ECG_PLATT", "ECG_PLATT_B",
                "ECG_PLATT_ENSEMBLE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(app_mod, "REPO_ROOT", tmp_path / "repo")
    return monkeypatch


def test_loader_reads_a_and_b_from_the_explicit_env_path(env, tmp_path):
    ckpt = _tiny_checkpoint(tmp_path / "best.pt", 1)
    path = _platt_csv(tmp_path / "platt.csv")
    env.setenv("ECG_PLATT", str(path))
    values, source = app_mod._load_platt_for_app(ckpt, str(tmp_path))
    assert source == str(path.resolve())
    assert values[NAMES[0]] == {"a": pytest.approx(0.9), "b": pytest.approx(-1.0)}


def test_loader_ignores_files_without_all_classes_or_without_columns(env, tmp_path):
    ckpt = _tiny_checkpoint(tmp_path / "best.pt", 1)
    env.setenv("ECG_PLATT", str(_platt_csv(tmp_path / "partial.csv", names=NAMES[:5])))
    assert app_mod._load_platt_for_app(ckpt, str(tmp_path)) == (None, None)
    pd.DataFrame({"class": NAMES, "temperature": [1.0] * 12}).to_csv(tmp_path / "t.csv", index=False)
    env.setenv("ECG_PLATT", str(tmp_path / "t.csv"))
    assert app_mod._load_platt_for_app(ckpt, str(tmp_path)) == (None, None)


def test_model_b_and_ensemble_use_their_own_env_vars(env, tmp_path):
    ckpt = _tiny_checkpoint(tmp_path / "best.pt", 1)
    env.setenv("ECG_PLATT_B", str(_platt_csv(tmp_path / "b.csv", a=2.0)))
    env.setenv("ECG_PLATT_ENSEMBLE", str(_platt_csv(tmp_path / "e.csv", a=3.0)))
    assert app_mod._load_platt_for_app(ckpt, str(tmp_path)) == (None, None)
    vb, _ = app_mod._load_platt_for_app(ckpt, str(tmp_path), env_var="ECG_PLATT_B")
    ve, _ = app_mod._load_platt_for_app(ckpt, str(tmp_path), env_var="ECG_PLATT_ENSEMBLE",
                                        filename="platt_ensemble_validation.csv")
    assert vb[NAMES[0]]["a"] == pytest.approx(2.0) and ve[NAMES[0]]["a"] == pytest.approx(3.0)


def test_health_reports_platt(env, tmp_path):
    env.setenv("ECG_PLATT", str(_platt_csv(tmp_path / "platt.csv")))
    app = app_mod.create_app()
    app.config["TESTING"] = True
    data = app.test_client().get("/health").get_json()
    assert data["platt_found"] is True and data["platt_source"].endswith("platt.csv")
    assert data["platt_ensemble_found"] is False
