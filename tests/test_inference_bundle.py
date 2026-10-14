"""Bundle de inferencia: mismas predicciones que el checkpoint original, más pequeño y sin rutas locales."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from ecg import load, network, predict
from tools import export_inference_bundle as bundle

NAMES = load.CLASS_NAMES


def _checkpoint(path, seed, with_optimizer=True, schema=load.LABEL_SCHEMA):
    torch.manual_seed(seed)
    model = network.build_network(num_leads=12, num_classes=12)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.zeros(2, 12, 5000)).sum().backward()
    optimizer.step()                                              # crea estado de Adam (pesa el doble que el modelo)
    payload = {"model_state_dict": model.state_dict(), "class_names": NAMES.copy(), "num_classes": 12,
               "config": {"conv_dropout": 0.3, "train": r"C:\Users\someone\data\train.h5", "save_dir": "saved/x"},
               "epoch": 7, "val_loss": 0.42, "problem_type": "multilabel_sigmoid_bce"}
    if schema:
        payload["label_schema"] = schema
    if with_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, str(path))
    return path


def _eval_dir(path):
    path.mkdir()
    pd.DataFrame({"class": NAMES, "threshold": [0.5] * 12}).to_csv(path / "thresholds_validation.csv", index=False)
    pd.DataFrame({"split": ["test"], "f1_macro": [0.7]}).to_csv(path / "metrics_global.csv", index=False)
    pd.DataFrame({"class": NAMES, "split": ["test"] * 12, "f1": [0.5] * 12}).to_csv(path / "metrics_per_class.csv", index=False)
    pd.DataFrame({"class": NAMES, "fpr": [0.0] * 12, "tpr": [1.0] * 12}).to_csv(path / "roc_curves_test.csv", index=False)
    pd.DataFrame({"record_name": ["x"]}).to_csv(path / "predictions_test.csv", index=False)      # no debe copiarse
    (path / "evaluation_summary.json").write_text(json.dumps({
        "checkpoint_a": r"C:\Users\juerg\Investigacion\saved\a\best.pt", "checkpoint_b": "/home/me/saved/b/best.pt",
        "temperatures_a": "eval-resnet/temperatures_validation.csv", "temperatures_b": "eval-resnet-v2/temperatures_validation.csv",
        "ensemble_alpha": 0.6,
        "test": {"f1_macro": 0.7}}), encoding="utf-8")
    return path


def _temps(path):
    pd.DataFrame({"class": NAMES, "temperature": [1.1] * 12}).to_csv(path, index=False)
    return path


@pytest.fixture()
def built(tmp_path):
    a, b = _checkpoint(tmp_path / "a.pt", 1), _checkpoint(tmp_path / "b.pt", 2)
    reports = tmp_path / "reports"; reports.mkdir()
    pd.DataFrame({"model": ["a"]}).to_csv(reports / "model_comparison.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(reports / "bootstrap_ignored.csv", index=False)
    out = tmp_path / "out"
    manifest = bundle.build_bundle(a, b, _eval_dir(tmp_path / "ensemble-f1"), _temps(tmp_path / "ta.csv"),
                                   _temps(tmp_path / "tb.csv"), reports, out_root=out)
    return tmp_path, out, manifest


def test_bundle_predictions_equal_original_checkpoint(built):
    tmp_path, out, _ = built
    x = np.random.default_rng(0).normal(0, 0.3, (5000, 12)).astype(np.float32)
    for name, src in (("resnet34_v1.pt", "a.pt"), ("resnet34_v2.pt", "b.pt")):
        m_old, _, _ = predict.load_model(tmp_path / src, device=torch.device("cpu"))
        m_new, ck, classes = predict.load_model(out / "models" / name, device=torch.device("cpu"))
        assert classes == NAMES and ck["label_schema"] == load.LABEL_SCHEMA
        assert np.array_equal(predict.predict_array(m_old, x), predict.predict_array(m_new, x))


def test_bundle_is_smaller_and_loads_without_pickle_execution(built):
    tmp_path, out, _ = built
    full, slim = (tmp_path / "a.pt").stat().st_size, (out / "models" / "resnet34_v1.pt").stat().st_size
    assert slim < 0.6 * full                                      # sin estado de Adam
    ckpt = torch.load(out / "models" / "resnet34_v1.pt", map_location="cpu", weights_only=True)
    assert set(ckpt) == {"model_state_dict", "class_names", "label_schema", "config", "epoch", "val_loss", "problem_type"}
    assert ckpt["config"]["conv_dropout"] == 0.3 and "train" not in ckpt["config"]    # sin rutas de datos


def test_manifest_matches_files_on_disk(built):
    _, out, manifest = built
    for key, fname in (("a", "resnet34_v1.pt"), ("b", "resnet34_v2.pt")):
        meta = manifest["models"][key]
        assert meta["sha256"] == hashlib.sha256((out / "models" / fname).read_bytes()).hexdigest()
        assert meta["bytes"] == (out / "models" / fname).stat().st_size and meta["epoch"] == 7
    on_disk = json.loads((out / "deploy" / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["ensemble_alpha"] == 0.6 and on_disk["label_schema"] == load.LABEL_SCHEMA


def test_published_json_has_no_local_paths(built):
    _, out, _ = built
    summary = json.loads((out / "deploy/results/ensemble-f1/evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["checkpoint_a"] == "best.pt" and summary["checkpoint_b"] == "best.pt"
    assert summary["temperatures_a"] == "temperatures_validation.csv" and summary["test"] == {"f1_macro": 0.7}
    text = "".join(p.read_text(encoding="utf-8") for p in (out / "deploy").rglob("*") if p.suffix in {".json", ".env"})
    assert "juerg" not in text and "C:\\" not in text and "/home/" not in text


def test_only_needed_result_files_are_copied(built):
    _, out, _ = built
    copied = {p.name for p in (out / "deploy/results/ensemble-f1").iterdir()}
    assert "predictions_test.csv" not in copied and {"metrics_global.csv", "thresholds_validation.csv", "roc_curves_test.csv"} <= copied
    assert {p.name for p in (out / "deploy/reports").iterdir()} == {"model_comparison.csv"}


def test_env_file_points_at_bundle_paths(built):
    _, out, _ = built
    env = dict(line.split("=", 1) for line in (out / "deploy/ensemble.env").read_text(encoding="utf-8").splitlines()
               if line and not line.startswith("#"))
    assert env["ECG_MODEL"] == "models/resnet34_v1.pt" and env["ECG_MODEL_B"] == "models/resnet34_v2.pt"
    assert env["ECG_SAVED"] == "models" and env["ECG_ALPHA"] == "0.6"
    assert env["ECG_THRESHOLDS"] == "deploy/results/ensemble-f1/thresholds_validation.csv"
    for key in ("ECG_TEMPERATURES", "ECG_TEMPERATURES_B", "ECG_EVAL_DIR", "ECG_EXP_DIR"):
        assert not env[key].startswith(("/", "C:")), key                   # rutas relativas al repo


def test_refuses_models_without_schema_or_with_mixed_schemas(tmp_path):
    no_schema = _checkpoint(tmp_path / "n.pt", 1, schema=None)
    with pytest.raises(ValueError, match="label_schema"):
        bundle.export_model(no_schema, tmp_path / "x.pt")
    a, b = _checkpoint(tmp_path / "a.pt", 1), _checkpoint(tmp_path / "b.pt", 2, schema=load.LEGACY_LABEL_SCHEMA)
    with pytest.raises(ValueError, match="esquemas distintos"):
        bundle.build_bundle(a, b, _eval_dir(tmp_path / "ev"), _temps(tmp_path / "ta.csv"), _temps(tmp_path / "tb.csv"),
                            out_root=tmp_path / "o")


def test_missing_required_result_files_fail_loudly(tmp_path):
    (tmp_path / "ev").mkdir()
    with pytest.raises(FileNotFoundError, match="Faltan"):
        bundle.export_results(tmp_path / "ev", tmp_path / "out")


@pytest.mark.parametrize("raw,expected", [(r"C:\Users\x\a\best.pt", "best.pt"), ("/home/x/a/best.pt", "best.pt"), ("best.pt", "best.pt")])
def test_base_name_handles_windows_and_posix(raw, expected):
    assert bundle._base_name(raw) == expected


# --- calibración afín (Platt) en el bundle ---------------------------------------------------------------------------

def _platt(path, a=1.1):
    pd.DataFrame({"class": NAMES, "a": [a] * 12, "b": [-0.5] * 12}).to_csv(path, index=False)
    return path


@pytest.fixture()
def built_platt(tmp_path):
    a, b = _checkpoint(tmp_path / "a.pt", 1), _checkpoint(tmp_path / "b.pt", 2)
    eval_dir = _eval_dir(tmp_path / "ensemble-f1-platt")
    summary = json.loads((eval_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    summary.update({"temperatures_a": None, "temperatures_b": None,
                    "platt_a": r"C:\Users\juerg\x\platt_validation.csv", "platt_b": "/home/me/y/platt_validation.csv",
                    "platt_ensemble": r"C:\Users\juerg\x\platt_ensemble_validation.csv", "recalibrate_ensemble": True})
    (eval_dir / "evaluation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    out = tmp_path / "out"
    manifest = bundle.build_bundle(a, b, eval_dir, platt_a=_platt(tmp_path / "pa.csv", 1.1),
                                   platt_b=_platt(tmp_path / "pb.csv", 1.2), platt_ensemble=_platt(tmp_path / "pe.csv", 1.3),
                                   out_root=out)
    return tmp_path, out, manifest


def test_platt_bundle_ships_the_three_calibration_files_and_no_temperatures(built_platt):
    _, out, manifest = built_platt
    for name, a in (("platt_a.csv", 1.1), ("platt_b.csv", 1.2), ("platt_ensemble.csv", 1.3)):
        assert pd.read_csv(out / "deploy" / name)["a"].iloc[0] == pytest.approx(a)
    assert not (out / "deploy" / "temperatures_a.csv").exists()
    assert manifest["calibration"]["method"] == "platt+ensemble"


def test_platt_env_points_at_relative_bundle_paths(built_platt):
    _, out, _ = built_platt
    env = dict(line.split("=", 1) for line in (out / "deploy/ensemble.env").read_text(encoding="utf-8").splitlines()
               if line and not line.startswith("#"))
    assert env["ECG_PLATT"] == "deploy/platt_a.csv" and env["ECG_PLATT_B"] == "deploy/platt_b.csv"
    assert env["ECG_PLATT_ENSEMBLE"] == "deploy/platt_ensemble.csv"
    assert "ECG_TEMPERATURES" not in env and "ECG_TEMPERATURES_B" not in env


def test_platt_paths_are_sanitized_in_the_published_summary(built_platt):
    _, out, _ = built_platt
    summary = json.loads((out / "deploy/results/ensemble-f1-platt/evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["platt_a"] == "platt_validation.csv" and summary["platt_ensemble"] == "platt_ensemble_validation.csv"
    text = "".join(p.read_text(encoding="utf-8") for p in (out / "deploy").rglob("*") if p.suffix in {".json", ".env"})
    assert "juerg" not in text and "C:\\" not in text and "/home/" not in text


def test_every_model_needs_a_calibration_and_platt_files_are_validated(tmp_path):
    a, b = _checkpoint(tmp_path / "a.pt", 1), _checkpoint(tmp_path / "b.pt", 2)
    ev = _eval_dir(tmp_path / "ev")
    with pytest.raises(ValueError, match="calibración"):
        bundle.build_bundle(a, b, ev, platt_a=_platt(tmp_path / "pa.csv"), out_root=tmp_path / "o1")
    pd.DataFrame({"class": NAMES, "temperature": [1.0] * 12}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="class,a,b"):
        bundle.build_bundle(a, b, ev, platt_a=tmp_path / "bad.csv", platt_b=_platt(tmp_path / "pb.csv"), out_root=tmp_path / "o2")


# --- la evaluación debe corresponder a la calibración que se empaqueta -------------------------------------------------

def _platt_summary(eval_dir, **recorded):
    """Resumen de una evaluación hecha con calibración afín; ``recorded`` = rutas que dice haber usado."""
    summary = json.loads((eval_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    summary.update({"temperatures_a": None, "temperatures_b": None, "platt_a": None, "platt_b": None, "platt_ensemble": None})
    summary.update({k: str(v) for k, v in recorded.items()})
    (eval_dir / "evaluation_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_refuses_an_eval_dir_computed_with_another_calibration(tmp_path):
    """Umbrales de la evaluación con temperatura (MI 0,56) no valen para probabilidades con calibración afín (MI 0,33)."""
    a, b = _checkpoint(tmp_path / "a.pt", 1), _checkpoint(tmp_path / "b.pt", 2)
    ev = _eval_dir(tmp_path / "ensemble-f1")                          # su resumen declara temperaturas A y B
    with pytest.raises(ValueError) as error:
        bundle.build_bundle(a, b, ev, platt_a=_platt(tmp_path / "pa.csv"), platt_b=_platt(tmp_path / "pb.csv"),
                            out_root=tmp_path / "out")
    message = str(error.value)
    assert "temperatures_a" in message and "platt_a" in message and "no corresponde" in message
    assert not (tmp_path / "out" / "models").exists()                  # falla antes de escribir nada


def test_refuses_an_unevaluated_ensemble_recalibration_and_a_stale_calibration_file(tmp_path):
    a, b = _checkpoint(tmp_path / "a.pt", 1), _checkpoint(tmp_path / "b.pt", 2)
    ev = _eval_dir(tmp_path / "ensemble-f1-platt")
    _platt_summary(ev, platt_a=_platt(tmp_path / "recorded_a.csv", a=1.1), platt_b=_platt(tmp_path / "recorded_b.csv", a=1.2))
    kwargs = dict(platt_b=_platt(tmp_path / "pb.csv", a=1.2), out_root=tmp_path / "out")
    with pytest.raises(ValueError, match="platt_ensemble"):             # se empaqueta una recalibración que no se evaluó
        bundle.build_bundle(a, b, ev, platt_a=_platt(tmp_path / "pa.csv", a=1.1), platt_ensemble=_platt(tmp_path / "pe.csv"), **kwargs)
    with pytest.raises(ValueError, match="ya no tiene el contenido"):  # recalibrado después de evaluar
        bundle.build_bundle(a, b, ev, platt_a=_platt(tmp_path / "pa2.csv", a=1.5), **kwargs)


def test_accepts_an_eval_dir_computed_with_exactly_that_calibration(tmp_path):
    a, b = _checkpoint(tmp_path / "a.pt", 1), _checkpoint(tmp_path / "b.pt", 2)
    ev = _eval_dir(tmp_path / "ensemble-f1-platt")
    _platt_summary(ev, platt_a=_platt(tmp_path / "recorded_a.csv", a=1.1), platt_b=_platt(tmp_path / "recorded_b.csv", a=1.2))
    manifest = bundle.build_bundle(a, b, ev, platt_a=_platt(tmp_path / "pa.csv", a=1.1), platt_b=_platt(tmp_path / "pb.csv", a=1.2),
                                   out_root=tmp_path / "out")
    assert manifest["calibration"]["method"] == "platt"


def test_the_committed_bundle_passes_its_own_check():
    """El bundle versionado es coherente consigo mismo (las rutas de runs/ del resumen pueden no existir en otra máquina)."""
    deploy = bundle.REPO_ROOT / "deploy"
    supplied = {"temperatures_a": None, "temperatures_b": None, "platt_a": deploy / "platt_a.csv",
                "platt_b": deploy / "platt_b.csv", "platt_ensemble": deploy / "platt_ensemble.csv"}
    bundle.check_eval_matches_calibration(deploy / "results" / "ensemble-f1-platt", supplied)


def test_refuses_a_blend_weight_other_than_the_one_the_thresholds_were_chosen_with(tmp_path):
    a, b = _checkpoint(tmp_path / "a.pt", 1), _checkpoint(tmp_path / "b.pt", 2)
    ev = _eval_dir(tmp_path / "ensemble-f1")                          # su resumen dice ensemble_alpha = 0.6
    with pytest.raises(ValueError, match="ensemble_alpha.*0.6.*0.7"):
        bundle.build_bundle(a, b, ev, _temps(tmp_path / "ta.csv"), _temps(tmp_path / "tb.csv"), alpha=0.7, out_root=tmp_path / "out")
    assert bundle.build_bundle(a, b, ev, _temps(tmp_path / "ta.csv"), _temps(tmp_path / "tb.csv"), alpha=0.6,
                               out_root=tmp_path / "ok")["ensemble_alpha"] == 0.6
