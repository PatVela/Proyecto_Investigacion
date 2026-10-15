"""Búsqueda de thresholds_validation.csv / temperatures_validation.csv en la webapp."""

import pandas as pd
import pytest

pytest.importorskip("flask")

from ecg import load, network
from webapp import app as app_mod

NAMES = load.CLASS_NAMES
LEGACY = load.LEGACY_CLASS_NAMES


def _csv(path, names, column, value):
    pd.DataFrame({"class": names, column: [value + i * 0.01 for i, _ in enumerate(names)]}).to_csv(path, index=False)
    return path


@pytest.fixture()
def env(monkeypatch, tmp_path):
    for var in ("ECG_THRESHOLDS", "ECG_TEMPERATURES", "ECG_TEMPERATURES_B", "ECG_EVAL_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(app_mod, "REPO_ROOT", tmp_path / "repo")      # sin carpeta runs/ por defecto
    return monkeypatch


def _checkpoint(tmp_path):
    import torch

    d = tmp_path / "ckpt"
    d.mkdir()
    torch.save({"model_state_dict": network.build_network(num_leads=12, num_classes=12).state_dict(),
                "class_names": NAMES, "label_schema": load.LABEL_SCHEMA}, str(d / "best.pt"))
    return d / "best.pt"


def test_nothing_found_returns_none_pair(env, tmp_path):
    ckpt = _checkpoint(tmp_path)
    assert app_mod._load_thresholds_for_app(str(ckpt), str(tmp_path)) == (None, None)
    assert app_mod._load_temperatures_for_app(str(ckpt), str(tmp_path)) == (None, None)


def test_explicit_env_path_wins_over_eval_dir(env, tmp_path):
    ckpt = _checkpoint(tmp_path)
    eval_dir = tmp_path / "eval"; eval_dir.mkdir()
    _csv(eval_dir / "thresholds_validation.csv", NAMES, "threshold", 0.1)
    explicit = _csv(tmp_path / "mine.csv", NAMES, "threshold", 0.6)
    env.setenv("ECG_EVAL_DIR", str(eval_dir)); env.setenv("ECG_THRESHOLDS", str(explicit))
    values, source = app_mod._load_thresholds_for_app(str(ckpt), str(tmp_path))
    assert source == str(explicit.resolve()) and values[NAMES[0]] == pytest.approx(0.6)


def test_file_next_to_checkpoint_is_found(env, tmp_path):
    ckpt = _checkpoint(tmp_path)
    side = _csv(ckpt.parent / "temperatures_validation.csv", NAMES, "temperature", 1.2)
    values, source = app_mod._load_temperatures_for_app(str(ckpt), str(tmp_path))
    assert source == str(side.resolve()) and values[NAMES[1]] == pytest.approx(1.21)


def test_eval_dir_is_searched(env, tmp_path):
    ckpt = _checkpoint(tmp_path)
    eval_dir = tmp_path / "eval"; eval_dir.mkdir()
    _csv(eval_dir / "thresholds_validation.csv", NAMES, "threshold", 0.3)
    env.setenv("ECG_EVAL_DIR", str(eval_dir))
    values, source = app_mod._load_thresholds_for_app(str(ckpt), str(tmp_path))
    assert values and source.endswith("thresholds_validation.csv")


def test_invalid_candidates_are_skipped_in_favor_of_a_valid_one(env, tmp_path):
    ckpt = _checkpoint(tmp_path)
    bad_cols = tmp_path / "bad_cols.csv"; pd.DataFrame({"x": [1]}).to_csv(bad_cols, index=False)
    missing_class = _csv(tmp_path / "missing.csv", NAMES[:-1], "threshold", 0.5)
    good = _csv(tmp_path / "eval_thr.csv", NAMES, "threshold", 0.4)
    # el explícito es inválido -> se cae al siguiente candidato (carpeta del checkpoint)
    _csv(ckpt.parent / "thresholds_validation.csv", NAMES, "threshold", 0.4)
    for broken in (bad_cols, missing_class, tmp_path / "does_not_exist.csv"):
        env.setenv("ECG_THRESHOLDS", str(broken))
        values, source = app_mod._load_thresholds_for_app(str(ckpt), str(tmp_path))
        assert values is not None and "thresholds_validation.csv" in source
    assert good.exists()


def test_legacy_class_names_accepted_for_thresholds_but_not_temperatures(env, tmp_path):
    ckpt = _checkpoint(tmp_path)
    _csv(ckpt.parent / "thresholds_validation.csv", LEGACY, "threshold", 0.5)
    _csv(ckpt.parent / "temperatures_validation.csv", LEGACY, "temperature", 1.0)
    thr, _ = app_mod._load_thresholds_for_app(str(ckpt), str(tmp_path))
    tmp, _ = app_mod._load_temperatures_for_app(str(ckpt), str(tmp_path))
    assert thr is not None and set(thr) == set(LEGACY)
    assert tmp is None


def test_model_b_uses_its_own_env_var(env, tmp_path):
    ckpt = _checkpoint(tmp_path)
    a = _csv(tmp_path / "a.csv", NAMES, "temperature", 1.0)
    b = _csv(tmp_path / "b.csv", NAMES, "temperature", 2.0)
    env.setenv("ECG_TEMPERATURES", str(a)); env.setenv("ECG_TEMPERATURES_B", str(b))
    va, _ = app_mod._load_temperatures_for_app(str(ckpt), str(tmp_path))
    vb, sb = app_mod._load_temperatures_for_app(str(ckpt), str(tmp_path), env_var="ECG_TEMPERATURES_B")
    assert va[NAMES[0]] == pytest.approx(1.0) and vb[NAMES[0]] == pytest.approx(2.0) and sb == str(b.resolve())


def test_the_parameter_count_excludes_the_batchnorm_buffers(env, tmp_path):
    ckpt = _checkpoint(tmp_path)
    model = network.build_network(num_leads=12, num_classes=12)
    info = app_mod._checkpoint_info(str(ckpt), str(tmp_path))
    assert info["num_parameters"] == {"total": sum(p.numel() for p in model.parameters())}
