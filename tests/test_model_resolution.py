"""Un checkpoint indicado explícitamente que no existe es un error, nunca un reemplazo silencioso."""

import pytest
import torch

pytest.importorskip("flask")

from ecg import load, network
from webapp import prediction as pred
from webapp.app import create_app


def _ckpt(path, val_loss=0.5):
    torch.save({"model_state_dict": network.build_network(num_leads=12, num_classes=12).state_dict(),
                "class_names": load.CLASS_NAMES, "label_schema": load.LABEL_SCHEMA, "epoch": 1, "val_loss": val_loss}, str(path))
    return path


def test_explicit_existing_path_is_used(tmp_path):
    chosen = _ckpt(tmp_path / "mine.pt", val_loss=9.0)
    _ckpt(tmp_path / "best.pt", val_loss=0.1)
    assert pred.resolve_model_path(str(chosen), str(tmp_path)) == chosen.resolve()


def test_explicit_missing_path_raises_even_if_other_models_exist(tmp_path):
    _ckpt(tmp_path / "best.pt")
    with pytest.raises(FileNotFoundError, match="no existe"):
        pred.resolve_model_path(str(tmp_path / "typo.pt"), str(tmp_path))


def test_no_explicit_path_selects_automatically(tmp_path):
    _ckpt(tmp_path / "a" / "best.pt", 0.9) if (tmp_path / "a").mkdir() is None else None
    _ckpt(tmp_path / "b.pt", 0.2)
    for empty in (None, "", "   "):
        assert pred.resolve_model_path(empty, str(tmp_path)).name in {"b.pt", "best.pt"}


def test_empty_search_dir_reports_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="Directorio de búsqueda"):
        pred.resolve_model_path(None, str(tmp_path))


@pytest.fixture()
def env(monkeypatch, tmp_path):
    for key in ("ECG_MODEL", "ECG_MODEL_B"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ECG_SAVED", str(tmp_path / "saved"))
    monkeypatch.setenv("ECG_UPLOADS", str(tmp_path / "up"))
    monkeypatch.setenv("ECG_RESULTS", str(tmp_path / "res"))
    return monkeypatch


@pytest.mark.parametrize("variable", ["ECG_MODEL", "ECG_MODEL_B"])
def test_app_refuses_to_start_with_a_missing_configured_model(env, tmp_path, variable):
    env.setenv(variable, str(tmp_path / "does_not_exist.pt"))
    with pytest.raises(RuntimeError, match=variable):
        create_app()


def test_app_starts_with_valid_models_and_without_any(env, tmp_path):
    create_app()                                                       # sin modelo configurado: selección automática
    a, b = _ckpt(tmp_path / "a.pt"), _ckpt(tmp_path / "b.pt")
    env.setenv("ECG_MODEL", str(a)); env.setenv("ECG_MODEL_B", str(b))
    assert create_app().config["ECG_MODEL_B"] == str(b)


def test_use_model_with_unknown_path_is_a_404_and_keeps_current_model(env, tmp_path):
    saved = tmp_path / "saved"; saved.mkdir()
    good = _ckpt(saved / "best.pt")
    env.setenv("ECG_MODEL", str(good))
    client = create_app().test_client()
    resp = client.post("/use_model", json={"model": str(saved / "typo.pt")})
    assert resp.status_code == 404 and "no existe" in resp.get_json()["error"]
    assert client.application.config["ECG_MODEL"] == str(good)
