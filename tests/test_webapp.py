"""Tests de humo de la webapp Flask (sin checkpoint ni resultados)."""

import pytest

flask = pytest.importorskip("flask")

from webapp.app import create_app


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert "temperatures_found" in data


def test_index_renders(client):
    html = client.get("/").get_data(as_text=True)
    assert "Clasificador de ECG" in html
    assert "Curvas ROC/PR interactivas" in html
    assert "Historial de sesión" in html
    assert "results/cinc2020" not in html


def test_metrics_endpoint_shape(client):
    data = client.get("/metrics").get_json()
    assert data["ok"] is True
    assert "per_class_test" in data


def test_curves_endpoint_empty_without_data(client):
    data = client.get("/curves").get_json()
    assert data["ok"] is True
    assert data["found"] is False
    assert data["classes"] == []


def test_extra_dirs_are_found_first(tmp_path, monkeypatch):
    import pandas as pd

    from webapp import app as app_mod

    (tmp_path / "metrics_global.csv").write_text(
        "split,samples,f1_macro,f1_micro,auroc_macro,auprc_macro,exact_match_ratio\n"
        "test,10,0.5,0.6,0.7,0.5,0.4\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ECG_EVAL_DIR", str(tmp_path))
    assert app_mod._load_metrics()["found"] is True

    exp = tmp_path / "exp"
    exp.mkdir()
    pd.DataFrame([{"model": "resnet", "split": "test", "samples": 10, "f1_macro": 0.6,
                   "f1_micro": 0.7, "sensitivity_macro": 0.6, "specificity_macro": 0.9,
                   "auroc_macro": 0.8}]).to_csv(exp / "model_comparison.csv", index=False)
    monkeypatch.setenv("ECG_EXP_DIR", str(exp))
    experiments = app_mod._load_experiment_results()
    assert experiments["comparison_found"] is True
    assert experiments["comparison_winner"] == "resnet"
