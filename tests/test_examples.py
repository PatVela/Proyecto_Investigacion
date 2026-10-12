"""Registros de ejemplo: integridad, API segura y coincidencia con la evaluación."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

pytest.importorskip("flask")

from ecg import load, network, schema, signal
from webapp import examples as store

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "webapp" / "examples"
BUNDLE_PRESENT = (REPO_ROOT / "deploy" / "ensemble.env").exists() and (REPO_ROOT / "models").is_dir()


@pytest.fixture(scope="module")
def manifest():
    return json.loads((EXAMPLES_DIR / "examples.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# integridad de los datos incluidos
# --------------------------------------------------------------------------- #
def test_every_example_is_complete_and_consistent_with_its_header(manifest):
    assert len(manifest) >= 6
    assert len({m["id"] for m in manifest}) == len(manifest)
    assert {m["kind"] for m in manifest} == {"correct", "mixed"}
    for m in manifest:
        assert store.SAFE_NAME.match(m["id"]) and m["record"] == m["id"]
        hea, mat = (EXAMPLES_DIR / m["id"] / f"{m['id']}{s}" for s in (".hea", ".mat"))
        assert hea.is_file() and mat.is_file()
        meta = signal.parse_header(hea)
        assert (meta["sampling_freq"], meta["num_samples"], meta["num_leads"]) == (500.0, 5000, 12)
        assert meta["dx_codes"] == m["dx_codes"]
        assert schema.matched_class_names(meta["dx_codes"]) == m["true_classes"] and m["true_classes"]
        assert set(m["reference_probabilities"]) == set(load.CLASS_NAMES)
        assert all(0.0 <= p <= 1.0 for p in m["reference_probabilities"].values())
        assert m["source_db"] != "unknown"


def test_examples_are_small_and_cover_distinct_classes(manifest):
    total = sum(p.stat().st_size for p in EXAMPLES_DIR.rglob("*") if p.is_file())
    assert total < 2 * 1024 * 1024
    singles = {m["true_classes"][0] for m in manifest if m["kind"] == "correct"}
    assert len(singles) >= 5                                            # varias clases distintas
    for m in manifest:
        if m["kind"] == "correct":
            assert m["reference_predicted_classes"] == m["true_classes"] or \
                set(m["reference_predicted_classes"]) == set(m["true_classes"])
        else:
            assert set(m["reference_predicted_classes"]) != set(m["true_classes"])


def test_license_and_citation_travel_with_the_examples():
    text = (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8")
    assert "CC BY 4.0" in text and "10.13026/dvyd-kd57" in text and "Perez Alday" in text


# --------------------------------------------------------------------------- #
# almacén de ejemplos: nunca un identificador se convierte en ruta
# --------------------------------------------------------------------------- #
def test_load_manifest_drops_unsafe_or_incomplete_entries(tmp_path):
    for name in ("good", "nomat"):
        (tmp_path / name).mkdir()
        (tmp_path / name / f"{name}.hea").write_text("x")
    (tmp_path / "good" / "good.mat").write_text("x")
    entries = [{"id": "good", "record": "good"}, {"id": "nomat", "record": "nomat"},
               {"id": "../good", "record": "../good"}, {"id": "good", "record": "other"}, {"id": "a/b", "record": "a/b"}]
    (tmp_path / "examples.json").write_text(json.dumps(entries), encoding="utf-8")
    assert [e["id"] for e in store.load_manifest(tmp_path)] == ["good"]
    assert store.load_manifest(tmp_path / "missing") == []


def test_example_files_only_serves_manifest_ids(tmp_path):
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "ok.hea").write_text("h")
    (tmp_path / "ok" / "ok.mat").write_bytes(b"m")
    (tmp_path / "secret.hea").write_text("secret")
    manifest = [{"id": "ok", "record": "ok"}]
    assert store.example_files(manifest, tmp_path, "ok") == [("ok.hea", b"h"), ("ok.mat", b"m")]
    for bad in ("../secret", "secret", "ok/../ok", ""):
        with pytest.raises(KeyError):
            store.example_files(manifest, tmp_path, bad)


def test_public_listing_hides_reference_data(manifest):
    listing = store.public_listing(manifest)
    assert listing and all(set(item) <= set(store.PUBLIC_FIELDS) for item in listing)
    assert not any("reference_probabilities" in item or "dx_codes" in item for item in listing)


# --------------------------------------------------------------------------- #
# API con un modelo de juguete
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client(monkeypatch, tmp_path):
    from webapp.app import create_app

    for key in [k for k in os.environ if k.startswith("ECG_")]:
        monkeypatch.delenv(key, raising=False)
    torch.save({"model_state_dict": network.build_network(num_leads=12, num_classes=12).state_dict(),
                "class_names": load.CLASS_NAMES, "label_schema": load.LABEL_SCHEMA, "epoch": 1, "val_loss": 0.5},
               str(tmp_path / "m.pt"))
    monkeypatch.setenv("ECG_MODEL", str(tmp_path / "m.pt"))
    monkeypatch.setenv("ECG_SAVED", str(tmp_path / "saved"))
    monkeypatch.setenv("ECG_UPLOADS", str(tmp_path / "up"))
    monkeypatch.setenv("ECG_RESULTS", str(tmp_path / "res"))
    monkeypatch.setenv("ECG_RATE_LIMIT_MAX", "0")
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_examples_endpoint_lists_every_example(client, manifest):
    data = client.get("/examples").get_json()
    assert data["ok"] is True and [e["id"] for e in data["examples"]] == [m["id"] for m in manifest]


def test_predicting_a_builtin_example_uses_the_same_pipeline(client, manifest):
    chosen = manifest[0]
    body = client.post("/predict", data={"example": chosen["id"]}).get_json()
    assert body["ok"] is True and body["record_name"] == chosen["id"] and body["input_type"] == "wfdb_hea_mat"
    comparison = body["label_comparison"]
    assert comparison["available"] and comparison["source"] == "wfdb_header_dx"
    assert comparison["true_classes"] == sorted(chosen["true_classes"], key=load.CLASS_NAMES.index)
    assert len(body["predictions"]) == 12


@pytest.mark.parametrize("bad", ["nope", "../../etc/passwd", "..\\..\\x", "A3763/../A3763", " "])
def test_unknown_or_malicious_example_ids_are_rejected(client, bad):
    resp = client.post("/predict", data={"example": bad})
    assert resp.status_code == 400 and resp.get_json()["ok"] is False


def test_page_offers_the_example_selector_with_both_languages_and_valid_js(client):
    html = client.get("/").get_data(as_text=True)
    assert 'id="exampleSelect"' in html and "examples.attribution" in html
    js_path = REPO_ROOT / "webapp" / "static" / "app.js"
    js = js_path.read_text(encoding="utf-8")
    for key in ("examples.title", "examples.choose", "examples.kind.correct", "examples.kind.mixed", "examples.attribution"):
        assert len(re.findall(rf'"{re.escape(key)}":', js)) == 2, key
    node = shutil.which("node")
    if node:
        assert subprocess.run([node, "--check", str(js_path)], capture_output=True, text=True).returncode == 0


# --------------------------------------------------------------------------- #
# con el bundle real: la app reproduce la evaluación
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not BUNDLE_PRESENT, reason="falta el bundle")
def test_examples_reproduce_the_evaluation_with_the_real_bundle(tmp_path, manifest):
    saved = dict(os.environ)
    try:
        for key in [k for k in os.environ if k.startswith("ECG_")]:
            del os.environ[key]
        os.environ.update({"ECG_UPLOADS": str(tmp_path / "up"), "ECG_RESULTS": str(tmp_path / "res"),
                           "ECG_RATE_LIMIT_MAX": "0", "CUDA_VISIBLE_DEVICES": "-1"})
        import webapp.wsgi as wsgi
        client = wsgi.create_wsgi_app().test_client()
        worst = 0.0
        for m in manifest:
            body = client.post("/predict", data={"example": m["id"]}).get_json()
            assert body["ok"], body
            probs = {p["class"]: p["probability"] for p in body["predictions"]}
            diff = max(abs(probs[c] - ref) for c, ref in m["reference_probabilities"].items())
            worst = max(worst, diff)
            assert diff < 5e-3, (m["id"], diff)                         # fp16 en GPU vs fp32 en CPU: ~7e-4 medido
            thresholds = {p["class"]: p["threshold"] for p in body["predictions"]}
            for cls, prob in probs.items():                             # misma decisión salvo empate numérico
                web_positive = prob >= thresholds[cls]
                ref_positive = cls in m["reference_predicted_classes"]
                if web_positive != ref_positive:
                    assert abs(prob - thresholds[cls]) < 5e-3, (m["id"], cls, prob, thresholds[cls])
        assert worst < 5e-3
    finally:
        os.environ.clear()
        os.environ.update(saved)
