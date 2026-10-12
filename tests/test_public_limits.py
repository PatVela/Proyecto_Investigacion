"""Límites para uso público: tamaño, tasa por IP, borrado de subidas, caducidad de informes y aviso."""

import io
import os
import re
import time
from pathlib import Path

import pytest
import torch

pytest.importorskip("flask")

from ecg import load, network
from tools.smoke_deployment import synthetic_csv
from webapp.app import create_app
from webapp.limits import SlidingWindowLimiter, cleanup_old_jobs

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# SlidingWindowLimiter
# --------------------------------------------------------------------------- #
class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_limiter_blocks_after_max_and_recovers_when_the_window_slides():
    clock = Clock()
    lim = SlidingWindowLimiter(3, 60, clock=clock)
    assert [lim.check("a")[0] for _ in range(3)] == [True, True, True]
    allowed, retry = lim.check("a")
    assert allowed is False and retry == pytest.approx(60.0)
    clock.t += 30
    assert lim.check("a") == (False, pytest.approx(30.0))
    clock.t += 31
    assert lim.check("a")[0] is True


def test_limiter_keys_are_independent_and_disabled_when_zero():
    lim = SlidingWindowLimiter(1, 60, clock=Clock())
    assert lim.check("a")[0] and lim.check("b")[0] and not lim.check("a")[0]
    off = SlidingWindowLimiter(0, 60)
    assert off.enabled is False and all(off.check("x")[0] for _ in range(100))


def test_limiter_memory_is_bounded_for_many_clients():
    clock = Clock()
    lim = SlidingWindowLimiter(1, 10, clock=clock)
    for i in range(10_050):
        lim.check(f"ip{i}")
        if i == 10_000:
            clock.t += 11                                                # las anteriores ya expiraron
    assert len(lim._hits) < 10_050


# --------------------------------------------------------------------------- #
# cleanup_old_jobs
# --------------------------------------------------------------------------- #
def test_cleanup_removes_only_old_job_directories(tmp_path):
    old, new = tmp_path / "aaaaaaaaaaaa", tmp_path / "bbbbbbbbbbbb"
    keep_name, keep_file = tmp_path / "not-a-job", tmp_path / "cccccccccccc"
    for d in (old, new, keep_name):
        d.mkdir()
        (d / "x.pdf").write_text("x")
    keep_file.write_text("a file, not a dir")
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    os.utime(keep_name, (time.time() - 7200, time.time() - 7200))
    os.utime(keep_file, (time.time() - 7200, time.time() - 7200))
    assert cleanup_old_jobs(tmp_path, 3600) == 1
    assert not old.exists() and new.exists() and keep_name.exists() and keep_file.exists()


def test_cleanup_is_disabled_with_zero_ttl_and_tolerates_missing_root(tmp_path):
    old = tmp_path / "aaaaaaaaaaaa"
    old.mkdir()
    os.utime(old, (0, 0))
    assert cleanup_old_jobs(tmp_path, 0) == 0 and old.exists()
    assert cleanup_old_jobs(tmp_path / "missing", 60) == 0


# --------------------------------------------------------------------------- #
# aplicación
# --------------------------------------------------------------------------- #
def _checkpoint(path):
    torch.save({"model_state_dict": network.build_network(num_leads=12, num_classes=12).state_dict(),
                "class_names": load.CLASS_NAMES, "label_schema": load.LABEL_SCHEMA, "epoch": 1, "val_loss": 0.5}, str(path))
    return path


@pytest.fixture()
def make_client(monkeypatch, tmp_path):
    def build(**env):
        for key in [k for k in os.environ if k.startswith("ECG_")]:
            monkeypatch.delenv(key, raising=False)
        base = {"ECG_UPLOADS": str(tmp_path / "up"), "ECG_RESULTS": str(tmp_path / "res"),
                "ECG_SAVED": str(tmp_path / "saved"), "ECG_MODEL": str(_checkpoint(tmp_path / "m.pt"))}
        base.update({k: str(v) for k, v in env.items()})
        for key, value in base.items():
            monkeypatch.setenv(key, value)
        app = create_app()
        app.config["TESTING"] = True
        return app.test_client()
    return build


def _post_csv(client, **kwargs):
    return client.post("/predict", data={"files": (io.BytesIO(synthetic_csv()), "r.csv")},
                       content_type="multipart/form-data", **kwargs)


def test_oversized_upload_gets_json_413(make_client):
    client = make_client(ECG_MAX_UPLOAD_MB=1)
    big = io.BytesIO(b"0" * (2 * 1024 * 1024))
    resp = client.post("/predict", data={"files": (big, "big.csv")}, content_type="multipart/form-data")
    assert resp.status_code == 413
    body = resp.get_json()
    assert body["ok"] is False and "1 MB" in body["error"]


def test_default_upload_limit_is_5_mb(make_client):
    assert make_client().application.config["MAX_CONTENT_LENGTH"] == 5 * 1024 * 1024


def test_rate_limit_returns_429_with_retry_after_then_isolates_by_ip(make_client):
    client = make_client(ECG_RATE_LIMIT_MAX=2, ECG_RATE_LIMIT_WINDOW=60)
    assert [_post_csv(client).status_code for _ in range(2)] == [200, 200]
    blocked = _post_csv(client)
    assert blocked.status_code == 429 and int(blocked.headers["Retry-After"]) >= 1
    assert blocked.get_json()["ok"] is False and "Demasiadas" in blocked.get_json()["error"]
    other = _post_csv(client, environ_base={"REMOTE_ADDR": "203.0.113.9"})
    assert other.status_code == 200                                     # otra IP, otro cupo
    assert client.get("/health").status_code == 200                      # solo /predict está limitado


def test_rate_limit_can_be_disabled(make_client):
    client = make_client(ECG_RATE_LIMIT_MAX=0)
    assert all(_post_csv(client).status_code == 200 for _ in range(4))


def test_forwarded_ip_is_used_only_when_a_proxy_is_trusted(make_client):
    headers_a, headers_b = {"X-Forwarded-For": "198.51.100.1"}, {"X-Forwarded-For": "198.51.100.2"}
    direct = make_client(ECG_RATE_LIMIT_MAX=1)                          # sin proxy: el encabezado se ignora
    assert _post_csv(direct, headers=headers_a).status_code == 200
    assert _post_csv(direct, headers=headers_b).status_code == 429
    proxied = make_client(ECG_RATE_LIMIT_MAX=1, ECG_TRUST_PROXY=1)      # con proxy: cada cliente real tiene su cupo
    assert _post_csv(proxied, headers=headers_a).status_code == 200
    assert _post_csv(proxied, headers=headers_b).status_code == 200
    assert _post_csv(proxied, headers=headers_a).status_code == 429


def test_uploaded_ecg_is_deleted_after_analysis_but_report_is_kept(make_client, tmp_path):
    client = make_client(ECG_RATE_LIMIT_MAX=0)
    body = _post_csv(client).get_json()
    assert body["ok"] is True
    assert not any((tmp_path / "up").rglob("*.csv"))                    # el ECG subido ya no está en el servidor
    assert (tmp_path / "res" / body["job_id"]).is_dir()
    assert client.get(body["pdf_url"]).status_code == 200               # el informe sigue descargable


def test_uploaded_ecg_is_deleted_even_when_the_analysis_fails(make_client, tmp_path):
    client = make_client(ECG_RATE_LIMIT_MAX=0)
    bad = io.BytesIO(b"# Sampling Rate: 0.01 Hz\n" + synthetic_csv().split(b"\n", 1)[1])
    resp = client.post("/predict", data={"files": (bad, "bad.csv")}, content_type="multipart/form-data")
    assert resp.status_code >= 400
    assert not any((tmp_path / "up").rglob("*"))


def test_keep_uploads_flag_preserves_them_for_debugging(make_client, tmp_path):
    client = make_client(ECG_RATE_LIMIT_MAX=0, ECG_KEEP_UPLOADS=1)
    assert _post_csv(client).get_json()["ok"] is True
    assert any((tmp_path / "up").rglob("*.csv"))


def test_expired_reports_are_purged_on_the_next_request(make_client, tmp_path):
    client = make_client(ECG_RATE_LIMIT_MAX=0, ECG_RESULT_TTL_SECONDS=3600)
    stale = tmp_path / "res" / "deadbeefdead"
    stale.mkdir(parents=True)
    (stale / "old_reporte.pdf").write_text("x")
    os.utime(stale, (time.time() - 7200, time.time() - 7200))
    body = _post_csv(client).get_json()
    assert not stale.exists() and (tmp_path / "res" / body["job_id"]).exists()


def test_privacy_notice_is_in_the_page_and_in_both_languages(make_client):
    html = make_client().get("/").get_data(as_text=True)
    assert 'id="privacyNote"' in html and 'data-i18n="upload.privacy"' in html
    js = (REPO_ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
    assert len(re.findall(r'"upload\.privacy":', js)) == 2
    assert "pacientes reales" in js and "real patient data" in js
