"""Contenido y UX: control de umbral honesto, nota del modelo, ayuda de desarrollo oculta, tipografía y confirmaciones."""

import io
import os
import re
from pathlib import Path

import pytest

pytest.importorskip("flask")

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC = REPO_ROOT / "webapp" / "static"
JS = (STATIC / "app.js").read_text(encoding="utf-8")
BUNDLE_ENV = REPO_ROOT / "deploy" / "ensemble.env"
BUNDLE_PRESENT = BUNDLE_ENV.exists() and (REPO_ROOT / "models").is_dir()


def _bundle_env():
    return dict(line.split("=", 1) for line in BUNDLE_ENV.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#"))


@pytest.fixture()
def make_client(monkeypatch, tmp_path):
    def build(bundle=False, **env):
        from webapp.app import create_app

        for key in [k for k in os.environ if k.startswith("ECG_")]:
            monkeypatch.delenv(key, raising=False)
        values = {"ECG_UPLOADS": tmp_path / "up", "ECG_RESULTS": tmp_path / "res", "ECG_SAVED": tmp_path / "saved"}
        if bundle:
            values = {**{k: str(REPO_ROOT / v) if k not in {"ECG_ALPHA"} else v for k, v in _bundle_env().items()},
                      "ECG_UPLOADS": tmp_path / "up", "ECG_RESULTS": tmp_path / "res"}
        values.update(env)
        for key, value in values.items():
            monkeypatch.setenv(key, str(value))
        app = create_app()
        app.config["TESTING"] = True
        return app.test_client()
    return build


def _page(client):
    return client.get("/").get_data(as_text=True)


# --------------------------------------------------------------------------- #
# el umbral global no aparenta controlar algo que no controla
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not BUNDLE_PRESENT, reason="falta el bundle")
def test_global_threshold_is_disabled_and_explained_when_class_thresholds_are_active(make_client):
    html = _page(make_client(bundle=True))
    field = re.search(r'<input type="number" id="threshold"[^>]*>', html).group(0)
    assert " disabled" in field and 'aria-describedby="thresholdHint"' in field and 'value=""' in field
    assert 'data-i18n="upload.thresholdCalibrated"' in html and "no se aplica" in html


def test_global_threshold_is_editable_when_no_class_thresholds_exist(make_client):
    html = _page(make_client())
    field = re.search(r'<input type="number" id="threshold"[^>]*>', html).group(0)
    assert " disabled" not in field and 'value="0.50"' in field
    assert 'data-i18n="upload.thresholdGlobal"' in html


def test_threshold_is_only_sent_when_the_field_is_active_and_calibrated_value_is_really_ignored():
    assert 'if (!threshold.disabled) fd.append("threshold"' in JS


@pytest.mark.skipif(not BUNDLE_PRESENT, reason="falta el bundle")
def test_a_client_threshold_really_has_no_effect_with_class_thresholds(make_client):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    client = make_client(bundle=True, ECG_RATE_LIMIT_MAX=0)
    results = []
    for value in ("0.05", "0.95"):
        body = client.post("/predict", data={"example": "HR09549", "threshold": value}).get_json()
        assert body["ok"] and body["using_class_thresholds"] is True
        results.append([p["class"] for p in body["predictions"] if p["prediction"] == 1])
    assert results[0] == results[1]                        # por eso el campo se deshabilita en la interfaz


# --------------------------------------------------------------------------- #
# nota del modelo: describe lo que realmente se sirve
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not BUNDLE_PRESENT, reason="falta el bundle")
def test_model_note_describes_the_ensemble_with_calibrated_thresholds(make_client):
    html = _page(make_client(bundle=True))
    assert 'data-i18n="upload.modelNote.ensemble"' in html and 'data-i18n="upload.modelNote.calibrated"' in html


def test_model_note_for_a_plain_setup_and_no_stale_text(make_client):
    html = _page(make_client())
    assert 'data-i18n="upload.modelNote.single"' in html and 'data-i18n="upload.modelNote.global"' in html
    for stale in ("carpeta saved/", "mejor checkpoint disponible", "upload.autoModel"):
        assert stale not in html and stale not in JS, stale


# --------------------------------------------------------------------------- #
# ayuda de desarrollo oculta al público
# --------------------------------------------------------------------------- #
def test_public_page_hides_developer_commands_and_dev_flag_shows_them(make_client, monkeypatch, tmp_path):
    from webapp import app as app_mod

    monkeypatch.setattr(app_mod, "REPO_ROOT", tmp_path / "sin_resultados")      # independiente de lo que haya en runs/ localmente
    public = _page(make_client())
    assert "python examples/cinc2020" not in public and "evaluate.py" not in public and "$resnet" not in public
    assert public.count('data-i18n="public.unavailable"') >= 4                    # métricas, comparación, robustez y curvas
    dev = _page(make_client(ECG_DEV_HELP=1))
    assert "python examples/cinc2020/evaluate.py" in dev and "compare_models.py" in dev and "robustness.py" in dev
    assert 'data-i18n="public.unavailable"' not in dev


# --------------------------------------------------------------------------- #
# datos opcionales plegables
# --------------------------------------------------------------------------- #
def test_optional_fields_live_in_a_collapsible_that_opens_on_desktop_only(make_client):
    html = _page(make_client())
    details = re.search(r'<details id="optionalData".*?</details>', html, flags=re.S).group(0)
    for field in ("patName", "patAge", "threshold", "trueLabels"):
        assert f'id="{field}"' in details
    assert "<summary" in details and "open" not in re.match(r"<details[^>]*>", details).group(0).split()
    assert 'matchMedia("(min-width: 881px)").matches) optionalData.open = true' in JS


# --------------------------------------------------------------------------- #
# textos: un solo idioma, claros y con tipografía cuidada
# --------------------------------------------------------------------------- #
def _dictionary(lang):
    start = JS.index("  es: {") if lang == "es" else JS.index("  en: {")
    end = JS.index("  en: {") if lang == "es" else len(JS)
    return dict(re.findall(r'"([\w.]+)": "((?:[^"\\]|\\.)*)"', JS[start:end]))


def test_both_languages_define_the_same_keys():
    es, en = _dictionary("es"), _dictionary("en")
    assert set(es) == set(en), (sorted(set(es) ^ set(en)))


def test_spanish_ui_has_no_leftover_english_terms():
    es = _dictionary("es")
    offenders = {k: v for k, v in es.items() if re.search(r"\b(threshold|header|fallback)\b", v, flags=re.I)}
    assert not offenders, offenders
    for old_inline in ("fallback NSR, threshold", "· threshold original", "< threshold ", '" · fallback"', 'row("Threshold"', 'row("Fallback NSR"'):
        assert old_inline not in JS, old_inline


def test_units_use_non_breaking_spaces():
    html = (REPO_ROOT / "webapp" / "templates" / "index.html").read_text(encoding="utf-8")
    texts = [v for d in (_dictionary("es"), _dictionary("en")) for v in d.values()]
    bad = [t for t in texts if re.search(r"\d (MB|KB|Hz|mV|ms)\b", t)]
    assert not bad, bad
    assert not re.search(r">[^<{]*\d (MB|KB|Hz|mV|ms)\b[^<]*<", html)
    assert "5\\u00a0MB" in JS and "1\\u00a0h" in JS or "1 hora" in JS


def test_code_tokens_are_marked_as_untranslatable(make_client):
    html = _page(make_client())
    assert "<code" in html and not re.search(r"<code(?![^>]*translate=\"no\")", html)
    assert 'class="prob-name" translate="no"' in JS and JS.count('<span class="chip" translate="no"') == 2


# --------------------------------------------------------------------------- #
# mensajes con siguiente paso y acciones destructivas con confirmación
# --------------------------------------------------------------------------- #
def test_oversized_upload_message_says_what_to_do(make_client):
    client = make_client(ECG_MAX_UPLOAD_MB=1)
    resp = client.post("/predict", data={"files": (io.BytesIO(b"0" * (2 * 1024 * 1024)), "big.csv")},
                       content_type="multipart/form-data")
    error = resp.get_json()["error"]
    assert resp.status_code == 413 and "1 MB" in error and "Reduzca" in error and "10 s" in error


def test_clearing_the_history_asks_for_confirmation():
    assert 'if (!window.confirm(tr("history.confirmClear"))) return;' in JS
    clear_handler = JS[JS.index('const btn = $("btnClearHistory")'):]
    assert clear_handler.index("window.confirm") < clear_handler.index("localStorage.removeItem")
