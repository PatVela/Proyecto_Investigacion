"""Carga, caché, contraste, tema/idioma y navegación por URL de la interfaz (PR 2 de la auditoría web)."""

import colorsys
import itertools
import os
import re
from pathlib import Path

import pytest

pytest.importorskip("flask")

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC = REPO_ROOT / "webapp" / "static"
CSS = (STATIC / "style.css").read_text(encoding="utf-8")
JS = (STATIC / "app.js").read_text(encoding="utf-8")


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from webapp.app import create_app

    for key in [k for k in os.environ if k.startswith("ECG_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ECG_UPLOADS", str(tmp_path / "up"))
    monkeypatch.setenv("ECG_RESULTS", str(tmp_path / "res"))
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# --------------------------------------------------------------------------- #
# Plotly bajo demanda y caché de estáticos
# --------------------------------------------------------------------------- #
def test_plotly_is_not_loaded_with_the_page_and_is_fetched_on_demand(client):
    html = client.get("/").get_data(as_text=True)
    assert "<script src" in html and not re.search(r'<script[^>]+plotly', html)           # ya no bloquea la página
    assert re.search(r'<body[^>]+data-plotly-src="/static/plotly\.min\.js\?v=[0-9a-f]{10}"', html)
    for needle in ("function ensurePlotly", "document.createElement(\"script\")", "await ensurePlotly()", "chart.unavailable"):
        assert needle in JS, needle
    assert JS.count("ensurePlotly()") >= 4                                                   # carga + precarga + dos secciones


def test_plotly_loading_failure_is_reported_and_retryable():
    assert re.search(r"script\.onerror\s*=\s*\(\)\s*=>\s*\{\s*plotlyPromise\s*=\s*null", JS)   # un fallo no queda cacheado
    assert "barsLoaded = false;" in JS and 'role="alert"' in JS


def test_html_references_versioned_assets(client):
    html = client.get("/").get_data(as_text=True)
    for name in ("style.css", "app.js", "escudo-unsa.png"):
        assert re.search(rf'/static/{re.escape(name)}\?v=[0-9a-f]{{10}}', html), name


def test_versioned_static_files_are_cached_for_a_year_and_dynamic_responses_never(client):
    html = client.get("/").get_data(as_text=True)
    url = re.search(r'href="(/static/style\.css\?v=[0-9a-f]{10})"', html).group(1)
    versioned = client.get(url)
    assert versioned.headers["Cache-Control"] == "public, max-age=31536000, immutable" and versioned.headers.get("ETag")
    assert client.get("/static/style.css").headers["Cache-Control"] == "no-cache"         # sin versión: revalida
    for path in ("/", "/health", "/metrics", "/curves", "/examples", "/models"):
        assert client.get(path).headers["Cache-Control"] == "no-store", path


def test_unversioned_static_requests_get_a_304_when_unchanged(client):
    first = client.get("/static/app.js")
    again = client.get("/static/app.js", headers={"If-None-Match": first.headers["ETag"]})
    assert again.status_code == 304 and len(again.data) == 0


def test_static_version_changes_only_when_the_file_changes(client, tmp_path):
    app = client.application
    folder = tmp_path / "static"
    folder.mkdir()
    (folder / "a.js").write_text("one")
    app.static_folder = str(folder)
    with app.test_request_context("/"):
        v1 = app.jinja_env.globals["static_v"]("a.js")
        assert app.jinja_env.globals["static_v"]("a.js") == v1
        (folder / "a.js").write_text("two!")
        assert app.jinja_env.globals["static_v"]("a.js") != v1


def test_results_and_uploads_are_never_cached(client):
    assert client.get("/results/nope.pdf").headers["Cache-Control"] == "no-store"


# --------------------------------------------------------------------------- #
# contraste (WCAG AA)
# --------------------------------------------------------------------------- #
def _theme(selector):
    block = re.search(re.escape(selector) + r"\s*\{(.*?)\n\}", CSS, flags=re.S).group(1)
    return dict(re.findall(r"--([\w-]+)\s*:\s*([^;]+);", block))


LIGHT = _theme(':root, [data-theme="light"]')
DARK = {**LIGHT, **_theme('[data-theme="dark"]')}


def _luminance(hex_):
    r, g, b = [int(hex_.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _ratio(a, b):
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


TEXT_PAIRS = [("ink", "card"), ("ink", "bg"), ("ink", "panel-2"), ("ink-soft", "panel"), ("ink-soft", "card"), ("ink-soft", "accent-soft"),
              ("muted", "card"), ("muted", "bg"), ("muted", "panel"), ("muted", "panel-2"), ("muted", "code-bg"), ("muted", "accent-soft"),
              ("accent-text", "card"), ("accent-text", "bg"), ("accent-text", "panel"), ("accent-text", "panel-2"),
              ("accent-text", "accent-soft"), ("accent-text", "code-bg"),
              ("ok-fg", "ok-bg"), ("bad-fg", "bad-bg"), ("bad-fg", "card"), ("warn-fg", "warn-bg"), ("notice-fg", "notice-bg"),
              ("side-ink", "side-bg"), ("side-muted", "side-bg")]


@pytest.mark.parametrize("theme_name,theme", [("claro", LIGHT), ("oscuro", DARK)])
def test_text_colors_meet_wcag_aa(theme_name, theme):
    failures = [f"{fg} sobre {bg}: {_ratio(theme[fg], theme[bg]):.2f}" for fg, bg in TEXT_PAIRS
                if _ratio(theme[fg], theme[bg]) < 4.5]
    assert not failures, f"tema {theme_name}: {failures}"


@pytest.mark.parametrize("theme_name,theme", [("claro", LIGHT), ("oscuro", DARK)])
def test_white_text_on_accent_buttons_meets_wcag_aa(theme_name, theme):
    for bg in ("accent", "accent-h"):
        assert _ratio("#ffffff", theme[bg]) >= 4.5, (theme_name, bg)


def test_accent_is_never_used_as_a_text_color_directly():
    # --accent es para fondos y bordes; el texto usa --accent-text (que cambia por tema para mantener el contraste)
    offenders = re.findall(r"(?<![-\w])color:\s*var\(--accent\)\s*[;}]", CSS)
    assert not offenders, offenders


def test_class_chip_palette_has_readable_white_text_and_distinguishable_colors():
    palette = re.search(r"const PALETTE = \[(.*?)\];", JS, flags=re.S).group(1)
    colors = re.findall(r'"(#[0-9a-fA-F]{6})"', palette)
    assert len(colors) == 12
    too_light = [(c, round(_ratio("#ffffff", c), 2)) for c in colors if _ratio("#ffffff", c) < 4.5]
    assert not too_light, too_light
    dist = lambda a, b: sum((int(a[i:i + 2], 16) - int(b[i:i + 2], 16)) ** 2 for i in (1, 3, 5)) ** 0.5
    assert min(dist(a, b) for a, b in itertools.combinations(colors, 2)) >= 40
    assert re.search(r"--chip-text:\s*#fff", CSS)


# --------------------------------------------------------------------------- #
# tema e idioma
# --------------------------------------------------------------------------- #
def test_theme_and_language_are_decided_before_first_paint(client):
    html = client.get("/").get_data(as_text=True)
    head = html[:html.index("</head>")]
    script = head.index("<script>")
    assert script < head.index('rel="stylesheet"')                                  # antes del CSS: sin parpadeo
    for needle in ("prefers-color-scheme: dark", "navigator.language", 'localStorage.getItem("ecg-theme")', 'localStorage.getItem("ecg-lang")'):
        assert needle in head, needle
    assert '<meta name="theme-color"' in head


def test_css_declares_color_scheme_for_native_controls():
    assert re.search(r':root, \[data-theme="light"\] \{\s*color-scheme: light;', CSS)
    assert re.search(r'\[data-theme="dark"\] \{\s*color-scheme: dark;', CSS)


def test_script_follows_the_system_theme_until_the_user_chooses_one():
    assert 'matchMedia("(prefers-color-scheme: dark)").addEventListener("change"' in JS
    assert "hasSavedTheme()" in JS and "THEME_COLORS" in JS and '"theme-color"' in JS
    assert 'document.documentElement.lang === "en" ? "en" : "es"' in JS


# --------------------------------------------------------------------------- #
# navegación con URL y diseño móvil
# --------------------------------------------------------------------------- #
def test_navigation_uses_real_links_that_reflect_the_section_in_the_url(client):
    html = client.get("/").get_data(as_text=True)
    links = re.findall(r'<a class="nav-link[^"]*" href="#(\w+)" data-sec="(\w+)"', html)
    assert links == [(s, s) for s in ("upload", "vis", "det", "exp")]
    assert "<button type=\"button\" class=\"nav-link" not in html
    for needle in ("history.pushState(null", "popstate", "hashchange", "sectionFromHash()", "e.ctrlKey || e.metaKey"):
        assert needle in JS, needle
    assert re.search(r"a\.nav-link\s*\{\s*text-decoration:\s*none", CSS)


def test_numbers_use_tabular_figures_in_tables():
    assert re.search(r"table\s*\{\s*font-variant-numeric:\s*tabular-nums", CSS)


def test_mobile_header_is_compact():
    block = re.search(r"@media \(max-width: 880px\) \{(.*?)\n\}", CSS, flags=re.S).group(1)
    assert "flex-wrap: nowrap" in block and "overflow-x: auto" in block and ".foot-meta { display: none; }" in block
