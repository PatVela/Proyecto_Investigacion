"""Accesibilidad y teclado de la interfaz (comprobaciones estáticas sobre HTML, CSS y JS).

El comportamiento real (foco, Enter/Espacio, lectores de pantalla) se verifica además en un
navegador; estas pruebas impiden que las regresiones simples vuelvan a entrar.
"""

import os
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

pytest.importorskip("flask")

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC = REPO_ROOT / "webapp" / "static"
CSS = (STATIC / "style.css").read_text(encoding="utf-8")
JS = (STATIC / "app.js").read_text(encoding="utf-8")


class Page(HTMLParser):
    """Recoge los elementos relevantes del HTML renderizado."""

    def __init__(self):
        super().__init__()
        self.elements = []          # (tag, attrs, texto directo)
        self.stack = []
        self.headings = []          # (nivel, id de la sección contenedora)
        self.section_stack = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "section" and attrs.get("id"):
            self.section_stack.append(attrs["id"])
        if re.fullmatch(r"h[1-6]", tag):
            self.headings.append((int(tag[1]), self.section_stack[-1] if self.section_stack else None))
        self.stack.append([tag, attrs, ""])
        self.elements.append(self.stack[-1])

    def handle_data(self, data):
        if self.stack:
            for entry in self.stack:                 # el texto cuenta para todos los ancestros abiertos
                entry[2] += data

    def handle_endtag(self, tag):
        while self.stack:
            if self.stack.pop()[0] == tag:
                break
        if tag == "section" and self.section_stack:
            self.section_stack.pop()

    def find(self, tag, **attrs):
        return [e for e in self.elements if e[0] == tag and all(e[1].get(k) == v for k, v in attrs.items())]


@pytest.fixture(scope="module")
def page():
    for key in [k for k in os.environ if k.startswith("ECG_")]:
        os.environ.pop(key)
    os.environ["ECG_UPLOADS"] = os.environ.get("TEMP", "/tmp") + "/ui_a11y_up"
    os.environ["ECG_RESULTS"] = os.environ.get("TEMP", "/tmp") + "/ui_a11y_res"
    # con los resultados del bundle se renderizan todas las tablas (métricas, comparación, robustez)
    os.environ["ECG_EVAL_DIR"] = str(REPO_ROOT / "deploy" / "results" / "ensemble-f1-platt")
    os.environ["ECG_EXP_DIR"] = str(REPO_ROOT / "deploy" / "reports")
    from webapp.app import create_app

    html = create_app().test_client().get("/").get_data(as_text=True)
    parsed = Page()
    parsed.feed(html)
    parsed.html = html
    for key in [k for k in os.environ if k.startswith("ECG_")]:
        os.environ.pop(key)
    return parsed


# --------------------------------------------------------------------------- #
# estructura y nombres
# --------------------------------------------------------------------------- #
def test_skip_link_is_first_and_targets_the_main_landmark(page):
    first_link = next(e for e in page.elements if e[0] == "a")
    assert first_link[1]["href"] == "#main" and "skip-link" in first_link[1]["class"]
    main = page.find("main")[0]
    assert main[1]["id"] == "main" and main[1].get("tabindex") == "-1"


def test_navigation_is_labelled_and_marks_the_current_section(page):
    nav = page.find("nav")[0]
    assert nav[1].get("aria-label")
    current = [e for e in page.elements if e[0] in {"a", "button"} and "nav-link" in e[1].get("class", "") and e[1].get("aria-current")]
    assert len(current) == 1 and current[0][1]["aria-current"] == "page"


def test_every_decorative_icon_is_hidden_from_assistive_technology(page):
    icons = [e for e in page.elements if e[0] == "svg" and "ico" in e[1].get("class", "").split()]
    assert len(icons) >= 10
    assert all(e[1].get("aria-hidden") == "true" for e in icons)
    assert len(re.findall(r'<svg class="ico[^"]*">', JS)) == 0         # tampoco los que genera el script


def test_every_button_has_an_accessible_name(page):
    nameless = []
    for tag, attrs, text in page.elements:
        if tag == "button" and not text.strip() and not (attrs.get("aria-label") or attrs.get("data-i18n-aria-label")):
            nameless.append(attrs.get("id"))
    assert not nameless, f"botones sin nombre accesible: {nameless}"
    theme = page.find("button", id="themeToggle")[0][1]
    assert theme["aria-label"] and theme["aria-pressed"] == "false"


def test_language_buttons_expose_their_state_and_language(page):
    es, en = page.find("button", id="langEs")[0][1], page.find("button", id="langEn")[0][1]
    assert (es["lang"], en["lang"]) == ("es", "en") and (es["aria-pressed"], en["aria-pressed"]) == ("true", "false")
    assert 'setAttribute("aria-pressed"' in JS


def test_headings_do_not_skip_levels_within_each_panel(page):
    by_section = {}
    for level, section in page.headings:
        by_section.setdefault(section, []).append(level)
    assert by_section, "sin encabezados"
    for section, levels in by_section.items():
        if section is None:
            continue
        assert levels[0] == 1, (section, levels)
        for before, after in zip(levels, levels[1:]):
            assert after <= before + 1, f"salto de nivel en {section}: {levels}"
    assert not [lvl for lvl, _ in page.headings if lvl == 3]


# --------------------------------------------------------------------------- #
# teclado
# --------------------------------------------------------------------------- #
def test_dropzone_is_a_label_for_a_focusable_file_input(page):
    dz = page.find("label", id="dropzone")
    assert dz and dz[0][1]["for"] == "file"
    file_input = page.find("input", id="file")[0][1]
    assert "visually-hidden-input" in file_input["class"]
    assert not re.search(r"\.visually-hidden-input\s*\{[^}]*display\s*:\s*none", CSS)    # sigue siendo enfocable
    assert re.search(r"\.dropzone:focus-within\s*\{[^}]*outline", CSS)
    assert 'dropzone.addEventListener("click"' not in JS                                  # el <label> abre el selector solo


def test_format_help_is_a_real_button_with_expanded_state(page):
    toggle = page.find("button", id="fmtHelp")
    assert toggle and toggle[0][1]["aria-expanded"] == "false" and toggle[0][1]["aria-controls"] == "fmtHelpBox"
    assert "aria-expanded" in JS
    assert not page.find("span", id="fmtHelp")


def test_focus_follows_the_section_and_motion_respects_the_system(page):
    assert 'setAttribute("aria-current", "page")' in JS and "heading.focus(" in JS
    assert "prefers-reduced-motion" in JS
    assert re.search(r'behavior:\s*prefersReducedMotion\(\)\s*\?\s*"auto"\s*:\s*"smooth"', JS)
    assert 'statusEl.setAttribute("role", cls === "err" ? "alert" : "status")' in JS


# --------------------------------------------------------------------------- #
# formularios, tablas y gráficos
# --------------------------------------------------------------------------- #
def test_free_text_fields_do_not_trigger_autofill_or_spellcheck(page):
    for field in ("patName", "patAge", "threshold", "trueLabels"):
        assert page.find("input", id=field)[0][1]["autocomplete"] == "off", field
    for field in ("patName", "trueLabels"):
        assert page.find("input", id=field)[0][1]["spellcheck"] == "false", field


def test_placeholders_end_with_an_ellipsis_in_both_languages(page):
    for field in ("patName", "patAge", "trueLabels"):
        assert page.find("input", id=field)[0][1]["placeholder"].endswith("…"), field
    for key in ("upload.patientPh", "upload.agePh", "upload.trueLabelsPh"):
        values = re.findall(rf'"{re.escape(key)}": "([^"]*)"', JS)
        assert len(values) == 2 and all(v.endswith("…") for v in values), (key, values)


def test_tables_have_captions_and_scoped_headers(page):
    tables = page.find("table")
    assert len(tables) >= 7                                     # ficha, modelo, esquema, 2 de métricas, comparación y robustez
    html = page.html
    assert html.count("<caption") >= 7
    assert not re.search(r"<thead>(?:(?!</thead>).)*<th(?![a-z])(?![^>]*scope=)", html, flags=re.S)   # todo <th> de cabecera lleva scope
    assert 'scope="col"' in JS and 'scope=\\"row\\"' in JS or 'scope="row"' in JS


def test_charts_and_plot_image_have_accessible_names():
    for needle in ('"role", "img"', "chart.bars", "chart.roc", "chart.pr", "aria-describedby"):
        assert needle in JS, needle
    assert 'alt="ECG paper grid' not in JS
    for key in ("plot.alt", "chart.bars", "chart.roc", "chart.pr", "a11y.skip", "a11y.nav", "a11y.theme"):
        assert len(re.findall(rf'"{re.escape(key)}":', JS)) == 2, key        # es y en


# --------------------------------------------------------------------------- #
# CSS: foco, movimiento y objetivos táctiles
# --------------------------------------------------------------------------- #
def test_css_keeps_focus_visible_and_never_removes_outlines():
    assert ":focus-visible" in CSS
    assert not re.search(r"outline\s*:\s*(none|0)\b", CSS)


def test_css_respects_reduced_motion_and_animates_only_cheap_properties():
    assert re.search(r"@media \(prefers-reduced-motion: reduce\)", CSS)
    assert "transition: all" not in CSS.replace("transition:all", "transition: all")
    layout_props = re.compile(r"\b(padding(?:-\w+)?|margin(?:-\w+)?|width|height|top|left|right|bottom)\b")
    for declaration in re.findall(r"transition\s*:\s*([^;}]+)", CSS):
        for part in declaration.split(","):
            prop = part.strip().split(" ")[0]
            assert not layout_props.fullmatch(prop) or prop == "top", f"transición de propiedad de layout: {declaration}"


def test_css_gives_touch_targets_44px_and_hover_states():
    assert re.search(r"@media \(pointer: coarse\)[^{]*\{[^}]*min-height:\s*44px", CSS, flags=re.S)
    assert "touch-action: manipulation" in CSS
    assert ".seg-group .seg:hover" in CSS and ".icon-btn:hover" in CSS


def test_long_file_names_can_wrap():
    assert re.search(r"\.file-list\s*\{[^}]*min-width:\s*0", CSS)
    assert re.search(r"\.fname\s*\{[^}]*overflow-wrap:\s*anywhere", CSS)
