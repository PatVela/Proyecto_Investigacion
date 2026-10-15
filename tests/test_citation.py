"""CITATION.cff: trae los campos obligatorios y su versión es la que muestra la webapp."""

import re
from pathlib import Path

from webapp.project_info import PROJECT_INFO

TEXT = (Path(__file__).resolve().parent.parent / "CITATION.cff").read_text(encoding="utf-8")


def test_citation_has_the_required_cff_fields():
    for key in ("cff-version", "message", "title", "authors"):
        assert re.search(rf"^{key}:", TEXT, flags=re.M), key


def test_citation_version_is_the_version_the_webapp_shows():
    assert re.search(rf'^version: "{re.escape(PROJECT_INFO["version"])}"$', TEXT, flags=re.M)
