"""Los .md no deben tener enlaces relativos rotos ni apuntar a un repositorio o rama que ya no se usa."""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _tracked_markdown():
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=REPO_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("no es un repositorio git")
    return [REPO_ROOT / name for name in out.stdout.split()]


def _anchors(text):
    found = set(re.findall(r'<a id="([^"]+)"', text))
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", text, flags=re.M):
        slug = re.sub(r"[^\w\s-]", "", heading.lower()).strip()
        found.add(re.sub(r"\s+", "-", slug))
    return found


def test_relative_links_and_anchors_resolve():
    broken = []
    for md in _tracked_markdown():
        text = md.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'(?:href|src)="([^"]+)"', text):
            if target.startswith(("http", "mailto:")):
                continue
            path, _, fragment = target.partition("#")
            dest = (md.parent / path).resolve() if path else md
            if not dest.exists():
                broken.append(f"{md.relative_to(REPO_ROOT)} -> {target}")
            elif fragment and dest.suffix == ".md" and fragment not in _anchors(dest.read_text(encoding="utf-8")):
                broken.append(f"{md.relative_to(REPO_ROOT)} -> {target} (ancla)")
    assert not broken, broken


def test_docs_point_to_the_current_repository_and_branch():
    stale = []
    for md in _tracked_markdown():
        text = md.read_text(encoding="utf-8")
        for needle in ("Proyectos-Universidad", "Paper_Replica_V3"):
            if needle in text:
                stale.append(f"{md.relative_to(REPO_ROOT)}: {needle}")
    assert not stale, stale
