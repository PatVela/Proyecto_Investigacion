"""Los valores por defecto deben apuntar a runs/, no a la carpeta results/ anterior."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES = [*REPO_ROOT.glob("examples/**/*.py"), *REPO_ROOT.glob("webapp/*.py"), *REPO_ROOT.glob("ecg/*.py"), *REPO_ROOT.glob("tools/*.py")]
STALE = re.compile(r"""(default\s*=\s*["']results/|REPO_ROOT\s*/\s*["']results["']|--output(?:-dir)?\s+results/)""")


def test_no_script_default_points_to_results_dir():
    offenders = [
        f"{p.relative_to(REPO_ROOT)}:{n}: {line.strip()}"
        for p in SOURCES if "official" not in p.parts
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if STALE.search(line)
    ]
    assert not offenders, "\n".join(offenders)


def test_webapp_default_discovery_looks_under_runs(monkeypatch):
    from webapp import app as app_mod

    monkeypatch.delenv("ECG_EVAL_DIR", raising=False)
    dirs = [p.as_posix() for p in app_mod._candidate_result_dirs()]
    assert dirs and all("/runs/eval/" in d for d in dirs)


def test_local_tool_settings_are_ignored_by_git():
    """.claude/ (ajustes locales de la herramienta) no debe poder entrar en un commit con ``git add .``."""
    import subprocess

    ignored = subprocess.run(["git", "check-ignore", "-q", ".claude/settings.json"], cwd=REPO_ROOT)
    assert ignored.returncode == 0
