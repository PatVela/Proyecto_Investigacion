"""Todo paquete de terceros importado por el repo debe estar declarado en requirements*.txt."""

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"data", "runs", "dataset2017", "dataset2020", "saved", ".git"}
# módulo -> distribución en requirements
DIST = {"sklearn": "scikit-learn", "iterstrat": "iterative-stratification", "flask": "flask", "h5py": "h5py"}
# paquetes que llegan como dependencia directa de uno declarado (matplotlib -> pillow)
TRANSITIVE = {"PIL"}


def _local_modules() -> set[str]:
    """Paquetes y scripts del propio repo (conftest.py agrega examples/ y la raíz al path)."""
    local = {d.name for d in REPO_ROOT.iterdir() if d.is_dir()}
    local |= {p.stem for p in REPO_ROOT.rglob("*.py") if p.relative_to(REPO_ROOT).parts[0] not in SKIP_DIRS}
    return local


def _declared() -> set[str]:
    names = set()
    for fname in ("requirements.txt", "requirements-dev.txt"):
        for line in (REPO_ROOT / fname).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-")):
                names.add(re.split(r"[=<>\[ ]", line)[0].lower().replace("_", "-"))
    return names


def _imported_modules() -> dict[str, str]:
    found = {}
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if rel.parts[0] in SKIP_DIRS or "official" in rel.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module.split(".")[0]]
            for m in mods:
                found.setdefault(m, str(rel))
    return found


def test_every_third_party_import_is_declared():
    declared = _declared()
    local = _local_modules()
    missing = {}
    for module, where in _imported_modules().items():
        if module in sys.stdlib_module_names or module in local or module in TRANSITIVE or module == "__future__":
            continue
        dist = DIST.get(module, module).lower().replace("_", "-")
        if dist not in declared:
            missing[module] = where
    assert not missing, f"Importados pero no declarados en requirements*.txt: {missing}"


def test_scripts_launching_waitress_find_it_declared():
    uses = [p for p in (REPO_ROOT / "webapp").glob("*.sh") if "-m waitress" in p.read_text(encoding="utf-8")]
    assert uses, "se esperaba que run_*.sh usen waitress"
    assert "waitress" in _declared()


def test_no_unused_packages_pinned_back():
    declared = _declared()
    assert not ({"gevent", "six", "scikit-multilearn", "pip-tools"} & declared)
