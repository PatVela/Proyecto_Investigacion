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
    launcher = REPO_ROOT / "deploy" / "start_local.ps1"
    assert "waitress" in launcher.read_text(encoding="utf-8"), "se esperaba que el lanzador use waitress"
    assert "waitress" in _declared()


def test_no_unused_packages_pinned_back():
    declared = _declared()
    assert not ({"gevent", "six", "scikit-multilearn", "pip-tools"} & declared)


def _torch_pin(fname: str) -> str:
    text = (REPO_ROOT / fname).read_text(encoding="utf-8")
    return re.search(r"^torch==([0-9][^\s+]*)", text, re.M).group(1)


def test_torch_pin_is_the_same_everywhere_and_the_one_the_bundle_was_made_with():
    """Antes requirements.txt fijaba 2.8.0 y todo (suite, bundle, humo) se probó con 2.14.0."""
    import json

    assert _torch_pin("requirements.txt") == _torch_pin("requirements-cuda.txt")
    manifest = json.loads((REPO_ROOT / "deploy" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["torch_version"].split("+")[0] == _torch_pin("requirements.txt")


def test_packages_only_research_scripts_need_are_not_installed_for_the_demo():
    runtime = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    dev = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8").lower()
    for package in ("optuna", "onnx", "onnxruntime", "wfdb"):
        assert not re.search(rf"^{package}==", runtime, re.M) and re.search(rf"^{package}==", dev, re.M), package


def test_ci_workflow_installs_the_dev_requirements_and_runs_pytest():
    """Sin esto las PR llegaban a main con 0 comprobaciones; si alguien cambia el flujo, que siga haciendo lo esencial."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8").replace("\r\n", "\n")
    for needed in ("pull_request:", "requirements-dev.txt", "download.pytorch.org/whl/cpu", "python -m pytest",
                   "tools/check_docs_commands.py", "permissions:\n  contents: read"):
        assert needed in workflow, needed
    assert workflow.index("permissions:") < workflow.index("jobs:")           # permisos a nivel de flujo, de solo lectura
    assert "@v4" not in workflow and "@v5" not in workflow                    # acciones en Node 20 (en desuso); las actuales usan Node 24
