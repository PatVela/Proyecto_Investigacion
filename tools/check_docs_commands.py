"""Comprueba que los comandos de los .md siguen existiendo: scripts, módulos y flags.

Para cada comando ``python <script>.py ...`` o ``python -m <modulo> ...`` (en bloques de código y en código en línea)
verifica que el script o módulo está en el repositorio y que cada ``--flag`` que se le pasa figura en algún
``add_argument`` de ese script. Los scripts se analizan con ``ast``: no se ejecuta nada.

También exige que las rutas versionadas que aparecen en un comando (``examples/...``, ``deploy/...``) existan.
Las rutas de salida o de datos que se generan (``runs/``, ``saved/``, ``data/``, ``dataset2020/``) y los marcadores de
posición (``<...>``, ``$variable``, ``*``, ``…``) no se comprueban.

Uso (desde la raíz del repositorio)::

    python tools/check_docs_commands.py            # revisa todos los .md versionados
    python tools/check_docs_commands.py README.md  # o solo los indicados
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.S | re.M)
INLINE = re.compile(r"`([^`\n]+)`")
GENERATED_ROOTS = ("runs/", "saved/", "data/", "dataset2020/", "webapp/uploads/", "webapp/results/")
CHECKED_ROOTS = ("ecg/", "examples/", "tools/", "webapp/", "tests/", "deploy/", "docs/", "models/")
CHECKED_SUFFIXES = (".py", ".json", ".ps1", ".sh", ".env", ".md", ".txt")
PLACEHOLDER = re.compile(r"[<>*$…{}]|\.\.\.")


def command_lines(text: str):
    """Comandos candidatos de un .md: líneas de bloques de código (con continuaciones unidas) y código en línea."""
    for block in FENCE.findall(text):
        buffer = ""
        for raw in block.splitlines():
            line = raw.rstrip()
            if line.endswith("\\") or line.endswith("`"):          # continuación de bash o de PowerShell
                buffer += line[:-1] + " "
                continue
            yield (buffer + line).strip()
            buffer = ""
        if buffer.strip():
            yield buffer.strip()
    for span in INLINE.findall(FENCE.sub("", text)):
        yield span.strip()


def _strip_quotes(line: str) -> str:
    return re.sub(r'"[^"]*"|\'[^\']*\'', "STR", line)


def parse_python_command(line: str):
    """``(objetivo, [tokens])`` si la línea invoca un script o módulo; ``objetivo`` = ruta .py o módulo con puntos."""
    for part in re.split(r"\s*(?:&&|;|\|)\s*", _strip_quotes(line)):
        part = re.sub(r"\s#\s.*$", "", part).strip()
        part = re.sub(r"^(?:PS>|\$|>)\s*", "", part)
        tokens = part.split()
        if not tokens:
            continue
        if tokens[0] in ("python", "python3", "py"):
            rest = tokens[1:]
            if rest[:1] == ["-m"] and len(rest) >= 2:
                return rest[1], rest[2:]
            while rest and rest[0] in ("-I", "-X", "utf8", "-u", "-B"):
                rest = rest[1:]
            if rest and rest[0].endswith(".py"):
                return rest[0], rest[1:]
        elif tokens[0].endswith(".py") and any(t.startswith("--") for t in tokens[1:]):
            return tokens[0], tokens[1:]                              # código en línea sin "python": `calibrate.py --method platt`
    return None


def tracked_files(root: Path) -> list[str]:
    """Archivos del repositorio: versionados y nuevos aún sin versionar (los ignorados por git no cuentan)."""
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root,
                         capture_output=True, text=True, check=True)
    return sorted({name for name in out.stdout.split("\n") if name})


def _imported_repo_modules(tree: ast.AST, files: list[str]) -> list[str]:
    """Archivos .py del repositorio que importa un módulo (``import a.b``, ``from a.b import c``, ``from a import b``)."""
    found = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
        for name in names:
            candidate = name.replace(".", "/") + ".py"
            if candidate in files:
                found.append(candidate)
    return found


def declared_flags(script: Path, files: list[str] | None = None, root: Path = REPO_ROOT, _seen: set | None = None) -> set[str]:
    """Opciones declaradas con ``add_argument`` en cualquier parte del script (análisis estático).

    Si el script no declara ninguna (p. ej. ``build_datasets.py`` solo llama a ``ecg.load.main``), se siguen sus
    imports dentro del repositorio hasta encontrar el ``argparse`` que de verdad atiende los flags.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-"):
                    flags.add(arg.value)
    if flags or files is None:
        return flags
    seen = _seen if _seen is not None else {script.resolve()}
    for rel in _imported_repo_modules(tree, files):
        path = (root / rel).resolve()
        if path not in seen:
            seen.add(path)
            flags |= declared_flags(root / rel, files, root, seen)
    return flags


def resolve_target(target: str, files: list[str]) -> str | None:
    """Ruta del script dentro del repo a partir de ``a/b.py``, ``b.py`` (si es único) o ``a.b`` (módulo)."""
    if target.endswith(".py"):
        norm = target.replace("\\", "/").lstrip("./")
        if norm in files:
            return norm
        matches = [f for f in files if f.endswith("/" + norm) or f == norm]
        if "/" not in norm:
            matches = [f for f in files if f.endswith(".py") and Path(f).name == norm and not f.startswith("tests/")]
        return matches[0] if len(matches) == 1 else None
    candidate = target.replace(".", "/") + ".py"
    return candidate if candidate in files else None


def check_command(line: str, files: list[str], root: Path = REPO_ROOT) -> list[str]:
    parsed = parse_python_command(line)
    if parsed is None:
        return []
    target, tokens = parsed
    script = resolve_target(target, files)
    if script is None:
        if target.endswith(".py") and "/" in target and not PLACEHOLDER.search(target) \
                and not target.replace("\\", "/").startswith(GENERATED_ROOTS):
            return [f"script inexistente: {target}"]
        return []                                                     # módulo externo (pytest, waitress…) o no verificable
    problems = []
    known = declared_flags(root / script, files, root)
    for token in tokens:
        flag = token.split("=", 1)[0]
        if re.fullmatch(r"--[A-Za-z][A-Za-z0-9-]*", flag) and flag not in known and flag != "--help":
            problems.append(f"{script}: flag inexistente {flag}")
        elif token.endswith(tuple(CHECKED_SUFFIXES)) and not PLACEHOLDER.search(token):
            norm = token.replace("\\", "/").strip("\"'")
            if norm.startswith(CHECKED_ROOTS) and not norm.startswith(GENERATED_ROOTS) and norm not in files:
                problems.append(f"{script}: la ruta versionada {norm} no existe")
    return problems


def check_markdown(paths: list[Path] | None = None, root: Path = REPO_ROOT) -> list[str]:
    files = tracked_files(root)
    md_files = paths or [root / name for name in files if name.endswith(".md")]
    problems = []
    for md in md_files:
        seen = set()
        for line in command_lines(md.read_text(encoding="utf-8")):
            for problem in check_command(line, files, root):
                if (line, problem) not in seen:
                    seen.add((line, problem))
                    problems.append(f"{md.relative_to(root)}: {problem}  <- {line[:110]}")
    return problems


def count_checked_commands(paths: list[Path] | None = None, root: Path = REPO_ROOT) -> int:
    """Cuántos comandos python/script se examinaron (para que un 0 de problemas no sea un 0 de cobertura)."""
    files = tracked_files(root)
    md_files = paths or [root / name for name in files if name.endswith(".md")]
    return sum(1 for md in md_files for line in command_lines(md.read_text(encoding="utf-8"))
               if parse_python_command(line) is not None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifica los comandos de los .md contra el código del repositorio.")
    parser.add_argument("files", nargs="*", help=".md a revisar (por defecto, todos los versionados)")
    args = parser.parse_args()
    paths = [Path(f).resolve() for f in args.files] or None
    problems = check_markdown(paths)
    for problem in problems:
        print(problem)
    print(f"{count_checked_commands(paths)} comandos revisados; {len(problems)} problema(s) en la documentación.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
