"""Toda variable de entorno ECG_* que lee el código debe estar documentada en webapp/README.md."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# lecturas reales del entorno: os.environ.get("X"), os.environ["X"], setdefault/pop("X"), int_env("X"), float_env("X"), _resolve_dir_env("X") y env_var="X"
# (el nombre que app.py pasa a los cargadores de archivos de calibración, que hacen os.environ.get(env_var))
READ = re.compile(r"""(?:environ(?:\.get|\.setdefault|\.pop)?[\(\[]|(?:int|float)_env\(|_resolve_dir_env\(|env_var(?::\s*str)?\s*=)\s*["'](ECG_[A-Z0-9_]+)["']""")


def _used_in_code() -> set[str]:
    names = set()
    for path in (REPO_ROOT / "webapp").glob("*.py"):
        names |= set(READ.findall(path.read_text(encoding="utf-8")))
    return names


def test_the_scanner_finds_the_variables_it_should():
    names = _used_in_code()
    assert {"ECG_MODEL", "ECG_MODEL_B", "ECG_RATE_LIMIT_MAX", "ECG_ENV_FILE", "ECG_DEBUG_ERRORS", "ECG_ALPHA", "ECG_NORMAL_FALLBACK_MIN_PROB"} <= names
    assert "ECG_SAVED_DIR" not in names                      # clave de app.config, no variable de entorno


def test_the_scanner_sees_the_calibration_variables_read_through_env_var():
    assert {"ECG_TEMPERATURES", "ECG_TEMPERATURES_B", "ECG_PLATT", "ECG_PLATT_B", "ECG_PLATT_ENSEMBLE"} <= _used_in_code()


def test_every_cli_path_option_maps_to_a_variable_the_app_reads():
    from webapp.app import CLI_TO_ENV_PATHS

    assert set(CLI_TO_ENV_PATHS.values()) <= _used_in_code()


def test_every_environment_variable_is_documented():
    readme = (REPO_ROOT / "webapp" / "README.md").read_text(encoding="utf-8")
    missing = sorted(name for name in _used_in_code() if name not in readme)
    assert not missing, f"Variables sin documentar en webapp/README.md: {missing}"


def test_bundle_env_file_only_uses_known_variables():
    known = _used_in_code()
    env = (REPO_ROOT / "deploy" / "ensemble.env").read_text(encoding="utf-8")
    declared = {line.split("=", 1)[0] for line in env.splitlines() if line and not line.startswith("#")}
    assert declared and declared <= known, f"Variables desconocidas en deploy/ensemble.env: {sorted(declared - known)}"
