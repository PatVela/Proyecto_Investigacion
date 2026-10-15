"""``python webapp/app.py`` traduce cada opción de línea de comandos a su variable ECG_* (única entrada de desarrollo)."""

import pytest

pytest.importorskip("flask")

from webapp import app as app_module


class _Stop(Exception):
    """Detiene main() justo después de fijar el entorno, sin arrancar el servidor."""


@pytest.fixture()
def cli(monkeypatch):
    variables = [*app_module.CLI_TO_ENV_PATHS.values(), "ECG_ALPHA", "ECG_NORMAL_FALLBACK_MIN_PROB"]
    for name in variables:
        # main() escribe en os.environ directamente: setenv + delenv hace que monkeypatch RESTAURE (borre) la variable al
        # terminar; con delenv(raising=False) sobre una variable ausente no se registraba nada y la fuga rompía otras pruebas
        # (ECG_MODEL_B=algun/valor.csv hacía fallar create_app en las siguientes).
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setattr(app_module, "create_app", lambda: (_ for _ in ()).throw(_Stop()))

    def run(*argv):
        with pytest.raises(_Stop):
            app_module.main(list(argv))

    return run


@pytest.mark.parametrize("option,variable", sorted(app_module.CLI_TO_ENV_PATHS.items()))
def test_each_path_option_sets_its_environment_variable(cli, option, variable):
    import os

    cli(f"--{option.replace('_', '-')}", "algun/valor.csv")
    assert os.environ[variable] == "algun/valor.csv"


def test_options_that_are_not_given_leave_the_environment_alone(cli):
    import os

    cli("--platt", "p.csv")
    assert os.environ["ECG_PLATT"] == "p.csv"
    assert "ECG_PLATT_B" not in os.environ and "ECG_MODEL" not in os.environ and "ECG_ALPHA" not in os.environ


def test_alpha_and_fallback_are_validated_and_forwarded(cli):
    import os

    cli("--alpha", "0.3", "--normal-fallback-min-prob", "0.4")
    assert os.environ["ECG_ALPHA"] == "0.3" and os.environ["ECG_NORMAL_FALLBACK_MIN_PROB"] == "0.4"
    with pytest.raises(SystemExit):
        app_module.main(["--alpha", "1.5"])                               # fuera de [0, 1]: lo rechaza el analizador
    for value in ("2", "-0.1", "nan"):                                    # igual el fallback: error de uso de una línea, no un traceback
        with pytest.raises(SystemExit):
            app_module.main(["--normal-fallback-min-prob", value])


def test_every_cli_path_option_is_documented_as_an_environment_variable():
    from pathlib import Path

    readme = (Path(__file__).resolve().parent.parent / "webapp" / "README.md").read_text(encoding="utf-8")
    for variable in app_module.CLI_TO_ENV_PATHS.values():
        assert variable in readme, variable


def test_the_fixture_leaves_no_variable_behind(monkeypatch):
    """Regresión de la fuga: tras una prueba que fija variables con main(), el entorno queda como estaba."""
    import os

    variables = [*app_module.CLI_TO_ENV_PATHS.values(), "ECG_ALPHA", "ECG_NORMAL_FALLBACK_MIN_PROB"]
    assert not [name for name in variables if name in os.environ]


def test_a_wrong_environment_does_not_break_the_command_line_entry_point():
    """Importar webapp.app ya no construye una app: con un ECG_MODEL inexistente en el entorno, --help y --model siguen vivos."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    env = {**os.environ, "ECG_MODEL": "models/no_existe.pt", "CUDA_VISIBLE_DEVICES": "-1"}
    out = subprocess.run([sys.executable, str(repo / "webapp" / "app.py"), "--help"], cwd=repo, env=env,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    assert out.returncode == 0 and "usage" in out.stdout.lower(), out.stderr[-800:]


@pytest.mark.parametrize("alpha", ["1.5", "-0.1", "nan"])
def test_an_alpha_outside_0_1_in_the_environment_stops_the_start(monkeypatch, alpha):
    monkeypatch.setenv("ECG_ALPHA", alpha)
    with pytest.raises(RuntimeError, match="ECG_ALPHA"):
        app_module.create_app()


def test_an_alpha_with_a_decimal_comma_stops_the_start_and_names_the_variable(monkeypatch):
    monkeypatch.setenv("ECG_ALPHA", "0,5")
    with pytest.raises(ValueError, match="ECG_ALPHA"):
        app_module.create_app()


@pytest.mark.parametrize("value", ["1.5", "-0.2", "nan"])
def test_a_fallback_probability_outside_0_1_in_the_environment_stops_the_start(monkeypatch, value):
    monkeypatch.setenv("ECG_NORMAL_FALLBACK_MIN_PROB", value)
    with pytest.raises(RuntimeError, match="ECG_NORMAL_FALLBACK_MIN_PROB"):
        app_module.create_app()


def test_a_fallback_probability_that_is_not_a_number_stops_the_start_and_names_the_variable(monkeypatch):
    monkeypatch.setenv("ECG_NORMAL_FALLBACK_MIN_PROB", "0,4")
    with pytest.raises(ValueError, match="ECG_NORMAL_FALLBACK_MIN_PROB"):
        app_module.create_app()
