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
