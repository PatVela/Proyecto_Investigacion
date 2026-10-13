"""Los comandos de los .md deben seguir funcionando: el script existe y los flags que se le pasan los define.

Regresión: la documentación pedía ``calibrate.py``/``evaluate.py`` con flags ya renombrados o inexistentes y
``tools/audit_signal_scaling.py`` se anunciaba como ``python audit_signal_scaling.py``.
"""

import pytest

from tools import check_docs_commands as checker

FILES = ["examples/cinc2020/evaluate.py", "examples/cinc2020/calibrate.py", "ecg/train.py", "ecg/load.py", "ecg/datasets.py",
         "examples/cinc2020/build_datasets.py", "deploy/start_local.ps1", "examples/cinc2020/config.json", "tools/audit_signal_scaling.py"]


def _problems(command):
    return checker.check_command(command, FILES)


def test_valid_commands_pass_including_module_form_and_inline_without_python():
    assert _problems("python examples/cinc2020/evaluate.py examples/cinc2020/config.json $resnet --threshold-beta 1.0 --platt x.csv") == []
    assert _problems("python -m ecg.train examples/cinc2020/config.json -e nombre --epochs 1 --seed 3") == []
    assert _problems("calibrate.py --method platt") == []                       # código en línea sin "python"
    assert _problems("python examples/cinc2020/evaluate.py --help") == []


def test_unknown_flag_is_reported_with_the_script_name():
    problems = _problems("python examples/cinc2020/evaluate.py --threshold-bet 1.0")
    assert problems == ["examples/cinc2020/evaluate.py: flag inexistente --threshold-bet"]


def test_flag_given_with_equals_is_checked_too():
    assert _problems("python examples/cinc2020/evaluate.py --platt=x.csv") == []
    assert _problems("python examples/cinc2020/evaluate.py --nope=1") != []


def test_flags_that_live_behind_a_reexported_main_are_followed():
    """build_datasets.py solo hace ``from ecg.load import main``; sus flags están en otro módulo."""
    assert _problems("python examples/cinc2020/build_datasets.py --data-dir dataset2020 --workers 4") == []
    assert _problems("python examples/cinc2020/build_datasets.py --workers-x 4") != []


def test_missing_script_and_missing_versioned_path_are_reported():
    assert _problems("python tools/no_existe.py --a 1") == ["script inexistente: tools/no_existe.py"]
    problems = _problems("python examples/cinc2020/evaluate.py examples/cinc2020/config_fantasma.json")
    assert problems == ["examples/cinc2020/evaluate.py: la ruta versionada examples/cinc2020/config_fantasma.json no existe"]


def test_generated_paths_placeholders_and_foreign_commands_are_ignored():
    assert _problems("python examples/cinc2020/evaluate.py runs/eval/x/modelo.json data/x.csv") == []
    assert _problems("python examples/cinc2020/evaluate.py examples/cinc2020/<config>.json $cfg.json") == []
    assert _problems("python -m pytest -q tests/test_foo.py") == []
    assert _problems("python -m waitress --port=1 webapp.wsgi:app") == []
    assert _problems("pip install -r requirements.txt") == []
    assert _problems("git commit -m 'python mal.py --x'") == []


def test_command_lines_join_continuations_and_read_inline_code():
    text = "Texto `python examples/cinc2020/evaluate.py --platt a.csv` y más.\n\n```powershell\n" \
           "python examples/cinc2020/evaluate.py `\n  --threshold-beta 1.0 `\n  --platt a.csv\n" \
           "python tools/audit_signal_scaling.py \\\n  --sample-per-db 20\n```\n"
    lines = list(checker.command_lines(text))
    assert any("--threshold-beta 1.0" in line and "--platt a.csv" in line for line in lines)
    assert any("--sample-per-db 20" in line for line in lines)
    assert "python examples/cinc2020/evaluate.py --platt a.csv" in lines


def test_chained_commands_are_split():
    assert checker.parse_python_command("cd x && python examples/cinc2020/evaluate.py --platt a.csv | tee log")[0] \
        == "examples/cinc2020/evaluate.py"


def test_every_command_in_the_tracked_markdown_files_is_valid():
    problems = checker.check_markdown()
    assert not problems, "\n".join(problems)
    assert checker.count_checked_commands() >= 80                         # que "0 problemas" no sea "0 comandos revisados"


@pytest.mark.parametrize("name", ["README.md", "webapp/README.md", "examples/cinc2020/README.md", "docs/experimentos_cinc2020.md"])
def test_the_main_guides_do_contain_commands(name):
    from pathlib import Path

    assert checker.count_checked_commands([checker.REPO_ROOT / Path(name)]) >= 3
