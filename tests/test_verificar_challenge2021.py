"""tools/verificar_challenge2021.py con registros sintéticos: acepta un conjunto completo y rechaza faltantes, vacíos y conteos distintos."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("wfdb")

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import verificar_challenge2021 as vc
from test_dataset_build import _write_record


@pytest.fixture
def root(tmp_path):
    """training/ con 2 registros de Chapman-Shaoxing y 1 de Ningbo."""
    base = tmp_path / "training"
    _write_record(base / "chapman_shaoxing" / "g1", "JS00001", ["426783006"])
    _write_record(base / "chapman_shaoxing" / "g1", "JS00002", ["426783006"])
    _write_record(base / "ningbo" / "g1", "JS00003", ["426783006"])
    return base


def run(root, esperados=3):
    return vc.main(["--data-dir", str(root), "--esperados", str(esperados)])


def test_a_complete_set_passes(root, capsys):
    assert run(root) == 0
    assert "SUPERARON LA VALIDACI" in capsys.readouterr().out


def test_a_missing_signal_file_fails(root, capsys):
    (root / "ningbo" / "g1" / "JS00003.mat").unlink()
    assert run(root) == 1
    out = capsys.readouterr().out
    assert "JS00003.hea" in out and "Registros con errores: 1" in out


def test_an_empty_signal_file_fails(root):
    (root / "chapman_shaoxing" / "g1" / "JS00001.mat").write_bytes(b"")
    assert run(root) == 1


def test_a_different_record_count_fails(root):
    assert run(root, esperados=4) == 1


def test_a_missing_subset_folder_fails(root):
    for f in (root / "ningbo").rglob("*"):
        if f.is_file():
            f.unlink()
    assert run(root) == 1
