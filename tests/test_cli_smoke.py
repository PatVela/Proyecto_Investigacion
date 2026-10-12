"""Smoke: todos los CLI responden ``--help`` (cableado intacto tras depuración)."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

HELP_CASES = [
    ("examples/audit_dx_codes", ["examples/cinc2020/audit_dx_codes.py"]),
    ("examples/benchmark_latency", ["examples/cinc2020/benchmark_latency.py"]),
    ("examples/bootstrap_ci", ["examples/cinc2020/bootstrap_ci.py"]),
    ("examples/build_datasets", ["examples/cinc2020/build_datasets.py"]),
    ("examples/calibrate", ["examples/cinc2020/calibrate.py"]),
    ("examples/challenge_score", ["examples/cinc2020/challenge_score.py"]),
    ("examples/compare_models", ["examples/cinc2020/compare_models.py"]),
    ("examples/debug_record_prediction", ["examples/cinc2020/_archive/debug_record_prediction.py"]),
    ("examples/diagnose_prediction", ["examples/cinc2020/_archive/diagnose_prediction.py"]),
    ("examples/ensemble_evaluate", ["examples/cinc2020/ensemble_evaluate.py"]),
    ("examples/error_analysis", ["examples/cinc2020/error_analysis.py"]),
    ("examples/evaluate", ["examples/cinc2020/evaluate.py"]),
    ("examples/export_onnx", ["examples/cinc2020/export_onnx.py"]),
    ("examples/make_figures", ["examples/cinc2020/make_figures.py"]),
    ("examples/make_synthetic", ["examples/cinc2020/make_synthetic.py"]),
    ("examples/robustness", ["examples/cinc2020/robustness.py"]),
    ("examples/sex_breakdown", ["examples/cinc2020/sex_breakdown.py"]),
    ("examples/tune_optuna", ["examples/cinc2020/tune_optuna.py"]),
    ("tools/audit_signal_scaling", ["tools/audit_signal_scaling.py"]),
    ("tools/check_missing", ["tools/check_missing.py"]),
    ("tools/download_missing", ["tools/download_missing.py"]),
    ("ecg/load", ["-m", "ecg.load"]),
    ("ecg/predict", ["-m", "ecg.predict"]),
    ("ecg/train", ["-m", "ecg.train"]),
    ("webapp/app", ["webapp/app.py"]),
]


@pytest.mark.parametrize("label,argv", HELP_CASES)
def test_cli_help(label, argv):
    proc = subprocess.run(
        [sys.executable, *argv, "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"{label}:\n{proc.stderr[-2000:]}"
    assert "usage:" in proc.stdout.lower(), f"{label}: sin usage en --help"


def test_lib_modules_compile():
    """Módulos sin CLI (bibliotecas/auditorías) compilan."""
    import py_compile

    for rel in (
        "examples/cinc2020/_archive/_record_diag.py",
        "examples/cinc2020/official/evaluate_12ECG_score.py",
        "tools/check_dataset2020.py",
    ):
        py_compile.compile(REPO_ROOT / rel, doraise=True)
