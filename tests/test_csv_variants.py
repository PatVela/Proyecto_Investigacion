"""Lectura del CSV de entrada: frecuencia de muestreo declarada, BOM, CSV de Excel en español y codificación."""

import numpy as np
import pytest

from ecg import load, signal
from ecg.predict import prepare_csv_windows

LEADS = signal.STANDARD_LEAD_ORDER
SIGNAL = np.round(np.random.default_rng(0).normal(0, 0.3, size=(5000, 12)), 4)


def _csv(comment=None, delim=",", decimal=".", header=True):
    lines = [comment] if comment else []
    if header:
        lines.append(delim.join(LEADS))
    lines += [delim.join(f"{v:.4f}".replace(".", decimal) for v in row) for row in SIGNAL]
    return "\n".join(lines) + "\n"


def _write(tmp_path, text, encoding="utf-8", prefix=b""):
    path = tmp_path / "ecg.csv"
    path.write_bytes(prefix + text.encode(encoding))
    return path


@pytest.mark.parametrize("comment,expected", [
    ("# Sampling Rate: 250 Hz", 250.0),
    ("# Sampling Rate: 250 Hz" + "," * 11, 250.0),
    ("# Sampling Rate: 250" + "," * 11, 250.0),           # Excel: celda sin «Hz» exportada con comas de relleno (antes: 500 Hz)
    ("# Sampling Rate 250 Hz", 250.0),
    ("# sampling rate: 257.5", 257.5),
])
def test_the_declared_sampling_rate_is_read(tmp_path, comment, expected):
    _, fs, _, _ = load.read_csv_ecg(_write(tmp_path, _csv(comment=comment)))
    assert fs == expected


@pytest.mark.parametrize("comment", ["# Sampling Rate: abc Hz", "# Sampling Rate: Hz", "# Sampling Rate: -250 Hz",
                                     "# Sampling Rate:" + "," * 11])
def test_an_unreadable_sampling_rate_is_rejected_instead_of_assuming_500_hz(tmp_path, comment):
    with pytest.raises(ValueError, match="frecuencia de muestreo"):
        load.read_csv_ecg(_write(tmp_path, _csv(comment=comment)))


def test_without_the_comment_500_hz_is_still_assumed(tmp_path):
    _, fs, _, _ = load.read_csv_ecg(_write(tmp_path, _csv()))
    assert fs == signal.TARGET_FS


@pytest.mark.parametrize("comment", [None, "# Sampling Rate: 250 Hz"])
def test_a_utf8_bom_does_not_hide_the_comment_or_the_first_column(tmp_path, comment):
    values, fs, names, _ = load.read_csv_ecg(_write(tmp_path, _csv(comment=comment), prefix=b"\xef\xbb\xbf"))
    assert names == list(LEADS) and values.shape == (5000, 12)
    assert fs == (250.0 if comment else signal.TARGET_FS)


def test_the_semicolon_csv_of_spanish_excel_gives_the_same_signal(tmp_path):
    reference, _, names_ref, _ = load.read_csv_ecg(_write(tmp_path, _csv()))
    values, _, names, _ = load.read_csv_ecg(_write(tmp_path, _csv(delim=";", decimal=",")))
    assert names == names_ref == list(LEADS)
    np.testing.assert_allclose(values, reference, atol=1e-6)


def test_the_semicolon_csv_reaches_the_model_input_unchanged(tmp_path):
    comma, _ = prepare_csv_windows(_write(tmp_path, _csv()), norm_mode="physical")
    semicolon, _ = prepare_csv_windows(_write(tmp_path, _csv(delim=";", decimal=",")), norm_mode="physical")
    np.testing.assert_allclose(semicolon, comma, atol=1e-5)


@pytest.mark.parametrize("encoding", ["cp1252", "utf-16"])
def test_a_csv_that_is_not_utf8_says_so(tmp_path, encoding):
    text = _csv(comment="# Paciente: Peña")
    with pytest.raises(ValueError, match="UTF-8"):
        load.read_csv_ecg(_write(tmp_path, text, encoding=encoding))
