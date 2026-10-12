"""Límite de frecuencia de muestreo declarada (evita pedir decenas de GB al remuestrear)."""

import numpy as np
import pytest

from ecg import signal


def _ecg(n=5000):
    return np.zeros((n, 12), dtype=np.float32)


@pytest.mark.parametrize("bad", [0.01, 1.0, 49.9, 10_000.1, 1e9, 0.0, -500.0, float("nan"), float("inf")])
def test_resample_rejects_out_of_range_rates(bad):
    with pytest.raises(ValueError, match="fuera de rango"):
        signal.resample_signal(_ecg(), bad)


@pytest.mark.parametrize("fs", [signal.MIN_SAMPLING_RATE_HZ, 250, 257, 500, 1000, signal.MAX_SAMPLING_RATE_HZ])
def test_resample_accepts_realistic_rates(fs):
    out = signal.resample_signal(_ecg(int(fs * 10)), fs)
    assert out.shape == (signal.TARGET_FS * 10, 12)


def test_preprocess_propagates_the_limit():
    with pytest.raises(ValueError, match="fuera de rango"):
        signal.preprocess_ecg_array(_ecg(), sampling_rate=0.01, lead_names=signal.STANDARD_LEAD_ORDER)


def test_csv_with_absurd_sampling_rate_is_rejected_quickly(tmp_path):
    from ecg.predict import prepare_csv_windows

    path = tmp_path / "x.csv"
    np.savetxt(str(path), np.zeros((5000, 12)), delimiter=",",
               header=",".join(signal.STANDARD_LEAD_ORDER), comments="")
    text = "# Sampling Rate: 0.01 Hz\n" + path.read_text()
    path.write_text(text)
    with pytest.raises(ValueError, match="fuera de rango"):
        prepare_csv_windows(path, norm_mode="physical")
