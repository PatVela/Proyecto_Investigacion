"""Rejilla del trazado ECG: posiciones equivalentes a MultipleLocator y figura válida."""

import numpy as np
import pytest
from matplotlib.ticker import MultipleLocator

from ecg import load
from webapp import plotting


@pytest.mark.parametrize("lo,hi,step", [(0.0, 10.0, 0.04), (-1.5, 2.0, 0.10), (-2.5, 3.0, 0.10), (0.0, 10.0, 0.20)])
def test_grid_positions_match_multiple_locator(lo, hi, step):
    expected = MultipleLocator(step).tick_values(lo, hi)
    expected = expected[(expected >= lo - 1e-9) & (expected <= hi + 1e-9)]
    assert np.allclose(plotting._grid_positions(lo, hi, step), expected)


def test_grid_positions_skip_major_lines():
    minor_x = plotting._grid_positions(0.0, 10.0, 0.04, major=0.20)
    assert not np.any(np.isclose(minor_x % 0.20, 0.0, atol=1e-9) | np.isclose(minor_x % 0.20, 0.20, atol=1e-9))
    assert len(minor_x) == 250 - 50          # 250 pasos de 0.04 s menos los 50 mayores
    minor_y = plotting._grid_positions(-1.0, 1.0, 0.10, major=0.50)
    assert len(minor_y) == 20 - 4            # 20 pasos de 0.1 mV menos los mayores (0.5)


def test_plot_ecg_array_writes_valid_png(tmp_path):
    from PIL import Image

    signal = np.random.default_rng(0).normal(0, 0.3, (load.WINDOW_LENGTH, load.NUM_LEADS)).astype(np.float32)
    path = plotting.plot_ecg_array(signal, tmp_path, "rec")
    assert path.exists() and path.suffix == ".png"
    with Image.open(path) as img:
        assert img.size[0] > 2000 and img.size[1] > 1500


def test_plot_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        plotting.plot_ecg_array(np.zeros((10, 12), dtype=np.float32), tmp_path, "bad")
