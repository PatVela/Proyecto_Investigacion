"""Trazado de las 12 derivaciones en papel milimetrado (PNG de paleta) para la interfaz y el informe PDF."""

from __future__ import annotations

import io
import os
import tempfile
import threading
from pathlib import Path

# matplotlib necesita un directorio de configuración; en contenedores mínimos puede no
# existir un "home" y la importación fallaría (RuntimeError) antes de arrancar la app.
if not os.environ.get("MPLCONFIGDIR"):
    try:
        Path.home()
    except RuntimeError:
        os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "ecg-mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from werkzeug.utils import secure_filename

from ecg import load

LEADS = load.STANDARD_LEAD_ORDER
# pyplot mantiene estado global y no es seguro entre hilos (waitress usa varios).
_PLOT_LOCK = threading.Lock()


def _robust_ylim(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(values, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        lo, hi = -2.5, 2.5
    pad = max(0.5, 0.12 * float(hi - lo))
    lo = float(np.floor((lo - pad) * 2) / 2)
    hi = float(np.ceil((hi + pad) * 2) / 2)
    return lo, hi


_MINOR_GRID_ZORDER = 1.6


def _grid_positions(lo: float, hi: float, step: float, major: float | None = None) -> np.ndarray:
    """Múltiplos de ``step`` dentro de [lo, hi] (equivale a MultipleLocator).

    Con ``major`` se omiten los que caen sobre una línea mayor, igual que hace
    Matplotlib con los ticks menores que coinciden con los mayores.
    """
    first = int(np.ceil(lo / step - 1e-9))
    last = int(np.floor(hi / step + 1e-9))
    idx = np.arange(first, last + 1)
    if major is not None:
        ratio = round(major / step)
        idx = idx[idx % ratio != 0]
    return idx * step


def plot_ecg_array(signal: np.ndarray, output_dir: str | Path, stem: str) -> Path:
    """Grafica las 12 derivaciones con estilo de papel milimetrado ECG."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape != (load.WINDOW_LENGTH, load.NUM_LEADS):
        raise ValueError(f"Shape inválida para gráfico ECG: {signal.shape}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(signal.shape[0]) / load.TARGET_FS

    with _PLOT_LOCK:
        return _plot_ecg_figure(signal, t, stem, output_dir)


def _plot_ecg_figure(signal: np.ndarray, t: np.ndarray, stem: str, output_dir: Path) -> Path:
    from matplotlib.ticker import MultipleLocator

    fig, axes = plt.subplots(6, 2, figsize=(16, 11.5), sharex=True)
    fig.patch.set_facecolor("#fffafa")
    fig.suptitle(
        f"ECG de 12 derivaciones — {stem} · papel milimetrado",
        fontsize=15,
        fontweight="bold",
        color="#7f1d1d",
    )

    for i, ax in enumerate(axes.flat):
        ax.set_facecolor("#fffafa")
        ax.plot(t, signal[:, i], linewidth=0.75, color="#111827", zorder=3)
        ax.set_title(LEADS[i], fontsize=10, loc="left", color="#7f1d1d", fontweight="bold")
        ax.set_xlim(0, load.WINDOW_SECONDS)
        y0, y1 = _robust_ylim(signal[:, i])
        ax.set_ylim(y0, y1)
        ax.xaxis.set_major_locator(MultipleLocator(0.20))
        ax.yaxis.set_major_locator(MultipleLocator(0.50))
        ax.grid(which="major", color="#e48b8b", linewidth=0.75)
        # Cuadrícula fina como líneas directas: 250 ticks menores por eje hacían
        # que la figura tardara ~5 s; así es el mismo dibujo en una fracción.
        ax.vlines(_grid_positions(0.0, load.WINDOW_SECONDS, 0.04, major=0.20), y0, y1,
                  colors="#f7c8c8", linewidths=0.35, zorder=_MINOR_GRID_ZORDER)
        ax.hlines(_grid_positions(y0, y1, 0.10, major=0.50), 0.0, load.WINDOW_SECONDS,
                  colors="#f7c8c8", linewidths=0.35, zorder=_MINOR_GRID_ZORDER)
        ax.tick_params(axis="both", which="major", labelsize=7, colors="#7f1d1d", length=0)
        for spine in ax.spines.values():
            spine.set_color("#e48b8b")
            spine.set_linewidth(0.6)
        ax.set_ylabel("z", fontsize=8, color="#7f1d1d")
    axes[-1, 0].set_xlabel("Tiempo (s)", fontsize=9, color="#7f1d1d")
    axes[-1, 1].set_xlabel("Tiempo (s)", fontsize=9, color="#7f1d1d")
    fig.text(
        0.5,
        0.008,
        "Cuadrícula ECG: 0.04 s por cuadro pequeño y 0.20 s por cuadro grande en el eje temporal. Amplitud mostrada normalizada por derivación.",
        ha="center",
        fontsize=8,
        color="#7f1d1d",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_path = output_dir / f"{secure_filename(stem)}_ecg_papel_milimetrado.png"
    _save_figure_png(fig, out_path)
    plt.close(fig)
    return out_path


PLOT_DPI = 160
PNG_PALETTE_COLORS = 256


def _save_figure_png(fig, path: Path) -> None:
    """Guarda la figura como PNG de paleta (≤ 256 colores) sin canal alfa.

    El trazado solo tiene cuadrícula rosa, texto y un trazo negro con suavizado; con 256 colores el error máximo
    es 14/255 en < 0,01 % de los píxeles (PSNR ≈ 59 dB) y el archivo pasa de ≈ 762 KB a ≈ 306 KB, que es lo
    que se descarga y lo que se incrusta en el PDF. La resolución (2545 × 1824) no cambia.
    """
    from PIL import Image

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=PLOT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    buffer.seek(0)
    with Image.open(buffer) as raw:
        rgb = raw.convert("RGB")                       # el fondo es opaco: el canal alfa solo estorba
    palette = rgb.quantize(colors=PNG_PALETTE_COLORS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    palette.save(path, format="PNG", optimize=True, compress_level=9, dpi=(PLOT_DPI, PLOT_DPI))
