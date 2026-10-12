"""Construye HDF5 train/val/test para CINC2020-12.

Ejemplo:
    python examples/cinc2020/build_datasets.py \
      --data-dir dataset2020 \
      --output-dir data/cinc2020_12 \
      --workers 6
"""

from __future__ import annotations

import sys

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecg.load import main


if __name__ == "__main__":
    main()
