"""Construye un HDF5 de solo prueba con todos los registros de un origen (sin reparto).

Sirve para evaluar un sistema ya entrenado en una base que no intervino en su diseño, con el mismo
preprocesamiento que ``build_datasets.py``. No exige positivos en todas las clases y conserva los
registros sin ninguna de las 12. Escribe el HDF5, ``<salida>.records.csv`` y ``<salida>.summary.json``.

Ejemplo (Challenge 2021, carpeta ``chapman_shaoxing``):
    python examples/cinc2020/build_external_test.py \
      --data-dir dataset2021/physionet.org/files/challenge-2021/1.0.3 \
      --origin chapman_shaoxing \
      --output data/cinc2021_externo/test_c.h5 \
      --workers 6
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import freeze_support
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecg.datasets import build_external_test
from ecg.schema import DEFAULT_NORM_MODE


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Construye un HDF5 de prueba con todos los registros de un origen")
    parser.add_argument("--data-dir", required=True, help="Raíz con training/<origen>/ (.hea/.mat)")
    parser.add_argument("--output", required=True, help="Ruta del HDF5 de salida")
    parser.add_argument("--origin", default=None, help="Carpeta que sigue a training/ (p. ej. chapman_shaoxing); sin él, todo lo escaneado")
    parser.add_argument("--exclude-record", action="append", default=[], metavar="NOMBRE",
                        help="Nombre del .hea sin extensión que se excluye (repetible); se cuenta y se lista en el resumen")
    parser.add_argument("--workers", type=int, default=6, help="Procesos de preprocesamiento")
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--write-batch-size", type=int, default=32)
    parser.add_argument("--norm-mode", choices=["physical", "global_zscore", "per_lead_zscore"], default=DEFAULT_NORM_MODE)
    parser.add_argument("--no-bandpass", dest="bandpass", action="store_false", default=True, help="Desactiva el pasa-banda")
    args = parser.parse_args(argv)

    summary = build_external_test(
        data_dir=args.data_dir,
        output_path=args.output,
        origin=args.origin,
        exclude_records=args.exclude_record,
        workers=args.workers,
        chunksize=args.chunksize,
        write_batch_size=args.write_batch_size,
        norm_mode=args.norm_mode,
        bandpass=args.bandpass,
    )
    print(f"\nRegistros escritos: {summary['records']:,}; excluidos: {summary['excluded']}; "
          f"sin ninguna clase: {summary['records_without_any_class']:,}")


if __name__ == "__main__":
    freeze_support()
    main()
