"""Verifica que Chapman-Shaoxing y Ningbo del Challenge 2021 estén completos y se puedan leer.

Para cada ``.hea`` de las carpetas ``chapman_shaoxing`` y ``ningbo`` comprueba que se lee con ``wfdb``, que los archivos de señal que
referencia existen y no están vacíos, que la señal tiene las dimensiones del encabezado y que sus valores son finitos. Complementa a
``descargar_challenge2021.sh`` (raíz). No compara con ``SHA256SUMS.txt`` ni mira las etiquetas::

    python tools/verificar_challenge2021.py
    python tools/verificar_challenge2021.py --data-dir dataset2021/physionet.org/files/challenge-2021/1.0.3/training

Código de salida 0 si todos los registros son válidos y hay exactamente ``--esperados`` (45152 en la versión 1.0.3: 10 247 de
Chapman-Shaoxing y 34 905 de Ningbo); 1 si falta algo o hay errores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import wfdb

DATA_DIR = "dataset2021/physionet.org/files/challenge-2021/1.0.3/training"
SUBSETS = ("chapman_shaoxing", "ningbo")
ESPERADOS = 45152


def validar_registro(hea_path: Path) -> None:
    """Lanza una excepción si el registro no se puede leer completo."""
    record_path = hea_path.with_suffix("")
    header = wfdb.rdheader(str(record_path))

    for filename in header.file_name:                                   # archivos de señal referenciados
        signal_file = hea_path.parent / filename
        if not signal_file.is_file():
            raise FileNotFoundError(f"No existe: {signal_file}")
        if signal_file.stat().st_size == 0:
            raise ValueError(f"Archivo vacío: {signal_file}")

    signal = wfdb.rdrecord(str(record_path), physical=False).d_signal   # lectura completa del ECG
    if signal is None:
        raise ValueError("No se pudo obtener la señal")
    if signal.shape != (header.sig_len, header.n_sig):
        raise ValueError(f"Dimensiones incorrectas: {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError("Señal con valores no finitos")


def verificar_subconjunto(folder: Path, subset: str) -> tuple[int, int]:
    """Valida todos los registros de ``folder``; devuelve (registros encontrados, registros con error)."""
    print(f"\n{'=' * 60}")
    print(f"VALIDANDO: {subset}")
    print(f"{'=' * 60}")

    headers = sorted(folder.rglob("*.hea"))
    valid = 0
    errors = []
    for i, hea_path in enumerate(headers, 1):
        try:
            validar_registro(hea_path)
            valid += 1
        except Exception as e:                                          # un registro dañado no debe detener la revisión del resto
            errors.append((hea_path.name, str(e)))
        if i % 100 == 0 or i == len(headers):
            print(f"\rProgreso: {i}/{len(headers)} | Válidos: {valid} | Errores: {len(errors)}", end="", flush=True)

    print(f"\n\nResultados de {subset}:")
    print(f"Registros encontrados: {len(headers)}")
    print(f"Registros válidos: {valid}")
    print(f"Registros con errores: {len(errors)}")
    if errors:
        print("\nPrimeros errores:")
        for name, error in errors[:20]:
            print(f"  {name}: {error}")
    return len(headers), len(errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=DATA_DIR, help="carpeta training/ del Challenge 2021 (por defecto, la que deja descargar_challenge2021.sh)")
    parser.add_argument("--esperados", type=int, default=ESPERADOS, help="total de registros que debe haber entre las dos carpetas")
    args = parser.parse_args(argv)

    root = Path(args.data_dir)
    total_records = total_errors = 0
    for subset in SUBSETS:
        found, bad = verificar_subconjunto(root / subset, subset)
        total_records += found
        total_errors += bad

    print("\n" + "=" * 60)
    print("RESULTADO GENERAL")
    print("=" * 60)
    print(f"Total de registros: {total_records}")
    print(f"Registros válidos: {total_records - total_errors}")
    print(f"Registros con errores: {total_errors}")

    if total_errors == 0 and total_records == args.esperados:
        print("\nRESULTADO: TODOS LOS REGISTROS SUPERARON LA VALIDACIÓN.")
        return 0
    print(f"\nRESULTADO: REVISAR REGISTROS FALTANTES O CON ERRORES (se esperaban {args.esperados}).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
