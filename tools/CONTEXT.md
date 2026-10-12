# tools/ — utilidades de datos

**Trabajo:** verificar y completar el dataset CINC2020 y auditar el escalado de señal. Se corren desde la raíz.

| Script | Hace | Test |
|---|---|---|
| `check_dataset2020.py` | Valida los HDF5 generados por `build_datasets.py` (usa `ecg/load.py`) | `tests/test_tools_check.py` |
| `check_missing.py` | Integridad .hea/.mat de `dataset2020/` (`--kind`) | `tests/test_tools_missing.py` |
| `download_missing.py` | Descarga los .hea/.mat faltantes desde PhysioNet (`--kind`) | — |
| `audit_signal_scaling.py` | Audita ganancia/normalización de señal | `tests/test_tools_audit.py` |

Complementa a `descargar_datos_2020.sh` (raíz). Chequeo humano: revisar el reporte de faltantes antes de construir los HDF5 (etapa 01).
