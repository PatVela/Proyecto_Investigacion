# tools/ — utilidades de datos y despliegue

**Trabajo:** verificar y completar el dataset CINC2020, auditar el escalado de señal y construir/probar el paquete desplegable. Se corren desde la raíz.

| Script | Hace | Test |
|---|---|---|
| `check_dataset2020.py` | Valida los HDF5 generados por `build_datasets.py` (formas, etiquetas, `EXPECTED_CLASSES = 12`). No recibe argumentos: cualquier flag (incluido `--help`) lo ejecuta | `tests/test_tools_check.py` |
| `check_missing.py` | Integridad .hea/.mat de `dataset2020/` (`--kind`) | `tests/test_tools_missing.py` |
| `download_missing.py` | Descarga los .hea/.mat faltantes desde PhysioNet (`--kind`) | — |
| `audit_signal_scaling.py` | Audita ganancia/normalización de señal de `dataset2020/` (`python tools/audit_signal_scaling.py`) | `tests/test_tools_audit.py` |
| `check_docs_commands.py` | Comprueba que cada `python …` de los .md apunta a un script que existe y usa flags que ese script define | `tests/test_docs_commands.py` |
| `export_inference_bundle.py` | Convierte los dos checkpoints en `models/*.pt` (solo pesos) y copia umbrales, calibración (temperaturas o `--platt-a/-b/--platt-ensemble`) y reportes a `deploy/` sin rutas locales; se niega si `--eval-dir` no se calculó con esa calibración y ese `--alpha` | `tests/test_inference_bundle.py` |
| `build_examples.py` | Elige 7 registros del test (seed 0) y los copia a `webapp/examples/` con las probabilidades de referencia; `--refresh-references` solo actualiza las referencias de los mismos registros | `tests/test_examples.py` |
| `smoke_deployment.py` | 19 comprobaciones por HTTP contra una URL (solo stdlib): `python tools/smoke_deployment.py URL --expect-ensemble` | `tests/test_deploy_*.py` |

Complementa a `descargar_datos_2020.sh` (raíz). Chequeo humano: revisar el reporte de faltantes antes de construir los HDF5 (etapa 01); tras regenerar el bundle, correr `smoke_deployment.py` contra `deploy\start_local.ps1`.
