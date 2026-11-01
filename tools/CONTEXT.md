# tools/ — utilidades de datos y despliegue

**Trabajo:** verificar y completar el dataset CINC2020, auditar el escalado de señal y construir/probar el paquete desplegable. Se corren desde la raíz.

| Script | Hace | Test |
|---|---|---|
| `check_dataset2020.py` | Valida los HDF5 generados por `build_datasets.py` (formas, etiquetas, `EXPECTED_CLASSES = 12`). No recibe argumentos: cualquier flag (incluido `--help`) lo ejecuta | `tests/test_tools_check.py` |
| `check_missing.py` | Integridad .hea/.mat de `dataset2020/` (`--kind`); comparte `obtener_archivos_remotos` con `download_missing.py` | `tests/test_tools_missing.py` |
| `download_missing.py` | Descarga los .hea/.mat faltantes desde PhysioNet (`--kind`); baja a `.part` y renombra solo si llegó entero, y no afirma "dataset completo" si no pudo leer el índice (código de salida 1) | `tests/test_tools_missing.py` |
| `verificar_challenge2021.py` | Comprueba que `chapman_shaoxing` y `ningbo` del Challenge 2021 (v1.0.3, bajados con `descargar_challenge2021.sh`) estén completos: cada `.hea` se lee con `wfdb`, sus archivos de señal existen y no están vacíos, las dimensiones coinciden y los valores son finitos; `--data-dir` y `--esperados` (45152). Código de salida 1 si falta algo. No compara con `SHA256SUMS.txt` | `tests/test_verificar_challenge2021.py` |
| `audit_signal_scaling.py` | Audita ganancia/normalización de señal de `dataset2020/` (`python tools/audit_signal_scaling.py`) | `tests/test_tools_audit.py` |
| `check_patient_overlap.py` | Mide cuántos registros de test de PTB-XL y PTB comparten paciente con train o val (T11) y, con las predicciones guardadas, el F1 con y sin solape. Necesita `ptbxl_database.csv` (PTB-XL 1.0.3) y `RECORDS` (PTB 1.0.0) de PhysioNet, que no se versionan (se guardan en `data/externo/`), y se niega a medir si la correspondencia de nombres no se verifica con la edad y el sexo de los encabezados | `tests/test_patient_overlap.py` |
| `make_patient_groups.py` | Escribe `data/externo/patient_groups.csv` (`record_name`, `source_db`, `group_id`) con el paciente de cada registro de PTB-XL y PTB, para el reparto por paciente de `build_datasets.py --groups-csv`; se niega a escribir si los nombres del Challenge no corresponden a los identificadores de PhysioNet. Necesita `ptbxl_database.csv` y `RECORDS` | `tests/test_grouped_split.py` |
| `check_docs_commands.py` | Comprueba que cada `python …` de los .md apunta a un script que existe y usa flags que ese script define | `tests/test_docs_commands.py` |
| `export_inference_bundle.py` | Convierte los dos checkpoints en `models/*.pt` (solo pesos) y copia umbrales, calibración (temperaturas o `--platt-a/-b/--platt-ensemble`) y reportes a `deploy/` sin rutas locales; se niega si `--eval-dir` no se calculó con esa calibración y ese `--alpha` | `tests/test_inference_bundle.py` |
| `build_examples.py` | Elige 7 registros del test (seed 0) y los copia a `webapp/examples/` con las probabilidades de referencia; `--refresh-references` solo actualiza las referencias de los mismos registros | `tests/test_examples.py` |
| `smoke_deployment.py` | 19 comprobaciones por HTTP contra una URL (solo stdlib): `python tools/smoke_deployment.py URL --expect-ensemble` | `tests/test_deploy_*.py` |

Complementa a `descargar_datos_2020.sh` y `descargar_challenge2021.sh` (raíz). Chequeo humano: revisar el reporte de faltantes antes de construir los HDF5 (etapa 01); tras regenerar el bundle, correr `smoke_deployment.py` contra `deploy\start_local.ps1`.
