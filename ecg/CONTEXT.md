# ecg/ — librería

**Trabajo:** esquema de etiquetas, lectura/preprocesamiento de señal, construcción de datasets, red, entrenamiento, calibración e inferencia. Sin estado propio: no escribe fuera de las rutas que le pasan los scripts.

## Mapa de módulos (de la base hacia arriba)

| Módulo | Rol | Depende de |
|---|---|---|
| `schema.py` | 12 clases SNOMED, constantes de normalización, `codes_to_vector` | — |
| `signal.py` | Lectura WFDB/CSV, filtro, remuestreo a 500 Hz, ventanas de 5000 | `schema` |
| `datasets.py` | Escaneo, splits estratificados, construcción HDF5 (CLI) | `schema`, `signal` |
| `load.py` | **Fachada** que reexporta schema/signal/datasets (compatibilidad) | los tres anteriores |
| `augment.py` | Aumentación solo de train, preserva etiquetas | — |
| `network.py` | ResNet-34 tipo Hannun y CNN convencional | torch |
| `calibration.py` | Temperature scaling por clase | — |
| `util.py` | Seeds, paths, checkpoints, conteo de parámetros | — |
| `train.py` | Entrenamiento reproducible (`python -m ecg.train`) | `load`, `network`, `augment`, `util` |
| `predict.py` | Inferencia desde checkpoint (CLI/CSV), usada por webapp | `load`, `network`, `util` |

## Qué toca un cambio

- **`schema.py` (clases, normalización):** única fuente de verdad (`CLASS_GROUPS_12` → `CLASS_NAMES`, `NUM_CLASSES`, `codes_to_vector`). Si cambia el significado de las salidas, subir `LABEL_SCHEMA`. Invalida HDF5 (`build_datasets.py`) y checkpoints; hay que reconstruir y reentrenar. Además:
  - `examples/cinc2020/config*.json` fijan `label_schema`/`num_classes`; `train.py` aborta si no coinciden con `NUM_CLASSES`, y `predict.py` rechaza checkpoints con otro número de clases.
  - **Valores duplicados a mano** (no se actualizan solos): alias v2→v1 (`AxisDev→LAD`, `BBB→RBBB`) en `webapp/prediction.py` y en `examples/cinc2020/_archive/{diagnose,debug_record}_prediction.py`; el texto "27 diagnósticos … 12 clases" en `webapp/report_pdf.py`; `EXPECTED_CLASSES = 12` en `tools/check_dataset2020.py`; el 12 en `tests/` (`test_network`, `test_schema`, `test_robustness`, `test_make_figures`, `test_tools_check`).
  - Evaluación: `challenge_score.py` mapea los 27 códigos oficiales a grupos; `audit_dx_codes.py` marca códigos puntuados sin clase.
  - Salidas generadas (`runs/`, `webapp/static/metrics/metrics.json`) quedan obsoletas hasta regenerarlas.
- **`signal.py` (preproceso):** debe ser idéntico en entrenamiento e inferencia (causa histórica del bug en `docs/fix_prediccion_cinc2020.md`); revisar `predict.py`, `webapp/prediction.py`, `tools/audit_signal_scaling.py`.
- **`network.py`:** cambia el formato de `best.pt`; revisar `predict.load_model`, `export_onnx.py`.
- **`calibration.py`:** afecta `calibrate.py`, `evaluate.py`, `robustness.py`, webapp.

## Chequeo humano

Tras tocar `schema`/`signal`: correr `pytest tests/test_schema.py tests/test_accuracy.py` y comparar una predicción CLI contra la webapp.
