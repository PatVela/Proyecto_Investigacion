# ecg/ — librería

**Trabajo:** esquema de etiquetas, lectura/preprocesamiento de señal, construcción de datasets, red, entrenamiento, calibración e inferencia. Sin estado propio: no escribe fuera de las rutas que le pasan los scripts.

## Mapa de módulos (de la base hacia arriba)

| Módulo | Rol | Depende de |
|---|---|---|
| `schema.py` | 12 clases SNOMED, constantes de normalización, `codes_to_vector` | — |
| `signal.py` | Lectura WFDB/CSV, filtro, remuestreo a 500 Hz, ventanas de 5000; en inferencia rechaza lo que no es un ECG (`require_usable_signal`: < 2 s o más de 6 derivaciones planas) | `schema` |
| `datasets.py` | Escaneo, splits estratificados, construcción HDF5 (CLI); `tests/test_dataset_build.py` lo prueba con un mini-Challenge sintético | `schema`, `signal` |
| `load.py` | **Fachada** que reexporta schema/signal/datasets (compatibilidad) | los tres anteriores |
| `augment.py` | Aumentación solo de train, preserva etiquetas | — |
| `hdf5_dataset.py` | `HDF5ECGDataset` único (train, evaluación, ensemble): valida el HDF5 antes de precargarlo | `util`, `schema`, `signal`, `augment` |
| `network.py` | ResNet-34 tipo Hannun y CNN convencional | `schema`, `signal`, torch |
| `calibration.py` | Temperature scaling y calibración afín (`a·logit + b`) por clase, ECE, Brier y `ensemble_probabilities` (promedio ponderado; lo usan `ensemble_evaluate.py` y la webapp) | — |
| `util.py` | Seeds, dispositivo, `dataloader_kwargs`, `decode_h5_string`, guardar/cargar checkpoints (`load_checkpoint(..., trusted=False)` usa `weights_only=True`) | `schema` |
| `train.py` | Entrenamiento reproducible (`python -m ecg.train`) | `hdf5_dataset`, `load`, `network`, `util` |
| `predict.py` | Inferencia desde checkpoint (CLI/CSV/HDF5; `predict_hdf5` devuelve también los logits crudos), usada por webapp | `hdf5_dataset`, `load`, `network`, `util` |

## Qué toca un cambio

- **`schema.py` (clases, normalización):** única fuente de verdad (`CLASS_GROUPS_12` → `CLASS_NAMES`, `NUM_CLASSES`, `codes_to_vector`). Si cambia el significado de las salidas, subir `LABEL_SCHEMA`. Invalida HDF5 (`build_datasets.py`) y checkpoints; hay que reconstruir y reentrenar. Además:
  - `examples/cinc2020/config*.json` fijan `label_schema`/`num_classes`; `train.py` aborta si no coinciden con `NUM_CLASSES`, y `predict.py` rechaza checkpoints con otro número de clases.
  - **Valores duplicados a mano** (no se actualizan solos): alias v2→v1 (`AxisDev→LAD`, `BBB→RBBB`) en `webapp/labels.py` (dos veces: `V2_TO_V1_CLASS` y el inverso dentro de `split_reference_label_text`) y en `examples/cinc2020/_archive/{diagnose,debug_record}_prediction.py`; el texto "27 diagnósticos … 12 clases" en `webapp/report_pdf.py`; `EXPECTED_CLASSES = 12` en `tools/check_dataset2020.py`; el 12 en `tests/` (`test_network`, `test_schema`, `test_robustness`, `test_make_figures`, `test_tools_check`).
  - Cobertura de etiquetas: `audit_dx_codes.py` marca códigos puntuados sin clase.
  - Salidas generadas (`runs/`) y el bundle de despliegue (`models/`, `deploy/`, `webapp/examples/`) quedan obsoletos hasta regenerarlos (`tools/export_inference_bundle.py`, `tools/build_examples.py`).
- **`signal.py` (preproceso):** debe ser idéntico en entrenamiento e inferencia (causa histórica del bug en `docs/fix_prediccion_cinc2020.md`); revisar `predict.py`, `webapp/prediction.py`, `tools/audit_signal_scaling.py`. `load_mat_signal` rechaza un `.mat` que declare más de `MAX_MAT_ELEMENTS` (25 M) elementos antes de leerlo (la subida web usa su propio tope, `WEB_MAX_MAT_ELEMENTS` = 6 M en `webapp/prediction.py`, vía `max_elements`) y convierte una `val` que no es numérica en `ValueError`; `resample_signal` solo acepta frecuencias de 50 a 10 000 Hz (`MIN/MAX_SAMPLING_RATE_HZ`); `preprocessing_for_schema` avisa si el checkpoint no declara esquema y falla si es desconocido.
- **Reparto y máscara de celdas:** `datasets.grouped_origin_split` (`build_datasets.py --holdout-origin/--groups-csv`) reparte por origen y por grupo de paciente y retiene un origen como `test_b`; con `mask_unannotated` el dataset devuelve un tercer elemento (máscara por clase) y `train.MaskedBCEWithLogits` promedia solo sobre celdas anotadas. Sin esas opciones nada cambia (`tests/test_grouped_split.py`, `tests/test_masked_loss.py`).
- **`train.py` (configuración):** una clave nueva que lea el código va en `KNOWN_CONFIG_KEYS`; el resto se ignora con un `AVISO` al empezar (`tests/test_train_monitor.py` lo exige).
- **`network.py`:** cambia el formato de `best.pt`; revisar `predict.load_model`, `export_onnx.py` y regenerar `models/*.pt`.
- **`calibration.py`:** afecta `calibrate.py`, `evaluate.py`, `ensemble_evaluate.py`, `robustness.py` (solo temperatura), webapp y el bundle (`deploy/platt_*.csv`).

## Chequeo humano

Tras tocar `schema`/`signal`: correr `pytest tests/test_schema.py tests/test_accuracy.py` y comparar la probabilidad que da la webapp para un registro de test con su fila de `predictions_test.csv` del ensemble (no con `ecg.predict`: es un solo modelo sin calibrar y da otra probabilidad a propósito).
