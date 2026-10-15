# examples/cinc2020/ — scripts del pipeline

**Trabajo:** ejecutar las etapas 01–08 de `../../CONTEXT.md`. Todos se corren desde la **raíz del repo** (`python examples/cinc2020/<script>.py ...`). Receta de comandos paso a paso: `README.md` (esta carpeta).

## Etapa → script

| Etapa | Scripts |
|---|---|
| 01 Datos | `build_datasets.py`, `audit_dx_codes.py`, `make_synthetic.py` |
| 03 Calibrar | `calibrate.py` |
| 04 Evaluar | `evaluate.py`, `ensemble_evaluate.py` |
| 05 Comparar | `compare_models.py`, `bootstrap_ci.py` |
| 06 Robustez y sesgo | `robustness.py`, `sex_breakdown.py`, `error_analysis.py` |
| 07 Figuras | `make_figures.py` |
| 08 Despliegue | `export_onnx.py`, `benchmark_latency.py` |
| Búsqueda HP (opcional) | `tune_optuna.py` |
| Archivado (`_archive/`) | `diagnose_prediction.py`, `debug_record_prediction.py`, `_record_diag.py`: depuración puntual, ver `docs/fix_prediccion_cinc2020.md`. Siguen ejecutándose desde su nueva ruta; no forman parte de la secuencia. |

## Factory aquí

`config.json` (ResNet v1), `config_resnet_v2.json`, `config_regular_cnn.json`, `config_synthetic.json`; `official/` (tabla oficial de códigos, **no modificar**).

## Salidas

Escriben fuera de esta carpeta (git-ignored), todo bajo `runs/`: `eval/`, `figs/`, `reports/`, `models/`. Rutas por defecto en cada `--help`.

## Chequeo humano

Antes de pasar de 04 a 05+, revisar `metrics_global.csv` y `thresholds_validation.csv` de la evaluación. Umbrales y calibración (temperaturas o parámetros afines) se ajustan **solo en validación**, nunca en test. Compare modelos solo con el mismo `threshold_beta`; el F1 sin enmascarar por origen es el titular y el enmascarado un diagnóstico.
