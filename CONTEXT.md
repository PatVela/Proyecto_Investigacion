# Pipeline CINC2020-12

Secuencia que se repite por cada modelo/receta. Detalle de comandos: `examples/cinc2020/README.md`. Estado = qué existe en la columna "Salida" (el sistema de archivos es la máquina de estados).

| # | Etapa | Script / módulo | Entrada | Salida | Chequeo humano |
|---|---|---|---|---|---|
| 01 | Datos | `descargar_datos_2020.sh`, `tools/check_dataset2020.py` (sin argumentos: valida los HDF5), `examples/cinc2020/build_datasets.py`, `audit_dx_codes.py` | `dataset2020/` (WFDB) | `data/cinc2020_12/{train,val,test}.h5` + CSV de distribución, `runs/reports/dx_code_audit.csv` | Distribución de clases, sin registros faltantes y sin códigos puntuados sin clase |
| 02 | Entrenar | `python -m ecg.train <config> -e <nombre>` | `config*.json`, HDF5 | `saved/cinc2020/<nombre>/*/best.pt` | Curva de val sin sobreajuste |
| 03 | Calibrar | `calibrate.py` (`--method temperature` o `platt`) | checkpoint, val.h5 | `temperatures_validation.csv` o `platt_validation.csv`, reporte ECE/Brier | ECE razonable (< 0.02 con la afín) |
| 04 | Evaluar | `evaluate.py`, `ensemble_evaluate.py`, `challenge_score.py` | checkpoint, calibración (`--temperatures` o `--platt`) | `runs/eval/eval-*/` (métricas, umbrales, curvas, predicciones), `runs/eval/ensemble*/` | F1-macro vs. `docs/resultados_finales.md` |
| 05 | Comparar | `compare_models.py`, `bootstrap_ci.py` | dos `runs/eval/eval-*/` con el mismo `threshold_beta` | `runs/reports/model_comparison.csv`, `runs/reports/bootstrap_*.json` | IC 95 % de la diferencia |
| 06 | Robustez y sesgo | `robustness.py`, `sex_breakdown.py`, `error_analysis.py` | checkpoint, test.h5 | `runs/reports/robustness.csv`, `runs/reports/sex_breakdown.json`, `runs/reports/errores_origen.csv` | Degradación aceptable por perturbación y sexo |
| 07 | Figuras | `make_figures.py` | `runs/eval/eval-*/`, `runs/reports/` | `runs/figs/figs_eval_v1/`, `runs/figs/figs_eval_v2/` | Figuras legibles para la tesis |
| 08 | Despliegue | `export_onnx.py`, `benchmark_latency.py`, `tools/export_inference_bundle.py`, `tools/build_examples.py`, `tools/smoke_deployment.py`, `webapp/`, `deploy/start_local.ps1` | checkpoints, `runs/eval/ensemble-f1-platt/`, calibración afín | `runs/models/resnet12.onnx`, latencias, bundle versionado (`models/`, `deploy/`, `webapp/examples/`), app en :5002 | Probabilidad web de un registro de test = su fila de `predictions_test.csv` del ensemble (≈ 1e-3; no la de `ecg.predict`, que es un solo modelo sin calibrar); `smoke_deployment.py` 19/19 |

Opcional, fuera de la secuencia: `tools/check_docs_commands.py` (valida los comandos de los .md), `tune_optuna.py` (búsqueda de hiperparámetros → `runs/models/optuna_cinc2020.db`), `make_synthetic.py` (dataset sintético para smoke tests).

## Factory (estable entre corridas)

`ecg/` (librería), `examples/cinc2020/config*.json` (recetas), `examples/cinc2020/official/` (métrica oficial, vendored), `requirements*.txt`, `constraints.txt` (versiones exactas de las transitivas), `tests/`. El bundle de despliegue (`models/`, `deploy/`, `webapp/examples/`) también va en git, pero se regenera cuando cambia el modelo final.

## Product (nuevo en cada corrida, git-ignored)

Todo lo listado en la columna "Salida" que cuelga de `runs/`, más `saved/` y `data/`. Respaldo: no está en git; si importa, copiarlo aparte.

## Recetas existentes

`config.json` (ResNet-34 v1), `config_resnet_v2.json` (v2 anti-sobreajuste, modelo recomendado), `config_regular_cnn.json` (línea base CNN), `config_synthetic.json` (smoke tests).
