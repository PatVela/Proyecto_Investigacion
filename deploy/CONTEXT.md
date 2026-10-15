# deploy/ y models/ — bundle de inferencia versionado

**Trabajo:** que la webapp corra en esta PC sin `saved/` ni `runs/`. `models/` guarda los dos pesos del ensemble (`resnet34_v1.pt` = A, `resnet34_v2.pt` = B, ≈19 MB c/u, sin estado del optimizador) y `deploy/` la calibración, los umbrales, las métricas y el lanzador de Windows.

- **Entradas (solo al regenerar):** los `best.pt` de A y B en `saved/`; `runs/eval/ensemble-f1-platt/`; `platt_validation.csv` de cada modelo y `platt_ensemble_validation.csv`; `runs/reports/` (`model_comparison.csv`, `robustness.csv`).
- **Proceso:** `tools/export_inference_bundle.py` (comando completo en su docstring y en `webapp/README.md`, Ejecución local). Se niega a escribir si `--eval-dir` no corresponde a la calibración y al `--alpha` indicados. Solo escribe los archivos listados abajo: no vacía la carpeta.
- **Salidas (generadas, no se editan):** `models/*.pt`; `deploy/manifest.json` (SHA-256, época, val_loss, alpha, versiones); `deploy/ensemble.env` (variables `ECG_*` que carga `webapp/wsgi.py`, rutas relativas a la raíz del repo); `deploy/platt_a.csv`, `platt_b.csv`, `platt_ensemble.csv`; `deploy/results/<eval>/` (métricas, curvas ROC/PR y umbrales, sin predicciones por registro); `deploy/reports/`.
- **A mano:** este archivo y `deploy/start_local.ps1` (waitress en `127.0.0.1`; `-Lan`, `-Tunnel`, `-UseGpu`). El exportador no los toca.
- **Qué toca un cambio:** si cambia el significado de las salidas del modelo, regenerar el bundle además de reentrenar. Los nombres `models/resnet34_v*.pt` los citan pruebas y documentos: no renombrarlos sin actualizarlos.
- **Chequeo humano:** `tools/smoke_deployment.py` (19/19) y la comparación con `predictions_test.csv` de la etapa 08 de `CONTEXT.md`; las pruebas `tests/test_inference_bundle.py`, `tests/test_deploy_config.py` y `tests/test_dev_server.py` comprueban que el bundle sea coherente.
