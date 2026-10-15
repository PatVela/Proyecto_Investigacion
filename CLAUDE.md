# ECG CINC2020-12 · ResNet tipo Hannun

Réplica académica de Hannun et al. (Nat Med 2019): clasificador multilabel de ECG de 12 derivaciones (12 clases SNOMED agrupadas) sobre PhysioNet/CinC 2020. Python 3.10+, PyTorch. Rama principal: `main`.

## Dónde está cada cosa

| Carpeta | Qué es | Contrato |
|---|---|---|
| `ecg/` | Librería: esquema, señal, datasets, red, entrenamiento, calibración, inferencia | `ecg/CONTEXT.md` |
| `examples/cinc2020/` | Scripts del pipeline experimental + configs (`config*.json`) + `official/` (vendored, no tocar) | `examples/cinc2020/CONTEXT.md` |
| `webapp/` | Demo Flask de inferencia + informe PDF + ejemplos incluidos | `webapp/CONTEXT.md` |
| `tools/` | Chequeos y descarga de datos CINC2020, bundle de despliegue, prueba de humo | `tools/CONTEXT.md` |
| `tests/` | Suite pytest | `tests/CONTEXT.md` |
| `.github/workflows/` | CI (`tests.yml`): pytest y `check_docs_commands.py` en cada PR y push a `main`; `audit.yml`: `pip-audit` semanal (complementa las alertas de Dependabot, el escaneo de secretos y la protección de envío de GitHub) | `tests/CONTEXT.md` |
| `docs/` | Informe técnico, experimentos, resultados, fix histórico | `docs/CONTEXT.md` |
| `models/`, `deploy/` | **Versionados.** Bundle de inferencia (pesos `models/*.pt`, `deploy/ensemble.env`, umbrales, métricas, lanzador de Windows `deploy/start_local.ps1`) con el que la webapp corre sin `saved/` ni `runs/`. Se regenera con `tools/export_inference_bundle.py`; no editar a mano (salvo `deploy/CONTEXT.md` y `start_local.ps1`) | `deploy/CONTEXT.md` |
| `tasks/` | Plan y lista de tareas para ejecutar la demo en esta PC (`plan.md`, `todo.md`) y la auditoría de mejoras (`analisis_mejoras.md`) | `tasks/plan.md` |

La definición del pipeline (etapas 01–08, entradas, salidas, chequeos humanos) está en `CONTEXT.md`.

## Producto generado (git-ignored, solo existe en disco local)

Todo lo que produce el pipeline vive bajo `runs/` y es la columna "Salida" de `CONTEXT.md`. No se edita a mano; se regenera con el pipeline. Fuera de `runs/`, también ignorados: `data/` (HDF5), `dataset2020/` (crudos), `saved/` (checkpoints completos), `webapp/uploads/` y `webapp/results/` (temporales de la webapp).

## Enrutamiento por tarea

- Entrenar o cambiar el modelo → `CONTEXT.md` etapas 02–03, luego `ecg/CONTEXT.md`.
- Evaluar / comparar / robustez / figuras → `CONTEXT.md` etapas 04–07.
- Cambiar etiquetas o preprocesamiento → `ecg/CONTEXT.md` (sección "qué toca un cambio"); afecta checkpoints ya entrenados.
- Demo web → `webapp/CONTEXT.md`. Ejecutarla en esta PC (Windows 11) → `webapp/README.md` (Ejecución local) y `tasks/todo.md`. Resultados citables → `docs/resultados_finales.md`.

## Reglas

- Comandos desde la raíz del repo. Documentación y mensajes en español.
- Compara modelos solo con el mismo `threshold_beta` y la misma calibración (`compare_models.py` lo exige). El F1 sin enmascarar es el titular; el enmascarado por origen es un diagnóstico.
- Al cambiar un comando o un flag, actualiza los `.md` que lo citan: `python tools/check_docs_commands.py` (y `tests/test_docs_commands.py`) detectan scripts y flags inexistentes.
- `examples/cinc2020/official/` es código oficial vendored: no modificar.
- Las rutas bajo `runs/` están hardcodeadas en README, docs y defaults de scripts; no renombrarlas sin actualizar todos los referentes.
- `examples/cinc2020/_archive/` guarda scripts de depuración fuera de la secuencia; no los uses como referencia del pipeline.
- Si cambia el significado de las salidas del modelo, regenerar el bundle (`models/`, `deploy/`) además de reentrenar; `tests/test_inference_bundle.py` y `tests/test_deploy_config.py` comprueban que el bundle sea coherente.
