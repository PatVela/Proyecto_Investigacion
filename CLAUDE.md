# ECG CINC2020-12 · ResNet tipo Hannun

Réplica académica de Hannun et al. (Nat Med 2019): clasificador multilabel de ECG de 12 derivaciones (12 clases SNOMED agrupadas) sobre PhysioNet/CinC 2020. Python 3.10+, PyTorch. Rama principal: `main`.

## Dónde está cada cosa

| Carpeta | Qué es | Contrato |
|---|---|---|
| `ecg/` | Librería: esquema, señal, datasets, red, entrenamiento, calibración, inferencia | `ecg/CONTEXT.md` |
| `examples/cinc2020/` | Scripts del pipeline experimental + configs (`config*.json`) + `official/` (vendored, no tocar) | `examples/cinc2020/CONTEXT.md` |
| `webapp/` | Demo Flask de inferencia + informe PDF | `webapp/CONTEXT.md` |
| `tools/` | Chequeos y descarga de datos CINC2020 | `tools/CONTEXT.md` |
| `tests/` | Suite pytest | `tests/CONTEXT.md` |
| `docs/` | Informe técnico, experimentos, resultados, fix histórico | `docs/CONTEXT.md` |

La definición del pipeline (etapas 01–08, entradas, salidas, chequeos humanos) está en `CONTEXT.md`.

## Producto generado (git-ignored, solo existe en disco local)

Todo lo que produce el pipeline vive bajo `runs/` (ignorado por git y Docker). No editar a mano; se regenera con el pipeline.

| Ruta | Contenido |
|---|---|
| `runs/eval/` | `eval-*/`, `ensemble*/`, `runs/eval/metrica-challenge/` (salidas de `evaluate.py`, `ensemble_evaluate.py`, `challenge_score.py`) |
| `runs/figs/` | `runs/figs/figs_eval_v1/`, `runs/figs/figs_eval_v2/` |
| `runs/reports/` | `model_comparison.csv`, `robustness.csv`, `bootstrap_*.json`, `runs/reports/sex_breakdown.json`, `*.csv` de auditoría y errores |
| `runs/models/` | `runs/models/resnet12.onnx`, `runs/models/optuna_cinc2020.db` |

Fuera de `runs/` (también ignorado): `data/` (HDF5), `dataset2017/`, `dataset2020/` (crudos), `saved/` (checkpoints).

## Enrutamiento por tarea

- Entrenar o cambiar el modelo → `CONTEXT.md` etapas 02–03, luego `ecg/CONTEXT.md`.
- Evaluar / comparar / robustez / figuras → `CONTEXT.md` etapas 04–07.
- Cambiar etiquetas o preprocesamiento → `ecg/CONTEXT.md` (sección "qué toca un cambio"); afecta checkpoints ya entrenados.
- Demo web → `webapp/CONTEXT.md`. Resultados citables → `docs/resultados_finales.md`.

## Reglas

- Comandos desde la raíz del repo. Documentación y mensajes en español.
- `examples/cinc2020/official/` es código oficial vendored: no modificar.
- Las rutas bajo `runs/` están hardcodeadas en README, docs y defaults de scripts; no renombrarlas sin actualizar todos los referentes.
- `examples/cinc2020/_archive/` guarda scripts de depuración fuera de la secuencia; no los uses como referencia del pipeline.
