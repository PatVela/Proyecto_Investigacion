# ECG CINC2020-12 · ResNet tipo Hannun

Réplica académica de Hannun et al. (Nat Med 2019): clasificador multilabel de ECG de 12 derivaciones (12 clases SNOMED agrupadas) sobre PhysioNet/CinC 2020. Python 3.10+, PyTorch. Rama de trabajo: `Paper_Replica_V3`.

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

`data/`, `dataset2017/`, `dataset2020/`, `saved/` (checkpoints), `eval-*/`, `ensemble*/`, `exp-files/`, `metrica-challenge/`, `figs_*/`, y sueltos en la raíz: `bootstrap_*.json`, `*.csv`, `sex_breakdown.json`, `resnet12.onnx`, `optuna_cinc2020.db`. No editar a mano; se regeneran con el pipeline.

## Enrutamiento por tarea

- Entrenar o cambiar el modelo → `CONTEXT.md` etapas 02–03, luego `ecg/CONTEXT.md`.
- Evaluar / comparar / robustez / figuras → `CONTEXT.md` etapas 04–07.
- Cambiar etiquetas o preprocesamiento → `ecg/CONTEXT.md` (sección "qué toca un cambio"); afecta checkpoints ya entrenados.
- Demo web → `webapp/CONTEXT.md`. Resultados citables → `docs/resultados_finales.md`.

## Reglas

- Comandos desde la raíz del repo. Documentación y mensajes en español.
- `examples/cinc2020/official/` es código oficial vendored: no modificar.
- Las rutas de salida (`eval-*`, `exp-files`, ...) están hardcodeadas en README, docs y scripts; no renombrarlas sin actualizar todos los referentes.
