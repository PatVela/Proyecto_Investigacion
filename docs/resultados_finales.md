# Resultados finales — ResNet-34 CINC2020-12

## Configuración experimental

- Datos: PhysioNet/CinC Challenge 2020 (6 subconjuntos), esquema agrupado v2 de 12 clases SNOMED.
- Splits: train 33 759 / validation 6 486 / test 6 421 ventanas (10 s, 12 derivaciones, 500 Hz).
- Registros: 43 101 sin exclusiones (30 194/6 486/6 421; 70/15/15).
- Punto de operación: calibración por clase (tuneada en validación) + umbrales por clase optimizados en validación con **F1** (β = 1.0, el punto recomendado y el de la demo) o con **F0.5** (β = 0.5, menos falsos positivos).
  **Toda comparación entre modelos usa el mismo β**: con F0.5 en un modelo y F1 en otro la diferencia mezcla modelo y umbral (v2 − v1 pasa de +3.5 a +1.2 puntos al igualarlos). `compare_models.py` se niega a comparar evaluaciones con distinto `threshold_beta`.
- Calibración: temperatura por clase (la original) y calibración afín por clase `sigmoid(a·logit + b)` ajustada solo en validación (la que usa la demo; ver hallazgo 4). En el ensemble, cada modelo se calibra por separado y el promedio se vuelve a calibrar.
- Modelos: ResNet v1 (receta base, 80 épocas máx., selección por `val_loss`), ResNet v2 (aumentación + label smoothing + 150 épocas máx. + selección por F1-macro), Ensemble (promedio v1+v2, α = 0.5).

## Tabla principal (test, n = 6421), punto F1 común

| Métrica | ResNet v1 | ResNet v2 | Ensemble (temperatura) | Ensemble (calibración afín) |
|---|---|---|---|---|
| F1-macro | 0.7341 | 0.7463 | 0.7556 | **0.7562** |
| F1-micro | 0.7607 | 0.7771 | 0.7833 | **0.7846** |
| Precisión-macro | 0.7163 | 0.7172 | 0.7334 | **0.7473** |
| Recall-macro | 0.7592 | 0.7817 | **0.7834** | 0.7681 |
| Especificidad-macro | 0.9376 | 0.9438 | 0.9455 | **0.9476** |
| AUROC-macro | 0.9434 | 0.9377 | 0.9505 | **0.9506** |
| AUPRC-macro | 0.7838 | 0.7794 | 0.8052 | **0.8078** |
| Exact-match | 0.4264 | 0.4689 | 0.4724 | **0.4755** |

Fuentes: `runs/eval/eval-resnet-f1/`, `runs/eval/eval-resnet-v2-f1/`, `runs/eval/ensemble-f1/`, `runs/eval/ensemble-f1-platt/` (`metrics_global.csv`). Los tres primeros usan la temperatura de cada modelo; la última, la calibración afín. En validación (n = 6486) el ensemble con calibración afín da F1-macro 0.7702, F1-micro 0.7945, AUROC 0.9532, AUPRC 0.8178 y exact-match 0.5017.

La calibración afín no mueve la clasificación: la diferencia de F1-macro entre las dos últimas columnas es +0.0005 (IC95 bootstrap [−0.0035, +0.0044], `runs/reports/bootstrap_ensemble-platt-vs-temperatura_f1.json`). Lo que cambia es la calidad de las probabilidades (hallazgo 4) y el balance precisión/recall con el que quedan los umbrales.

## Punto F0.5 (menos falsos positivos)

| Métrica | ResNet v1 | ResNet v2 | Ensemble |
|---|---|---|---|
| F1-macro | 0.6835 | 0.7180 | 0.7102 |
| F1-micro | 0.7309 | 0.7445 | 0.7469 |
| Precisión-macro | 0.8180 | 0.7887 | **0.8239** |
| Recall-macro | 0.5931 | **0.6730** | 0.6289 |
| Especificidad-macro | 0.9758 | 0.9741 | **0.9773** |
| AUROC-macro | 0.9434 | 0.9377 | **0.9505** |
| AUPRC-macro | 0.7838 | 0.7794 | **0.8052** |
| Exact-match | 0.4368 | **0.4590** | 0.4548 |

Fuentes: `runs/eval/eval-resnet/`, `runs/eval/eval-resnet-v2/`, `runs/eval/ensemble/` (temperatura, sin regla NSR-exclusivo; regenerado con `ensemble_evaluate.py --threshold-beta 0.5`). Estas filas no son comparables con las de la tabla anterior.

## F1 por clase (test, F1 común)

| Clase | Soporte | v1 | v2 | Ensemble (temperatura) | Ensemble (afín) |
|---|---|---|---|---|---|
| NSR | 3249 | 0.917 | 0.925 | 0.925 | 0.925 |
| AxisDev | 1072 | 0.731 | 0.765 | 0.764 | 0.767 |
| MI | 1651 | 0.723 | 0.743 | 0.753 | 0.751 |
| TAb | 1786 | 0.663 | 0.688 | 0.693 | 0.692 |
| AF | 568 | 0.891 | 0.900 | 0.899 | 0.903 |
| LVH | 800 | 0.632 | 0.635 | 0.659 | 0.658 |
| VEctopy | 395 | 0.669 | 0.671 | 0.693 | 0.702 |
| AVBlock | 421 | 0.715 | 0.712 | 0.729 | 0.724 |
| STach | 360 | 0.819 | 0.823 | 0.843 | 0.839 |
| BBB | 1527 | 0.726 | 0.745 | 0.747 | 0.748 |
| SB | 398 | 0.766 | 0.787 | 0.777 | 0.782 |
| AEctopy_Junctional | 374 | 0.557 | 0.560 | 0.584 | 0.583 |

Las clases más débiles son AEctopy_Junctional, LVH, TAb y VEctopy. MI no lo es con F1 (0.75; AUPRC 0.839): solo parecía débil con los umbrales F0.5.

## Hallazgos

1. **Receta v2: +1.2 pts de F1-macro con el mismo punto de operación.** Con F0.5 la diferencia es +3.46 (0.6835 → 0.7180), pero con F1 en ambos es +1.22 pts (0.7341 → 0.7463; IC95 bootstrap [+0.60, +1.88], `runs/reports/bootstrap_v2-v1_f1.json`). El AUROC-macro de v2 es menor que el de v1 (0.9377 vs 0.9434), así que buena parte de la mejora de la receta es de umbral y recall. La v1 se eligió por `val_loss` en la época 25 de 35 (gap train/val 0.051 en la mejor; 0.137 al final); la v2 usa aumentación, smoothing, regularización y selección por F1. Solo hay una corrida por receta (semillas 42 y 43): la variabilidad entre semillas no se midió, así que diferencias de ≈ 0.4–1 punto pueden ser ruido de entrenamiento (el IC es solo de muestreo del test).
2. **Ablación negativa NSR-exclusivo.** Forzar "solo NSR" cuando P(NSR) ≥ 0.90 sobre el ensemble F0.5 bajó el F1-macro de 0.7102 a 0.6044 (−10.6 pts) y la precisión-macro solo subió de 0.8239 a 0.8284. Causa: NSR describe el ritmo y coexiste legítimamente con clases morfológicas (NSR+LVH, NSR+BBB…). Regla descartada. Salidas conservadas en `runs/eval/ensemble-nsr090/`. El daño se concentró en PTB-XL (etiquetado multietiqueta completo); fue neutra en CPSC (etiquetado casi single-label).
3. **Ensemble: mejor discriminación y, con F1, mejor F1.** AUROC +1.3 y AUPRC +2.6 sobre v2. Con F1 común el ensemble gana +0.94 pts de F1-macro a v2 (IC95 [+0.50, +1.35], `runs/reports/bootstrap_ensemble-v2_f1.json`) y +2.15 a v1; con F0.5 pierde 0.8 pts frente a v2 y gana 3.5 de precisión. El resultado depende del punto de operación. Con F1 el ensemble supera a v2 en 8 de 12 clases (MI, TAb, LVH, VEctopy, AVBlock, STach, BBB, AEctopy), empata en NSR y pierde 1 pt o menos en AxisDev, AF y SB.
4. **Calibración.**
   - **Ablación 2×2** (v2): a umbral fijo 0.5 la temperatura no cambia ninguna decisión (F1-macro 0.7090 idéntico con/sin escalar, las 12 clases iguales). El sistema con temperatura + umbrales F0.5 da 0.7180. El antiguo 0.670 era una corrida mismatched (umbrales-temp sobre probs crudas) y se retiró.
   - **La temperatura no arregla el ECE.** Se entrena con `pos_weight = neg/pos` y label smoothing 0.05: la probabilidad óptima de un negativo no baja de `pw·s/2 / (pw·s/2 + 1 − s/2)`, que vale 0.21–0.32 en AF, VEctopy, AVBlock, STach, SB y AEctopy_Junctional. Es un desplazamiento (término `b`), y una temperatura solo escala. La calibración afín por clase (`a·logit + b`, a > 0, ajustada en validación sobre los logits crudos) sí lo corrige.
   - **Resultados** (test, 15 bins uniformes): 

     | Sistema | ECE sin calibrar | ECE temperatura | ECE afín | Brier sin calibrar | Brier temperatura | Brier afín |
     |---|---|---|---|---|---|---|
     | v1 | 0.0879 | 0.0922 | **0.0085** | 0.0788 | 0.0783 | **0.0547** |
     | v2 | 0.1910 | 0.1142 | **0.0140** | 0.1011 | 0.0774 | **0.0527** |
     | Ensemble | — | 0.1031 | **0.0086** | — | 0.0715 | **0.0504** |

     Con la afín el ECE máximo por clase es 0.024 (v2) y 0.014 (ensemble), y por origen 0.070 en PTB (n = 99) y ≤ 0.052 en el resto. Fuentes: `calibrate.py --method platt` (`platt_validation.csv`, `calibration_report_platt.json`) y `ensemble_evaluate.py --recalibrate-ensemble` (`platt_ensemble_validation.csv`). Un desplazamiento de prior sin parámetros, `−log(pos_weight)`, da ECE 0.0121 en el ensemble; la afín lo mejora 0.0035.
   - **Efectos laterales.** F1 y AUROC no cambian (F1 +0.0002 a +0.0006 al reoptimizar umbrales; es la misma ordenación de puntuaciones). El AUPRC de v2 sube 0.7794 → 0.7855 porque la temperatura T < 1 satura probabilidades a 1.0 exacto (671 en test) y crea empates. Los umbrales cambian de escala: con la afín quedan en 0.26–0.47 (con temperatura, 0.51–0.93), así que los CSV de umbrales de una y otra calibración no son intercambiables.
5. **Robustez matched** (ResNet v2 a F0.5 con temperatura, F1-macro; base 0.7180): tolera perder 1 derivación (−1.8) y escala ×1.5 (−2.5); degradación moderada con deriva 0.10 (−7.6), dropout ×3 (−7.1), ruido 0.05 (−8.9) y escala ×0.5 (−12.1, coherente con clases de voltaje como LVH); cae con ruido 0.10 (−16.1) y deriva 0.20 (−20.0). Precisión estable (0.73–0.78, base 0.79) con recall guiando la caída: perfil prudente. `robustness.py` solo conoce la temperatura; no se repitió con la calibración afín.
6. **Por origen** (ensemble con calibración afín, F1-macro de test, `runs/reports/errores_origen_ensemble-f1-platt.csv`):

   | Origen | n | F1-macro (12 clases) | F1-macro (solo clases que el origen anota) |
   |---|---|---|---|
   | PTB-XL | 3221 | 0.764 | 0.764 (12) |
   | CPSC | 998 | 0.452 | 0.775 (7) |
   | Georgia | 1588 | 0.692 | 0.692 (12) |
   | CPSC-extra | 505 | 0.546 | 0.546 (12) |
   | PTB | 99 | 0.272 | 0.407 (8) |
   | INCART | 10 | 0.254 | 0.277 (11), n no concluyente |

   El F1 por origen con 12 clases cuenta como 0 las clases que ese origen no anota: CPSC (0.452) es un artefacto de anotación parcial, no un mal rendimiento (0.775 sobre sus 7 clases). La regla de "sin anotar" sale solo de train (celda origen×clase sin ningún positivo; cpsc_2018 no tiene AxisDev, MI, LVH, STach ni SB; ptb no tiene AxisDev, TAb, SB ni AEctopy_Junctional). Esas celdas se entrenan como negativas, y 201 de los 3061 falsos positivos del ensemble caen en ellas. Enmascararlas en la evaluación sube el F1-macro global de 0.7562 a 0.7657: **es un diagnóstico, no el titular** (solo elimina falsos positivos, así que siempre sube).
7. **Punto F1 para screening.** Con β = 1.0 el recall-macro del ensemble sube a 0.783 frente a 0.629 con F0.5 (a costa de precisión 0.733 vs 0.824), y todas las clases mejoran en F1 (SB 0.658 → 0.777, TAb 0.654 → 0.693 con la tabla de F0.5 del ensemble). Motivado por el caso Q1033 (TAb 0.687 bajo el umbral F0.5 de 0.69, por dilución v1/v2: 0.80 vs 0.57). Con la calibración afín el mismo F1 se reparte distinto: precisión 0.747 y recall 0.768 (con temperatura, 0.733 y 0.783).
8. **Significación estadística (bootstrap, B = 1000, por registro, test).** v2 − v1 con F1: +0.0122 [+0.0060, +0.0188]; ensemble − v2 con F1: +0.0094 [+0.0050, +0.0135]; ensemble afín − ensemble temperatura: +0.0005 [−0.0035, +0.0044]. Con F0.5: v2 0.7180 [0.7090, 0.7270] vs v1 0.6835 [0.6740, 0.6928], diferencia +0.0346 [+0.0255, +0.0430] (`runs/reports/bootstrap_v2-v1.json`, IC por clase incluidos). Son intervalos de muestreo con el sistema fijo: no incluyen la varianza de entrenamiento (una sola semilla por receta).
9. **Sin brecha por sexo; tiempo real.** F1-macro 0.717 mujeres (n = 3041) vs 0.716 hombres (n = 3379), Δ = 0.001 (v2 a F0.5). Inferencia: 5.7 ms/ventana de 10 s en GPU (batch 1, 175 ventanas/s), 4 806 668 parámetros.
10. **Solape de pacientes entre train/val y test (T11).** El reparto es por registro, no por paciente. `tools/check_patient_overlap.py` lo mide con los metadatos de PhysioNet (`ptbxl_database.csv` de PTB-XL 1.0.3 y `RECORDS` de PTB 1.0.0). La correspondencia de nombres del Challenge se comprobó antes de medir: en PTB-XL, edad y sexo coinciden en 21 506 de 21 506 registros comparables (`HR<n>` = `ecg_id` n); en PTB, los registros de un mismo paciente de `RECORDS` (`S<k>` = línea k) comparten sexo en el 100 % y edad en el 99.8 % de 468 pares.

    | Origen | Registros con paciente conocido | Pacientes | Test con paciente en train | Test con paciente en train o val |
    |---|---|---|---|---|
    | PTB-XL | 21 799 de 21 837 | 18 869 | 598 de 3 215 (18.6 %) | 673 (20.9 %) |
    | PTB | 516 | 262 | 62 de 99 (62.6 %) | 65 (65.7 %) |

    En PTB-XL, 38 registros del Challenge (6 de test) no figuran en la versión 1.0.3 y no se pueden asignar. Con las decisiones del ensemble (calibración afín, F1) y bootstrap de 1 000 remuestreos: PTB-XL tiene F1-macro 0.7592 en los registros cuyo paciente aparece en train o val (n = 673) y 0.7638 en los demás (n = 2 542); la diferencia (sin − con) es +0.0046, IC95 [−0.0260, +0.0385], así que no hay inflación detectable. PTB tiene 0.8095 con solape (n = 65) y 0.4208 sin él (n = 34), IC95 de la diferencia [−0.5771, −0.1556]: allí el rendimiento depende de haber visto al paciente. Los subconjuntos tienen distinta mezcla de clases (el macro solo cuenta las clases con positivos), de modo que es compatible con fuga, no una prueba. CPSC, CPSC-extra, Georgia e INCART no traen identificador de paciente: no se pudo medir. Al menos 738 de 6 421 registros de test (11.5 %) comparten paciente con train o val. La condición de la tarea T11 (≥ 5 % en un origen con n ≥ 500) se cumple en PTB-XL; el reparto agrupado por paciente invalidaría HDF5, checkpoints, calibración y bundle, y queda como decisión del autor. Salida: `runs/reports/solape_pacientes.json`.

## Corridas v3 (búsqueda de hiperparámetros, no incluidas en el modelo final)

La búsqueda con Optuna (`runs/models/optuna_cinc2020.db`) hizo 3 trials: dos completos y uno fallido sin causa registrada. La receta de cada trial solo existe en `saved/tune_v3/*/*/config_used.json` (`examples/cinc2020/config_resnet_v3.json` no se escribió ni se versionó). El trial 0 (lr 1.07e-3, weight decay 3.7e-5, dropout 0.19, label smoothing 0.055, semilla 43, 40 épocas) dio F1-macro 0.7499 y AUROC 0.9454 en test con F1; continuarlo 33 épocas más (v3-long) dio 0.7501 y AUROC 0.9403. Frente a v2 con el mismo punto de operación (0.7463) la diferencia no es significativa (+0.0036, IC95 [−0.0021, +0.0090]). Como tercer miembro, el ensemble v1+v2+v3-long llega a 0.7625 de F1-macro (+0.0069 sobre v1+v2); es una observación sin réplicas de semilla y la demo no soporta tres modelos, así que no forma parte del modelo recomendado. Salidas: `runs/eval/eval-v3-trial0/`, `eval-v3-trial1/`, `eval-v3-long/`.

## Modelo final recomendado

**Ensemble v1+v2** (promedio simple, α = 0.5) con calibración afín por modelo y recalibración del promedio. Dos puntos de operación: **F1** (F1-macro 0.7562, recall 0.768, precisión 0.747; recomendado para screening y para la demo webapp, donde un FN cuesta más que un FP) y **F0.5** (precisión 0.824, titular de precisión/discriminación; calculado con temperatura). ResNet v2 sola queda como mejor modelo individual. La webapp demuestra el ensemble (`--model` + `--model-b` + `--platt`/`--platt-b`/`--platt-ensemble` + `--eval-dir runs/eval/ensemble-f1-platt`; las variantes con temperatura siguen disponibles con `--temperatures`/`--temperatures-b`). La demo desplegable (`models/`, `deploy/`) empaqueta el punto **F1** con calibración afín.

## Artefactos

- `runs/eval/eval-resnet/`, `runs/eval/eval-resnet-v2/` (F0.5, temperatura; contienen además `platt_validation.csv` y `calibration_report_platt.json`), `runs/eval/eval-resnet-f1/`, `runs/eval/eval-resnet-v2-f1/` (F1, temperatura), `runs/eval/eval-resnet-platt-f1/`, `runs/eval/eval-resnet-v2-platt-f1/` (F1, afín): métricas, umbrales, predicciones y curvas.
- `runs/eval/ensemble/` (F0.5, temperatura), `runs/eval/ensemble-f1/` (F1, temperatura), `runs/eval/ensemble-f1-platt/` (F1, afín; incluye `platt_ensemble_validation.csv`). `deploy/results/ensemble-f1-platt/` es la copia versionada de esta última para la demo. `runs/eval/ensemble-nsr090/`: ablación NSR-exclusivo.
- `runs/reports/model_comparison.csv` (v1 vs v2, F0.5), `model_comparison_f1.csv` (v1 vs v2, F1), `comparacion_ensemble.csv` (v2 vs ensemble, F0.5), `comparacion_ensemble_f1.csv` (v2 vs ensemble, F1), `comparacion_calibracion_f1.csv` (temperatura vs afín). Todas llevan la columna `threshold_beta`.
- `runs/reports/solape_pacientes.json`: solape de pacientes (hallazgo 10; `tools/check_patient_overlap.py`, con los metadatos de PhysioNet en `data/externo/`).
- `runs/reports/bootstrap_*.json`: IC. `runs/reports/errores_origen*.csv` y `*.masked.json`: errores por origen y diagnóstico de celdas sin anotar. `runs/reports/robustness.csv`, `runs/reports/sex_breakdown.json`.
- `runs/figs/figs_eval_v2/`: figuras empíricas (generadas con la tabla de F0.5; no se regeneraron con las evaluaciones nuevas).
- `runs/figs/figs_eval_final/`: las mismas figuras regeneradas con el sistema final (ensemble con calibración afín, F1) a partir de la copia versionada de su evaluación: `python examples/cinc2020/make_figures.py --eval-dir deploy/results/ensemble-f1-platt --outdir runs/figs/figs_eval_final`.
