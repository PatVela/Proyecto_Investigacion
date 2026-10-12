# Resultados finales — ResNet-34 CINC2020-12

## Configuración experimental

- Datos: PhysioNet/CinC Challenge 2020 (6 subconjuntos), esquema agrupado v2 de 12 clases SNOMED.
- Splits: train 33 759 / validation 6 486 / test 6 421 ventanas (10 s, 12 derivaciones, 500 Hz).
- Registros: 43 101 sin exclusiones (30 194/6 486/6 421; 70/15/15).
- Punto de operación: temperature scaling por clase (tuneado en validación) + umbrales por clase
  optimizados con **F0.5** (objetivo: menos falsos positivos con F1-macro alto).
- Modelos: ResNet v1 (receta base, 80 épocas, selección por `val_loss`), ResNet v2
  (aumentación + label smoothing + 150 épocas + selección por F1-macro), Ensemble (promedio v1+v2).

## Tabla principal (test, n = 6421)

| Métrica | ResNet v1 | ResNet v2 | Ensemble F0.5 | Ensemble F1 |
|---|---|---|---|---|
| F1-macro | 0.683 | 0.718 | 0.710 | **0.756** |
| F1-micro | 0.731 | 0.745 | 0.747 | **0.783** |
| Precisión-macro | 0.818 | 0.789 | **0.824** | 0.733 |
| Recall-macro | 0.593 | 0.673 | 0.629 | **0.783** |
| Especificidad-macro | 0.976 | 0.974 | **0.977** | 0.945 |
| AUROC-macro | 0.943 | 0.938 | **0.951** | **0.951** |
| AUPRC-macro | 0.784 | 0.779 | **0.805** | **0.805** |
| Exact-match | 0.437 | 0.459 | 0.455 | **0.472** |

Ensemble F1 = mismo promedio v1+v2 y mismas temperaturas, con umbrales optimizados con
beta=1.0 (`ensemble-f1/`). En validación (n=6486): F1-macro 0.768, F1-micro 0.792,
AUROC 0.953, AUPRC 0.814, exact-match 0.499.

## F1 por clase (test)

| Clase | Soporte | v1 | v2 | Ensemble |
|---|---|---|---|---|
| NSR | 3249 | 0.906 | 0.898 | 0.903 |
| AF | 568 | 0.886 | 0.901 | 0.893 |
| STach | 360 | 0.741 | 0.823 | 0.813 |
| SB | 398 | 0.668 | 0.787 | 0.658 |
| AVBlock | 421 | 0.638 | 0.718 | 0.674 |
| AxisDev | 1072 | 0.708 | 0.708 | 0.695 |
| MI | 1651 | 0.681 | 0.692 | 0.679 |
| BBB | 1527 | 0.671 | 0.658 | 0.699 |
| VEctopy | 395 | 0.660 | 0.675 | 0.671 |
| TAb | 1786 | 0.608 | 0.623 | 0.654 |
| LVH | 800 | 0.525 | 0.594 | 0.620 |
| AEctopy_Junctional | 374 | 0.510 | 0.540 | 0.562 |

## Hallazgos

1. **Receta v2: +3.45 pts F1-macro** (0.683 → 0.718). La v1 se estancó en época 37 con gap
   train/val de 0.125 (sobreajuste); la v2 (aumentación, smoothing, regularización, selección por F1)
   generaliza mejor, sobre todo en recall (+8 pts).
2. **Ablación negativa NSR-exclusivo.** Forzar "solo NSR" cuando P(NSR) ≥ 0.90 costó
   **−11.3 pts F1-macro sin ganancia de precisión** (0.605 vs 0.718). Causa: NSR describe el ritmo y
   coexiste legítimamente con clases morfológicas (NSR+LVH, NSR+BBB…). Regla descartada.
   El daño se concentró en PTB-XL (etiquetado multietiqueta completo); fue neutra en CPSC
   (etiquetado casi single-label).
3. **Ensemble: mejor discriminación y precisión.** AUROC +1.3, AUPRC +2.6 y precisión +3.5 sobre v2,
   a costa de −0.8 F1-macro y −4.4 recall. Eleva las clases morfológicas débiles (TAb, LVH, BBB)
   pero diluye SB/AVBlock donde v2 es muy superior. Bajo el objetivo F0.5 declarado, el ensemble gana.
4. **Calibración (ablación 2×2).** A umbral fijo 0.5, la temperatura no cambia ninguna
   decisión (F1-macro 0.7090 idéntico con/sin escalar, las 12 clases iguales). El sistema
   final (temp + umbrales F0.5) da 0.7180. La temperatura aporta probabilidades fiables
   (ECE 0.191→0.114, Brier 0.099→0.075) y escala comparable para el ensemble. El antiguo
   0.670 era una corrida mismatched (umbrales-temp sobre probs crudas) y se retiró.
5. **Robustez matched** (ResNet v2, F1-macro; base 0.7180 con temps): tolera perder
   1 derivación (−1.8) y escala ×1.5 (−2.5); degradación moderada con deriva 0.10 (−7.6),
   dropout ×3 (−7.1), ruido 0.05 (−8.9) y escala ×0.5 (−12.1, coherente con clases de
   voltaje como LVH); cae con ruido 0.10 (−16.1) y deriva 0.20 (−20.0). Precisión estable
   (0.73–0.78, base 0.79) con recall guiando la caída: perfil prudente.
6. **Por origen** (F1-macro): PTB-XL 0.725 (n=3221) > Georgia 0.630 > CPSC-extra 0.515 >
   CPSC 0.433 (clases ausentes hunden el macro pese a exact-match 0.642) > PTB 0.258 (n=99) >
   INCART 0.160 (n=10, no concluyente).
7. **Métrica Challenge 27 códigos: 0.2348** (F1-macro-27 0.302). Referencia interna solamente:
   no comparable al leaderboard (esquema agrupado de 12 + punto conservador).
8. **Punto de operación F1 para screening.** Con beta=1.0, el ensemble sube a F1-macro 0.756
   (+4.5 sobre F0.5) y recall 0.783 (+15.4), con todas las clases mejorando en F1
   (SB 0.658→0.777, TAb 0.654→0.693); costo: precisión 0.824→0.733. Motivado por el caso
   Q1033 (TAb 0.687 bajo el umbral F0.5 de 0.69, por dilución v1/v2: 0.80 vs 0.57).
9. **v2−v1 significativo (bootstrap, B=1000, test).** v2 0.7180 [0.7090, 0.7270] vs
   v1 0.6835 [0.6740, 0.6928]; diferencia +0.0346 [+0.0255, +0.0430], P(diff>0)=1.000.
   IC por clase en `bootstrap_v2-v1.json`. Solo variabilidad muestral (sistema fijo).
10. **Sin brecha por sexo; tiempo real.** F1-macro 0.717 mujeres (n=3041) vs 0.716
    hombres (n=3379), Δ=0.001. Inferencia: 5.7 ms/ventana de 10 s en GPU (batch 1,
    175 ventanas/s), 4 806 668 parámetros.

## Modelo final recomendado

**Ensemble v1+v2** (promedio simple, temperaturas por modelo). Dos puntos de operación:
**F0.5** (precisión 0.824, titular de precisión/discriminación) y **F1** (F1-macro 0.756,
recall 0.783, recomendado para screening y para la demo webapp, donde un FN cuesta más
que un FP). ResNet v2 sola queda como mejor modelo individual en F1-macro con F0.5 (0.718).
La webapp demuestra el ensemble (`--model` + `--model-b` + `--temperatures`/`--temperatures-b` +
`--eval-dir ensemble` o `--eval-dir ensemble-f1`).

## Artefactos

- `eval-resnet/`, `eval-resnet-v2/`, `ensemble-noexcl/`: métricas, umbrales, predicciones, curvas (`eval-resnet-v2-nsrexcl/` conserva la ablación NSR).
- `figs_eval_v2/`: figuras empíricas. `bootstrap_v2-v1.json`: IC. `sex_breakdown.json`: desglose por sexo.
- `ensemble-f1/`: mismo ensemble con umbrales F1 (punto de screening).
- `exp-files/model_comparison.csv`: v1 vs v2. `comparacion_ensemble.csv`: v2 vs ensemble.
- `exp-files/robustness.csv`, `metrica-challenge/`, `errores_origen.csv`.
