# Análisis de mejoras — ECG CinC2020-12 (auditoría de solo lectura de `main`)

- Fecha: 2026-10-13. Rama `main`, HEAD `f6b10a225b379a72f2d5d4399625e83c1c34d9ef` (Merge PR #15). Árbol limpio salvo `.claude/`.
- T11 ejecutada el 2026-10-15: resultado en `docs/resultados_finales.md` (hallazgo 10). Se cumple la condición de la tarea (≥ 5 % en un origen con n ≥ 500: PTB-XL 18.6 %), pero el F1 de PTB-XL no cambia con y sin solape; el reparto agrupado por paciente queda como decisión del autor.
- La métrica Challenge se eliminó el 2026-10-15 (del Challenge solo se usa su base de datos, no se compara con sus resultados): `challenge_score.py`, `official/weights.csv` y `official/evaluate_12ECG_score.py` ya no existen y las menciones de abajo son históricas.
- Estado posterior: M1 (T1–T4) y M2 (T7) se ejecutaron en la rama `feat/protocolo-comun-y-calibracion-afin`; sus resultados medidos están en `docs/resultados_finales.md`. M3 en adelante no se han ejecutado. Las cifras de este informe son las de la auditoría (solo lectura), salvo las correcciones señaladas en 2.3 y la sección 8.
- Alcance: etapas 01–08 de `CONTEXT.md`. No se entrenó, no se evaluó con los scripts del pipeline y no se escribió nada en `runs/`, `saved/`, `data/`, `models/` ni `deploy/`. Este es el único archivo escrito.
- Rutas en texto plano o `code`, sin enlaces relativos (`tests/test_docs_links.py` valida los `.md` versionados).
- Etiquetas de certeza:
  - `[VERIFICADO]`: leído o recalculado por mí con una herramienta en esta auditoría (la fuente va en la misma línea).
  - `[VERIFICADO-SUB]`: reproducido solo por un subagente de solo lectura; yo no lo re-ejecuté. Es una desviación deliberada del esquema de dos etiquetas pedido, para no inflar la certeza. En la sección 5 estas evidencias cuentan como `[HIPÓTESIS]` salvo que yo las haya confirmado aparte.
  - `[HIPÓTESIS]`: mecanismo plausible sin prueba directa.
- Scripts de cálculo: viven en el directorio temporal de la sesión (no versionados). Leen `runs/eval/*/predictions_*.csv`, `data/cinc2020_12/*` y `saved/**/history.csv` en solo lectura. Las cifras salen de esos archivos, no de la documentación.

## 1. Resumen ejecutivo

1. Mejor sistema documentado: ensemble v1+v2 con umbrales F1, F1-macro test 0.7556 (val 0.7685), AUROC 0.9505, AUPRC 0.8052 (`runs/eval/ensemble-f1/metrics_global.csv`). Brecha val→test 0.013, del tamaño de la desviación bootstrap (0.004) por tres.
2. Las cifras titulares mezclan puntos de operación: con F0.5 v2−v1 = +0.035; con F1 común es +0.0125 [0.006, 0.019]. Además `runs/eval/ensemble/` en disco es la ablación NSR-exclusivo (F1 0.6044), no el F0.5 = 0.710 que cita `docs/resultados_finales.md`.
3. Fuga entre splits: ninguna ventana ni registro repetido, pero 3.0 % del test y 2.1 % de val tienen una señal idéntica (md5) a otra de un split anterior; quitarlos mueve el F1 en −0.0002. La fuga por paciente no se puede medir (las cabeceras no traen ID de paciente).
4. Métrica Challenge 0.2348 es un artefacto de procedimiento (`challenge_score.py` no acepta `--temperatures`); con probabilidades calibradas y las 3 clases equivalentes fusionadas (24) salen ~0.289 (v2 con umbrales F0.5) y ~0.351 (v2 y ensemble con umbrales F1: sin ventaja del ensemble). El techo del esquema de 12 grupos es 0.503. Desglose reproducido solo por subagentes.
5. ECE 0.114 (v2, test): causa identificada, `pos_weight=neg/pos` × label smoothing fija un suelo de probabilidad de 0.21–0.32 en 6 clases. Una calibración afín por clase lo baja a 0.0135 sin cambiar la clasificación (F1 β=1 +0.0003 a +0.0007, AUROC idéntico).
6. Clases débiles reales: AEctopy_Junctional (F1 0.584), LVH (0.659), TAb (0.693) y VEctopy (0.693); MI no lo es (0.753). Orígenes: CPSC-extra y PTB; CPSC "0.449" es artefacto de clases sin anotar (0.769 sobre las 7 que anota).
7. Top-5 por impacto/esfuerzo: M1 protocolo único de reporte, M2 calibración afín, M3 ensemble de 3 miembros (+0.0069 F1 test, IC [0.0037, 0.0100]; con selección y umbrales remuestreados el revisor obtiene +0.0045, IC [0.000, +0.009]), M4 réplicas de semilla, M5 selección de checkpoint alineada con AUROC. Solo M1–M3 tienen efecto medido; M3 y M5 quedan como hipótesis hasta M4, que es la condición para decidir el resto.
8. Mayor incertidumbre: el techo parece venir de varianza de entrenamiento y de datos (AUROC val ≈ 0.94 en las tres recetas), no de capacidad, y la variabilidad entre semillas nunca se midió; con ±0.004 de ruido, varias diferencias del repo no son concluyentes.
9. Seguridad aparte (sección 7): ReDoS cuadrático en `clean_path_text` y `.mat` sin tope de descompresión.
10. Contra-examen (Fase 5, secciones 5.2–5.3): la BCE enmascarada por origen y el criterio de selección se degradaron a hipótesis; el F1 enmascarado por origen queda solo como diagnóstico (no titular) y el ensemble de 3 miembros bajó a prioridad 1.0.

## 2. Línea base de métricas

### 2.1 Test (n = 6421 ventanas = registros), tal como están en disco

| Sistema | Punto de operación | F1-macro | AUROC-m | AUPRC-m | Exact-match | Fuente |
|---|---|---|---|---|---|---|
| ResNet v1 | F0.5 + T | 0.6835 | 0.9434 | 0.7838 | 0.4368 | `runs/eval/eval-resnet/metrics_global.csv` |
| ResNet v2 | F0.5 + T | 0.7180 | 0.9377 | 0.7794 | 0.4590 | `runs/eval/eval-resnet-v2/metrics_global.csv` |
| Ensemble v1+v2 | F0.5 (cifra de los docs) | 0.7102 | 0.9505 | 0.8052 | 0.4548 | `runs/reports/comparacion_ensemble.csv` (cita `ensemble-noexcl`, directorio inexistente) |
| `runs/eval/ensemble/` en disco | F0.5 + NSR-exclusivo 0.9 | 0.6044 | 0.9505 | 0.8052 | 0.3875 | `runs/eval/ensemble/metrics_global.csv`, `evaluation_summary.json` |
| Ensemble v1+v2 | F1 | 0.7556 | 0.9505 | 0.8052 | 0.4724 | `runs/eval/ensemble-f1/metrics_global.csv` |
| v3-trial0 (40 ép.) | F1 | 0.7499 | 0.9454 | 0.7917 | 0.4569 | `runs/eval/eval-v3-trial0/metrics_global.csv` |
| v3-long (trial0 + 33 ép.) | F1 | 0.7501 | 0.9403 | 0.7897 | 0.4708 | `runs/eval/eval-v3-long/metrics_global.csv` |
| v3-trial1 | F1 | 0.7296 | 0.9388 | 0.7742 | 0.4283 | `runs/eval/eval-v3-trial1/metrics_global.csv` |

Validación (n = 6486): ensemble F1 0.7685, AUROC 0.9530, AUPRC 0.8139 (`runs/eval/ensemble-f1/metrics_global.csv`).

### 2.2 Todos al mismo punto de operación (F1, β = 1; umbrales de validación) `[VERIFICADO]`

Recalculado desde `runs/eval/*/predictions_{validation,test}.csv` con `optimize_thresholds(beta=1.0)` de `examples/cinc2020/evaluate.py:55-69`. Reproduce los oficiales de v3-long (0.7501), v3-trial0 (0.7499) y ensemble F1 (0.7556).

| Sistema | F1 val | F1 test | brecha | AUROC test | AUPRC test |
|---|---|---|---|---|---|
| v1 | 0.7454 | 0.7341 | 0.0113 | 0.9434 | 0.7838 |
| v2 | 0.7626 | 0.7463 | 0.0163 | 0.9377 | 0.7794 |
| v3-trial0 | 0.7551 | 0.7499 | 0.0052 | 0.9454 | 0.7917 |
| v3-long | 0.7600 | 0.7501 | 0.0099 | 0.9403 | 0.7897 |
| v1+v2 | 0.7685 | 0.7556 | 0.0129 | 0.9505 | 0.8052 |
| v1+v2+v3-long | 0.7741 | 0.7625 | 0.0116 | 0.9526 | 0.8121 |

Bootstrap pareado por registro, B = 1000, semilla 0, test (solo variabilidad muestral, sistema fijo):

- v2 − v1 = +0.0125, IC95 [+0.0061, +0.0189].
- (v1+v2+tl) − (v1+v2) = +0.0069, IC95 [+0.0037, +0.0100].
- v2 − (v1+v2) = −0.0092 [−0.0134, −0.0048].

### 2.3 Contraste con `docs/resultados_finales.md`

| Afirmación del documento | Qué hay en disco | Estado |
|---|---|---|
| Ensemble F0.5 = 0.710 en `runs/eval/ensemble/` | ese directorio da 0.6044 (NSR-exclusivo); 0.7102 solo se reproduce con sus probabilidades y umbrales | `[VERIFICADO]` discrepancia |
| v2 − v1 = +0.0346 | +0.0346 con F0.5 para v2/v1; +0.0125 con F1 común; AUROC de v2 < v1 (−0.0057) | `[VERIFICADO]` el punto de operación explica 2/3 |
| "v1 se estancó en época 37, gap 0.125" | `saved/**/history.csv`: 35 épocas, mejor época 25, gap 0.051 en la mejor (0.137 al final) | `[VERIFICADO]` impreciso |
| ECE 0.191→0.114, Brier 0.099→0.075 (v2) | ECE test 0.1142 reproducido; Brier test 0.0774 (el documento dice 0.075). Al ejecutar M2 se recalculó el "antes" desde los logits crudos: ECE 0.191 (reproducido) y Brier 0.1011 (el documento dice 0.099); falta `runs/eval/eval-resnet-v2/calibration_report.json` | `[VERIFICADO]` (corrige la versión anterior de esta fila) |
| Challenge 0.2348 | `runs/eval/metrica-challenge/challenge_metric.json`; ver H04.2 | cifra `[VERIFICADO]`; que sea un artefacto de procedimiento es `[VERIFICADO-SUB]` |
| Hallazgo 3: el ensemble cuesta −0.8 pts de F1-macro frente a v2 | con F1 común el ensemble supera a v2 en +0.0092 (v2 − ensemble = −0.0092, IC [−0.0134, −0.0048]); el signo se invierte | `[VERIFICADO]` (el revisor obtiene +0.009) |
| Modelo final: ensemble v1+v2 | `deploy/ensemble.env`: `ECG_MODEL=models/resnet34_v1.pt`, `ECG_MODEL_B=models/resnet34_v2.pt`, `ECG_ALPHA=0.5` | coherente |
| v3 | ausente de `docs/`; solo `examples/cinc2020/README.md:72` dice que `tune_optuna.py` escribe `config_resnet_v3.json`, "no versionado" | ver pista 2 |

### 2.4 Tres clases más débiles y dos orígenes más débiles (ensemble F1, test) `[VERIFICADO]`

Fuente: `runs/eval/ensemble-f1/metrics_per_class.csv`.

| Clase | F1 | AUPRC | Precisión / Recall | Soporte | Umbral |
|---|---|---|---|---|---|
| AEctopy_Junctional | 0.584 | 0.602 | 0.536 / 0.642 | 374 | 0.77 |
| LVH | 0.659 | 0.715 | 0.617 / 0.708 | 800 | 0.73 |
| TAb | 0.693 | 0.739 | 0.657 / 0.734 | 1786 | 0.56 |
| VEctopy (empata con TAb) | 0.693 | 0.727 | 0.767 / 0.633 | 395 | 0.93 |
| MI (referencia) | 0.753 | 0.839 | 0.708 / 0.804 | 1651 | 0.56 |

Con v2 a F0.5 (tabla de los docs) las tres más débiles son las mismas: AEctopy 0.540, LVH 0.594, TAb 0.623; MI queda en 0.692.

Por origen: `runs/eval/ensemble-f1/predictions_test.csv` + `data/cinc2020_12/split_assignments.csv`.

| Origen | n | F1-macro (12 clases) | F1-macro (solo clases con positivos) |
|---|---|---|---|
| ptb-xl | 3221 | 0.764 | 0.764 (12) |
| georgia | 1588 | 0.697 | 0.697 (12) |
| cpsc_2018 | 998 | 0.449 | 0.769 (7) |
| cpsc_2018_extra | 505 | 0.531 | 0.579 (11) |
| ptb | 99 | 0.271 | 0.542 (6) |
| st_petersburg_incart | 10 | 0.219 | 0.439 (6) — n demasiado pequeño |

Con v2 a F0.5 (`runs/reports/errores_origen.csv`): ptb-xl 0.725, georgia 0.630, cpsc_extra 0.515, cpsc 0.433, ptb 0.258, incart 0.160.

### 2.5 Calibración (test, 15 bins uniformes, `ecg.calibration`) `[VERIFICADO]`

| Sistema | ECE | Brier | ECE tras Platt (ajuste en val) | Brier tras Platt | Prob. media / prevalencia |
|---|---|---|---|---|---|
| v1 | 0.0922 | 0.0783 | 0.0086 | 0.0547 | 0.254 / 0.164 |
| v2 | 0.1142 | 0.0774 | 0.0135 | 0.0526 | 0.278 / 0.164 |
| ensemble F1 | 0.1031 | 0.0715 | 0.0095 | 0.0506 | 0.266 / 0.164 |
| v3-long | 0.1232 | 0.0800 | 0.0151 | 0.0523 | 0.286 / 0.164 |

Fuente: `runs/eval/*/predictions_test.csv`, ajuste afín `a·logit(p)+b` por clase sobre `predictions_validation.csv`. `runs/eval/eval-resnet/calibration_report.json`: la temperatura **empeoró** el ECE de val de v1 (0.0861 → 0.0909).

### 2.6 Otras líneas base

- Métrica Challenge (27 códigos): 0.2348, F1-macro-27 0.3022 (`runs/eval/metrica-challenge/challenge_metric.json`).
- `pytest -q -p no:cacheprovider`: 360 passed en 372.63 s (6:12). Salida en el directorio temporal de la sesión.
- Entrenamiento (`saved/**/history.csv`): v1 35 épocas, mejor 25, 52 s/época; v2 107 épocas, mejor 92, 66 s/época (1.97 h); v3-long 33 épocas más, mejor 58, 67 s/época.

## 3. Hallazgos por etapa 01–08

Formato: hallazgo · archivo:línea · evidencia · métrica afectada · certeza.

### Etapa 01 — Datos

- **H01.1** Split por registro, estratificado por vector de etiquetas, sin agrupar por paciente ni origen · `ecg/datasets.py:138-171`. La expansión a ventanas ocurre después del split (`:351-355`, `:571-582`); 0 `record_name` compartidos entre `train/val/test.h5` y entre los tres splits de `split_assignments.csv` (comprobado por mí con intersecciones de conjuntos) · afecta la validez de todas las métricas · `[VERIFICADO]`.
- **H01.2** Fuga por paciente: no medible. PTB-XL tiene 21 799 registros de 18 869 pacientes (https://physionet.org/content/ptb-xl/1.0.3/) y PTB 549 registros de 290 sujetos (https://physionet.org/content/ptbdb/1.0.0/; la misma página también habla de 294 sujetos). Las cabeceras `.hea` no traen ID de paciente. Una simulación con reparto uniforme dio ~19 % del test PTB-XL y ~60 % del test PTB con un registro del mismo paciente en train (supuesto, no medido) · afecta todas las métricas, sobre todo PTB y la elección de umbrales en val · `[HIPÓTESIS]`.
- **H01.3** Duplicados exactos entre splits (md5 de la ventana en `data/cinc2020_12/*.h5`): 326 ventanas idénticas entre splits y 511/18/17 duplicados dentro de train/val/test. 133 registros de val (2.1 %) tienen copia en train; 193 del test (3.0 %) tienen copia en un split anterior (152 en train); 61 y 81 con etiqueta distinta a la de su copia. Origen: CPSC (91) y CPSC-extra (61) dominan el test. F1-macro test del ensemble F1 sin ellos: 0.7554 vs 0.7556 (−0.0002); exact-match de los 81 con etiqueta distinta: 0.21 · afecta ruido de etiqueta y estimación, no el F1 titular · `[VERIFICADO]`.
- **H01.4** 24 registros "todo cero" solo en test (18 son PTB), ninguno en train/val · `data/cinc2020_12/split_summary.csv` (`ALL_ZERO`) · efecto global < 0.4 % · `[VERIFICADO]` (la cifra por origen es `[VERIFICADO-SUB]`).
- **H01.5** Anotación parcial por origen. `cpsc_2018` tiene 0 positivos en train de AxisDev, MI, LVH, STach y SB; `ptb` en AxisDev, TAb, SB y AEctopy_Junctional. Esas celdas "sin anotar" se entrenan como negativas (`ecg/train.py:260`, BCE sin máscara) · 218 de los 3220 FP del ensemble F1 en test caen en celdas sin positivos en train; en CPSC: MI 73, SB 37 (otros 9 de PTB), STach 31, LVH 27, AxisDev 11 `[VERIFICADO]`. Evaluar cada clase solo en orígenes con ≥ 1 positivo en train (regla sin mirar test): F1-macro test del ensemble F1 0.7556 → 0.7655 (+0.0098; STach +0.035, SB +0.045, MI +0.016, LVH +0.011) · `[VERIFICADO]`. Que el entrenamiento aprenda la huella del origen es `[HIPÓTESIS]`: el contraste AxisDev predicho 1.1 % en CPSC vs 30 % en PTB-XL no la prueba, porque la prevalencia real en PTB-XL es ~27 % (train, `split_assignments.csv`) y la de CPSC es desconocida. La regla "0 positivos" tampoco enmascara NSR ni AxisDev en CPSC-extra (11 y 1 positivos de 2464), así que es frágil. Cota superior si desaparecieran todos esos FP sin máscara en la evaluación: +0.0084 de F1-macro (revisor, `[VERIFICADO-SUB]`); con la tasa de FP de PTB-XL y Georgia como contrafactual, los FP de AxisDev/MI/LVH subirían +62/+48/+48 y el efecto neto sería −0.004.
- **H01.6** Val/test usan 1 ventana central (`ecg/datasets.py:581`, `ecg/signal.py:271-298`) pero la etiqueta es del registro completo. 63 % de CPSC y 59 % de CPSC-extra duran > 10 s. Recall de VEctopy en test: 0.43 en registros > 10 s (n = 107) vs 0.71 en 10 s (n = 288) · `[VERIFICADO-SUB]`. La webapp agrega hasta 4 ventanas con máximo (`webapp/prediction.py:51-52,663-668`) con umbrales ajustados a 1 ventana: sesgo a FP en registros ≥ 20 s (solo demo) · `[VERIFICADO-SUB]`. **Corregido** (revisión de umbrales): en 1019 registros largos de test, FP 426 (ventana centrada) → 511 (máximo) → 433 (media de calibradas por ventana); F1 macro 0.6108 / 0.6052 / 0.6109. La webapp usa ahora la media de las ventanas calibradas (`webapp/prediction.py:aggregate_window_probabilities`); no requiere reentrenar.
- **H01.7** Los grupos agrupan códigos heterogéneos (`ecg/schema.py:47-140`): 197/800 positivos de LVH en test son solo atriales (recall 0.61 vs 0.74); 456/1651 de MI son solo isquemia; en BBB, "QRS anormal" es el 31 % (LQRSV recall 0.29); AEctopy: SVPB recall 0.24, precisión 0.54 · explica parte del techo de LVH, MI, BBB y AEctopy · `[VERIFICADO-SUB]`, causalidad `[HIPÓTESIS]`.
- **H01.8** Preprocesado coherente entre orígenes (desviación 0.14–0.15 mV por origen); clip ±5 mV afecta 1.56 % (CPSC-extra 5 %); padding en 12/6486 de val · `[VERIFICADO-SUB]`. Pasa-banda 0.5–50 Hz (`ecg/schema.py:212-213`): atenúa espigas de marcapasos y melladuras del QRS · `[HIPÓTESIS]` (no pude confirmar el ancho de banda recomendado por la AHA).
- **H01.9** 0 códigos puntuados sin clase; 191 registros con algún código sin clase (WPW 88, CHD 37) · `[VERIFICADO-SUB]`. `tools/audit_signal_scaling.py:50-51` apunta a `tools/dataset2020`, que no existe · `[VERIFICADO-SUB]`, no afecta métricas.
- **H01.10** Val y test tienen los mismos positivos por clase (p. ej. NSR 3249, AxisDev 1072, MI 1651) salvo SB (397/398): propiedad de la estratificación iterativa con tamaños iguales, no fuga (0 `record_name` comunes) · `data/cinc2020_12/split_summary.csv` · `[VERIFICADO]`.

### Etapa 02 — Entrenar

- **H02.1** El criterio de selección (F1-macro a umbral 0.5 sobre probabilidades crudas, `ecg/train.py:173-189,280-324`) diverge de AUROC/AUPRC. v3-trial0 (mejor época ≈ 40) vs v3-long (mejor época 58): F1@0.5 de val sube de 0.684 a 0.711 mientras AUROC test baja 0.9454 → 0.9403 (`runs/eval/eval-v3-trial0/` y `eval-v3-long/`, `metrics_global.csv`) y el F1 con umbrales β=1 queda igual (0.7499 → 0.7501) · AUROC, AUPRC · `[VERIFICADO]`. En v2, mejor época 92 vs última 107: −0.0033 AUROC test · `[VERIFICADO-SUB]`.
- **H02.2** `pos_weight = neg/pos` sin tope (`ecg/train.py:69-80,260`, rango 1.18–18.07) × label smoothing 0.05 (`train.py:113-118`, `examples/cinc2020/config_resnet_v2.json`). El óptimo analítico de la probabilidad de un negativo es `pw·s/2 / (pw·s/2 + 1 − s/2)`: 0.208 AF, 0.246 VEctopy, 0.275 AVBlock, 0.317 STach, 0.291 SB, 0.277 AEctopy (calculado con los `pw` de `data/cinc2020_12/train.h5`). Las temperaturas de v2 son 0.245 (STach), 0.354 (AF), 0.421 (SB), 0.459 (AVBlock), 0.499 (VEctopy) (`runs/eval/eval-resnet-v2/temperatures_validation.csv`) · explica ECE 0.114 y la saturación (H03.2) · mecanismo `[VERIFICADO]`; que además cueste AUROC en VEctopy/AEctopy/AVBlock frente a v1 está confundido con la aumentación · `[HIPÓTESIS]`. La documentación de PyTorch fija `pos_weight = negativos/positivos` por clase y advierte que valores > 1 aumentan el recall: https://docs.pytorch.org/docs/2.8/generated/torch.nn.BCEWithLogitsLoss.html
- **H02.3** El techo parece de varianza y datos, no de capacidad. AUROC val ≈ 0.94 en v1, v2 y v3; sin aumentación el AUROC de train es 0.989 (v2) frente a 0.941 en val; 4.8 M parámetros y campo receptivo de 7657 muestras (> ventana de 5000) · `[VERIFICADO-SUB]`.
- **H02.4** Schedule: `ReduceLROnPlateau` sobre `val_loss` (`ecg/train.py:266-272`); v2 llega a `min_lr = 5e-5` en la época 53 y pasa 55 épocas en el piso mientras el AUROC cae · `[VERIFICADO-SUB]`. Al reanudar, `set_seed` repite el orden de los datos (`ecg/train.py:286-299`) · `[VERIFICADO-SUB]`.
- **H02.5** Optuna (`runs/models/optuna_cinc2020.db`, abierto `mode=ro`): 3 trials, 2 COMPLETE (0.6839 y 0.6347) y 1 FAIL (#2, sin causa registrada). Solo se muestrean `learning_rate`, `weight_decay`, `conv_dropout` y `label_smoothing`; `max_epochs = 40`; la semilla cambia con el trial (43/44/45 en `saved/tune_v3/*/*/config_used.json`), así que semilla e hiperparámetros van confundidos. Con 10 trials de arranque aleatorio del TPE por defecto, los 3 fueron aleatorios · `[VERIFICADO]`; valores por defecto de TPE/MedianPruner `[VERIFICADO-SUB]` (`examples/cinc2020/tune_optuna.py:34-39,70,118-125`).
- **H02.6** Semillas: v1 = 42, v2 = 43, v3 trial0 y v3-long = 43, trial1 = 44, trial2 = 45; ninguna réplica. Desviación de F1 de val en la meseta de v2 ≈ 0.0037; el máximo de val infla 0.0074 sobre la media · `[VERIFICADO]` (semillas), `[VERIFICADO-SUB]` (ruido).
- **H02.7** Documentación: `docs/resultados_finales.md` dice "época 37, gap 0.125"; `history.csv` da 35 épocas, mejor 25, gap 0.051 · `[VERIFICADO]`.

### Etapa 03 — Calibrar

- **H03.1** El ECE 0.114 viene de H02.2, no de falta de temperatura: una T por clase solo escala el logit y no corrige el sesgo (media predicha 0.278 vs prevalencia 0.164). Platt por clase (`a·logit(p)+b`, ajustado en val) lleva el ECE test de 0.1142 a 0.0135 (v2), de 0.0922 a 0.0086 (v1) y de 0.1031 a 0.0095 (ensemble); Brier de v2 0.0774 → 0.0526 · `[VERIFICADO]` (sección 2.5). Guo et al. (2017) describen la temperatura como una variante de un solo parámetro de Platt y atribuyen la mala calibración a decisiones de arquitectura y entrenamiento (https://arxiv.org/abs/1706.04599), coherente con que aquí falte el término de sesgo.
- **H03.2** T « 1 satura: en float32 aparecen empates en p = 1.0 (STach 322, AF 231) y el AUPRC de STach cae de 0.866 a 0.815 · `[VERIFICADO-SUB]`.
- **H03.3** El ensemble no se recalibra tras promediar (`examples/cinc2020/ensemble_evaluate.py:81-105`); su ECE es 0.1031 · `[VERIFICADO]`.
- **H03.4** Falta `runs/eval/eval-resnet-v2/calibration_report.json` (v1, trial0, trial1 y long sí lo tienen) · `[VERIFICADO]`.

### Etapa 04 — Evaluar

- **H04.1** Umbrales y temperaturas salen solo de validación (`examples/cinc2020/evaluate.py:55-69,250-294`); malla `np.linspace(0.01, 0.99, 99)`, primer máximo con `>` estricto. A F0.5, 6 de 12 clases de v2 quedan en el tope 0.99 de la malla (subirlo a 0.999 mejora VEctopy 0.683 → 0.730 y STach 0.778 → 0.811 en test, `[VERIFICADO-SUB]`). En el punto F1 los umbrales llegan a 0.93 (`runs/eval/ensemble-f1/thresholds_validation.csv`), así que no afecta al punto recomendado · `[VERIFICADO]`.
- **H04.2** `examples/cinc2020/challenge_score.py:186-195` no acepta `--temperatures`: aplica umbrales F0.5 ajustados sobre probabilidades calibradas a probabilidades crudas (`:204-212`). Con probabilidades calibradas (sin fusionar clases): 0.2988 (v2 con umbrales F0.5) y 0.3588 (ensemble con umbrales F1), que mezclan puntos de operación; con F1 en ambos, v2 = 0.358 y ensemble = 0.359, es decir sin mejora; con crudas y umbral 0.5: 0.3638 · `[VERIFICADO-SUB]` (C y revisor concuerdan). El techo de un clasificador perfecto de grupos es 0.503 (`:80-99,129-143`) · `[VERIFICADO-SUB]`. Usa 27 columnas; el código oficial fusiona tres pares equivalentes (`equivalent_classes = [['713427006','59118001'], ['284470004','63593006'], ['427172004','17338001']]` en https://github.com/physionetchallenges/evaluation-2020/blob/master/evaluate_12ECG_score.py), es decir 24, y la diferencia infla ~+0.009: con fusión v2 con F0.5 = 0.289, v2 con F1 = 0.351 y ensemble = 0.351 · `[VERIFICADO-SUB]`. `challenge_metric.json` cita `eval-resnet-v2-noexcl` y una ruta de checkpoint de otro checkout; ninguno existe · `[VERIFICADO]`.
- **H04.3** `runs/eval/ensemble/` es la ablación NSR-exclusivo (`evaluation_summary.json`: `nsr_exclusive_min_prob = 0.9`, F1 test 0.6044), no el F0.5 de los docs · `[VERIFICADO]`.
- **H04.4** `examples/cinc2020/ensemble_evaluate.py:149-150` admite exactamente 2 checkpoints (`checkpoint_a`, `checkpoint_b`); `--alpha` por defecto 0.5, sin barrido en el repo ni en `docs/` · `[VERIFICADO]`.

### Etapa 05 — Comparar

- **H05.1** `examples/cinc2020/compare_models.py:19-30` compara filas con distinto `threshold_beta` sin avisar; `runs/reports/model_comparison.csv` y `comparacion_ensemble.csv` citan directorios inexistentes (`eval-resnet-v2-noexcl`, `ensemble-noexcl`) · `[VERIFICADO]`.
- **H05.2** Los JSON de bootstrap existentes cubren solo v1 y v2−v1 (`runs/reports/bootstrap_v1.json`, `bootstrap_v2-v1.json`); no hay IC del ensemble, del punto F1 ni de v3. El bootstrap es por registro con el sistema fijo (`examples/cinc2020/bootstrap_ci.py:88-107`): no incluye varianza de entrenamiento; v1 usa semilla 42 y v2 la 43, así que receta y semilla van confundidas · `[VERIFICADO]`.

### Etapa 06 — Robustez y sesgo

- **H06.1** `examples/cinc2020/robustness.py:62-71`: solo v2 a F0.5, una semilla y un sorteo por perturbación, sin IC; perturba después del preprocesado, así que la deriva de línea base no pasa por el pasa-banda · `[VERIFICADO-SUB]`.
- **H06.2** `examples/cinc2020/sex_breakdown.py:46-66`: usa v2 a F0.5 y reporta solo macro (0.717 vs 0.716). Por clase, en el ensemble F1, hay brechas que se cancelan: MI 0.72 mujeres / 0.775 hombres, BBB 0.70 / 0.77, TAb 0.72 / 0.67, AEctopy 0.62 / 0.55 · `[VERIFICADO-SUB]`.
- **H06.3** `examples/cinc2020/error_analysis.py:74` cuenta como 0 las clases ausentes del origen: "CPSC 0.433" es artefacto (0.769 sobre sus 7 clases presentes) · `[VERIFICADO]` (sección 2.4).

### Etapa 07 — Figuras

- Sin hallazgos sobre el contenido de las figuras: no las abrí. `tests/test_make_figures.py:36-56` solo comprueba existencia y tamaño · `[VERIFICADO-SUB]`. Las figuras se regeneran con `make_figures.py` cuando cambien las tablas (T4).

### Etapa 08 — Despliegue

- **H08.1** El bundle (`deploy/ensemble.env`) empaqueta v1+v2 con α = 0.5 y el punto F1 (`deploy/results/ensemble-f1/`); v3 no está. `ensemble_evaluate.py` y la webapp (`--model-b`) solo admiten 2 modelos: un ensemble de 3 requiere código · `[VERIFICADO]`.
- **H08.2** Entrenamiento ≠ inferencia solo en agregación: ambos usan el mismo `preprocess_ecg_array`; el CLI (`ecg/predict.py:304-317`) usa un `--threshold` global y no aplica temperaturas ni ensemble, así que su salida no coincide con la de `evaluate.py` ni con la web · `[VERIFICADO-SUB]`.
- **H08.3** `torch` 2.14.0+cu130 instalado frente al pin `torch==2.8.0` de `requirements.txt:4`; `deploy/manifest.json` registra la versión usada · `[VERIFICADO-SUB]`. Sin CI (`.github/` no existe) · `[VERIFICADO-SUB]`.
- Hallazgos de seguridad: sección 7.

## 4. Veredicto de las 4 pistas

### Pista 1 — Fuga entre splits

| Sub-afirmación | Veredicto | Evidencia |
|---|---|---|
| Ventanas del mismo registro en splits distintos | **Descartada** | 0 `record_name` compartidos; las ventanas se expanden después del split (`ecg/datasets.py:351-355`); val/test usan 1 ventana central (`:581`). `[VERIFICADO]` |
| Split estratificado por vector de etiquetas, no por paciente ni origen | **Confirmada** como diseño | `ecg/datasets.py:138-171` (MultilabelStratifiedShuffleSplit). `[VERIFICADO]` |
| Esa fuga por paciente infla las métricas | **No verificable** | Sin ID de paciente en `.hea`. PTB-XL: 21 799 registros / 18 869 pacientes; PTB: 549 / 290. Magnitud simulada ~19 % (PTB-XL) y ~60 % (PTB) del test con registro hermano en train bajo reparto uniforme: supuesto. `[HIPÓTESIS]` |
| (Hallazgo añadido) señales idénticas entre splits | **Confirmada, efecto nulo** | 3.0 % del test y 2.1 % de val; F1 −0.0002 al quitarlas. `[VERIFICADO]` |

Veredicto de la pista: **parcialmente confirmada**; el riesgo restante (paciente) es **no verificable** con los datos actuales.

### Pista 2 — Corridas v3

| Sub-afirmación | Veredicto | Evidencia |
|---|---|---|
| Existen `saved/tune_v3`, `eval-v3-*` y la base Optuna con 3 trials | **Confirmada** | `ls saved/tune_v3` (trial-000/001/002 y `cinc2020_resnet_v3_long`); `runs/eval/eval-v3-{trial0,trial1,long}`; sqlite `mode=ro`: 2 COMPLETE + 1 FAIL. `[VERIFICADO]` |
| Ausentes de `docs/resultados_finales.md` y del ensemble recomendado | **Confirmada** | `grep` en `docs/`: sin menciones; `deploy/ensemble.env` = v1+v2. `[VERIFICADO]` |
| `config_resnet_v3.json` sin versionar | **Confirmada, y peor**: no existe en disco | `ls examples/cinc2020`: no está; `git ls-files`: sin rastro. La receta solo vive en `saved/tune_v3/*/*/config_used.json`. `[VERIFICADO]` |
| eval-v3-long F1 ≈ 0.750 vs v2 ≈ 0.718 | **Descartada como ventaja**: es un artefacto del punto de operación | v3-long usa β = 1.0 y v2 β = 0.5. Con β = 1.0: 0.7501 vs 0.7463 (+0.0038, IC [−0.002, +0.010], no significativo, `[VERIFICADO-SUB]`); trial0 con 40 épocas ya da 0.7499. `[VERIFICADO]` (cifras puntuales) |

Lo útil de v3 es otra cosa: como tercer miembro del ensemble sube el F1 test de 0.7556 a 0.7625 (M3).

### Pista 3 — Clases/orígenes débiles, ECE, Challenge

| Afirmación | Veredicto | Evidencia |
|---|---|---|
| Clases débiles LVH, AEctopy_Junctional, TAb | **Confirmada** | F1 0.659 / 0.584 / 0.693 (sección 2.4). `[VERIFICADO]` |
| Clase débil MI | **Descartada** en el punto F1 (F1 0.753, AUPRC 0.839); solo débil con F0.5 (0.692). VEctopy (0.693; AUPRC 0.727) ocupa su lugar | sección 2.4. `[VERIFICADO]` |
| Orígenes débiles CPSC, CPSC-extra, PTB | **Parcial**: CPSC-extra (0.579 en sus 11 clases) y PTB (0.542 en 6, n = 99) sí; CPSC es **artefacto** (0.449 → 0.769 sobre sus 7 clases anotadas); Georgia (0.697) es la siguiente con muestra grande | sección 2.4 y H01.5. `[VERIFICADO]` |
| ECE 0.114 | **Confirmada** (0.1142 en test) y con causa: H02.2/H03.1 | `[VERIFICADO]` |
| Challenge 0.2348 | **Confirmada** la cifra; **es un artefacto de procedimiento** (sin temperaturas, 27 vs 24 clases, techo 0.503); con F1 común el ensemble no supera a v2 en esta métrica | H04.2. `[VERIFICADO-SUB]` el desglose, `[VERIFICADO]` la cifra |

### Pista 4 — Ensemble y variabilidad entre semillas

| Sub-afirmación | Veredicto | Evidencia |
|---|---|---|
| El ensemble es solo el promedio simple v1+v2 | **Confirmada** | `examples/cinc2020/ensemble_evaluate.py:27-31,81-105`; `deploy/ensemble.env` (`ECG_ALPHA=0.5`); sin barrido de α en el repo ni en los docs. `[VERIFICADO]` |
| La variabilidad entre semillas no está medida | **Confirmada** | semillas 42/43/43/44/45 sin réplicas de receta; el bootstrap es por registro con sistema fijo. `[VERIFICADO]` |

## 5. Mejoras priorizadas

Prioridad = Impacto (1–5, sobre la métrica que mejora, con la evidencia de la tabla) ÷ Esfuerzo (S = 1, M = 2, L = 3). Esfuerzo S/M/L es de adopción completa; el costo de GPU de M4 se comparte con M5, y sus réplicas pueden servir de miembros para M3. Orden tras el contra-examen (5.2–5.3): M1 > M2 > {M3, M4, M5, M6, M7, M8} con 1.0 (desempate: M4 primero, porque fija el umbral de decisión de los demás) > {M9, M10}. Los experimentos mínimos usan scripts y configs del repo; las salidas van a `runs/` en la ejecución real.

### 5.1 Tabla

| ID | Área | Evidencia | Métrica que mejora | Esf. | ¿Invalida checkpoints o bundle? | Experimento mínimo | Prior. |
|---|---|---|---|---|---|---|---|
| M1 | Evaluación (etapas 04–06) `[VERIFICADO]` | v2−v1 +0.0346 (F0.5) vs +0.0125 (F1); `compare_models.py:19-30`; `challenge_score.py:186-212`; `error_analysis.py:74`; máscara por origen +0.0098 | Fidelidad del reporte: F1 comparable entre modelos; Challenge 0.2348 → ~0.289 (F0.5) / ~0.351 (F1) con 24 clases `[HIPÓTESIS]` (solo subagentes); F1 enmascarado por origen solo como diagnóstico | S | No invalida checkpoints; el bundle no cambia; sí cambia `docs/resultados_finales.md` | `evaluate.py --threshold-beta 1.0 --temperatures …` para v1 y v2; `bootstrap_ci.py --predictions … --predictions-baseline …`; `challenge_score.py` con temperaturas | 4.0 |
| M2 | Calibración (03) `[VERIFICADO]` | ECE 0.1142 → 0.0135, Brier 0.0774 → 0.0526 (test, ajuste en val); con umbrales reoptimizados el F1 β=1 cambia +0.0003 a +0.0007 y el AUROC nada; `ecg/calibration.py:24-38,47-64`; `ensemble_evaluate.py:81-105` | ECE y Brier (no F1, no AUROC, no AUPRC: transformación monótona por clase) | M | Sí al bundle: `deploy/temperatures_*.csv`, umbrales y webapp; los pesos no cambian | Ajustar `a, b` por clase con `calibrate.py` ampliado sobre `predictions_validation.csv` y medir ECE en `predictions_test.csv` | 1.5 |
| M3 | Ensemble (04, 08) `[VERIFICADO]` en la cifra, `[HIPÓTESIS]` en la generalización | v1+v2+tl 0.7625 vs 0.7556, +0.0069 [0.0037, 0.0100]; AUPRC 0.8121 vs 0.8052; AUROC 0.9526 vs 0.9505; 8 mejores combinaciones por val dan test 0.7592–0.7637; con selección y umbrales remuestreados +0.0045 [0.000, +0.009] (revisor, `[VERIFICADO-SUB]`); `v2+tl` (2 miembros, sin código nuevo) +0.0036 [−0.0006, +0.0077] pero con AUROC 0.9468 y AUPRC 0.8031 por debajo de v1+v2; `ensemble_evaluate.py:149-150` solo admite 2 | F1-macro, AUPRC-macro, AUROC-macro (efecto esperado +0.002 a +0.007) | M | Sí al bundle (miembro nuevo, umbrales y temperaturas); checkpoints actuales intactos | `ensemble_evaluate.py` generalizado a N miembros; mismo protocolo que M1 | 1.0 |
| M4 | Reproducibilidad (02) `[VERIFICADO]` | semillas 42/43/43/44/45 sin réplicas; sd val ≈ 0.0037, sd bootstrap 0.0042 | Capacidad de decidir entre diferencias de 0.004–0.012; entrega 3 miembros | L | No invalida; añade checkpoints | 4 réplicas de `examples/cinc2020/config_resnet_v2.json` con `--seed` 44–47 y `aug_seed` distinto en copias del config (hoy fijo en 123): 4 × ≈ 1.97 h de GPU, n = 5 con la v2 original | 1.0 |
| M5 | Entrenamiento (02) `[HIPÓTESIS]` | `train.py:173-189,280-324`: entre trial0 y long el F1@0.5 de val sube y el AUROC de val/test baja (0.9478 → 0.9429 / 0.9454 → 0.9403); el AUPRC de val no los separa (0.8045 vs 0.8050), así que el criterio debe usar AUROC | AUROC macro (≤ 0.005 esperado); F1 β=1 no cambia | S (código) | No invalida los existentes; exige reentrenar (comparte las corridas de M4) | Guardar además `best_auroc.pt` (y `best_auprc.pt`) en las corridas de M4 y comparar | 1.0 |
| M6 | Datos/inferencia (01, 08) `[HIPÓTESIS]` | `datasets.py:581`; recall VEctopy 0.43 (> 10 s) vs 0.71 (10 s) `[VERIFICADO-SUB]` | Recall de VEctopy y AEctopy; coherencia web/evaluación | M | Sí: val/test HDF5, umbrales y calibración (no los pesos) | Inferencia solo-lectura con `ecg.predict` sobre registros > 10 s de `dataset2020/`, agregación media vs máximo | 1.0 |
| M7 | Entrenamiento (02) `[HIPÓTESIS]` | suelo analítico 0.21–0.32 `[VERIFICADO]`; efecto sobre AUROC confundido con la aumentación | AUROC de VEctopy/AEctopy/AVBlock; ECE | M | Reentrenar; bundle si se adopta | `pos_weight` acotado (p. ej. potencia 0.5) con la receta v2, semilla 43, vs referencia | 1.0 |
| M8 | Datos (01) `[HIPÓTESIS]` | PTB-XL 21 799 registros / 18 869 pacientes; duplicados 3.0 %; etiquetas en conflicto 81 | Validez de las métricas; sin efecto medible en F1 (−0.0002) | M (medir) / L (re-split) | Re-split invalida todo (HDF5, checkpoints, bundle) | Medir solape de pacientes con metadatos de PTB-XL (requiere permiso de descarga) | 1.0 |
| M9 | Entrenamiento (01–02) `[HIPÓTESIS]` | anotación parcial `train.py:260`; +0.0098 solo de evaluación | F1 en celdas anotadas; FP de CPSC | L | Sí: reentrenar y regenerar bundle | BCE con máscara origen×clase (regla: 0 positivos en train) con 3 semillas | 0.67 |
| M10 | Esquema (01) `[HIPÓTESIS]` | grupos heterogéneos (LVH 197/800 solo atrial); techo Challenge 0.503 | Métrica Challenge; clases LVH, MI, BBB, AEctopy | L | Sí: `LABEL_SCHEMA`, HDF5, checkpoints y bundle | Cabeza auxiliar de subclases/27 códigos | 0.67 |

No se reproponen la regla NSR-exclusivo ni la ablación 2×2 de calibración (`docs/resultados_finales.md`, hallazgos 2 y 4). M2 es una variante distinta de lo ya probado: la ablación 2×2 comparó temperatura sí/no con umbral fijo/optimizado y concluyó que la temperatura no cambia decisiones; M2 cambia la familia de la calibración (suma el término de sesgo `b`) para atacar el ECE y espera, igual que ellas, F1 sin cambios.

### 5.2 Contra-examen (Fase 5): motivo más fuerte por el que cada mejora podría no mejorar o empeorar

Orden provisional previo al contra-examen: M1, M2, M3, M4 y la BCE enmascarada por origen (hoy M9). La BCE enmascarada no resistió y bajó al puesto 9; el criterio de selección (M5) entró al top-5 por la fórmula, y como no pasó por el revisor externo recibió el contra-examen propio de esta tabla. Los números de esta tabla los recalculé yo salvo indicación.

| ID | Motivo más fuerte en contra (categoría) | Evidencia | Veredicto |
|---|---|---|---|
| M1 | Es medición, no mejora del modelo. Elegir β = 1 y la máscara después de ver el test sube el número "por construcción" (+0.0098) (bifurcación de decisiones). La regla "0 positivos en train del origen" confunde "no anotado" con "raro" (INCART, n = 52, sin AxisDev en train). (sobreajuste a lo observado; semántica de la métrica) | La regla usa solo train (sin test): `[VERIFICADO]`. De las 10 celdas origen×clase con 0 positivos en train, 5 son de CPSC (998 registros de test); el enmascarado solo elimina FP (218 de 3220). El β = 1 ya es el punto recomendado para la demo | **Mantener** como corrección de reporte; declarar el protocolo antes de recalcular y publicar ambas versiones (con y sin máscara; β 0.5 y 1.0). No cuenta como mejora del clasificador |
| M2 | No puede subir F1, AUROC ni AUPRC (transformación monótona por clase): solo ECE/Brier. (cambio del significado de las probabilidades de salida) | F1 con umbrales reoptimizados: v2 0.7463 → 0.7466, ensemble 0.7556 → 0.7562, v1 0.7341 → 0.7348; AUROC idéntico; AUPRC de v2 0.7794 → 0.7788. Parámetros ajustados en val y evaluados en test: sin fuga. `[VERIFICADO]` | **Mantener** con el alcance declarado (calibración, no clasificación). Cambia el bundle: umbrales y archivos de temperaturas |
| M3 | La ganancia depende de la calidad y del linaje del tercer miembro. v3-long es la continuación de v3-trial0 (misma semilla 43 y `aug_seed` 123 que v2) y se eligió entre 31 combinaciones por F1 de val (3.º de 31). El IC es solo muestral. (dentro del IC / selección sobre val) | Tercer miembro: t1 (independiente, más débil) +0.0021 [−0.0007, +0.0055]; t0 +0.0041 [+0.0011, +0.0071]; tl +0.0069 [+0.0035, +0.0101]. Las 8 mejores combinaciones por val dan test 0.7592–0.7637, todas > 0.7556: la selección no explica la ganancia. `v1+tl` solo +0.0012. `[VERIFICADO]` | **Degradar a `[HIPÓTESIS]`** (el revisor externo y mi comprobación coinciden): efecto esperado +0.002 a +0.007; +0.0069 es el mejor de los terceros miembros probados. Solo mejora F1, AUROC y AUPRC a la vez con v1+v2+tl (`v2+tl` sube el F1 y baja AUROC/AUPRC). Se decide con las réplicas de T6. Costo: inferencia ×1.5 y código N-miembros |
| M4 | No mejora ninguna métrica por sí misma y cuesta ≈ 7.9 h de GPU; si sd_seed ≤ 0.003 no cambiará ninguna decisión. (ruido de semilla) | Hoy: sd val 0.0037, sd bootstrap 0.0042, diferencias del repo de 0.004–0.012 (secciones 2.2, H02.6). Las réplicas sirven además como miembros para M3 | **Mantener** como prerrequisito de decisión, no como mejora de métricas |
| M5 | El AUPRC de val no separa trial0 de long (0.8045 vs 0.8050); el efecto esperado en AUROC (≤ 0.005) está dentro del ruido y el F1 β=1 no cambia (0.7499 → 0.7501). Seleccionar con una métrica de val que también calibra umbrales agrava el sobreajuste a val. (dentro del IC; sobreajuste a val) | `metrics_global.csv` de `eval-v3-trial0` y `eval-v3-long`; v1 ya se eligió por `val_loss` y quedó +0.004 F1 sobre la época 35 `[VERIFICADO-SUB]` | **Degradar a `[HIPÓTESIS]`**: el criterio debe usar AUROC, no solo AUPRC; se acepta o descarta con T6 |
| M6 | Georgia (registros de 10 s) también tiene recall 0.28 en VPB: la duración no es la única causa; n = 107 registros > 10 s con VEctopy en test deja un IC ancho; agregar ventanas cambia la calibración. (dentro del IC) | `[VERIFICADO-SUB]`; ganancia sin medir | **`[HIPÓTESIS]`** de baja confianza; solo se adopta si T9 supera los criterios |
| M7 | El mecanismo está verificado, pero su efecto sobre AUROC/F1 no: la caída de AUROC de v2 frente a v1 en VEctopy, AEctopy y AVBlock está confundida con la aumentación; el ECE ya se arregla con M2 sin reentrenar; bajar `pos_weight` baja el recall de clases raras y mueve el equilibrio que hoy se ajusta con umbrales. (cambio de la función de pérdida) | suelo 0.21–0.32 `[VERIFICADO]`; efecto `[HIPÓTESIS]` | **`[HIPÓTESIS]`**, tras T6 |
| M8 | La fuga por paciente no se puede medir hoy; la parte medible (duplicados) mueve el F1 en −0.0002. Re-dividir invalida todo y, si el solape es bajo, no cambia nada. (fuga; coste/beneficio) | H01.2, H01.3 | **Mantener solo la medición** (T11); el re-split queda condicionado |
| M9 | La máscara no distingue "no anotado" de "ausente"; sin máscara en la evaluación, quitar los negativos de CPSC puede subir los FP justo en las celdas que hoy se penalizan; la ganancia de +0.0098 es de evaluación, no de entrenamiento, y a lo sumo afecta al 15.5 % del test (CPSC, 998 registros). (cambio de etiquetas efectivas) | H01.5 | **Degradar a `[HIPÓTESIS]`** (era top-5 provisional) |
| M10 | Cambia el esquema de etiquetas (`LABEL_SCHEMA`): invalida HDF5, checkpoints, umbrales y bundle; el beneficio sobre las 12 clases agrupadas es incierto. (cambio de etiquetas) | techo Challenge 0.503 `[VERIFICADO-SUB]` | **`[HIPÓTESIS]`**, diferida |

### 5.3 Revisor adversarial independiente (doubt-driven, ciclo 1 de 3)

Se lanzó un subagente de contexto limpio con un prompt adversarial ("encuentra qué está mal"). Recibió las cinco propuestas con la evidencia citada y las reglas de solo lectura, pero no mis conclusiones ni los veredictos que esperaba. El primer intento falló por el límite de uso de la API (HTTP 429) sin producir resultados; este segundo completó (56 usos de herramientas, ≈ 14 min). Cada hallazgo se contrastó con los datos antes de aceptarlo. Sus propuestas P1–P5 corresponden a M1, M2, M3, M9 (BCE enmascarada) y M4.

| Hallazgo del revisor | Clasificación | Qué cambió en este informe |
|---|---|---|
| P1: el enmascarado solo elimina FP (218 de 3220); 0.7663 con la regla de test y 0.7655 con la de train | Válido y accionable. Lo reproduje (218/3220 y 0.7655) | El F1 enmascarado pasa a diagnóstico; el titular es el F1 sin máscara (M1, T3) |
| P1: con otra regla de máscara PTB pasa de 0.542 a 0.407 e INCART de 0.439 a 0.239 | Válido (sensibilidad) | T3 exige la sensibilidad (prevalencia < 0.5 %); no se concluye nada de PTB ni de INCART con la máscara |
| P1: el Challenge de 0.299/0.359 mezclaba umbrales F0.5 y F1 y no fusionaba clases; con F1 en ambos v2 = 0.358 y ensemble = 0.359; con fusión 0.289 / 0.351 / 0.351 | Válido y accionable; coincide con C, pero no lo reproduje yo | Corregidos H04.2, el resumen y T2; el Challenge queda como `[HIPÓTESIS]` |
| P1: con F1 común el ensemble supera a v2 (+0.009) y el hallazgo 3 del documento se invierte | Válido; lo reproduje (v2 − ensemble = −0.0092) | Fila nueva en 2.3 |
| P2: un desplazamiento de prior sin parámetros, −log(`pos_weight`), da ECE 0.0131 en el ensemble frente a 0.0095 de Platt | Válido; refuerza H02.2 como causa; no lo reproduje | T7 debe compararse con ese desplazamiento y adoptar el más simple si Platt no lo mejora en ≥ 0.002 de ECE |
| P2: v2 tiene 671 probabilidades exactamente 1.0 (AF 3.6 % de los registros) y la pendiente `a` depende del recorte | Válido y accionable; reproduje las 671 | T7 ajusta sobre los logits crudos y mide la sensibilidad a `eps` |
| P2: ECE por origen tras Platt: CPSC 0.036, CPSC-extra 0.052, PTB 0.069 | Válido, con compensación (calibración global, no por origen) | Criterio por origen ≤ 0.08 en T7 |
| P2: los umbrales bajan a 0.1–0.5 y hay que regenerar `deploy/temperatures_*.csv` y `webapp/app.py:272` | Ya reconocido (el bundle se invalida) | Sin cambio |
| P3: el argmax estricto de val entre 26 subconjuntos es v1+v2+t0+tl (0.7749); v1+v2+tl es 3.º | Ya sabido; coincide con mi tabla (3.º de 31 con los modelos solos) | Sin cambio |
| P3: con la selección en val remuestreada +0.0045 [+0.001, +0.007]; con los umbrales reajustados dentro del bootstrap [0.000, +0.009] | Válido y accionable; no lo reproduje | M3 baja a prioridad 1.0 y a `[HIPÓTESIS]`; efecto esperado +0.002 a +0.007 |
| P3: v2 y tl tienen correlación de logits 0.93 y comparten semilla y `aug_seed`; v2+tl da 0.7592 y 3 miembros vs v2+tl es +0.0024 [−0.003, +0.007] | Válido. Reproduje v2+tl: +0.0036 [−0.0006, +0.0077] con AUROC y AUPRC menores que v1+v2 | M3 documenta la variante de 2 miembros (sin código nuevo) y su costo en AUROC/AUPRC |
| P4: el +0.0107 es un cambio de métrica, no del modelo; cota superior sin máscara +0.0084; contrafactual con la tasa de FP de PTB-XL y Georgia: −0.004 | Válido; confirma la degradación de M9 | H01.5 y 5.2 |
| P4: AxisDev 1.1 % vs 30 % no es evidencia (prevalencia real en PTB-XL ≈ 27 %) | Válido; comprobé 27 % en train | H01.5 retira esa evidencia |
| P4: SB en CPSC son 37 FP, no 46 | Válido; comprobé 37 (más 9 de PTB) | H01.5 corregido |
| P4: la regla "0 positivos" no enmascara NSR (11/2464) ni AxisDev (1/2464) en CPSC-extra | Válido | T3 y T12 piden sensibilidad con prevalencia < 0.5 % |
| P4: `pos_weight`, el scheduler y el monitor usan `val_loss` sin máscara (`ecg/train.py:260,266,312`) | Válido y accionable | T12 amplía su alcance |
| P5: sd de meseta 0.0038 y sd bootstrap 0.0040; `aug_seed` fijo en 123 (`ecg/train.py:93`, leído de la config); 3 réplicas dejan 2 grados de libertad | Válido; comprobé que `--seed` no cambia `aug_seed` | T6 pasa a 4 réplicas (n = 5) con `aug_seed` distinto, ≈ 7.9 h de GPU |

Veredictos del revisor: P1 mantener el F1 único, retirar el enmascarado como titular y degradar el Challenge; P2 mantener, acotado a calidad de probabilidad; P3 y P4 degradar a hipótesis; P5 mantener con ≥ 5 semillas. Los adopté todos. De 17 hallazgos, 15 fueron válidos (accionables o con compensación) y 2 ya estaban recogidos; ninguno se descartó como ruido ni por contrato mal leído.

Parada: el ciclo 1 solo añadió salvedades y criterios y no cambió la lista de mejoras; lo que queda por resolver requiere entrenar (sección 8), así que no se lanzó un ciclo 2 sobre un artefacto casi igual. Segunda opinión de otro modelo (Gemini CLI, Codex CLI o revisión externa manual): queda ofrecida al usuario; no se invocó ninguna CLI externa porque nadie la autorizó.

## 6. Plan de experimentos

Este plan vive solo aquí: la restricción de la tarea impide crear o modificar `tasks/plan.md` y `tasks/todo.md` (que ya existen y son del plan de la demo local). Nada de lo siguiente se ejecutó. Los comandos escriben en `runs/`, `saved/` y `data/` y exigen la autorización del usuario cuando se ejecuten.

**Protocolo fijado antes de recalcular (condición de todas las tareas):** (a) todo sistema se reporta con β = 0.5 y β = 1.0, con la misma regla de temperaturas y umbrales de validación; (b) IC95 bootstrap pareado por registro, B = 1000, semilla 0; (c) la regla de máscara por origen se deriva solo de train; (d) los criterios de aceptación de cada tarea se escriben antes de ver el resultado; (e) las diferencias se declaran concluyentes solo si superan 2 × la desviación entre semillas medida en T6 (hasta entonces, el piso es 0.008, el doble del ruido bootstrap).

**Lista obligatoria para toda tarea que cambie etiquetas, preprocesamiento o el significado de las salidas** (T7, T8, T9 si se adopta, T10, T12, T13 y cualquier re-split): (1) reconstruir los HDF5 si aplica (`examples/cinc2020/build_datasets.py`); (2) reentrenar (donde los pesos no cambian se escribe "no aplica" con su justificación); (3) regenerar `models/` y `deploy/` con `tools/export_inference_bundle.py`; (4) actualizar `docs/resultados_finales.md`; (5) pasar `tests/test_inference_bundle.py` y `tests/test_deploy_config.py`.

### Fase A — Medición (no invalida nada)

| Tarea | Qué | Depende de | Criterio de aceptación (numérico) |
|---|---|---|---|
| T1 | M1. Reproducir v1 y v2 con `evaluate.py --threshold-beta 1.0 --temperatures …`; `compare_models.py` falla si difiere `threshold_beta`; regenerar `runs/eval/ensemble` sin `--nsr-exclusive-min-prob` | — | v1 F1 test 0.7341 ± 0.001, v2 0.7463 ± 0.001; IC95 de v2−v1 contiene +0.0125 ± 0.002; ensemble F0.5 regenerado = 0.7102 ± 0.001 con `nsr_exclusive_min_prob` nulo; test nuevo que falla con el `compare_models.py` actual; `model_comparison.csv` y `comparacion_ensemble.csv` citan solo directorios existentes (100 %) |
| T2 | M1. `challenge_score.py` con `--temperatures` y fusión de los 3 pares equivalentes (envoltorio propio; `examples/cinc2020/official/` no se toca) | T1 | con 24 clases fusionadas y probabilidades calibradas: v2 con umbrales F0.5 ≈ 0.289 ± 0.010, v2 con F1 ≈ 0.351 ± 0.010 y ensemble F1 ≈ 0.351 ± 0.010 (se reporta que no hay ventaja del ensemble); `challenge_metric.json` cita rutas existentes; test nuevo que falla si se omiten las temperaturas |
| T3 | M1. F1 por origen con máscara (regla fijada de antemano: 0 positivos de la clase en train del origen; sensibilidad con prevalencia < 0.5 %) en `error_analysis.py`, como diagnóstico junto al valor sin máscara, que sigue siendo el titular | T1 | ensemble F1 test 0.7655 ± 0.0005 con la regla de train (0.7663 con la de test; sin máscara 0.7556); CPSC 0.769 ± 0.002; el informe lista las 10 celdas origen×clase enmascaradas (cpsc_2018: 5, ptb: 4, st_petersburg_incart: 1) y los 218 FP que caen en ellas |
| T4 | M1. Actualizar `docs/resultados_finales.md`: tablas con β común, ensemble citado desde un directorio existente, "época 25, gap 0.051", v3 documentado con su receta real, Challenge con aviso, regeneración de `runs/figs/` | T1–T3 | `tests/test_docs_links.py` y `tests/test_docs_env_vars.py` verdes; 100 % de las rutas citadas en el documento existen (script de comprobación); `pytest -q` sin fallos (360 o más) |

**Punto de control A:** `pytest -q -p no:cacheprovider` → 0 fallos; el titular reportado es el F1 con β común y su IC.

### Fase B — Ruido de entrenamiento (condición para decidir el resto)

| Tarea | Qué | Depende de | Criterio de aceptación |
|---|---|---|---|
| T5 | M5. Guardar también `best_auroc.pt` y `best_auprc.pt` (selección por AUROC y AUPRC macro de val, calculados en la pasada de validación que ya existe) en `ecg/train.py`; prueba primero | T1 | prueba RED→GREEN: con `--epochs 2` aparecen `best.pt`, `best_auroc.pt` y `best_auprc.pt`; el resto de pruebas sigue verde |
| T6 | M4 + M5. Cuatro réplicas de la receta v2: `python -m ecg.train <copia de examples/cinc2020/config_resnet_v2.json con seed y aug_seed propios> -e cinc2020_resnet_v2_s44 --seed 44` y semillas 45, 46 y 47 (`aug_seed` distinto en cada copia; secuenciales; ≈ 1.97 h cada una, ≈ 7.9 h en total, VRAM de 4 GB) | T1, T5 | sd de F1 test β=1 sobre las 5 semillas (con la v2 original) publicada con n = 5 (4 grados de libertad); si sd > 0.008, toda diferencia < 0.016 se marca no concluyente. M5 se acepta solo si el AUROC macro de test con selección por AUROC supera a la selección por F1 en ≥ 0.002 de media y en ≥ 3 de las 4 semillas nuevas, con F1 β=1 no menor que −0.003; si no, se descarta |

### Fase C — Mejoras con efecto medido

| Tarea | Qué | Depende de | Criterio de aceptación |
|---|---|---|---|
| T7 | M2. `fit_platt_per_class` en `ecg/calibration.py` (pruebas con valores conocidos, no solo cotas), `calibrate.py --method`, aplicación en `evaluate.py`, `ensemble_evaluate.py` y la webapp; recalibrar tras promediar. **Lista obligatoria** (pesos sin cambio: reentrenar no aplica) | T1 | ECE test ≤ 0.02 en v2 y en el ensemble (hoy 0.114 y 0.103); ECE ≤ 0.05 en cada una de las 12 clases y ≤ 0.08 por origen (hoy tras Platt global: CPSC 0.036, CPSC-extra 0.052, PTB 0.069); Brier ≤ 0.056; ajuste sobre los logits crudos del checkpoint (`calibrate.py` en modo HDF5), no sobre las probabilidades ya escaladas, porque v2 tiene 671 probabilidades exactamente 1.0 en test (AF 3.6 %) y la pendiente `a` depende del recorte: `a` varía ≤ 10 % entre `eps` = 1e-6 y 1e-9 en las 12 clases; umbrales reoptimizados (bajan a 0.1–0.5); F1 β=1 igual ± 0.0010, AUROC ± 0.0005 y AUPRC ± 0.0010 con los umbrales reoptimizados en val (medido: F1 v2 0.7463 → 0.7466, ensemble 0.7556 → 0.7562, v1 0.7341 → 0.7348; AUROC idéntico; AUPRC de v2 0.7794 → 0.7788); prueba de que los parámetros se ajustan solo con val; Platt debe mejorar en ≥ 0.002 de ECE al desplazamiento de prior sin parámetros (−log `pos_weight`, ECE 0.0131 en el ensemble según el revisor), o se adopta el más simple |
| T8 | M3. `ensemble_evaluate.py`, webapp y `tools/export_inference_bundle.py` para N miembros; miembros elegidos solo con val. **Lista obligatoria** | T6, T7 | con (v1, v2, tl): F1 test 0.7625 ± 0.001 y AUPRC 0.8121 ± 0.001; ΔF1 vs v1+v2 con IC95 > 0 y ≥ 2·sd_seed (T6); latencia por registro en CPU ≤ 1.6 × la del ensemble de 2 (`examples/cinc2020/benchmark_latency.py`); probabilidades CLI y web idénticas ± 1e-4 para la misma señal. Si no llega a 2·sd_seed, no se adopta |
| T9 | M6. Primero solo inferencia en lectura sobre registros > 10 s; si pasa, adopción con **lista obligatoria** (val/test HDF5 con ≤ 4 ventanas, umbrales y calibración reajustados) | T1 | recall de VEctopy en registros > 10 s (n = 107) de 0.43 a ≥ 0.55 con agregación media; F1-macro global no cae más de 0.003; la webapp y `evaluate.py` quedan con la misma regla de agregación |

**Punto de control C:** `pytest -q` verde; `tests/test_inference_bundle.py` y `tests/test_deploy_config.py` verdes; `tools/smoke_deployment.py` 19/19 si cambia el bundle.

### Fase D — Hipótesis (solo si T6 muestra sd ≤ 0.005)

| Tarea | Qué | Depende de | Criterio de aceptación |
|---|---|---|---|
| T10 | M7. `pos_weight` acotado (p. ej. potencia 0.5), receta v2, semilla 43. **Lista obligatoria** | T6 | mediana de probabilidad de negativos ≤ 0.10 en AF, VEctopy, AVBlock, STach, SB y AEctopy (hoy 0.21–0.31); AUROC macro de val ≥ 0.9414 − 0.001; F1 test β=1 no inferior a −1·sd_seed; ECE antes de calibrar < 0.08 |
| T11 | M8. Medir solape de pacientes con metadatos de PTB-XL y PTB. Pedir permiso al usuario antes de descargar | permiso | porcentaje de registros de test con paciente presente en train, por origen; si ≥ 5 % en un origen con n ≥ 500, abrir la tarea de re-split agrupado (invalida todo; **lista obligatoria**) |
| T12 | M9. BCE con máscara origen×clase con 3 semillas; la máscara también debe entrar en `pos_weight` (`ecg/train.py:260`), en `val_loss` (`:266`) y en el monitor de parada (`:312`), o el scheduler y la selección cambiarán de significado. **Lista obligatoria** | T3, T6 | F1 enmascarado (T3) ≥ +0.005 sobre la referencia y F1 sin máscara no peor que −0.003; si el sin-máscara cae más, se rechaza |
| T13 | M10. Cabeza auxiliar de subclases. Diferida: requiere decidir el cambio de `LABEL_SCHEMA`. **Lista obligatoria** | decisión del usuario | no se define umbral numérico hasta decidir el esquema |

## 7. Seguridad y pruebas (aparte del ranking)

No entran en la prioridad de la sección 5 porque no mueven las métricas del clasificador. Ninguna se corrigió aquí. Para cualquier arreglo, escribir antes la prueba que falla con el código actual (`tests/CONTEXT.md`).

### 7.1 Seguridad (webapp, herramientas, despliegue)

| # | Severidad | Hallazgo | Evidencia |
|---|---|---|---|
| S1 | Alta | ReDoS cuadrático en `clean_path_text`; se llama con datos de la petición en `webapp/app.py:606,626,669,680,681` | `webapp/prediction.py:141`. Medido por mí con `"[" * N`: 0.018 s (N = 2000), 0.072 s, 0.273 s, 1.132 s (N = 16 000): ×4 al duplicar. Extrapolación cuadrática ≈ 18 min con 500 000 caracteres (no ejecutada a esa escala). La regex mantiene el GIL: un solo POST anónimo por túnel o LAN paraliza waitress. `[VERIFICADO]` |
| S2 | Alta | `.mat` sin tope de descompresión: `loadmat` sin límite | `ecg/signal.py:427,430`; razón medida por D2 1025× (5 MB comprimidos ≈ 5.4 GB); `num_samples` del `.hea` sin validar (`ecg/signal.py:58`). Efecto real no ejecutado a escala. `[VERIFICADO-SUB]` |
| S3 | Media | `/use_model` sin autenticación y sin uso en la interfaz: cualquier visitante cambia `ECG_MODEL` global (p. ej. ensemble v2+v2) | `webapp/app.py:603-618`; `static/app.js` solo consulta `/examples`, `/predict`, `/metrics`, `/curves`. `[VERIFICADO-SUB]` |
| S4 | Media | Sin tope de cuerpo antes de Flask: el valor por defecto de waitress es 1 073 741 824 bytes y `deploy/start_local.ps1:66` no pasa `--max-request-body-size`; `GET /` carga los `.pt` y calcula SHA-256 sin límite (el limitador solo cubre `POST /predict`, `webapp/app.py:499`) | valor por defecto leído por mí con `inspect` sobre `waitress.adjustments`; `MAX_CONTENT_LENGTH` en `webapp/app.py:468`. `[VERIFICADO]` (valor y línea), `[VERIFICADO-SUB]` (efecto de `GET /`) |
| S5 | Media | Los servidores de desarrollo escuchan en `0.0.0.0` por defecto | `webapp/app.py:787,842`; `webapp/wsgi.py:117,146`. `[VERIFICADO-SUB]` |
| S6 | Baja | Self-XSS por atributo: `escapeHtml` no escapa `"` y `record_name` sale del primer token del `.hea` subido | `static/app.js:906`, `:493`; `ecg/signal.py:55`. No probado en navegador. `[VERIFICADO-SUB]` |
| S7 | Baja | Sin CSP ni `X-Frame-Options`; rutas absolutas en el JSON de `/predict` y `/health` | `webapp/app.py:531-544`. `[VERIFICADO-SUB]` |
| S8 | Baja | Retención: la purga solo corre al arrancar y en cada `POST /predict`; los PDF con nombre de paciente pueden pasar la hora que promete la interfaz | `webapp/app.py:497,509`. `[VERIFICADO-SUB]` |
| S9 | Baja | `patient_name` sin límite: con 480 000 caracteres reportlab tarda 9.5 s y cae al PDF mínimo | `webapp/app.py:680`; `webapp/report_pdf.py:41-44`. `[VERIFICADO-SUB]` |
| S10 | Info | Dependencias con versiones exactas y sin hashes; Pillow sin fijar; `pip-audit` no ejecutado. `torch` 2.14.0 instalado vs `torch==2.8.0` en `requirements.txt:4` | `[VERIFICADO-SUB]` |

Verificado sin hallazgo `[VERIFICADO-SUB]`: `load_checkpoint` usa `weights_only=True` (`ecg/train.py:287`, `ecg/predict.py:57`); `/results` usa `send_from_directory`; `/use_model` exige un `.pt` dentro de `saved_root`; el texto de usuario va escapado en el PDF; `git grep` no encuentra secretos; el SHA-256 de `models/*.pt` coincide con `deploy/manifest.json` (la webapp no lo comprueba al cargar).

### 7.2 Pruebas (`tests/`; la suite pasa: 360 en 372.63 s)

Hallazgos de D1, todos `[VERIFICADO-SUB]` (mutaciones aplicadas en memoria por el subagente; yo no las repetí):

| # | Brecha | Evidencia |
|---|---|---|
| P1 | Ningún test importa el split (`ecg/datasets.py:138-171,333-455`); la mutación `val_frac_local = val_frac` pasa | solo `--help` en `tests/test_cli_smoke.py:15` |
| P2 | Nada impone "umbrales y temperaturas solo con validación": `evaluate.run_evaluation` (`examples/cinc2020/evaluate.py:250-294`) no corre en ninguna prueba; `optimize_thresholds(test[...])` pasa | `tests/test_accuracy.py:66-87` usa predicciones simuladas |
| P3 | `optimize_thresholds` no se prueba con valores; cambiar `>` por `>=` (`evaluate.py:65`) o no pasar `beta` (`:80`) pasa | `tests/test_decision_logic.py:98-103,130-136` |
| P4 | `tests/test_robustness.py:35-49` es trivial; quitar `apply_temperature` (`robustness.py:69-70`) pasa y reproduce la corrida "mismatched 0.670" que `docs/resultados_finales.md` retiró | — |
| P5 | Preprocesado sin pruebas numéricas (orden del filtro 3→4 y clip 5→10 mV pasan); único guardia: golden de 6 registros a 500 Hz y 10 s con tolerancia 5e-3 | `tests/test_examples.py:157-184`; no cubre remuestreo ni multiventana |
| P6 | Sin prueba de paridad entrenamiento/inferencia (`ecg/datasets.py:217-263` vs `load_wfdb_record_windows`); `norm_mode`/`bandpass` no se guardan en el checkpoint, así que un HDF5 con `--norm-mode global_zscore` se serviría como `physical` | `ecg/predict.py:276-297`, `ecg/train.py:106` |
| P7 | El mapeo SNOMED→clase no está fijado: mover un código entre grupos pasa | `ecg/schema.py:34-141`, `tests/test_signal_schema_core.py:80-87` |
| P8 | Calibración y bootstrap flojos: solo cotas (`tests/test_calibration.py:25,39-42`); `calibrate.py`, `load_temperatures_csv` y `bootstrap_ci.py:88-107` sin prueba | — |
| P9 | Pruebas triviales o frágiles: `tests/test_make_figures.py:36-56`, `test_sex_breakdown.py:36`, `test_serving_device.py:57-59`; `_free_port` con carrera (`tests/test_dev_server.py:24`); plazos de 120–180 s | — |
| P10 | Sin CI (`.github/` no existe); ninguna prueba usa `data/` ni `runs/` | — |

Orden sugerido (cada una con su prueba primero): S1 y S2 (prueba con entrada adversarial acotada en tiempo y memoria), P2 y P3 (fijan el protocolo del que depende M1), P1, S3–S5.

## 8. No verificado / datos no disponibles

- Magnitud de la fuga por paciente (PTB-XL y PTB): las cabeceras no traen ID de paciente y no se descargaron metadatos externos (requiere permiso del usuario, T11). Las cifras ~19 % / ~60 % son una simulación con reparto uniforme.
- Fuente primaria del detalle del split de Hannun et al. (Nat Med 2019): no se contrastó; este informe no compara el protocolo con ese artículo.
- `runs/eval/eval-resnet-v2/calibration_report.json` no existe. Corrección posterior (al ejecutar M2): recalculado desde los logits crudos, el ECE "antes" de 0.191 sí se reproduce en test; el Brier "antes" da 0.1011 (docs 0.099) y el Brier tras la temperatura 0.0774 (docs 0.075).
- Desglose de la métrica Challenge (0.2988 / 0.3588 con calibradas, techo 0.503, +0.009 por fusión) y brechas por sexo por clase: reproducidos solo por subagentes (para el Challenge, C y el revisor de la Fase 5 coinciden; las brechas por sexo, solo C).
- Efecto de la agregación multiventana sobre VEctopy/AEctopy (necesita inferencia, T9) y efecto del pasa-banda 0.5–50 Hz sobre espigas de marcapasos.
- Causa del FALLO del trial 2 de Optuna: sin `fail_reason`. Si la diferencia v2−v1 se debe a la receta o a la semilla: sin réplicas (T6).
- `.mat` como bomba de descompresión (S2) a escala real, self-XSS (S6) en navegador, `X-Forwarded-For` de Cloudflare y `pip-audit` (S10): no ejecutados.
- Figuras (`make_figures.py`, etapa 07): no inspeccionadas.
- Todo lo marcado `[VERIFICADO-SUB]` en las secciones 3, 4 y 7 depende de un informe de subagente que yo no re-ejecuté.
- Incidencias del proceso: el primer revisor adversarial de la Fase 5 falló por límite de uso de la API (HTTP 429) tras leer solo el comienzo; se relanzó (5.3). No se usó ninguna segunda opinión de otro modelo (no estaba autorizada).
