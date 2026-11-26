# Comparación exploratoria con el modelo publicado de Ribeiro et al. (propuesta, R11)

**Estado: propuesta sin publicar.** Documenta lo que se **ejecutó** en R11 y cómo repetirlo. Es una comparación **exploratoria y posterior** a los resultados de C y D del sistema principal (P3); no forma parte del protocolo con hash y no lo sustituye.

## Qué se comparó
- **Sistema principal P3** (ensemble de v1 y v2 con calibración afín y umbrales fijados en validación): se **leen** sus probabilidades guardadas en `runs/t19/eval/p3_C` y `p3_D` (`predictions_test.csv`); no se regeneran.
- **Modelo publicado de Ribeiro et al.** (Nat. Commun. 11, 1760, 2020; pesos en Zenodo 10.5281/zenodo.3625017, CC BY 4.0): se ejecuta **sin reentrenar, sin ajustar y sin calibrar** con C ni D. Como solo se publica en Keras y TensorFlow no está instalado, se usa la reimplementación en PyTorch de `examples/cinc2020/baseline_ribeiro/ribeiro_pytorch.py`, que **reproduce sus predicciones publicadas** sobre su conjunto de prueba (diferencia máxima de probabilidad 3.3×10⁻⁷ en 827 trazados; `validar_port_ribeiro.py`).
- **Registros:** exactamente los que P3 evaluó: 10,234 de Chapman-Shaoxing (C) y 34,830 de Ningbo (D) del Challenge 2021 v1.0.3 (descargados y verificados contra su `SHA256SUMS.txt`).

## Qué clases son comparables
Solo la taquicardia sinusal (ST ↔ STach) tiene el **mismo código SNOMED** (427084000). AF y SB de Ribeiro son subconjuntos de clases de P3 que agrupan más códigos (flutter y otros en AF; bradicardia y nodo sinusal en SB); 1dAVb, RBBB y LBBB son subconjuntos de AVBlock y BBB, que P3 no puede puntuar por separado; y las etiquetas de Ribeiro se filtraron con mediciones (PR ≥ 190 ms, QRS ≥ 115 ms) que las cabeceras del Challenge no traen. Por eso: **ST = comparación uno a uno; AF y SB = aproximadas; las otras tres = solo descriptivas.**

## Dos entradas del modelo publicado (fijadas antes de ejecutar; se informan las dos)
- **A:** su entrada propia (milivoltios sin filtrar, 500→400 Hz, relleno simétrico con ceros a 4 096 muestras).
- **B:** el preprocesamiento de este repositorio (`ecg.signal.load_wfdb_record_windows`: 0.5–50 Hz, recorte a ±5 mV), seguido del mismo remuestreo y relleno.

## Resultados (AUROC y AUPRC, etiquetas de Ribeiro derivadas de los códigos; entrada B)
Diferencia = Ribeiro − P3 con el percentil 2.5–97.5 de 1 000 remuestreos **pareados** de los registros (los mismos para ambos sistemas). Una diferencia es detectable solo si el intervalo excluye 0.

Etiquetas de Ribeiro (R); en D la definición R de AF **no tiene positivos** (los encabezados de Ningbo no usan el código 164889003), por eso su fila no tiene valor:

| Origen | Clase de Ribeiro (P3) | n+ | AUROC P3 | AUROC Ribeiro | Dif. [IC 95 %] | AUPRC P3 | AUPRC Ribeiro | Dif. [IC 95 %] |
|---|---|---|---|---|---|---|---|---|
| C | ST (STach) | 1566 | 0.974 | 0.983 | +0.009 [0.007 a 0.012] | 0.865 | 0.918 | +0.054 [0.039 a 0.069] |
| C | AF (AF) | 1778 | 0.984 | 0.992 | +0.008 [0.006 a 0.009] | 0.883 | 0.951 | +0.068 [0.048 a 0.085] |
| C | SB (SB) | 3888 | 0.990 | 0.944 | −0.046 [−0.051 a −0.042] | 0.984 | 0.878 | −0.106 [−0.117 a −0.096] |
| D | ST (STach) | 5683 | 0.966 | 0.988 | +0.021 [0.019 a 0.024] | 0.891 | 0.953 | +0.061 [0.055 a 0.068] |
| D | AF (AF) | 0 | — | — | — [—] | — | — | — [—] |
| D | SB (SB) | 12659 | 0.979 | 0.924 | −0.055 [−0.058 a −0.052] | 0.960 | 0.823 | −0.136 [−0.143 a −0.130] |

Etiquetas agrupadas de este trabajo (G), con las que sí se puede calcular AF en D; la media usa G en las tres clases:

| Origen | Clase de Ribeiro (P3) | n+ | AUROC P3 | AUROC Ribeiro | Dif. [IC 95 %] | AUPRC P3 | AUPRC Ribeiro | Dif. [IC 95 %] |
|---|---|---|---|---|---|---|---|---|
| C | AF (AF) | 2223 | 0.990 | 0.995 | +0.004 [0.003 a 0.006] | 0.967 | 0.983 | +0.016 [0.011 a 0.021] |
| C | media de ST, AF, SB | — | 0.985 | 0.974 | −0.011 [−0.013 a −0.009] | 0.939 | 0.926 | −0.012 [−0.018 a −0.006] |
| D | AF (AF) | 7600 | 0.992 | 0.988 | −0.004 [−0.006 a −0.003] | 0.978 | 0.976 | −0.002 [−0.004 a 0.000] |
| D | media de ST, AF, SB | — | 0.979 | 0.967 | −0.013 [−0.014 a −0.011] | 0.943 | 0.918 | −0.025 [−0.029 a −0.022] |

Las filas de la entrada A, las etiquetas agrupadas de este trabajo (G), el F1 informativo y las tres clases sin equivalente están en `comparacion_ribeiro_vs_P3.csv` (y en la Tabla S1 del material suplementario del artículo).

## Qué NO se afirma
Que un sistema sea «mejor» en general: se entrenaron con datos, prevalencias y definiciones de etiqueta distintas (Ribeiro: más de 2 millones de exámenes de una red de telemedicina brasileña; P3: 22 895 registros de cinco fuentes públicas), y la comparación es exploratoria. No se mezclan las cifras publicadas por Ribeiro et al. (sobre su propio conjunto de prueba) con estas, que se **ejecutaron**.

## Notas de la ejecución
- En Ningbo, el registro `S23074` tiene una cabecera que nombra el archivo de señal `JS23074.mat`; los scripts leen el archivo que nombra la cabecera.
- 1 registro de D (JS45386) tiene derivaciones planas: `ecg.signal.load_wfdb_record_windows` (ruta de inferencia) lo rechaza, pero la evaluación de P3 se hizo con los HDF5 de `build_external_test.py`, que no descartan registros por eso; para la entrada B se replica el mismo preprocesamiento sin esa comprobación.
- En Ningbo no aparece el código SNOMED 164889003 (fibrilación auricular) en ninguna cabecera; su clase AF del sistema principal se puebla con el código 164890007 (flúter auricular), que no se contrastó con la anotación original.

## Cómo repetirla
```powershell
python tools/descargar_challenge2021.py <carpeta>          # datos públicos (≈ 5.4 GB), reanudable
python tools/verificar_challenge2021.py --data-dir <carpeta>/physionet.org/files/challenge-2021/1.0.3/training
python examples/cinc2020/baseline_ribeiro/validar_port_ribeiro.py <datos_ribeiro> <salida>
python examples/cinc2020/baseline_ribeiro/comparar_ribeiro_cd.py <datos_ribeiro> <carpeta> runs/t19/eval <salida> .
```
Entorno de la ejecución de R11: Python 3.13, PyTorch 2.14.0 (CPU), NumPy 2.3.3, SciPy 1.16.2, 8 hilos; una ejecución completa tarda ≈ 36 min en C y ≈ 38 min en D en un i5-10300H.
