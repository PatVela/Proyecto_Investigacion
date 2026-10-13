<a id="readme-top"></a>
<!--
*** Plantilla base: https://github.com/othneildrew/Best-README-Template
*** Submódulo: pipeline de datos, entrenamiento y experimentos CINC2020-12.
-->

[![Python 3.10+][python-shield]][python-url]
[![PyTorch 2.8][pytorch-shield]][pytorch-url]

<br />
<div align="center">
  <a href="https://github.com/PatVela/Proyecto_Investigacion">
    <img src="../../images/escudo-unsa.png" alt="Escudo UNSA" height="90">
  </a>

<h3 align="center">Pipeline CINC2020-12</h3>

  <p align="center">
    Scripts de construcción de datos, entrenamiento, evaluación, comparación y robustez
    <br />
    para PhysioNet/CinC Challenge 2020 con 12 clases SNOMED multilabel.
    <br />
    <a href="../../README.md"><strong>Volver al README principal »</strong></a>
    <br />
    <br />
    <a href="../../webapp/README.md">Ver demo web</a>
    &middot;
    <a href="https://github.com/PatVela/Proyecto_Investigacion/issues/new?labels=bug">Reportar Bug</a>
    &middot;
    <a href="https://github.com/PatVela/Proyecto_Investigacion/issues/new?labels=enhancement">Pedir Funcionalidad</a>
  </p>
</div>

<details>
  <summary>Tabla de contenidos</summary>
  <ol>
    <li><a href="#sobre-este-módulo">Sobre este módulo</a></li>
    <li>
      <a href="#primeros-pasos">Primeros pasos</a>
      <ul>
        <li><a href="#prerrequisitos">Prerrequisitos</a></li>
      </ul>
    </li>
    <li><a href="#uso">Uso</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#licencia">Licencia</a></li>
  </ol>
</details>

## Sobre este módulo

Esta carpeta contiene el pipeline experimental completo del proyecto: desde la descarga de CINC2020 hasta la métrica oficial del Challenge. Los comandos se ejecutan desde la **raíz del repositorio**.

| Script | Propósito |
|---|---|
| `build_datasets.py` | Construye `train/val/test.h5` + CSV de distribución y resúmenes |
| `evaluate.py` | Métricas multilabel + umbrales calibrados en validación |
| `compare_models.py` | Compara dos evaluaciones con el mismo `threshold_beta` (p. ej. ResNet v1 vs v2) |
| `robustness.py` | Robustez ante perturbaciones controladas de la señal |
| `challenge_score.py` | Métrica oficial estilo Challenge 2020 (24 clases, equivalentes fusionadas) desde decisiones guardadas o un checkpoint |
| `_archive/diagnose_prediction.py` | Diagnóstico detallado de un registro |
| `_archive/debug_record_prediction.py` | Compara señal webapp vs señal HDF5 por registro |
| `calibrate.py` | Calibración por clase (temperatura o afín `--method platt`) + reporte ECE/Brier |
| `ensemble_evaluate.py` | Ensemble promedio de dos checkpoints |
| `error_analysis.py` | Errores por subconjunto de origen (hospital/fuente) y diagnóstico de celdas origen×clase sin anotar |
| `export_onnx.py` | Exportación a ONNX con verificación numérica |
| `audit_dx_codes.py` | Cobertura Dx→clase: frecuencias por código SNOMED |
| `make_figures.py` | Figuras ROC/PR empíricas + robustez + aprendizaje |
| `bootstrap_ci.py` | IC 95 % bootstrap de F1-macro y diferencia pareada |
| `sex_breakdown.py` | F1-macro por sexo (auditoría de sesgo) |
| `benchmark_latency.py` | Latencia de inferencia y conteo de parámetros |
| `tune_optuna.py` | Búsqueda de hiperparámetros con Optuna (escribe `config_resnet_v3.json`; ese config no está versionado) |
| `make_synthetic.py` | Dataset sintético para smoke tests |
| `config*.json` | Configs de ResNet, ResNet v2, regular y sintético |
| `official/` | Scripts y tablas oficiales (vendored, no modificar) |

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

## Primeros pasos

### Prerrequisitos

* Instalación base del proyecto (ver [README principal](../../README.md#instalación)).
* Datos CINC2020 descargados:
  ```sh
  bash descargar_datos_2020.sh
  ```

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

## Uso

Comandos PowerShell desde la **raíz del repositorio**, con variables para no escribir rutas de checkpoints a mano. Defínalas una vez por terminal (nombres según el [README principal](../../README.md#uso)):

```powershell
$resnet = Get-ChildItem saved/cinc2020/cinc2020_resnet/*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$resnet2 = Get-ChildItem saved/cinc2020/cinc2020_resnet_v2*/*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$mejor = $resnet2
$evalMejor = "runs/eval/eval-resnet-v2"
```

### 1. Construir HDF5

```powershell
python examples/cinc2020/build_datasets.py --data-dir dataset2020 --output-dir data/cinc2020_12 --workers 8 --norm-mode physical --train-windows-max 4
```

### 2. Entrenar ResNet-34 tipo Hannun

```powershell
python -m ecg.train examples/cinc2020/config.json -e cinc2020_resnet
```

### 3. Entrenar ResNet-34 v2 (receta anti-sobreajuste)

```powershell
python -m ecg.train examples/cinc2020/config_resnet_v2.json -e cinc2020_resnet_v2
```

Receta con aumentación de señal, label smoothing, más épocas y selección por F1-macro (ver §12).

### 4. Evaluar

```powershell
python examples/cinc2020/evaluate.py examples/cinc2020/config.json $resnet --output-dir runs/eval/eval-resnet --temperatures runs/eval/eval-resnet/temperatures_validation.csv --threshold-beta 0.5
```

Genera umbrales por clase, métricas por clase y globales, predicciones, matrices de confusión 2×2 y curvas ROC/PR por clase. Para v2 use su config, checkpoint y carpeta (`runs/eval/eval-resnet-v2`). Para cuantificar el fallback normal en todo el conjunto:

```powershell
python examples/cinc2020/evaluate.py examples/cinc2020/config.json $resnet --output-dir runs/eval/eval-resnet-fallback --normal-fallback-min-prob 0.40
```

Para ablaciones con umbral fijo (p. ej. tabla 2×2 temperatura × umbral):

```powershell
python examples/cinc2020/evaluate.py examples/cinc2020/config.json $resnet --output-dir runs/eval/eval-abl-t05 --fixed-threshold 0.5
```

### 5. Comparar modelos

```powershell
python examples/cinc2020/compare_models.py --eval-a runs/eval/eval-resnet --eval-b runs/eval/eval-resnet-v2 --label-a "ResNet v1" --label-b "ResNet v2" --output runs/reports/model_comparison.csv
```

Las etiquetas son opcionales (por defecto usa el nombre de cada directorio). Las dos evaluaciones deben tener el mismo `threshold_beta` (se lee de `evaluation_summary.json`); si no, el script se detiene, porque la diferencia de F1-macro mezclaría modelo y punto de operación (`--allow-different-beta` lo permite y deja la columna en el CSV).

### 6. Robustez controlada

```powershell
python examples/cinc2020/robustness.py $mejor data/cinc2020_12/test.h5 --thresholds "$evalMejor/thresholds_validation.csv" --temperatures "$evalMejor/temperatures_validation.csv" --output runs/reports/robustness.csv
```

Perturbaciones: ruido gaussiano, baseline wander, escalado de amplitud y apagado de derivaciones (incluye fila base `clean`). Pase siempre `--temperatures` del mismo eval-dir que los umbrales: sin él, los umbrales calibrados se aplican sobre probabilidades crudas (protocolo mismatched, base subestimada).

### 7. Diagnóstico de predicción

```powershell
$hea1 = Get-ChildItem dataset2020 -Recurse -Filter E00001.hea | Select-Object -First 1 -ExpandProperty FullName
$mat1 = [System.IO.Path]::ChangeExtension($hea1, '.mat')
python examples/cinc2020/_archive/diagnose_prediction.py --checkpoint $mejor --config examples/cinc2020/config.json --thresholds "$evalMejor/thresholds_validation.csv" --record E00001 --hea $hea1 --mat $mat1
```

### 8. Métrica oficial estilo Challenge 2020 (24 clases)

```powershell
python examples/cinc2020/challenge_score.py --predictions "$evalMejor/predictions_test.csv" --test-h5 data/cinc2020_12/test.h5 --output-dir runs/eval/metrica-challenge/mejor
```

Puntúa las decisiones ya guardadas de una evaluación (también las de un ensemble). Las clases equivalentes del evaluador oficial (CRBBB/RBBB, PAC/SVPB, PVC/VPB) se fusionan, así que se puntúan 24 clases (`--no-merge-equivalent` conserva las 27 columnas). Desde un checkpoint (`--checkpoint`) hay que pasar además `--temperatures` con las temperaturas con las que se ajustaron los umbrales; sin ellas los umbrales se aplican a probabilidades crudas y la métrica sale subestimada (así se obtuvo el antiguo 0.2348). El esquema de 12 grupos no puede llegar a 1: un clasificador perfecto de grupos puntúa ≈ 0.50.

### 9. Calibrar probabilidades (temperatura o calibración afín)

```powershell
python examples/cinc2020/calibrate.py --checkpoint $resnet --val-h5 data/cinc2020_12/val.h5 --output-dir runs/eval/eval-resnet
python examples/cinc2020/calibrate.py --method platt --checkpoint $resnet --val-h5 data/cinc2020_12/val.h5 --output-dir runs/eval/eval-resnet
```

La primera ajusta una temperatura por clase sobre validación y guarda `temperatures_validation.csv` + `calibration_report.json` (ECE/Brier). Aplíquela en evaluación con `--temperatures`. La segunda ajusta `sigmoid(a·logit + b)` por clase sobre los logits crudos del checkpoint y guarda `platt_validation.csv` + `calibration_report_platt.json` (no pisa los de la temperatura); se aplica con `--platt`. Entrenar con `pos_weight = neg/pos` y label smoothing sube la probabilidad base de todos los registros (un desplazamiento que una temperatura no corrige): el ECE de v2 pasa de 0.114 a 0.014. La calibración afín conserva el orden de las puntuaciones (a > 0), así que F1, AUROC y AUPRC no cambian; sí cambia la escala de los umbrales, que se reoptimizan. `--predictions predictions_validation.csv` también sirve, pero solo trae probabilidades y el ajuste es menos preciso (avisa). La webapp detecta cualquiera de las dos.

### 10. Exportar a ONNX

```powershell
python examples/cinc2020/export_onnx.py --checkpoint $mejor --output runs/models/resnet12.onnx
```

Usa `onnx`/`onnxruntime` de `requirements.txt`. Verifica el grafo y compara numéricamente contra PyTorch.

### 11. Análisis de errores por origen

```powershell
python examples/cinc2020/error_analysis.py --predictions "$evalMejor/predictions_test.csv" --test-h5 data/cinc2020_12/test.h5 --output runs/reports/errores_origen.csv
```

Reporta exact-match y F1 macro/micro por cada subconjunto de origen (hospital/fuente).

El diseño experimental completo está en [docs/experimentos_cinc2020.md](../../docs/experimentos_cinc2020.md).

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

### 12. Máxima accuracy: F-beta y ensemble

Umbrales orientados a precisión (menos falsos positivos) con F0.5:

```powershell
python examples/cinc2020/evaluate.py examples/cinc2020/config.json $resnet --output-dir runs/eval/eval-resnet --threshold-beta 0.5
```

`--threshold-beta 1.0` = F1 (equilibrio, valor por defecto); `0.5` = menos FPs; `2.0` = menos FNs. La opción `--nsr-exclusive-min-prob` (predecir solo NSR si P(NSR) es muy alta) resultó perjudicial en ablación (−11 pts de F1-macro sin ganancia de precisión, porque NSR coexiste legítimamente con clases morfológicas); no se recomienda. La columna `validation_objective` de `thresholds_validation.csv` registra el objetivo optimizado.

Ensemble promedio de dos checkpoints (mismo layout de salida que `evaluate.py`). Con `--threshold-beta 1.0` y `--output-dir runs/eval/ensemble-f1` se obtiene el punto de operación de screening (F1), el que usa la demo pública:

```powershell
python examples/cinc2020/ensemble_evaluate.py examples/cinc2020/config.json $resnet $resnet2 --output-dir runs/eval/ensemble --alpha 0.5 --threshold-beta 0.5 --temperatures-a runs/eval/eval-resnet/temperatures_validation.csv --temperatures-b runs/eval/eval-resnet-v2/temperatures_validation.csv
python examples/cinc2020/ensemble_evaluate.py examples/cinc2020/config.json $resnet $resnet2 --output-dir runs/eval/ensemble-f1-platt --alpha 0.5 --threshold-beta 1.0 --platt-a runs/eval/eval-resnet/platt_validation.csv --platt-b runs/eval/eval-resnet-v2/platt_validation.csv --recalibrate-ensemble
```

Con `--platt-a/--platt-b` cada modelo se calibra con su calibración afín; `--recalibrate-ensemble` ajusta, solo con validación, otra sobre el promedio (promediar probabilidades calibradas no deja una probabilidad calibrada) y la guarda como `platt_ensemble_validation.csv` en el directorio de salida. `runs/eval/ensemble-f1-platt` es el que se empaqueta para la demo.

Para re-entrenar la ResNet con receta anti-sobreajuste (más épocas, más regularización), usar `examples/cinc2020/config_resnet_v2.json` con `ecg.train`. Ese config selecciona el mejor checkpoint por F1-macro en validación (`early_stopping_metric: val_f1_macro`) en vez de por `val_loss`, porque ambas métricas pueden discrepar; `history.csv` registra ambas curvas en todos los runs. Además activa aumentación de señal (`augment: true`: ruido, deriva basal, escala, desplazamiento temporal, dropout de derivación) y label smoothing (`label_smoothing: 0.05`) para frenar el sobreajuste. Resultados finales (v1 vs v2 vs ensemble) en [docs/resultados_finales.md](../../docs/resultados_finales.md).

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

### 13. Auditoría de códigos Dx

```powershell
python examples/cinc2020/audit_dx_codes.py --data-dir dataset2020 --output runs/reports/dx_code_audit.csv
```

Recorre los `.hea`, cuenta en cuántos registros aparece cada código Dx y lo cruza con el esquema de 12 clases y el mapeo oficial de 27 puntuados. Reporta códigos del esquema con frecuencia 0 y códigos Dx sin clase asignada. En `dataset2020`: 111 códigos distintos, 27/27 oficiales presentes, 0 frecuencias 0 y 10 códigos sin clase (pre-excitación/WPW, comorbilidades no diagnosticables por ECG y hallazgos inespecíficos o técnicos).

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

### 14. Figuras empíricas del informe

```powershell
python examples/cinc2020/make_figures.py --eval-dir $evalMejor --outdir runs/figs/figs_eval --robustness runs/reports/robustness.csv
```

Traza curvas ROC/PR empíricas por clase con punto de operación (idéntico a la tabla F1 por construcción), clases destacadas y barras de robustez. Agregue `--history-v1`/`--history-v2` con los `history.csv` de cada run para la figura de aprendizaje.

### 15. Intervalos bootstrap (IC 95 %)

```powershell
python examples/cinc2020/bootstrap_ci.py --predictions $evalMejor/predictions_test.csv --predictions-baseline runs/eval/eval-resnet/predictions_test.csv --output runs/reports/bootstrap_v2-v1.json --n-bootstrap 1000
```

Remuestrea registros del test (sistema fijo): IC de F1-macro, por clase y de la diferencia pareada entre modelos. Sin `--predictions-baseline` reporta solo el modelo.

### 16. Auditorías: sexo, latencia y baseline

```powershell
python examples/cinc2020/sex_breakdown.py --predictions $evalMejor/predictions_test.csv --test-h5 data/cinc2020_12/test.h5 --output runs/reports/sex_breakdown.json
```

```powershell
python examples/cinc2020/benchmark_latency.py --checkpoint $mejor --device auto --iters 200
```

```powershell
python -m ecg.train examples/cinc2020/config_regular_cnn.json -e cinc2020_regular
```

Desglose de F1-macro por sexo, ms/ventana y parámetros, y entrenamiento de la CNN convencional (misma receta que v1) como línea base no residual.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

## Roadmap

- [x] Evaluación multilabel con umbrales calibrados
- [x] Comparación de evaluaciones + robustez controlada
- [x] Métrica oficial del Challenge sobre 27 códigos
- [x] Curvas ROC/PR por clase exportadas a CSV
- [x] Análisis de errores por subconjunto de origen
- [x] Calibración temperature scaling + reporte ECE/Brier
- [x] Exportación a ONNX verificada numéricamente
- [ ] Validación cruzada leave-one-source-out (generalización por hospital)
- [ ] Curvas de calibración (reliability diagrams) exportadas a CSV
- [x] Aumentación de señal durante el entrenamiento
- [x] Umbrales F-beta + selección de checkpoint por F1-macro
- [x] Ensemble promedio de checkpoints (evaluación e inferencia webapp)
- [x] Script de tuning con Optuna (`tune_optuna.py`; genera `config_resnet_v3.json`, no versionado)

Ver los [issues abiertos](https://github.com/PatVela/Proyecto_Investigacion/issues) para más propuestas.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

## Licencia

Distribuido bajo licencia GPL-3.0. Ver `LICENSE` en la raíz para más información.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

[python-shield]: https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/pytorch-2.8-ee4c2c?style=for-the-badge&logo=pytorch
[pytorch-url]: https://pytorch.org/
