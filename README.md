<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Plantilla base: https://github.com/othneildrew/Best-README-Template
*** Proyecto: réplica académica de Hannun et al. (Nat Med 2019) sobre CINC2020.
-->

<!-- PROJECT SHIELDS -->
<!--
*** Se usan "reference style" links para legibilidad.
*** Ver la declaración de variables al final del documento.
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPL-3.0 License][license-shield]][license-url]
[![Python 3.10+][python-shield]][python-url]
[![PyTorch 2.8][pytorch-shield]][pytorch-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/PatVela/Proyectos-Universidad">
    <img src="images/escudo-unsa.png" alt="Escudo UNSA" height="110">
  </a>

<h3 align="center">ECG CINC2020-12 · ResNet tipo Hannun</h3>

  <p align="center">
    Clasificador multilabel de ECG de 12 derivaciones con redes convolucionales profundas.
    <br />
    Réplica académica de Hannun et al. (<i>Nature Medicine</i>, 2019) sobre el dataset público PhysioNet/CinC Challenge 2020.
    <br />
    <a href="docs/informe_cinc2020_12.md"><strong>Explorar la documentación »</strong></a>
    <br />
    <br />
    <a href="webapp/README.md">Ver demo web</a>
    &middot;
    <a href="https://github.com/PatVela/Proyectos-Universidad/issues/new?labels=bug">Reportar Bug</a>
    &middot;
    <a href="https://github.com/PatVela/Proyectos-Universidad/issues/new?labels=enhancement">Pedir Funcionalidad</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Tabla de contenidos</summary>
  <ol>
    <li>
      <a href="#sobre-el-proyecto">Sobre el proyecto</a>
      <ul>
        <li><a href="#construido-con">Construido con</a></li>
      </ul>
    </li>
    <li>
      <a href="#primeros-pasos">Primeros pasos</a>
      <ul>
        <li><a href="#prerrequisitos">Prerrequisitos</a></li>
        <li><a href="#instalación">Instalación</a></li>
      </ul>
    </li>
    <li><a href="#uso">Uso</a></li>
    <li><a href="#esquema-de-12-clases">Esquema de 12 clases</a></li>
    <li><a href="#estructura-del-proyecto">Estructura del proyecto</a></li>
    <li><a href="#aplicación-web">Aplicación web</a></li>
    <li><a href="#experimentos">Experimentos</a></li>
    <li><a href="#pruebas-y-despliegue">Pruebas y despliegue</a></li>
    <li><a href="#diferencias-frente-a-hannun-et-al">Diferencias frente a Hannun et al.</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#licencia">Licencia</a></li>
    <li><a href="#agradecimientos">Agradecimientos</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## Sobre el proyecto

[![Informe de ejemplo][product-screenshot]](docs/informe_cinc2020_12.md)

El dataset original de Hannun et al. (Zio Patch/iRhythm) no es público, así que esta implementación adapta su enfoque ResNet-1D al dataset público **PhysioNet/Computing in Cardiology Challenge 2020**: ECG de 12 derivaciones con etiquetas diagnósticas SNOMED-CT.

El problema se formula como **clasificación multilabel real** —un ECG puede tener varios diagnósticos a la vez— con 12 salidas sigmoid independientes, umbrales calibrados por clase y preprocesamiento en unidades físicas (milivoltios + filtro pasa-banda 0.5–50 Hz). El repositorio incluye el pipeline completo: construcción de datasets HDF5, entrenamiento, evaluación, experimentos de comparación/robustez y una aplicación web con informe PDF.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

### Construido con

* [![Python][Python-badge]][Python-url]
* [![PyTorch][PyTorch-badge]][PyTorch-url]
* [![Flask][Flask-badge]][Flask-url]
* [![NumPy][NumPy-badge]][NumPy-url]
* [![pandas][pandas-badge]][pandas-url]
* [![scikit-learn][sklearn-badge]][sklearn-url]
* [![Matplotlib][Matplotlib-badge]][Matplotlib-url]
* [![WFDB][WFDB-badge]][WFDB-url]
* [![h5py][h5py-badge]][h5py-url]
* [![Plotly][Plotly-badge]][Plotly-url]

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- GETTING STARTED -->
## Primeros pasos

Para tener una copia local funcionando, siga estos pasos.

### Prerrequisitos

* Python 3.10 o superior.
* ~15 GB de disco libre si va a descargar CINC2020 completo.
* 16 GB de RAM recomendados (el pipeline precarga los HDF5 a memoria).
* GPU NVIDIA opcional (el código corre en CPU; con CUDA instale PyTorch según `requirements-cuda.txt`).

### Instalación

La instalación completa es el **Paso 0** de la sección [Uso](#uso): clonado, PyTorch con CUDA, dependencias, verificación de GPU y smoke test, en orden. Siga los pasos 0–10 sin saltarse la verificación de `cuda: True`.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- USAGE EXAMPLES -->
## Uso

Guía completa de replicación en PowerShell. Todos los comandos son de una sola línea y usan variables para no escribir rutas de checkpoints a mano.

Notas generales:

* Si la GPU se queda sin memoria (`CUDA out of memory`), baje `batch_size` a 32 en el JSON de configuración y reentrene.
* No modifique `num_workers: 0` (la precarga de HDF5 usa ~9 GB de RAM).
* PyTorch se instala aparte con CUDA (ver Paso 0); el proyecto no usa `torchvision`.

Nombres usados en la guía:

| Carpeta/archivo | Contenido |
|---|---|
| `eval-resnet` | Calibración + evaluación ResNet v1 |
| `eval-resnet-v2` | Calibración + evaluación ResNet v2 |
| `exp-files` | `model_comparison.csv` (v1 vs v2) + `robustness.csv` |
| `ensemble` | Ensemble ResNet v1 + v2 |
| `comparacion_ensemble.csv` | Ensemble vs mejor individual |
| `metrica-challenge` | Métrica oficial estilo Challenge |
| `errores_origen.csv` | Errores por hospital |
| `resnet12.onnx` | Modelo exportado |

### Paso 0 — Instalación y verificación

```powershell
git clone -b Paper_Replica_V3 https://github.com/PatVela/Proyectos-Universidad.git
cd Proyectos-Universidad
pip install -r requirements-cuda.txt
pip install -r requirements.txt
```

Verifique GPU **antes de seguir**:

```powershell
python -c "import torch; print(torch.__version__, '| cuda:', torch.cuda.is_available())"
```

Debe decir `cuda: True`. Si dice False: actualice el driver NVIDIA o instale la compilación de PyTorch correspondiente a su CUDA desde pytorch.org. Para CPU, omita `requirements-cuda.txt` e instale la rueda CPU de PyTorch.

Smoke test (~2 min):

```powershell
python examples/cinc2020/make_synthetic.py
python -m ecg.train examples/cinc2020/config_synthetic.json -e smoke_test --epochs 1 --device cpu --no-amp
pytest tests/ -q
```

Debe decir `78 passed`. No use el dataset sintético para reportar métricas científicas.

### Paso 1 — Descargar datos (~15 GB)

```powershell
bash descargar_datos_2020.sh
```

En Windows, ejecute esta línea en una terminal **Git Bash**.

Opcional, verificar que cada `.hea` tenga su `.mat`:

```powershell
python tools/check_missing.py --kind mats --data-dir dataset2020
```

### Paso 2 — Construir HDF5 (~20–40 min según CPU)

```powershell
python examples/cinc2020/build_datasets.py --data-dir dataset2020 --output-dir data/cinc2020_12 --workers 8 --norm-mode physical --train-windows-max 4
```

Ajuste `--workers` al número de núcleos de su CPU. Cada HDF5 contiene `signals → (N, 5000, 12)` y `labels → (N, 12)` (12 derivaciones → 500 Hz → mV + pasa-banda 0.5–50 Hz → recorte ±5 mV → 5000 muestras); en `train`, los registros largos aportan hasta 4 ventanas de 10 s; validación y test usan una ventana centrada.

### Paso 3 — Entrenar v1 + v2

```powershell
python -m ecg.train examples/cinc2020/config.json -e cinc2020_resnet
```

(~35 min en GPU). Al arrancar verifique: `Device: cuda`, `Save dir: ...cinc2020_resnet...`, `Época 1/80`. Luego:

```powershell
python -m ecg.train examples/cinc2020/config_resnet_v2.json -e cinc2020_resnet_v2
```

(~2.3 h en GPU). Verifique las 5 líneas del checklist antes de dejarlo solo:

```text
Device       : cuda
Save dir     : ...cinc2020_resnet_v2...
Aumentación activada (solo train): {...}
Monitorizando: val_f1_macro
Época 1/150
```

Opcional, segunda semilla para quedarse con la mejor:

```powershell
python -m ecg.train examples/cinc2020/config_resnet_v2.json -e cinc2020_resnet_v2b --seed 44
```

Línea base no residual (misma receta que v1, ~35 min en GPU):

```powershell
python -m ecg.train examples/cinc2020/config_regular_cnn.json -e cinc2020_regular
```

### Paso 4 — Checkpoints y calibración

```powershell
$resnet = Get-ChildItem saved/cinc2020/cinc2020_resnet/*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$resnet2 = Get-ChildItem saved/cinc2020/cinc2020_resnet_v2*/*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$resnet
$resnet2
```

**Lea las dos rutas**: deben apuntar a `cinc2020_resnet...` y `cinc2020_resnet_v2*...` respectivamente. Si ve un run abortado a medias, bórrelo primero con `Remove-Item -Recurse` apuntando a la carpeta de ese run.

```powershell
python examples/cinc2020/calibrate.py --checkpoint $resnet --val-h5 data/cinc2020_12/val.h5 --output-dir eval-resnet
python examples/cinc2020/calibrate.py --checkpoint $resnet2 --val-h5 data/cinc2020_12/val.h5 --output-dir eval-resnet-v2
```

### Paso 5 — Evaluar (~10 min c/u en GPU)

```powershell
python examples/cinc2020/evaluate.py examples/cinc2020/config.json $resnet --output-dir eval-resnet --temperatures eval-resnet/temperatures_validation.csv --threshold-beta 0.5
python examples/cinc2020/evaluate.py examples/cinc2020/config_resnet_v2.json $resnet2 --output-dir eval-resnet-v2 --temperatures eval-resnet-v2/temperatures_validation.csv --threshold-beta 0.5
```

Sin `--nsr-exclusive-min-prob`: esa regla resultó perjudicial en ablación (−11 pts de F1-macro sin ganancia de precisión) y no se usa. Cada directorio contendrá umbrales por clase, métricas por clase y globales, predicciones de validación/test, matrices de confusión 2×2, curvas ROC/PR y un resumen JSON (`evaluation_summary.json`).

### Paso 6 — Comparar v1 vs v2 y elegir el mejor

```powershell
python examples/cinc2020/compare_models.py --eval-a eval-resnet --eval-b eval-resnet-v2 --label-a "ResNet v1" --label-b "ResNet v2" --output exp-files/model_comparison.csv
```

Abra `exp-files/model_comparison.csv` en su visor de CSV, mire F1-macro en test y defina el ganador (ajuste según lo que vea; aquí se asume v2):

```powershell
$mejor = $resnet2
$evalMejor = "eval-resnet-v2"
```

### Paso 7 — Ensemble + experimentos (~20 min en GPU)

```powershell
python examples/cinc2020/ensemble_evaluate.py examples/cinc2020/config.json $resnet $resnet2 --output-dir ensemble --alpha 0.5 --threshold-beta 0.5 --temperatures-a eval-resnet/temperatures_validation.csv --temperatures-b eval-resnet-v2/temperatures_validation.csv
python examples/cinc2020/compare_models.py --eval-a "$evalMejor" --eval-b ensemble --label-a "ResNet mejor" --label-b "Ensemble" --output comparacion_ensemble.csv
python examples/cinc2020/robustness.py $mejor data/cinc2020_12/test.h5 --thresholds "$evalMejor/thresholds_validation.csv" --temperatures "$evalMejor/temperatures_validation.csv" --output exp-files/robustness.csv
python examples/cinc2020/challenge_score.py --checkpoint $mejor --test-h5 data/cinc2020_12/test.h5 --thresholds "$evalMejor/thresholds_validation.csv" --output-dir metrica-challenge
python examples/cinc2020/error_analysis.py --predictions "$evalMejor/predictions_test.csv" --test-h5 data/cinc2020_12/test.h5 --output errores_origen.csv
python examples/cinc2020/make_figures.py --eval-dir "$evalMejor" --outdir figs_eval --robustness exp-files/robustness.csv
python examples/cinc2020/bootstrap_ci.py --predictions "$evalMejor/predictions_test.csv" --predictions-baseline eval-resnet/predictions_test.csv --output bootstrap_ci.json --n-bootstrap 1000
```

### Paso 8 — Diagnóstico de registros ejemplo

```powershell
$hea1 = Get-ChildItem dataset2020 -Recurse -Filter E00001.hea | Select-Object -First 1 -ExpandProperty FullName
$mat1 = [System.IO.Path]::ChangeExtension($hea1, '.mat')
python examples/cinc2020/diagnose_prediction.py --checkpoint $mejor --config examples/cinc2020/config.json --thresholds "$evalMejor/thresholds_validation.csv" --record E00001 --hea $hea1 --mat $mat1
$hea14 = Get-ChildItem dataset2020 -Recurse -Filter E00014.hea | Select-Object -First 1 -ExpandProperty FullName
$mat14 = [System.IO.Path]::ChangeExtension($hea14, '.mat')
python examples/cinc2020/diagnose_prediction.py --checkpoint $mejor --config examples/cinc2020/config.json --thresholds "$evalMejor/thresholds_validation.csv" --record E00014 --hea $hea14 --mat $mat14
```

### Paso 9 — Webapp (ensemble final)

```powershell
python webapp/app.py --saved saved --eval-dir ensemble --exp-dir exp-files --model $resnet --model-b $resnet2 --temperatures eval-resnet/temperatures_validation.csv --temperatures-b eval-resnet-v2/temperatures_validation.csv --port 5002
```

Abra http://127.0.0.1:5002 y pruebe con `E00001.hea + E00001.mat`.

### Paso 10 — ONNX y Docker (opcionales)

```powershell
python examples/cinc2020/export_onnx.py --checkpoint $mejor --output resnet12.onnx
docker build -t ecg-cinc2020 .
docker run --rm -p 5002:5002 -v ./saved:/app/saved:ro ecg-cinc2020
```

En Linux/bash los comandos `python`, `pytest`, `bash` y `docker` son idénticos; solo cambia cómo se definen las variables (usar `$resnet`, `$mejor`, etc. dentro de los comandos funciona igual):

```bash
resnet=$(ls -t saved/cinc2020/cinc2020_resnet/*/best.pt | head -1)
resnet2=$(ls -t saved/cinc2020/cinc2020_resnet_v2*/*/best.pt | head -1)
mejor=$resnet2
evalMejor="eval-resnet-v2"
hea1=$(find dataset2020 -name E00001.hea | head -1)
mat1="${hea1%.hea}.mat"
```

### Si algo falla

* `CUDA out of memory`: baje `batch_size` a 32 en el JSON y reentrene.
* `Device: cpu` al arrancar: detenga, verifique `cuda: True` y relance. No entrene en CPU.
* Duda del config usado: abra `config_used.json` dentro de la carpeta `Save dir` del run.
* `pip install -r requirements.txt` falla en `gevent`: coméntelo con `#` (solo sirve para despliegue productivo).
* Época lenta (>3 min en GPU): confirme el mensaje `Precargando a RAM`; cierre aplicaciones pesadas.
* Webapp sin métricas: revise `--eval-dir`/`--exp-dir`; `/health` muestra qué detectó.

_Más ejemplos y comandos avanzados en [examples/cinc2020/README.md](examples/cinc2020/README.md) y [webapp/README.md](webapp/README.md)._

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- CLASS SCHEMA -->
## Esquema de 12 clases

Fuente de verdad: `ecg/schema.py` (re-exportado por `ecg/load.py`; esquema `cinc2020_12_grouped_snomed_v2`). Cubre las 27 clases puntuadas oficiales del Challenge 2020 (ver `examples/cinc2020/official/dx_mapping_scored.csv`); el esquema v1 solo cubría ~14 y dejaba ~21% de registros como all-zero.

| # | Clase | Códigos SNOMED-CT incluidos |
|--:|---|---|
| 0 | `NSR` | `426783006`, `427393009` |
| 1 | `AxisDev` | `39732003`, `445118002`, `47665007`, `445211001`, `251200008` |
| 2 | `MI` | `164865005`, `57054005`, `164867002`, `54329005`, `164861001`, `413444003`, `413844008`, `426434006`, `425419005`, `425623009`, `164917005` |
| 3 | `TAb` | `164934002`, `59931005`, `428750005`, `429622005`, `164931005`, `164930006`, `55930002`, `704997005`, `111975006`, `77867006`, `164937009`, `251259000`, `428417006` |
| 4 | `AF` | `164889003`, `164890007`, `195080001`, `282825002`, `426749004`, `314208002` |
| 5 | `LVH` | `164873001`, `370365005`, `446813000`, `67741000119109`, `253352002`, `195126007`, `89792004`, `266249003`, `253339007`, `446358003` |
| 6 | `VEctopy` | `427172004`, `17338001`, `164884008`, `11157007`, `251180001`, `251182009`, `75532003`, `81898007`, `164895002`, `425856008`, `164896001`, `111288001`, `49260003`, `13640000` |
| 7 | `AVBlock` | `270492004`, `195042002`, `233917008`, `27885002`, `54016002`, `204384007`, `164947007`, `65778007` |
| 8 | `STach` | `427084000` |
| 9 | `BBB` | `59118001`, `713427006`, `713426002`, `164909002`, `251120003`, `6374002`, `698252002`, `82226007`, `251146004`, `164951009` |
| 10 | `SB` | `426177001`, `426627000`, `60423000`, `74615001` |
| 11 | `AEctopy_Junctional` | `284470004`, `63593006`, `713422000`, `426664006`, `29320008`, `426995002`, `251164006`, `426648003`, `195101003`, `251268003`, `251170000`, `251168009`, `251173003`, `426761007`, `67198005`, `10370003`, `251266004` |

Notas:

* Las salidas son independientes: sigmoid por clase, no softmax.
* Quedan fuera: WPW/preexcitación (muy infrecuente y morfológicamente singular), diagnósticos clínicos no-ECG (HF/HVD/CHD/TIA) y ruido (sin SNOMED equivalente).
* Los registros sin ninguna de las 12 etiquetas (<1% en v2) se conservan como vectores all-zero; pueden excluirse con `--drop-no-selected-labels`.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- PROJECT STRUCTURE -->
## Estructura del proyecto

```text
.
├── ecg/                    # paquete: carga, red, entrenamiento e inferencia
│   ├── load.py             # fachada: re-exporta schema/signal/datasets
│   ├── schema.py           # esquema SNOMED 12 clases + mapeo a vectores
│   ├── signal.py           # lectores WFDB/CSV, normalización y ventanas
│   ├── datasets.py         # escaneo, splits y construcción de HDF5
│   ├── network.py          # ResNet-34 tipo Hannun
│   ├── predict.py          # inferencia desde checkpoints (CSV/HDF5)
│   ├── train.py            # entrenamiento + logs reproducibles
│   ├── augment.py          # aumentación de señal (preserva etiquetas)
│   ├── calibration.py      # temperature scaling por clase
│   └── util.py             # checkpoints, parámetros y seeds
├── examples/cinc2020/      # pipeline: datos, evaluación y experimentos
│   ├── build_datasets.py / evaluate.py / compare_models.py
│   ├── robustness.py / challenge_score.py
│   ├── calibrate.py / ensemble_evaluate.py
│   ├── error_analysis.py / export_onnx.py
│   ├── diagnose_prediction.py / debug_record_prediction.py
│   ├── audit_dx_codes.py     # cobertura Dx→clase (frecuencia SNOMED)
│   ├── make_figures.py / bootstrap_ci.py   # figuras empíricas e IC bootstrap
│   ├── sex_breakdown.py / benchmark_latency.py  # auditorías: sexo y latencia
│   ├── config*.json        # configs ResNet, ResNet v2, regular y sintético
│   └── official/           # scripts y tablas oficiales (vendored, no modificar)
├── tools/                  # descarga/verificación CINC2020 + auditorías de dataset
├── webapp/                 # dashboard Flask + informe PDF
│   ├── app.py / prediction.py / report_pdf.py / wsgi.py
│   ├── project_info.py     # datos institucionales
│   └── templates/ + static/
├── docs/                   # informe técnico y guías de experimentos
├── images/                 # logo y capturas para documentación
├── requirements.txt / requirements-cuda.txt
├── descargar_datos_2020.sh
└── README.md
```

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- WEBAPP -->
## Aplicación web

Dashboard Flask en español/inglés: carga CSV o par WFDB `.hea + .mat`, predicción multilabel con probabilidades por clase, trazado en papel milimetrado, comparación automática contra el campo `Dx`, informe PDF descargable y secciones de detalle técnico, métricas y experimentos.

```sh
python webapp/app.py --saved saved
```

Detalles, formatos y endpoints en [webapp/README.md](webapp/README.md).

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- EXPERIMENTS -->
## Experimentos

* **Evaluación multilabel**: AUROC/AUPRC por clase, sensibilidad, especificidad, F1 macro/micro y matrices 2×2 (`evaluate.py`).
* **Calibración**: temperature scaling por clase + umbrales F-beta orientados a precisión (`calibrate.py`).
* **Ensemble**: promedio ponderado de dos checkpoints con calibración por modelo (`ensemble_evaluate.py`).
* **Comparación de evaluaciones**: dos runs (p. ej. ResNet v1 vs v2) con idéntico dataset, split y métricas (`compare_models.py`).
* **Robustez controlada**: ruido gaussiano, baseline wander, escalado de amplitud y apagado de derivaciones (`robustness.py`).
* **Métrica oficial estilo Challenge 2020** sobre los 27 códigos puntuados (`challenge_score.py`).
* **Diagnóstico por registro**: inspección de casos individuales (`diagnose_prediction.py`, `debug_record_prediction.py`).
* **Figuras empíricas e IC bootstrap**: curvas ROC/PR con punto de operación, barras de robustez e intervalos por bootstrap (`make_figures.py`, `bootstrap_ci.py`).
* **Auditorías**: desglose por sexo y latencia de inferencia (`sex_breakdown.py`, `benchmark_latency.py`).

Los resultados se visualizan automáticamente en la webapp. Comandos en [examples/cinc2020/README.md](examples/cinc2020/README.md) y diseño experimental en [docs/experimentos_cinc2020.md](docs/experimentos_cinc2020.md). Resultados finales en [docs/resultados_finales.md](docs/resultados_finales.md) y diagnóstico del fix v1→v2 en [docs/fix_prediccion_cinc2020.md](docs/fix_prediccion_cinc2020.md).

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- TESTING -->
## Pruebas y despliegue

```sh
pytest tests/ -q
```

```sh
docker build -t ecg-cinc2020 .
docker run --rm -p 5002:5002 -v ./saved:/app/saved:ro ecg-cinc2020
```

Monte su carpeta local `saved/` con el checkpoint entrenado.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- DIFFERENCES -->
## Diferencias frente a Hannun et al.

1. Se reemplaza el dataset privado Zio Patch/iRhythm por CINC2020.
2. Se usan ECG de 12 derivaciones.
3. Las etiquetas se definen con SNOMED-CT agrupado en 12 clases.
4. El problema se conserva multilabel; no se fuerza una clase única.
5. La evaluación usa métricas por clase y matrices 2×2 independientes.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Esquema v2 con cobertura total de las 27 clases puntuadas
- [x] Preprocesamiento en unidades físicas (mV + pasa-banda)
- [x] Precarga de HDF5 a RAM para entrenamiento y evaluación
- [x] Webapp con informe PDF profesional y secciones ES/EN
- [x] Experimentos de comparación de evaluaciones y robustez
- [x] Ensemble promedio de checkpoints + inferencia ensemble en la webapp
- [x] Umbrales F-beta y selección de checkpoint por F1-macro
- [x] Calibración de probabilidades (temperature scaling por clase + reporte ECE/Brier)
- [x] Curvas ROC/PR por clase en CSV + gráficas interactivas Plotly en la webapp
- [x] Análisis de errores por subconjunto de origen (sesgo por hospital)
- [x] Exportación a ONNX verificada numéricamente contra PyTorch
- [x] Historial de análisis por sesión en la webapp
- [x] Suite de tests automatizados + Dockerfile CPU
- [ ] Entrenamiento distribuido multi-GPU
- [ ] Búsqueda de hiperparámetros con Optuna
- [ ] Tracking de experimentos con MLflow
- [ ] Informe comparativo multi-registro (lotes de ECG)
- [x] Aumentación de datos específica para ECG
- [ ] Evaluar migración de la webapp a Flet (quizás; habilitaría empaquetado móvil)

Ver los [issues abiertos](https://github.com/PatVela/Proyectos-Universidad/issues) para la lista completa de propuestas y problemas conocidos.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- LICENSE -->
## Licencia

Distribuido bajo licencia GPL-3.0. Ver `LICENSE` para más información.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Agradecimientos

* Hannun AY et al. Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. *Nature Medicine*. 2019;25(1):65-69. [DOI 10.1038/s41591-018-0268-3](https://doi.org/10.1038/s41591-018-0268-3)
* PhysioNet/Computing in Cardiology Challenge 2020: Classification of 12-lead ECGs.
* Universidad Nacional de San Agustín de Arequipa — Facultad de Ingeniería de Producción y Servicios.
* Plantilla [Best-README-Template](https://github.com/othneildrew/Best-README-Template) e insignias de [shields.io](https://shields.io).

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/PatVela/Proyectos-Universidad.svg?style=for-the-badge
[contributors-url]: https://github.com/PatVela/Proyectos-Universidad/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/PatVela/Proyectos-Universidad.svg?style=for-the-badge
[forks-url]: https://github.com/PatVela/Proyectos-Universidad/network/members
[stars-shield]: https://img.shields.io/github/stars/PatVela/Proyectos-Universidad.svg?style=for-the-badge
[stars-url]: https://github.com/PatVela/Proyectos-Universidad/stargazers
[issues-shield]: https://img.shields.io/github/issues/PatVela/Proyectos-Universidad.svg?style=for-the-badge
[issues-url]: https://github.com/PatVela/Proyectos-Universidad/issues
[license-shield]: https://img.shields.io/github/license/PatVela/Proyectos-Universidad.svg?style=for-the-badge
[license-url]: https://github.com/PatVela/Proyectos-Universidad/blob/Paper_Replica_V3/LICENSE
[python-shield]: https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/pytorch-2.8-ee4c2c?style=for-the-badge&logo=pytorch
[pytorch-url]: https://pytorch.org/
[product-screenshot]: images/informe-ejemplo.png
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[PyTorch-badge]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[Flask-badge]: https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/
[NumPy-badge]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[pandas-badge]: https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[pandas-url]: https://pandas.pydata.org/
[sklearn-badge]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[sklearn-url]: https://scikit-learn.org/
[Matplotlib-badge]: https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white
[Matplotlib-url]: https://matplotlib.org/
[WFDB-badge]: https://img.shields.io/badge/WFDB-4B8BBE?style=for-the-badge
[WFDB-url]: https://archive.physionet.org/physiobank/database/wfdbcal/
[h5py-badge]: https://img.shields.io/badge/h5py-00A6ED?style=for-the-badge
[h5py-url]: https://www.h5py.org/
[Plotly-badge]: https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white
[Plotly-url]: https://plotly.com/
