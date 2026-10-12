# Fix: predicción incorrecta en CINC2020 — diagnóstico y correcciones

Fecha: 2026-09-10. Casos reportados: `E00001` (Dx `426783006`=NSR) y `E00014`
(Dx `428750005,426177001,111975006` = NSSTTA + SB + LQT), ambos de Georgia.

## 1. Causas raíz encontradas (con evidencia)

### C1. Esquema de etiquetas v1 incompleto → 21% de registros all-zero (CAUSA PRINCIPAL)

El esquema v1 mapeaba ~40 códigos SNOMED a 12 clases. Medido sobre una muestra
aleatoria de 1438 headers (los 6 subconjuntos):

- **21.1% de registros quedan all-zero** (ninguna de las 12 clases), incluyendo
  ECG claramente anormales: LBBB (`164909002`), STD (`429622005`), STE
  (`164931005`), NSSTTA (`428750005`), OldMI (`164867002`), isquemias, LQT, etc.
- **13 de las 27 clases puntuadas oficiales quedan fuera**: LBBB, LAnFB, LQRSV,
  NSIVCB, PR, LPR, LQT, QAb, RAD, SA, IRBBB, TInv, Brady. El modelo no puede
  predecirlas y las ve como negativas/all-zero durante el entrenamiento.
- Clases capturadas parcialmente: `TAb` solo incluía `164934002` pero el
  dataset trae NSSTTA (155), STD (67), STIAb (66), TInv (37), STE (23) en la
  muestra → el modelo aprende TAb=0 para la mayoría de las anomalías ST-T
  reales. Igual para `MI` (falta OldMI/isquemias/QAb), `RBBB` (falta IRBBB),
  `SB` (falta Brady genérica), `AVBlock` (falta LPR).

Efecto en los casos reportados:

- `E00014` tiene NSSTTA + SB + LQT. En v1 solo mapea `SB`; NSSTTA y LQT se
  pierden (el modelo fue entrenado para predecir TAb=0 ante ST-T anómalos).
  En v2 mapea `SB + TAb`.

### C2. Normalización por derivación destruye voltaje y eje

El z-score **por derivación independiente** elimina la amplitud absoluta
(criterios de voltaje de LVH, bajo voltaje LQRSV, elevación ST) y las
amplitudes relativas entre derivaciones (eje eléctrico LAD/RAD). La ganancia
ADC del header se ignoraba (verificado: uniforme 1000/mV, baseline 0 en los
6 subconjuntos, por lo que la conversión a mV es segura y exacta).

### C3. Recorte centrado descarta el 42% de la información temporal

El 42% de los registros dura más de 10 s (CPSC hasta 60 s, PTB hasta 120 s,
INCART 30 min). El pipeline tomaba **solo los 10 s centrales** tanto en
entrenamiento como en inferencia: eventos focales (ectopia, episodios) fuera
del centro se pierden, pero la etiqueta del registro completo se mantiene
→ ruido de etiquetas + eventos perdidos en inferencia.

### C4. Riesgos silenciosos en inferencia (webapp/CLI)

- Auto-selección de checkpoint por menor `val_loss` global: un checkpoint de
  smoke-test sintético (pérdida artificialmente baja) podía ser elegido en
  lugar del modelo CINC2020 real → predicciones basura sin aviso.
- `load_model` solo validaba el **número** de clases, no los **nombres**: un
  checkpoint con otro orden/agrupación se cargaba sin advertencia.
- Thresholds podían cargarse de otro experimento (búsqueda por fecha en
  cualquier `thresholds_validation.csv` generado por `evaluate.py`) sin validar cobertura de clases.
- Re-subir el CSV convertido generaba **doble normalización** (en v2 físico
  sería catastrófico: dividir por 1000 dos veces).
- Unidades de CSV de usuario ambiguas (mV vs ADC) sin heurística.

## 2. Correcciones aplicadas

| # | Cambio | Archivos | ¿Requiere reentrenar? |
|---|--------|----------|----------------------|
| 1 | Esquema **v2**: 12 grupos expandidos que cubren las 27 clases puntuadas; all-zero 21%→0.6% | `ecg/schema.py` (antes `ecg/load.py` monolítico), README, docs | **Sí** (HDF5 + train) |
| 2 | Normalización **física**: mV vía ganancia/baseline + pasa-banda 0.5–50 Hz + clip ±5 mV (modos `global_zscore`/`per_lead_zscore` disponibles) | `ecg/load.py` | **Sí** |
| 3 | **Multi-ventana**: train hasta 4 ventanas/registro largo; inferencia deslizante con agregación max | `ecg/load.py`, `ecg/predict.py`, `webapp/prediction.py` | Parcial (train sí; inferencia mejora también checkpoints v1) |
| 4 | Compatibilidad v1↔v2: preprocesamiento según `label_schema` del checkpoint, traducción LAD/RBBB, CSV con marcador `# Preprocessed`, heurística de unidades | `ecg/load.py`, `ecg/predict.py`, `webapp/*` | No |
| 5 | Robustez: `best.pt` reales primero (sintéticos al final), validación de nombres de clase, chequeo esquema HDF5↔checkpoint, thresholds legacy | `ecg/util.py`, `ecg/predict.py`, `ecg/train.py`, `webapp/app.py` | No |
| 6 | **Métrica oficial** estilo Challenge (27 códigos + `weights.csv`) | `examples/cinc2020/challenge_score.py`, `official/` | No |
| 7 | **Diagnóstico integral** unificado | `examples/cinc2020/diagnose_prediction.py` | No |
| 8 | `torch` fuera de `requirements.txt`: se instala aparte con CUDA vía `requirements-cuda.txt` (ver README, Paso 0) | `requirements.txt`, `requirements-cuda.txt` | No |

Nota (post-depuración): `ecg/load.py` es hoy una fachada que re-exporta `ecg/schema.py` (esquema), `ecg/signal.py` (lectura/preprocesamiento) y `ecg/datasets.py` (HDF5/CLI).

## 3. Migración (pasos para reentrenar en v2)

```powershell
git pull
pip install -r requirements.txt   # torch se instala aparte (ver README, Paso 0)

# 0) Checkpoints (una vez por terminal)
$resnet = Get-ChildItem saved/cinc2020/cinc2020_resnet/*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName

# 1) Reconstruir HDF5 con esquema v2 (multi-ventana en train)
python examples/cinc2020/build_datasets.py --data-dir dataset2020 --output-dir data/cinc2020_12 --workers 8 --norm-mode physical --train-windows-max 4

# 2) Reentrenar (ResNet)
python -m ecg.train examples/cinc2020/config.json -e cinc2020_resnet

# 3) Calibrar + evaluar (thresholds en val, métricas en test)
python examples/cinc2020/calibrate.py --checkpoint $resnet --val-h5 data/cinc2020_12/val.h5 --output-dir eval-resnet
python examples/cinc2020/evaluate.py examples/cinc2020/config.json $resnet --output-dir eval-resnet --temperatures eval-resnet/temperatures_validation.csv --threshold-beta 0.5

# 4) Métrica oficial estilo Challenge (27 códigos)
python examples/cinc2020/challenge_score.py --checkpoint $resnet --test-h5 data/cinc2020_12/test.h5 --thresholds eval-resnet/thresholds_validation.csv --output-dir metrica-challenge

# 5) Verificar casos reportados
$hea14 = Get-ChildItem dataset2020 -Recurse -Filter E00014.hea | Select-Object -First 1 -ExpandProperty FullName
$mat14 = [System.IO.Path]::ChangeExtension($hea14, '.mat')
python examples/cinc2020/diagnose_prediction.py --checkpoint $resnet --config examples/cinc2020/config.json --thresholds eval-resnet/thresholds_validation.csv --record E00014 --hea $hea14 --mat $mat14

# 6) Webapp (usa automáticamente esquema/thresholds del checkpoint)
python webapp/app.py --saved saved
```

## 4. Mejoras sin reentrenar (checkpoint v1 actual)

1. Actualizar el código y reiniciar la webapp: la inferencia usará
   automáticamente `per_lead_zscore` para checkpoints v1 (sin mezcla de
   esquemas) + ventanas deslizantes en registros largos.
2. Fijar checkpoint explícito si hay varios en `saved/`:
   `python webapp/app.py --model $resnet --thresholds eval-resnet/thresholds_validation.csv`
   y verificar en `/health` el modelo y thresholds realmente cargados.
3. Correr `diagnose_prediction.py` sobre el checkpoint actual y revisar:
   clases del checkpoint vs esquema, thresholds, tasa all-negativo y
   calibración P-media vs prevalencia por clase.
4. Expectativa realista: con v1, `E00014` (LQT/NSSTTA) y cualquier registro
   con diagnósticos fuera de las ~14 clases cubiertas seguirá prediciéndose
   mal; eso solo se corrige con v2 + reentrenamiento.

## 5. Archivos a compartir para confirmar el diagnóstico

Si las métricas globales siguen bajas tras migrar, adjuntar:

- `eval-resnet/metrics_global.csv`, `metrics_per_class.csv`,
  `thresholds_validation.csv`, `metrica-challenge/challenge_metric.json`;
- `saved/.../training_summary.json` (o `history.csv`);
- salida de `diagnose_prediction.py` para `E00001` y `E00014`.
