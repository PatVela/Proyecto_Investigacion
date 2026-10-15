# Experimentos CINC2020-12

## Propósito

Adaptar el enfoque Hannun et al. a un dataset público de ECG de 12 derivaciones: PhysioNet/CinC Challenge 2020. El problema se mantiene multilabel con 12 clases agrupadas por SNOMED-CT.

Variables PowerShell usadas en todos los bloques (definir una vez por terminal):

```powershell
$resnet = Get-ChildItem saved/cinc2020/cinc2020_resnet/*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$resnet2 = Get-ChildItem saved/cinc2020/cinc2020_resnet_v2*/*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$mejor = $resnet2
$evalMejor = "runs/eval/eval-resnet-v2"
```

## Preparación

```powershell
bash descargar_datos_2020.sh
python examples/cinc2020/build_datasets.py --data-dir dataset2020 --output-dir data/cinc2020_12 --workers 8 --norm-mode physical --train-windows-max 4
```

En Windows, la primera línea corre en una terminal Git Bash.

## Experimento A — ResNet-34 tipo Hannun

```powershell
python -m ecg.train examples/cinc2020/config.json -e cinc2020_resnet
python examples/cinc2020/calibrate.py --checkpoint $resnet --val-h5 data/cinc2020_12/val.h5 --output-dir runs/eval/eval-resnet
python examples/cinc2020/evaluate.py examples/cinc2020/config.json $resnet --output-dir runs/eval/eval-resnet --temperatures runs/eval/eval-resnet/temperatures_validation.csv --threshold-beta 0.5
```

## Experimento B — ResNet-34 v2 (receta anti-sobreajuste)

```powershell
python -m ecg.train examples/cinc2020/config_resnet_v2.json -e cinc2020_resnet_v2
python examples/cinc2020/calibrate.py --checkpoint $resnet2 --val-h5 data/cinc2020_12/val.h5 --output-dir runs/eval/eval-resnet-v2
python examples/cinc2020/evaluate.py examples/cinc2020/config_resnet_v2.json $resnet2 --output-dir runs/eval/eval-resnet-v2 --temperatures runs/eval/eval-resnet-v2/temperatures_validation.csv --threshold-beta 0.5
```

## Comparación justa

Ambos modelos deben usar:

- **el mismo punto de operación** (`threshold_beta` y la misma calibración): con F0.5 en uno y F1 en otro la diferencia mezcla modelo y umbral (v2 − v1 es +3.5 pts con F0.5 y +1.2 con F1 en ambos); `compare_models.py` se detiene si no coinciden;
- mismo dataset HDF5;
- mismo split train/validation/test;
- mismo preprocesamiento;
- mismo número de clases;
- mismo batch size y optimizador; misma familia de pérdida (BCE, v2 añade label smoothing 0.05);
- mismas métricas de evaluación.

```powershell
python examples/cinc2020/compare_models.py --eval-a runs/eval/eval-resnet --eval-b runs/eval/eval-resnet-v2 --label-a "ResNet v1" --label-b "ResNet v2" --output runs/reports/model_comparison.csv   # mismo --threshold-beta en ambas (0.5)
python examples/cinc2020/evaluate.py examples/cinc2020/config.json $resnet --output-dir runs/eval/eval-resnet-f1 --temperatures runs/eval/eval-resnet/temperatures_validation.csv --threshold-beta 1.0
python examples/cinc2020/evaluate.py examples/cinc2020/config_resnet_v2.json $resnet2 --output-dir runs/eval/eval-resnet-v2-f1 --temperatures runs/eval/eval-resnet-v2/temperatures_validation.csv --threshold-beta 1.0
python examples/cinc2020/compare_models.py --eval-a runs/eval/eval-resnet-f1 --eval-b runs/eval/eval-resnet-v2-f1 --label-a "ResNet v1 (F1)" --label-b "ResNet v2 (F1)" --output runs/reports/model_comparison_f1.csv
python examples/cinc2020/bootstrap_ci.py --predictions runs/eval/eval-resnet-v2-f1/predictions_test.csv --predictions-baseline runs/eval/eval-resnet-f1/predictions_test.csv --output runs/reports/bootstrap_v2-v1_f1.json --n-bootstrap 1000
```

## Experimento C — Ensemble v1+v2 (modelo final)

```powershell
python examples/cinc2020/ensemble_evaluate.py examples/cinc2020/config.json $resnet $resnet2 --output-dir runs/eval/ensemble --alpha 0.5 --threshold-beta 0.5 --temperatures-a runs/eval/eval-resnet/temperatures_validation.csv --temperatures-b runs/eval/eval-resnet-v2/temperatures_validation.csv
python examples/cinc2020/compare_models.py --eval-a "$evalMejor" --eval-b runs/eval/ensemble --label-a "ResNet mejor" --label-b "Ensemble" --output runs/reports/comparacion_ensemble.csv
```

Promedio simple con calibración por modelo y umbrales F0.5 tuneados sobre el promedio (con `--platt-a/--platt-b --recalibrate-ensemble` la calibración es afín y el promedio se recalibra; ver `docs/resultados_finales.md`). El punto de operación F1 (recomendado para screening y para la demo) se obtiene con `--threshold-beta 1.0 --output-dir runs/eval/ensemble-f1`; no se usa `--nsr-exclusive-min-prob`.
Ver resultados en [resultados_finales.md](resultados_finales.md).

## Experimento D — Calibración afín del ensemble (modelo final de la demo)

```powershell
python examples/cinc2020/calibrate.py --method platt --checkpoint $resnet --val-h5 data/cinc2020_12/val.h5 --output-dir runs/eval/eval-resnet
python examples/cinc2020/calibrate.py --method platt --checkpoint $resnet2 --val-h5 data/cinc2020_12/val.h5 --output-dir runs/eval/eval-resnet-v2
python examples/cinc2020/ensemble_evaluate.py examples/cinc2020/config.json $resnet $resnet2 --output-dir runs/eval/ensemble-f1-platt --alpha 0.5 --threshold-beta 1.0 --platt-a runs/eval/eval-resnet/platt_validation.csv --platt-b runs/eval/eval-resnet-v2/platt_validation.csv --recalibrate-ensemble
python examples/cinc2020/compare_models.py --eval-a runs/eval/ensemble-f1 --eval-b runs/eval/ensemble-f1-platt --label-a "Ensemble F1 (temperatura)" --label-b "Ensemble F1 (afín)" --output runs/reports/comparacion_calibracion_f1.csv
```

Calibración `sigmoid(a·logit + b)` por clase, ajustada solo en validación sobre los logits crudos, y recalibración del promedio. Hipótesis y resultado: la temperatura no corrige el desplazamiento que introducen `pos_weight = neg/pos` y el label smoothing, la afín sí (ECE del ensemble en test 0.103 → 0.009) y no cambia la clasificación (F1 +0.0005, IC95 [−0.0035, +0.0044]). Cifras en [resultados_finales.md](resultados_finales.md).

## Métricas a reportar

- AUROC por clase;
- AUPRC por clase;
- sensibilidad;
- especificidad;
- precisión;
- F1 por clase;
- F1 macro/micro;
- matrices de confusión 2x2 por clase;
- ECE y Brier (con la calibración usada);
- F1 por origen, con y sin enmascarar las celdas sin anotar (`error_analysis.py --train-h5`);
- número de parámetros;
- tiempo total de entrenamiento.

## Robustez controlada

```powershell
python examples/cinc2020/robustness.py $mejor data/cinc2020_12/test.h5 --thresholds "$evalMejor/thresholds_validation.csv" --temperatures "$evalMejor/temperatures_validation.csv" --output runs/reports/robustness.csv
```

Perturbaciones incluidas: ruido gaussiano, baseline wander, escalado de amplitud y dropout de derivaciones.
