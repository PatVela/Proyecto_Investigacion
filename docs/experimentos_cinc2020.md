# Experimentos CINC2020-12

## Propósito

Adaptar el enfoque Hannun et al. a un dataset público de ECG de 12 derivaciones: PhysioNet/CinC Challenge 2020. El problema se mantiene multilabel con 12 clases agrupadas por SNOMED-CT.

Variables PowerShell usadas en todos los bloques (definir una vez por terminal):

```powershell
$resnet = Get-ChildItem saved/cinc2020/cinc2020_resnet/*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$resnet2 = Get-ChildItem saved/cinc2020/cinc2020_resnet_v2*/*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$mejor = $resnet2
$evalMejor = "eval-resnet-v2"
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
python examples/cinc2020/calibrate.py --checkpoint $resnet --val-h5 data/cinc2020_12/val.h5 --output-dir eval-resnet
python examples/cinc2020/evaluate.py examples/cinc2020/config.json $resnet --output-dir eval-resnet --temperatures eval-resnet/temperatures_validation.csv --threshold-beta 0.5
```

## Experimento B — ResNet-34 v2 (receta anti-sobreajuste)

```powershell
python -m ecg.train examples/cinc2020/config_resnet_v2.json -e cinc2020_resnet_v2
python examples/cinc2020/calibrate.py --checkpoint $resnet2 --val-h5 data/cinc2020_12/val.h5 --output-dir eval-resnet-v2
python examples/cinc2020/evaluate.py examples/cinc2020/config_resnet_v2.json $resnet2 --output-dir eval-resnet-v2 --temperatures eval-resnet-v2/temperatures_validation.csv --threshold-beta 0.5
```

## Comparación justa

Ambos modelos deben usar:

- mismo dataset HDF5;
- mismo split train/validation/test;
- mismo preprocesamiento;
- mismo número de clases;
- mismo batch size y optimizador; misma familia de pérdida (BCE, v2 añade label smoothing 0.05);
- mismas métricas de evaluación.

```powershell
python examples/cinc2020/compare_models.py --eval-a eval-resnet --eval-b eval-resnet-v2 --label-a "ResNet v1" --label-b "ResNet v2" --output exp-files/model_comparison.csv
```

## Experimento C — Ensemble v1+v2 (modelo final)

```powershell
python examples/cinc2020/ensemble_evaluate.py examples/cinc2020/config.json $resnet $resnet2 --output-dir ensemble --alpha 0.5 --threshold-beta 0.5 --temperatures-a eval-resnet/temperatures_validation.csv --temperatures-b eval-resnet-v2/temperatures_validation.csv
python examples/cinc2020/compare_models.py --eval-a "$evalMejor" --eval-b ensemble --label-a "ResNet mejor" --label-b "Ensemble" --output comparacion_ensemble.csv
```

Promedio simple con calibración por modelo y umbrales F0.5 tuneados sobre el promedio.
Ver resultados en [resultados_finales.md](resultados_finales.md).

## Métricas a reportar

- AUROC por clase;
- AUPRC por clase;
- sensibilidad;
- especificidad;
- precisión;
- F1 por clase;
- F1 macro/micro;
- matrices de confusión 2x2 por clase;
- número de parámetros;
- tiempo total de entrenamiento.

## Robustez controlada

```powershell
python examples/cinc2020/robustness.py $mejor data/cinc2020_12/test.h5 --thresholds "$evalMejor/thresholds_validation.csv" --temperatures "$evalMejor/temperatures_validation.csv" --output exp-files/robustness.csv
```

Perturbaciones incluidas: ruido gaussiano, baseline wander, escalado de amplitud y dropout de derivaciones.
