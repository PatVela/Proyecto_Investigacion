# Informe técnico — CINC2020-12 tipo Hannun

## Resumen

Se adaptó el enfoque de CNN residual profunda de Hannun et al. a PhysioNet/CinC Challenge 2020. El dataset original de Zio Patch/iRhythm no es público, por lo que se usa CINC2020 con ECG de 12 derivaciones y diagnósticos SNOMED-CT.

La implementación sigue la estructura canónica solicitada:

```text
ecg/        paquete principal
examples/   scripts de experimento
webapp/     Flask + inferencia + reportes
docs/       documentación experimental
```

## Etiquetas

El esquema v2 (`cinc2020_12_grouped_snomed_v2`) tiene 12 clases multilabel, está definido en `ecg/schema.py` (re-exportado por `ecg/load.py`) y cubre las 27 clases puntuadas oficiales del Challenge 2020.

1. `NSR`: 426783006, 427393009.
2. `AxisDev`: 39732003, 445118002, 47665007, 445211001, 251200008.
3. `MI`: 164865005, 57054005, 164867002, 54329005, 164861001, 413444003, 413844008, 426434006, 425419005, 425623009, 164917005.
4. `TAb`: 164934002, 59931005, 428750005, 429622005, 164931005, 164930006, 55930002, 704997005, 111975006, 77867006, 164937009, 251259000, 428417006.
5. `AF`: 164889003, 164890007, 195080001, 282825002, 426749004, 314208002.
6. `LVH`: 164873001, 370365005, 446813000, 67741000119109, 253352002, 195126007, 89792004, 266249003, 253339007, 446358003.
7. `VEctopy`: 427172004, 17338001, 164884008, 11157007, 251180001, 251182009, 75532003, 81898007, 164895002, 425856008, 164896001, 111288001, 49260003, 13640000.
8. `AVBlock`: 270492004, 195042002, 233917008, 27885002, 54016002, 204384007, 164947007, 65778007.
9. `STach`: 427084000.
10. `BBB`: 59118001, 713427006, 713426002, 164909002, 251120003, 6374002, 698252002, 82226007, 251146004, 164951009.
11. `SB`: 426177001, 426627000, 60423000, 74615001.
12. `AEctopy_Junctional`: 284470004, 63593006, 713422000, 426664006, 29320008, 426995002, 251164006, 426648003, 195101003, 251268003, 251170000, 251168009, 251173003, 426761007, 67198005, 10370003, 251266004.

## Exclusiones

Se excluyen WPW/preexcitación (muy infrecuente y morfológicamente singular), diagnósticos clínicos no-ECG (HF/HVD/CHD/TIA) y ruido (sin SNOMED equivalente). TSV/TV se agrupan dentro de `AEctopy_Junctional`/`VEctopy` respectivamente para no etiquetar taquiarritmias como sanas.

## Preprocesamiento

Implementación: `ecg/signal.py` + `ecg/datasets.py` (re-exportados por `ecg/load.py`) y wrapper `examples/cinc2020/build_datasets.py`.

Pasos:

1. leer `.hea` (incluye ganancia ADC y baseline por derivación);
2. extraer `# Dx`;
3. leer `.mat`;
4. exigir 12 derivaciones;
5. reordenar derivaciones estándar;
6. convertir a mV con ganancia/baseline;
7. remuestrear a 500 Hz;
8. pasa-banda 0.5–50 Hz + clip ±5 mV;
9. recorte centrado / multi-ventana o padding hasta 5000 muestras;
10. guardar HDF5 train/val/test.

El split usa `MultilabelStratifiedShuffleSplit`; no hay fallback por fuente.

## Modelo

Implementación: `ecg/network.py`.

- `ECGResNet34`: ResNet 1D tipo Hannun, 16 bloques residuales, salida de 12 logits.
- `ECGRegularCNN`: CNN convencional sin residuales (línea base con la misma receta que v1).

## Entrenamiento

Implementación: `ecg/train.py`.

- `BCEWithLogitsLoss`.
- `pos_weight` calculado solo desde train.
- Adam.
- ReduceLROnPlateau.
- Early stopping.
- AMP opcional.
- Logs reproducibles en `history.csv` y `training_summary.json`.

## Evaluación

Implementación: `examples/cinc2020/evaluate.py`.

- thresholds por clase optimizados en validation;
- AUROC por clase;
- AUPRC por clase;
- sensibilidad;
- especificidad;
- F1;
- matrices de confusión 2x2.

## Diferencias frente al paper original

1. Dataset público CINC2020 en lugar de dataset privado iRhythm.
2. ECG de 12 derivaciones.
3. Etiquetas SNOMED agrupadas en 12 clases.
4. Formulación multilabel real.
5. Métricas por clase en lugar de matriz multiclase única.
