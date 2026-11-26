# Reproducibilidad del sistema del artículo

> **Estado: propuesta sin publicar (R11, 19 de noviembre de 2026).** Este documento reúne lo que se comprobó y lo que no. En R11 se **ejecutó** la inferencia del sistema publicado y las pruebas del repositorio en un clon limpio (sección 3b); **no se volvió a entrenar**, de modo que sigue sin afirmarse que ejecutar este repositorio reproduzca las cifras del artículo: esa afirmación exigiría reentrenar y reevaluar (nivel 4 completo, abajo).

## 1. De qué sistema se habla

El sistema principal del artículo (P3) es el ensemble de dos ResNet-34 1D: la receta v1 (semilla 42) y la receta v2 (semilla 43), con calibración afín por clase y umbrales por clase fijados con F1 en validación. Se entrenó con el reparto por origen y por paciente (Georgia como prueba B), no con el reparto aleatorio por registro que sigue la guía de pasos 0–10 del [README](../README.md). Sus pesos están en `models/` y su calibración, umbrales y métricas de la prueba A en `deploy/` (ver [deploy/CONTEXT.md](../deploy/CONTEXT.md)).

## 2. Niveles de prueba (no son equivalentes)

| Nivel | Qué prueba | Qué no prueba |
|---|---|---|
| 0. Existencia | que el archivo está | nada sobre su contenido |
| 1. Coincidencia de hash | que el archivo es el mismo que se registró | cómo se produjo |
| 2. Coincidencia de métricas | que las cifras guardadas coinciden con otras guardadas, o con las recalculadas desde predicciones guardadas | que el modelo las produzca al ejecutarse |
| 3. Correspondencia de tensores | que los pesos publicados son idénticos a los del punto de control del entrenamiento | que el entrenamiento se pueda repetir |
| 4. Reproducción experimental | que reejecutar el procedimiento da las cifras | (es el único nivel que respalda «reproduce») |

## 3. Lo que se comprobó (solo lectura, sobre la copia local de los autores)

Las comprobaciones leen archivos y calculan SHA-256; la de tensores carga los pesos en CPU con `torch.load(weights_only=True)` y compara con `torch.equal`, sin ejecutar la red. No se descargó nada, no se entrenó, no se hizo ninguna inferencia nueva y no se ejecutó `pytest`.

| Comprobación | Nivel | Resultado |
|---|---|---|
| SHA-256 de `models/resnet34_v1.pt` y `models/resnet34_v2.pt` frente a `deploy/manifest.json` | 1 | coinciden los dos |
| Pesos del bundle frente a los `best.pt` de `saved/t19/t19_v1` (semilla 42) y `t19_v2` (semilla 43) | 3 | 271 de 271 tensores idénticos en cada archivo |
| Archivos de calibración, umbrales, curvas y métricas del bundle frente a los de `runs/t19/eval/` y `runs/t19/bundle/reports/` de los que se exportaron | 1 | 10 de 10 idénticos byte a byte; `evaluation_summary.json` idéntico salvo los campos de ruta |
| Ocho métricas globales de la prueba A de `deploy/results/ensemble-f1-platt/` frente a las de la evidencia del artículo (`t19_etapa1/metricas_globales.csv`, a seis decimales) | 2 | coinciden las 8 (compara números guardados; no se recalcularon desde los pesos ni desde las predicciones) |
| SHA-256 registrados de las predicciones por registro (224 entradas, 156 archivos distintos) frente a los archivos locales | 1 | coinciden todos |
| SHA-256 de checkpoints, configuraciones y calibraciones citados en los protocolos | 1 | coinciden los comprobables; un registro abreviado (`config_t19_v2_testb.json`) tiene el prefijo correcto y el sufijo mal transcrito. No se pudieron comprobar dos: un registro que creció después de anotar su hash (`protocolo_externo_t19_desviaciones.md`) y `data/externo/patient_groups.csv`, que no estaba |
| Hiperparámetros del artículo (18) frente a las configuraciones de entrenamiento usadas | 0/1 | coinciden los 18 |
| `examples/cinc2020/config.json`, `config_resnet_v2.json` y `config_regular_cnn.json` frente a las configuraciones de entrenamiento usadas | 0/1 | difieren solo en `train`, `dev`, `test` y `save_dir` |
| Los siete ECG de `webapp/examples/` | 0/1 | pertenecen a las pruebas A y B del sistema; el registro E00014 de la Fig. 6 del artículo está en la prueba B y no es uno de los siete |
| `python tools/check_docs_commands.py` | (herramienta del repositorio) | 133 comandos comprobados, 0 problemas |

Una **emulación estática** (análisis con `ast`, no una ejecución) de `tests/test_requirements.py` predice que la prueba de dependencias pasa con los archivos versionados en HEAD y falla en la copia local, solo porque esta tiene un archivo local no versionado que importa un paquete no declarado. Es una predicción, no un resultado de `pytest`.

## 3b. Lo que se ejecutó en R11 (clon limpio del commit citado, entorno aislado)

Clon limpio de `329da54` (árbol de trabajo sin cambios), Python 3.13.17, PyTorch 2.14.0 (CPU), dependencias de `requirements-dev.txt`, Windows 11, i5-10300H, 15.8 GB de RAM. Los registros de cada ejecución están en la carpeta `EVIDENCIAS_R11/06_software` del paquete de la revisión.

| Comprobación | Nivel | Resultado |
|---|---|---|
| `python tools/check_docs_commands.py` | (herramienta del repositorio) | 133 comandos, 0 problemas |
| `pytest tests/ -q`, un archivo por proceso, con el entorno virtual activado | (herramienta del repositorio) | 653 aprobadas, 0 falladas, 0 con error y 2 omitidas de 655 (las omitidas requieren Node.js, no instalado); 855 s |
| Descarga de Chapman-Shaoxing y Ningbo (45 152 registros) y comparación con el `SHA256SUMS.txt` oficial de PhysioNet | 1 | 90350 de 90350 archivos coinciden; el hash del propio `SHA256SUMS.txt` coincide con el que registraron los autores |
| Reejecución de la red del paquete publicado (v1, v2, calibración afín y ensemble) sobre registros de C | 4 (inferencia) | 1000 registros; diferencia máxima de probabilidad 2.298e-03; 0 de 12000 decisiones distintas |
| Reejecución de la red del paquete publicado (v1, v2, calibración afín y ensemble) sobre registros de D | 4 (inferencia) | 1000 registros; diferencia máxima de probabilidad 1.795e-03; 0 de 12000 decisiones distintas |

Nota sobre las pruebas: en una primera corrida, lanzada con el modo UTF-8 de Python y con un `python` del `PATH` sin las dependencias, 3 pruebas de `test_deploy_entrypoint.py` y `test_deploy_config.py` fallaron (el lanzador PowerShell busca `python` en el `PATH` y decodifica su salida); con el entorno virtual activado, como indica el README, pasan. Ese es el comportamiento documentado de los lanzadores, no un defecto del código, pero conviene saberlo si alguien ejecuta `pytest` sin activar el entorno.

Estas comprobaciones **reejecutan la inferencia** (nivel 4 para esos registros) y verifican los datos; **no** reentrenan ni recalculan las tablas del artículo.

## 3c. Lo que se comprobó en línea (R11B, solo lectura, 19 de noviembre de 2026)

- `main` apunta a `329da54` (`git ls-remote`) y no hay commits posteriores; un clon nuevo desde GitHub tiene el mismo árbol de archivos que el clon probado en 3b.
- La ejecución de GitHub Actions de ese commit (Ubuntu, Python 3.13.18, PyTorch 2.14.0 CPU) pasó `tools/check_docs_commands.py` (133 comandos, 0 problemas) y dio 652 pruebas aprobadas y 3 omitidas de 655. En 3b (Windows) fueron 653 aprobadas y 2 omitidas de 655: cambia qué omite cada plataforma (en Linux, las tres pruebas marcadas «solo Windows»; en 3b, las dos que necesitan Node.js).
- La auditoría de dependencias (`pip-audit`) de ese commit no encontró vulnerabilidades conocidas en las ejecuciones del 11 y del 16 de noviembre.
- **Etiqueta y commit.** La etiqueta `v3.1` (15 de octubre de 2026), que es la versión que nombra `CITATION.cff`, apunta a `bc4ed10`, un commit anterior cuyos pesos (`models/resnet34_v1.pt` y `models/resnet34_v2.pt`) son distintos de los de `329da54`: los SHA-256 de `deploy/manifest.json` no coinciden entre ambos. El artículo se refiere al **commit `329da54`**, no a la etiqueta `v3.1`; quien descargue `v3.1` no obtiene los pesos del artículo. Fijar una etiqueta sobre `329da54` (y, si se quiere, un número de versión nuevo) es decisión de quien publica.

## 4. Lo que no está disponible o no se pudo verificar

- Los datos crudos y los HDF5 no estaban en la copia revisada (`dataset2020/` vacío, sin HDF5): no se pudo reconstruir el reparto ni comprobar los hashes de los HDF5.
- `data/externo/patient_groups.csv` no estaba en la copia revisada; su SHA-256 está en el registro de los autores (valor de control en la sección 5).
- Los pesos de las otras variantes y semillas (CNN, enmascaradas, ablaciones, semillas 101–109) existen en la copia local de los autores, pero no están publicados en este repositorio; tampoco las configuraciones de las variantes enmascaradas ni de las ablaciones.
- Las predicciones por registro, los protocolos sellados, los manifiestos del reparto y los cuatro scripts que calculan las tablas y figuras del artículo (`analizar_t19.py`, `calcular_t19.py`, `hacer_figuras.py`, `enmascarada_vs_semillas.py`) están en una carpeta local de los autores que no se sube a GitHub.
- No se recalcularon las tablas ni las figuras del artículo desde el repositorio.
- Las versiones: el repositorio declara «Python 3.10 o superior» como mínimo, pero solo se fijan y se prueban Python 3.13 y PyTorch 2.14.0 (CUDA 13). En R11 se verificó con Python 3.13.17 y PyTorch 2.14.0 **en CPU**; no se verificó con otras versiones ni con CUDA.
- No se midió la sensibilidad al hardware ni al no determinismo de CUDA, de modo que no se sabe qué diferencia se obtendría al repetir el entrenamiento.

## 5. Procedimiento registrado por los autores (no ejecutado en esta revisión)

Los comandos y los valores de control salen de los registros de protocolo de los autores. **No se ejecutaron en la revisión**; se publican para que quien reproduzca pueda comparar.

Datos de 2020 (públicos, no se redistribuyen):

```powershell
bash descargar_datos_2020.sh
```

Identificadores de paciente de PTB-XL y PTB (necesita `ptbxl_database.csv` y `RECORDS` en `data/externo/`):

```powershell
python tools/make_patient_groups.py --ptbxl-database data/externo/ptbxl_database.csv --ptb-records data/externo/RECORDS
```

HDF5 con el reparto por origen y por paciente (Georgia como prueba B), con los parámetros registrados:

```powershell
python examples/cinc2020/build_datasets.py --data-dir dataset2020 --output-dir data/cinc2020_12_origen --holdout-origin georgia --groups-csv data/externo/patient_groups.csv --workers 8 --norm-mode physical --train-windows-max 4 --val-frac 0.15 --test-frac 0.15 --random-state 42
```

Valores de control registrados: tamaños de partición train 22 895, val 4 913, test 4 949 y test_b 10 344 registros (26 554 ventanas de entrenamiento); 0 pacientes en más de una partición; Georgia solo en `test_b`; SHA-256 de los grupos de paciente `93380f479662b3417ff5c502f0a639109809ff98f628735aac00e0b59c6daa46`; SHA-256 de `split_assignments.csv` `e43579d0fcf3024bcae97a5cf811bd6f41e7f8b25121e99809289a583e8f9d86`.

Configuraciones: se copian `examples/cinc2020/config.json` (v1) y `examples/cinc2020/config_resnet_v2.json` (v2) a `runs/t19/config/config_t19_v1.json` y `config_t19_v2.json`, y se cambian solo `train`, `dev`, `test` (a `data/cinc2020_12_origen/train.h5`, `val.h5` y `test.h5`) y `save_dir` (a `saved/t19`). Las semillas son 42 (v1) y 43 (v2), y `aug_seed` 123.

Entrenamiento (v1 y luego v2):

```powershell
python -m ecg.train runs/t19/config/config_t19_v1.json -e t19_v1
```

```powershell
python -m ecg.train runs/t19/config/config_t19_v2.json -e t19_v2
```

Valores de control registrados (hardware: una GPU de 4 GB): v1 paró por `val_loss` a las 34 épocas con mejor época 24 y mejor `val_loss` 0.4719; v2 paró por `val_f1_macro` a las 102 épocas con mejor época 87 y mejor `val_f1_macro` 0.6887.

Calibración afín por clase en validación, evaluación en la prueba A y ensemble (ejemplo para v1; `<best.pt>` es el punto de control de la corrida):

```powershell
python examples/cinc2020/calibrate.py --method platt --checkpoint <best.pt de v1> --val-h5 data/cinc2020_12_origen/val.h5 --output-dir runs/t19/eval/p1
```

```powershell
python examples/cinc2020/evaluate.py runs/t19/config/config_t19_v1.json <best.pt de v1> --output-dir runs/t19/eval/p1_A --platt runs/t19/eval/p1/platt_validation.csv --threshold-beta 1.0
```

```powershell
python examples/cinc2020/ensemble_evaluate.py runs/t19/config/config_t19_v1.json <best.pt de v1> <best.pt de v2> --output-dir runs/t19/eval/p3_A --alpha 0.5 --threshold-beta 1.0 --platt-a runs/t19/eval/p1/platt_validation.csv --platt-b runs/t19/eval/p2/platt_validation.csv --recalibrate-ensemble
```

La secuencia completa (con la prueba B y los demás sistemas) está en un script local de los autores (no publicado), que se detiene si una carpeta de salida ya existe para respetar «una sola evaluación por sistema».

Valores de control registrados para P3 en la prueba A (n = 4 949): F1 macro 0.7522 y AUROC macro 0.955. Los autores no han fijado qué diferencia respecto de estos valores se consideraría una reproducción; conviene decidirlo **antes** de ejecutar.

## 6. Qué falta para poder afirmar que el repositorio reproduce el artículo

1. ~~Clonar el commit vigente en un directorio limpio, instalar y ejecutar `pytest` y `check_docs_commands.py`~~ **Hecho en R11** (sección 3b): 653 pruebas aprobadas de 655, 2 omitidas; y comprobado en línea contra el CI de GitHub en R11B (sección 3c).
2. Descargar los datos, ejecutar la sección 5 y comparar con los valores de control y con las tablas del artículo, con una tolerancia fijada de antemano.
3. Decidir si se publican las predicciones, los protocolos con hash, los manifiestos y los cuatro scripts, por ejemplo en un depósito con DOI, y enlazarlos desde aquí.
