# Comparación exploratoria con el modelo publicado de Ribeiro et al.

**Estado: propuesta sin publicar (R11, 2026-11-19).** Scripts para ejecutar los pesos publicados de Ribeiro et al. (Nat. Commun. 11, 1760, 2020) sobre Chapman-Shaoxing y Ningbo y compararlos, registro a registro, con el sistema principal de este repositorio. **No es una evaluación confirmatoria**: se hizo después de conocer los resultados de C y D y no forma parte del protocolo sellado (ver `docs/COMPARACION_PUBLICADA.md`).

| Script | Qué hace |
|---|---|
| `ribeiro_pytorch.py` | Reimplementación en PyTorch de la arquitectura de su `model.py` y lector de sus pesos Keras HDF5 (no necesita TensorFlow) |
| `validar_port_ribeiro.py` | Comprueba la reimplementación contra las predicciones **publicadas** por los autores sobre sus 827 trazados |
| `escala_entrada_ribeiro.py` | Decide la unidad de entrada (mV) comparando amplitudes, sin etiquetas |
| `comparar_ribeiro_cd.py` | Ejecuta el modelo sobre los registros de C y D que evaluó el sistema principal y calcula AUROC y AUPRC por clase con IC pareados |
| `reproducir_predicciones_P3.py` | Reejecuta en CPU la red del paquete publicado (`models/`, `deploy/`) sobre los primeros N registros de C y D y compara sus probabilidades con las que guardó la evaluación (nivel 4 de `docs/REPRODUCIBILIDAD.md` para la inferencia) |

## Ejecución
Descargar los pesos y los datos de prueba de Zenodo (ver `docs/DATOS_Y_LICENCIAS.md`), descomprimirlos y validar la reimplementación:

```powershell
python examples/cinc2020/baseline_ribeiro/validar_port_ribeiro.py datos_ribeiro salida_validacion
```

Descargar Chapman-Shaoxing y Ningbo y ejecutar la comparación:

```powershell
python tools/descargar_challenge2021.py dataset2021 24
```

```powershell
python examples/cinc2020/baseline_ribeiro/comparar_ribeiro_cd.py datos_ribeiro dataset2021 runs/t19/eval salida_comparacion .
```

El orden de derivaciones es I, II, III, aVR, aVL, aVF, V1–V6. Los README de los autores dan dos órdenes distintos; solo ese reproduce sus predicciones publicadas.

## Límites
Solo la taquicardia sinusal (ST ↔ STach) tiene el mismo código SNOMED en ambos sistemas; AF y SB son clases del sistema principal con más códigos; 1dAVb, RBBB y LBBB son subconjuntos de AVBlock y BBB y se informan solo como descriptivas. Los dos sistemas se entrenaron con datos y definiciones de etiqueta distintos.
