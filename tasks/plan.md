# Plan: webapp de resultados que no dependa de localhost

## Resumen
Hoy la webapp solo corre en la máquina del autor (`python webapp/app.py`, `127.0.0.1`) y necesita
carpetas locales que no están en git (`saved/`, `runs/`). Objetivo: que cualquier lector del
trabajo de investigación pueda abrirla con una URL estable (o, como mínimo, ver los resultados
sin instalar nada), con costo cero o muy bajo y sin exponer datos de pacientes.

## Qué se midió (hechos, no suposiciones)
| Dato | Valor | Implicación |
|---|---|---|
| Modelo completo (`best.pt`) | 57,9 MB c/u | incluye estado del optimizador; no hace falta para inferir |
| Modelo solo para inferencia | **19,4 MB** c/u (2 modelos ≈ 39 MB) | cabe en GitHub/HF sin problemas |
| Resultados que muestra la app (`ensemble-f1`, reportes, estáticos) | ≈ 3,1 MB + 4,6 MB | despreciable |
| Memoria máxima del proceso (ensemble, solo CPU) | **≈ 1,04 GB** | descarta hosts de 256–512 MB |
| Latencia `/predict` solo CPU | 2,4–4,2 s (media 2,9 s) | aceptable para una demo sin GPU |
| Torch instalado vs fijado | 2.14.0+cu130 instalado, `requirements.txt` fija 2.8.0 | hay que probar con la versión que se despliegue |

## Opciones consideradas
| Opción | Qué es | A favor | En contra | Veredicto |
|---|---|---|---|---|
| **A. Demo en vivo en un servidor** | misma app Flask en un host con ≥ 2 GB RAM | todas las funciones (subir ECG, PDF, curvas); URL estable | hay que cuidar abuso, privacidad y modelos | **Recomendada** |
| **B. Sitio estático de resultados** | métricas, curvas y ejemplos precalculados (GitHub Pages) | gratis, permanente, sin servidor ni riesgo | no clasifica ECG nuevos | **Complemento** (respaldo permanente) |
| C. App de escritorio | PyInstaller/pywebview con torch dentro | funciona sin internet | 0,6–1,5 GB por instalador, un build por sistema operativo, avisos de SmartScreen | No recomendada para un artículo |
| D. Inferencia en el navegador | ONNX Runtime Web | sin servidor | hay que reescribir filtro, remuestreo y ventanas en JS y mantener paridad numérica | Descartada |
| E. Backend ligero con ONNX Runtime | reemplaza torch por `onnxruntime` en el servidor | menos RAM/imagen | reescribir la ruta de predicción y validar paridad | Opcional, solo si el host elegido no alcanza |

**Recomendación:** A + B. A da la demo interactiva; B asegura que los resultados sigan visibles
aunque el servidor gratuito se duerma o desaparezca. Todo lo común (modelos empaquetados,
configuración por variables de entorno, límites) se hace una vez y sirve a cualquier host.

## Decisiones de arquitectura
- **Empaquetar solo pesos de inferencia** (`models/`), generados por un script reproducible desde
  los checkpoints; la app no depende de `saved/` ni `runs/` en producción. Se cargan con
  `weights_only=True` (ya soportado).
- **Configuración solo por variables de entorno** (ya existen: `ECG_MODEL`, `ECG_MODEL_B`,
  `ECG_ALPHA`, `ECG_THRESHOLDS`, `ECG_TEMPERATURES[_B]`, `ECG_EVAL_DIR`, `ECG_EXP_DIR`); un
  manifiesto en `deploy/` documenta los valores del ensemble final.
- **Servidor de producción** (`waitress`, ya en `requirements.txt`) y `PORT` del entorno; sin servidor de desarrollo.
- **Modelos en caché** en memoria: hoy `/predict` vuelve a cargar los dos checkpoints en cada petición.
- **Sin persistencia de ECG subidos**: borrar archivos tras predecir; resultados con vida corta.
- **Privacidad:** la demo es académica; avisar que no se suban datos de pacientes reales.

## Riesgos y mitigaciones
| Riesgo | Impacto | Mitigación |
|---|---|---|
| Host gratuito con poca RAM o que se duerme | Alto | elegir ≥ 2 GB; cold start documentado; respaldo estático (Tarea 9) |
| Abuso público (cargas grandes, ráfagas) | Alto | límite de 5 MB, tasa por IP, concurrencia acotada (Tarea 4) |
| Datos sensibles subidos por un usuario | Alto | no persistir, aviso visible, ejemplos incluidos para no necesitar datos propios |
| Distinta versión de torch en el servidor | Medio | humo con la versión fijada; test de paridad del bundle (Tarea 1) |
| URL que deja de existir con el tiempo | Medio | snapshot con DOI en Zenodo (Tarea 10) |
| El host exige contenedor y se había descartado el Dockerfile | Medio | decisión abierta 2 (abajo) |

## Orden y dependencias
```
T1 bundle de modelos ─┬─ T2 config/manifiesto ─┬─ T5 limites publicos ─┐
                      ├─ T3 servidor prod      ├─ T4 cache de modelos ─┤
                      └─ T6 ejemplos ──────────┘                       ├─ CHECKPOINT 1 → T7 deploy → T8 humo remoto → CHECKPOINT 2
T9 modo estatico (independiente, tras T1/T6) ───────────────────────────┘                  T10 Zenodo/README
```

## Decisiones tomadas (autor)
1. **Público:** revisores/jurado con una URL → la demo en vivo (opción A) es obligatoria; B es su respaldo.
2. **Contenedor:** se permite un Dockerfile (habilita Hugging Face Spaces o Cloud Run).
3. **Alcance inmediato:** solo la Fase 1 (T1–T6); nada se publica todavía.
4. Sin servidor/VM de la universidad por ahora.

## Preguntas que quedan para la Fase 2
Qué host concreto (RAM, inactividad, límites y términos vigentes) y qué URL se citará en el artículo.

## (Histórico) Preguntas abiertas antes de decidir
1. **¿Quién va a abrir la demo?** (jurado/revisores con URL pública, o solo presentación en vivo).
   Si es solo presentación, basta B + A local; si es público, A es obligatoria.
2. **¿Se permite un contenedor (Dockerfile) para desplegar?** Hugging Face Spaces y Cloud Run lo
   piden; Render/Railway pueden usar `requirements.txt` + comando de inicio pero necesitan ≥ 2 GB (plan de pago).
3. **¿Hay servidor o VM de la universidad disponible?** Sería el host más estable y sin límites de inactividad.
4. **Políticas del host a verificar antes de elegir:** RAM, CPU, tiempo de inactividad, tamaño máximo de
   archivos y términos de uso (los límites cambian; no se asumen aquí).

## Fuera de alcance
App de escritorio (C) e inferencia en el navegador (D), salvo que la decisión 1 lo exija.
