# Tareas: webapp sin dependencia de localhost

Plan y justificación: `tasks/plan.md`. Verificación base de cada tarea: `pytest` en verde.

## Fase 1: núcleo desplegable (sirve a cualquier host)

- [ ] **T1. Bundle de modelos de inferencia** (M)
  - Script `tools/export_inference_bundle.py` que lee los dos checkpoints y escribe `models/*.pt` (solo `model_state_dict`, `class_names`, `label_schema`, `config`) + copia de umbrales/temperaturas del `ensemble-f1` a `deploy/`.
  - Criterios: cada modelo ≤ 20 MB; carga con `weights_only=True`; las probabilidades coinciden con las del checkpoint original (|dif| < 1e-6); `models/` entra en git (no está ignorado).
  - Verificación: test de paridad `tests/test_inference_bundle.py`; ejecutar el script y listar tamaños.
  - Archivos: `tools/export_inference_bundle.py`, `tests/test_inference_bundle.py`, `.gitignore`, `models/`, `deploy/`.
  - Depende de: nada.

- [ ] **T2. Configuración solo por entorno + manifiesto** (S)
  - `deploy/ensemble.env` (o `deploy/README.md`) con las variables del ensemble final apuntando a `models/` y `deploy/`; `webapp.wsgi` arranca con solo esas variables.
  - Criterios: `waitress-serve webapp.wsgi:app` en un clon limpio (sin `saved/` ni `runs/`) responde `/health` con `ensemble=true`, umbrales y temperaturas encontrados.
  - Verificación: `tools/smoke_deployment.py URL` (nuevo; reutiliza la prueba de extremo a extremo hecha a mano) pasa contra `http://127.0.0.1:PORT`.
  - Archivos: `deploy/`, `webapp/wsgi.py`, `tools/smoke_deployment.py`, `tests/test_smoke_script.py`.
  - Depende de: T1.

- [ ] **T3. Arranque de producción** (S)
  - Script/`Procfile` con `python -m waitress --port=$PORT webapp.wsgi:app`; `ECG_HOST`/`PORT` por entorno; la app deja de decir "development server".
  - Criterios: arranca con `PORT` arbitrario; `--help` documentado; un test comprueba que el comando de inicio importa `webapp.wsgi:app`.
  - Archivos: `deploy/start.sh`, `webapp/README.md`, `tests/test_deploy_entrypoint.py`.
  - Depende de: T2.

- [ ] **T4. Modelos en caché y concurrencia acotada** (M)
  - Cargar los dos modelos una vez (caché con candado) y limitar peticiones simultáneas de inferencia (p. ej. 2) para acotar la RAM.
  - Criterios: la 2.ª predicción no recarga checkpoints (test que cuenta llamadas a `load_model`); 5 peticiones concurrentes terminan sin error y RAM máxima < 1,5 GB; latencia media ≤ la actual (2,9 s CPU).
  - Archivos: `webapp/prediction.py`, `webapp/app.py`, `tests/test_model_cache.py`.
  - Depende de: T1.

- [ ] **T5. Límites y privacidad para uso público** (M)
  - `MAX_CONTENT_LENGTH` a 5 MB; límite de peticiones por IP; borrar los archivos subidos justo después de predecir; resultados (PDF/PNG/CSV) con vida de 1 h y limpieza periódica; aviso visible "uso académico, no subir datos de pacientes".
  - Criterios: archivo > 5 MB → 413; la 11.ª petición del minuto → 429; tras `/predict` no queda nada en `uploads/`; artefactos > 1 h se eliminan; el aviso aparece en la página.
  - Archivos: `webapp/app.py`, `webapp/prediction.py`, `webapp/templates/index.html`, `tests/test_public_limits.py`.
  - Depende de: T2.

- [ ] **T6. Ejemplos incluidos ("Probar con un ejemplo")** (M)
  - 4–6 registros públicos del conjunto de test (cubriendo varias clases) en `webapp/examples/`, endpoint `/examples` y botón en la interfaz que los carga.
  - Criterios: cada ejemplo da la predicción esperada (igual a `predictions_test.csv` dentro de 1e-3); un revisor puede probar la app sin tener un ECG propio; tamaño total < 2 MB.
  - Archivos: `webapp/examples/`, `webapp/app.py`, `webapp/static/app.js`, `webapp/templates/index.html`, `tests/test_examples.py`.
  - Depende de: T1.

### Punto de control 1: la app funciona sin las carpetas locales
- [ ] Clon limpio del repositorio (solo archivos versionados) + `models/` + `deploy/` arranca con T3 y pasa `tools/smoke_deployment.py`
- [ ] Suite completa en verde, sin avisos nuevos
- [ ] RAM máxima y latencia dentro de lo medido en el plan
- [ ] **Revisión con el autor antes de elegir host**

## Fase 2: publicar

- [ ] **T7. Desplegar en el host elegido** (M) — *depende de las decisiones 1–3*
  - Opción recomendada si se permite contenedor: Hugging Face Spaces (Docker, CPU) o equivalente con ≥ 2 GB; ajustar el contenedor (usuario no root, `waitress`, torch CPU, `models/` por Git LFS si el host lo exige).
  - Criterios: URL pública responde `/health`; variables del manifiesto aplicadas; límites de T5 activos; sin secretos en el repositorio.
  - Verificación: `tools/smoke_deployment.py https://<url>`.
  - Archivos: `Dockerfile` (solo si el host lo requiere), `deploy/`, `README.md`.
  - Depende de: Punto de control 1.

- [ ] **T8. Prueba remota y medición** (S)
  - Ejecutar el humo contra la URL pública; medir arranque en frío, latencia y RAM reales; documentarlas.
  - Criterios: 100 % de comprobaciones OK en remoto; arranque en frío y latencia anotados en `deploy/README.md`.
  - Depende de: T7.

### Punto de control 2: demo pública estable
- [ ] Humo remoto OK y medidas documentadas
- [ ] Un tercero (o una ventana de incógnito) completa el flujo con un ejemplo de T6
- [ ] **Revisión con el autor**

## Fase 3: permanencia para el artículo

- [ ] **T9. Modo "solo resultados" estático (respaldo permanente)** (M)
  - Exportar a archivos estáticos las vistas de métricas, curvas ROC/PR, experimentos y predicciones precalculadas de los ejemplos; publicar en GitHub Pages; la página indica que la clasificación en vivo está en la URL de T7.
  - Criterios: la página abre sin servidor, muestra las mismas cifras que `docs/resultados_finales.md` (test automático compara), sin llamadas a rutas inexistentes.
  - Archivos: `tools/export_static_site.py`, `webapp/static/app.js` (modo estático), `tests/test_static_export.py`.
  - Depende de: T1, T6.

- [ ] **T10. Versión citable** (S)
  - Etiquetar `v1.0`, archivar en Zenodo (DOI) y añadir una sección "Demo" al `README.md` con ambas URL y el DOI.
  - Criterios: el DOI resuelve; el README enlaza demo viva, sitio estático y snapshot.
  - Depende de: T8, T9.

### Punto de control final
- [ ] Todos los criterios de aceptación cumplidos
- [ ] Documentación (`README.md`, `deploy/README.md`) coherente con lo desplegado
- [ ] Listo para revisión

## Opcional (solo si lo exige la decisión 1 o el host no alcanza)
- [ ] **O1. Backend con ONNX Runtime** (L → dividir): sustituir torch por `onnxruntime` en la ruta de predicción; exige paridad numérica con torch y reduce RAM/imagen.
- [ ] **O2. Instalador de escritorio** (L → dividir): PyInstaller/pywebview; un build por sistema operativo; no recomendado.
