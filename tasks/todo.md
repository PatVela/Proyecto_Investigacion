# Tareas: webapp ejecutada en esta PC (Windows 11)

Plan y justificación: `tasks/plan.md`. Verificación base de cada tarea: `pytest` en verde.

## Fase 1: núcleo independiente de carpetas locales (hecha y fusionada en `main`)

- [x] **T1. Bundle de modelos de inferencia**: `tools/export_inference_bundle.py` → `models/*.pt` (solo pesos, ≤ 20 MB c/u, `weights_only=True`) y copia de umbrales, temperaturas y reportes a `deploy/`. Prueba de paridad: `tests/test_inference_bundle.py`.
- [x] **T2. Configuración solo por entorno + manifiesto**: `deploy/ensemble.env`, `webapp/wsgi.py`; `tools/smoke_deployment.py` (19 comprobaciones por HTTP).
- [x] **T3. Arranque con waitress** (`webapp.wsgi:app`), sin servidor de desarrollo.
- [x] **T4. Modelos en caché y concurrencia acotada**: `tests/test_model_cache.py`.
- [x] **T5. Límites y privacidad**: 5 MB, 20 solicitudes por minuto y por IP, subidas borradas, informes de 1 h: `tests/test_public_limits.py`.
- [x] **T6. Siete ejemplos incluidos** (CC BY 4.0): `tests/test_examples.py`.

### Punto de control 1
- [x] Un `git clone` real (sin `saved/`, `runs/` ni `data/`) arranca y pasa el humo 19/19.
- [x] Auditoría de interfaz fusionada (PRs #8–#10): teclado y lectores de pantalla, contraste AA en ambos temas, primera carga de 4,6 MB a 152 KB, tema/idioma sin parpadeo, secciones con URL propia. Sin probar aún con lector de pantalla real ni en otros navegadores.
- [x] Revisión con el autor: PRs fusionadas en `main` con la suite en verde.

## Fase 2: ejecutar en esta PC

- [x] **T7. Lanzador local y dispositivo de inferencia** (M)
  - `deploy/start_local.ps1`: waitress solo en `127.0.0.1`, CPU por defecto, `-Port`, `-Threads`, `-Lan`, `-Tunnel`, `-UseGpu`; comprueba Python, dependencias y puerto libre. `ECG_DEVICE=auto|cpu|cuda` (un valor inválido o `cuda` sin GPU impide arrancar).
  - Se eliminaron `Dockerfile`, `.dockerignore`, `Procfile`, `deploy/start.sh`, `webapp/run_https.sh` y `webapp/run_public.sh` (nube/contenedor; los dos últimos no funcionaban).
  - Verificado: `tests/test_serving_device.py` y `tests/test_deploy_entrypoint.py` (incluye arranque real del lanzador + humo 19/19 + `"device":"cpu"`); la IP de la red local no responde por defecto y sí con `-Lan`; `-Lan -Tunnel` y `-Tunnel` sin `cloudflared` se rechazan antes de arrancar nada.

- [x] **T8. Guía de uso bajo demanda y energía** (S)
  - `webapp/README.md` (Uso bajo demanda): iniciar con `deploy\start_local.ps1`, detener con Ctrl+C, y los `powercfg` opcionales si se deja corriendo varias horas. No hay arranque automático (decisión del autor).
  - **No se cambió la configuración de energía de la PC**: lo hace el autor si lo necesita.

### Punto de control 2: lista para usar
- [x] Suite de pruebas en verde con los cambios de la Fase 2
- [ ] T9 hecha por el autor

- [ ] **T9. Prueba de uso real** (S, requiere al autor)
  - Iniciar `deploy\start_local.ps1`, correr `python tools/smoke_deployment.py http://127.0.0.1:5002 --expect-ensemble` → 19/19, abrir la interfaz en un navegador (teclado, tema oscuro, móvil) y analizar un ejemplo.
  - Solo si se quiere mostrarla fuera de la PC: `-Lan` desde un teléfono de la misma red, o `winget install --id Cloudflare.cloudflared` y `-Tunnel` desde un teléfono con datos móviles, comprobando que dos visitantes distintos no comparten el límite.

## Opcional (solo si hace falta)
- [ ] **O1. Comprimir `plotly.min.js`** (4,3 MB sin comprimir) si se usa `-Tunnel` con muchos visitantes.
- [ ] **O2. Instantánea citable del repositorio** (DOI) si el autor quiere un respaldo de reproducibilidad; no es necesaria para la demo.
- [ ] **O3. Prueba con lector de pantalla y otros navegadores.**
