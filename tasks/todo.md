# Tareas: webapp ejecutada en esta PC (Windows 11)

Plan y justificación: `tasks/plan.md`. Verificación base de cada tarea: `pytest` en verde.

## Fase 1: núcleo independiente de carpetas locales (hecha y fusionada en `main`)

- [x] **T1. Bundle de modelos de inferencia**: `tools/export_inference_bundle.py` → `models/*.pt` (solo pesos, ≤ 20 MB c/u, `weights_only=True`) y copia de umbrales, calibración (afín o temperaturas) y reportes a `deploy/`. Prueba de paridad: `tests/test_inference_bundle.py`.
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
  - **Prueba real con túnel (2026-10-12, `cloudflared` 2026.10.0):** página, análisis de un ejemplo, consola vacía y 7 peticiones en 200. Se encontró y corrigió un fallo: `ECG_TRUST_PROXY=1` no bastaba porque waitress 3.0.2 elimina `X-Forwarded-For` (todos los visitantes compartían el límite); ahora `-Tunnel` pasa `--trusted-proxy=127.0.0.1 --trusted-proxy-count=1 --trusted-proxy-headers=x-forwarded-for` (`webapp.wsgi:app` debe ser el último argumento). Limitación: descargas grandes lentas o atascadas por el túnel (ver `webapp/README.md`).

- [x] **T8. Guía de uso bajo demanda y energía** (S)
  - `webapp/README.md` (Uso bajo demanda): iniciar con `deploy\start_local.ps1`, detener con Ctrl+C, y los `powercfg` opcionales si se deja corriendo varias horas. No hay arranque automático (decisión del autor).
  - **No se cambió la configuración de energía de la PC**: lo hace el autor si lo necesita.

### Punto de control 2: lista para usar
- [x] Suite de pruebas en verde con los cambios de la Fase 2
- [ ] T9 hecha por el autor

- [ ] **T9. Prueba de uso real** (S, requiere al autor)
  - Iniciar `deploy\start_local.ps1`, correr `python tools/smoke_deployment.py http://127.0.0.1:5002 --expect-ensemble` → 19/19, abrir la interfaz en un navegador (teclado, tema oscuro, móvil) y analizar un ejemplo.
  - Solo si se quiere mostrarla fuera de la PC: `-Lan` desde un teléfono de la misma red, o `-Tunnel` desde un teléfono con datos móviles (ya probado desde esta PC: ver T7 y las limitaciones de velocidad en `webapp/README.md`).

## Endurecimiento posterior (hecho)

- [x] **H1. Hallazgos de seguridad de la auditoría** (S1–S6): regex cuadrática de `clean_path_text`, tope de tamaño del `.mat`, `/use_model` apagado por defecto, tope de cuerpo en waitress (`--max-request-body-size`), servidores de desarrollo en `127.0.0.1`, `escapeHtml` con comillas y tope de longitud del nombre del paciente. Verificado con `tests/test_hardening.py` y en vivo (10 MB a `/predict` → 413 de waitress; `/use_model` → 403).

## Opcional (solo si hace falta)
- [x] **O1. Reducir las descargas pesadas** (hecho): PNG 762 → 306 KB, PDF 952 → 406 KB y Plotly 4,6 → 1,3 MB (gzip), sin cambiar resultados. Verificado con `tests/test_payload_size.py` y en el navegador (gráficos de Plotly y trazado ECG). Opcional pendiente: un Plotly parcial (solo barras y líneas) lo bajaría a ≈ 1 MB sin comprimir, pero exige descargar otro paquete.
- [ ] **O2. Instantánea citable del repositorio** (DOI) si el autor quiere un respaldo de reproducibilidad; no es necesaria para la demo.
- [ ] **O3. Prueba con lector de pantalla y otros navegadores.**
