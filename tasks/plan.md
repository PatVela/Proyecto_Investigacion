# Plan: webapp de resultados ejecutada en esta PC (Windows 11)

## Resumen
La webapp es una demostración de los resultados del trabajo de investigación. **Decisión del autor (2026-10-12):
se ejecuta en su propia PC**, que permanece encendida; no se usa nube, servidor ni contenedor. El artículo solo
menciona que existe una webapp (sin URL), así que no hace falta una dirección fija ni pública. La app ya no depende de
carpetas locales sueltas: todo lo que necesita está versionado en `models/`, `deploy/` y `webapp/examples/`.

## Qué se midió (hechos, no suposiciones)
PC: Windows 11, Intel Core i5-11400H, 24 GB de RAM, NVIDIA RTX 3050 Laptop (4 GB). Servida con waitress y el bundle:

| Dato | Valor | Implicación |
|---|---|---|
| Modelo solo para inferencia | 19,4 MB c/u (2 modelos ≈ 39 MB) | caben en git |
| Memoria del proceso (ensemble) | 0,89 GB en CPU; 1,17 GB con GPU | sobra con 24 GB |
| Latencia por análisis | CPU 2,36 s (primera 4,0 s); GPU 2,45 s | la GPU no ayuda |
| Inferencia pura por modelo y ventana | CPU 44 ms; GPU 9 ms | el tiempo está en el PDF y los gráficos (≈ 1 s el PDF) |
| 5 solicitudes simultáneas | 10,7 s en total (en cola) | suficiente para una demostración |
| Probabilidades CPU vs GPU | coinciden a 4 decimales | servir en CPU no cambia resultados |
| Plan de energía "Equilibrado" | suspende a las 3 h con corriente, 3 min con batería | solo importa si se deja corriendo (T8) |
| Escucha por defecto del lanzador | solo `127.0.0.1` (la IP de la red local no responde) | no se expone sin pedirlo |

## Opciones
| Opción | Veredicto |
|---|---|
| **A. Ejecutar en esta PC con waitress y el bundle** | **Elegida**: sin costo, sin cuentas, resultados idénticos |
| B. Red local (`-Lan`) | Disponible para mostrarla desde un teléfono o proyector en la misma red |
| C. Túnel temporal de Cloudflare (`-Tunnel`) | Opcional: URL `https://….trycloudflare.com` que cambia en cada ejecución; sin cuenta |
| D. Nube / contenedor (Hugging Face, Render, Cloud Run, Docker) | Descartada por el autor; se eliminaron `Dockerfile`, `.dockerignore`, `Procfile` y los scripts de host |
| E. Sitio estático en GitHub Pages | No necesario: el artículo no cita URL |
| F. App de escritorio / inferencia en el navegador / backend ONNX | Descartadas: no aportan frente a A |

## Decisiones de arquitectura
- **Pesos de inferencia empaquetados** (`models/`, generados con `tools/export_inference_bundle.py`); se cargan con `weights_only=True`.
- **Configuración por variables de entorno** (`deploy/ensemble.env` + `ECG_*`, documentadas en `webapp/README.md`).
- **Servidor waitress** lanzado por `deploy/start_local.ps1`: solo `127.0.0.1`, CPU (`ECG_DEVICE=cpu`), 4 hilos.
- **Arranque a demanda**: el servidor se inicia cuando se necesita y se detiene con Ctrl+C; sin arranque automático.
- **Modelos en caché** y concurrencia acotada; **sin persistencia** de ECG subidos (se borran al terminar; informes caducan a la hora).
- **Privacidad:** la demo es académica; la interfaz pide no subir datos de pacientes reales; hay siete ejemplos incluidos.
- **Túnel (opcional):** waitress borra `X-Forwarded-For` salvo que se le configure `--trusted-proxy` (medido: sin eso todos los visitantes comparten un límite de 20 solicitudes por minuto); el lanzador lo configura y solo permite el túnel con la app en `127.0.0.1`.

## Riesgos y mitigaciones
| Riesgo | Impacto | Mitigación |
|---|---|---|
| La PC se suspende o se apaga | Alto | `powercfg` sin suspensión con corriente y cargador conectado si se deja corriendo (T8); si está apagada, la demo no existe (aceptado) |
| Windows Update reinicia la PC | Bajo | se vuelve a iniciar el servidor a demanda; fijar las horas activas si se deja corriendo mucho tiempo |
| Exponer la app por error | Alto | escucha solo en `127.0.0.1`; `-Lan` y `-Tunnel` son opciones explícitas e incompatibles entre sí |
| Mucha gente a la vez | Bajo | cola de inferencia + límite por IP; ≈ 2 s por análisis |
| El túnel es lento (medido ≈ 10–25 KB/s). Con PNG 306 KB, PDF 406 KB y Plotly 1,3 MB gzip las descargas ya terminan (9 de 9; antes 1 de 3 y Plotly nunca), pero tardan 10–75 s | Medio | uso opcional y corto, mejor desde datos móviles; para mostrarla en la misma sala usar `127.0.0.1` o `-Lan` |
| Datos sensibles subidos por un usuario | Alto | no se persisten, aviso visible, ejemplos incluidos |
| Cambio de versión de torch | Medio | `requirements.txt` fija `torch==2.8.0`; el entorno actual usa 2.14.0 y el bundle pasa el humo con ella |

## Orden y dependencias
```
Fase 1 (hecha): T1 bundle → T2 config → T3 servidor → T4 caché → T5 límites → T6 ejemplos
Fase 2: T7 lanzador + ECG_DEVICE → T8 guía de uso y energía (documentada) → T9 prueba de uso real (autor)
```

## Fuera de alcance
Hosting en la nube, contenedores, URL pública estable, DOI/Zenodo, sitio estático, app de escritorio e inferencia en el navegador.
