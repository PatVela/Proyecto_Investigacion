# Datos, procedencia, licencias y política de disponibilidad

**Estado: propuesta sin publicar (R11, 2026-11-19).** Este documento no redistribuye datos: dice de dónde salen, bajo qué licencia, cómo obtenerlos y qué se publica y qué no.

## Política de disponibilidad
- **No se redistribuyen datos de pacientes ni señales.** Todos los datos de entrenamiento y evaluación son públicos y se descargan del sitio de origen con los scripts del repositorio.
- **Sí se publican** el código (GPL-3.0-only), los pesos del sistema principal (`models/`, ≈ 19 MB cada uno, sin estado del optimizador), su calibración, sus umbrales y las ocho métricas globales de la prueba A (`deploy/`).
- **No se publican en este repositorio** (están en una carpeta local de los autores que no se sube a GitHub): las predicciones por registro, los protocolos con hash, los manifiestos del reparto y los cuatro scripts que calculan las tablas y figuras del artículo. Su depósito (por ejemplo con DOI) es una decisión de los autores; hasta entonces el artículo declara que regenerar las tablas y figuras desde el repositorio solo no se ha verificado.
- No hay datos restringidos, privados ni con consentimiento pendiente en el repositorio.

## Fuentes de datos
| Datos | Dónde se obtienen | Licencia declarada | Uso en este trabajo |
|---|---|---|---|
| PhysioNet/CinC Challenge 2020 (6 subconjuntos; 43 101 registros) | `bash descargar_datos_2020.sh` (PhysioNet) | CC BY 4.0 (según la declaración del artículo) | Entrenamiento, validación, prueba A y prueba B (Georgia) |
| PhysioNet/CinC Challenge 2021 v1.0.3: carpetas `chapman_shaoxing` y `ningbo` | `bash descargar_challenge2021.sh` o `python tools/descargar_challenge2021.py <carpeta>` (reanudable, con hilos); comprobación con `python tools/verificar_challenge2021.py` | CC BY 4.0 | Evaluación externa (C y D); solo se usan estas dos carpetas |
| Pesos y conjunto de prueba del modelo publicado de Ribeiro et al. | Zenodo 10.5281/zenodo.3625017 (pesos, 320 MB) y 10.5281/zenodo.3625006 (prueba, 208 MB) | CC BY 4.0 (comprobada en la ficha de Zenodo); el código de los autores es MIT | Comparación exploratoria con un modelo publicado (`docs/COMPARACION_PUBLICADA.md`) |

## Procedencia y citas
Los datos del Challenge se citan en el artículo (Perez Alday et al. 2020; Zheng et al.; Reyna et al.). El modelo de Ribeiro et al. se cita como Nat. Commun. 11, 1760 (2020), DOI 10.1038/s41467-020-15432-4. Los archivos descargados se verifican con el `SHA256SUMS.txt` que publica PhysioNet (el artículo y `evidencia/reparto_externo.json` registran su SHA-256).

**Verificado en R11:** Chapman-Shaoxing y Ningbo (45 152 registros) se descargaron de nuevo; los **90 350 archivos** de esas carpetas coinciden con ese `SHA256SUMS.txt`, cuyo propio SHA-256 (`f29c2609…2473`) coincide con el que registraron los autores, y `python tools/verificar_challenge2021.py` termina sin errores (código 0). Una nota sobre los datos: en Ningbo, la cabecera `S23074.hea` nombra el archivo de señal `JS23074.mat` (no `S23074.mat`); ningún otro registro de Chapman-Shaoxing ni de Ningbo tiene un nombre de señal distinto del suyo, y los scripts de comparación leen el archivo que nombra la cabecera.

**No verificado:** los datos de entrenamiento del Challenge 2020 (≈ 15 GB; a la velocidad medida con PhysioNet en R11, ≈ 0.4–0.7 MB/s, serían ≈ 6 a 10 h) no se descargaron en R11.

## Licencias del código
Este repositorio: GPL-3.0-only (`LICENSE`). El código de Ribeiro et al. (MIT) **no se copia** aquí; `examples/cinc2020/baseline_ribeiro/ribeiro_pytorch.py` es una reimplementación propia de su arquitectura que lee sus pesos publicados.
