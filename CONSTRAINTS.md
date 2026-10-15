# Nivel de calidad

Última revisión: 2026-10-15. Cada umbral trae el comando que lo comprueba, dónde se hace cumplir y el valor medido hoy.
No confundir con [`constraints.txt`](constraints.txt) de la raíz: son los pins de pip, no este contrato.

## Umbrales

| Dimensión | Umbral | Comando de verificación | Dónde se aplica | Valor medido hoy (2026-10-15) |
|---|---|---|---|---|
| 1. CI en verde | `tests` en verde en cada PR y en cada push a `main` (una ejecución `cancelled` por un push posterior no cuenta como fallo) | `gh run list --workflow tests --limit 3` | CI: [`tests.yml`](.github/workflows/tests.yml) | las 3 últimas `success`; último push a `main` (af31e63): `success`, 5 min 29 s. Una ejecución `cancelled` es un push a `main` sustituido por el siguiente |
| 2. Pruebas omitidas | Solo los motivos de la lista de abajo; ninguna por dependencia faltante | `python -m pytest -q -rs -p no:cacheprovider` (las líneas `SKIPPED` solo pueden traer esos motivos) | CI (`python -m pytest -q -W ignore`) y manual | Windows, local: 625 pasadas, 0 omitidas, 6 min 7 s. CI Linux (af31e63): 598 pasadas y 3 omitidas (las 3 `@windows_only`) |
| 3. `pip-audit` limpio | 0 vulnerabilidades conocidas: el job de `audit` termina en `success` | `gh run view $(gh run list --workflow audit --limit 1 --json databaseId -q '.[0].databaseId') --json jobs -q '.jobs[].conclusion'` | CI: [`audit.yml`](.github/workflows/audit.yml), lunes 06:00 UTC en `main`, a mano y en PR que toquen dependencias | Ejecución manual sobre `main` (8534048, 2026-10-15): job `success`. La primera semanal: 2026-10-19 |
| 4. Documentación sin problemas | 0 problemas y todas las pruebas de enlaces y comandos pasan | `python tools/check_docs_commands.py` y `python -m pytest tests/test_docs_links.py tests/test_docs_commands.py -q -p no:cacheprovider` | CI (primer paso y suite) | 128 comandos, 0 problemas; 15 pruebas pasan |
| 5. Cobertura | Solo informativa, sin umbral | propuesta pendiente de aprobación (abajo) | ninguno | no se mide |

En una PR, `audit` sale `success` como ejecución aunque el paso `pip-audit` falle (`continue-on-error`): por eso el comando de la dimensión 3 mira el job, no la ejecución.

## Omisiones de pruebas permitidas (dimensión 2)

| Motivo en la línea `SKIPPED` | Origen |
|---|---|
| `requiere Windows con PowerShell` | 3 pruebas `@windows_only` de `tests/test_deploy_entrypoint.py`; el CI Linux las omite |
| `git no disponible` / `no es un repositorio git` | `tests/test_deploy_config.py` y `tests/test_docs_links.py` |
| `node no disponible` | `tests/test_hardening.py` |
| `falta el bundle` | checkout incompleto; el bundle está en git, así que en CI no debe aparecer |

No se admite ninguna omisión por dependencia faltante (`importorskip` de `flask`, `waitress`, `torch`, `pyflakes` u `optuna`): si aparece, el entorno está roto.

## Cobertura: propuesta pendiente de aprobación (no está instalada)

Cambio exacto, solo informativo (sin `--cov-fail-under`):

- `requirements-dev.txt`: añadir `pytest-cov==<X.Y.Z>` (fijar la versión al aprobar) y regenerar [`constraints.txt`](constraints.txt) como hace el CI; entrará también `coverage`.
- `.github/workflows/tests.yml`, paso «Pruebas»: `python -m pytest -q -W ignore --cov=ecg --cov=webapp --cov=tools --cov-report=term:skip-covered`.

## Medido, sin umbral

Pruebas recolectadas: 625 en local (en el último CI de `main`, 601: 598 pasadas y 3 omitidas). No debe bajar sin explicar qué se quitó y por qué.

## Regla anti-degradación

- Un umbral solo se baja con la justificación escrita en el mismo cambio (descripción de la PR y este archivo), nunca para que un cambio pase.
- No se silencian ni se omiten pruebas para llegar a verde: ningún `skip`, `skipif`, `xfail`, `# noqa` ni `# type: ignore` nuevo sin motivo escrito en el mismo cambio. Una omisión nueva debe encajar en la lista de arriba.
- Quien cambia este archivo o `.github/workflows/` lo dice en la descripción de la PR, y el cambio de este archivo se revisa antes de fusionar.
- Excepciones vigentes: ninguna.
