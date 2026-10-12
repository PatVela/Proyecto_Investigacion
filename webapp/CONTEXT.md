# webapp/ — demo de inferencia

**Trabajo:** servir un Flask (`app.py`, puerto 5002) que recibe un ECG (CSV/WFDB), predice con un checkpoint de `saved/` y genera informe PDF.

- **Entradas:** checkpoint en `saved/`, `thresholds`/`temperaturas` de un `eval-*/`, subidas del usuario (`uploads/`, git-ignored).
- **Código:** `prediction.py` (puente a `ecg.predict`, fallback normal), `report_pdf.py`, `project_info.py` + `static/metrics/` (métricas mostradas), `templates/index.html`, `static/app.js`.
- **Arranque:** `webapp/README.md`; `run_https.sh`, `run_public.sh`, `wsgi.py`, `Dockerfile` raíz.
- **Qué toca un cambio:** preprocesamiento debe coincidir con `ecg/signal.py`; `static/metrics/metrics.json` debe reflejar `docs/resultados_finales.md`. Tests: `tests/test_webapp*.py`, `tests/test_report.py`.
- **Chequeo humano:** misma señal por CLI y por la web debe dar las mismas probabilidades.
