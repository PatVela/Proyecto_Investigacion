# docs/ — documentación del proyecto

**Una casa por hecho:** el README raíz es la portada; el detalle vive aquí.

| Archivo | Contenido | Úsese para |
|---|---|---|
| `informe_cinc2020_12.md` | Informe técnico: etiquetas, exclusiones, preprocesamiento, modelo | Metodología |
| `experimentos_cinc2020.md` | Experimentos A/B, comparación justa, ablaciones | Reproducir corridas |
| `resultados_finales.md` | Tablas en test (n = 6421) con F1 común y F0.5, F1 por clase, calibración, Challenge, modelo recomendado | Cifras citables |
| `fix_prediccion_cinc2020.md` | Postmortem (2026-09-10): esquema v1 incompleto, normalización, recorte. Documento **histórico**: describe el estado de ese día | Contexto de `schema`/`signal` |

Chequeo humano: si cambian los resultados, actualizar `resultados_finales.md` **y** regenerar el bundle de la webapp (`deploy/results/`, `deploy/reports/`; ver `webapp/README.md`, Ejecución local). Los documentos de esta carpeta son metodología y cifras.
