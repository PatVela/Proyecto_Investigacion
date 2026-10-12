# docs/ — documentación del proyecto

**Una casa por hecho:** el README raíz es la portada; el detalle vive aquí.

| Archivo | Contenido | Úsese para |
|---|---|---|
| `informe_cinc2020_12.md` | Informe técnico: etiquetas, exclusiones, preprocesamiento, modelo | Metodología (tesis) |
| `experimentos_cinc2020.md` | Experimentos A/B, comparación justa, ablaciones | Reproducir corridas |
| `resultados_finales.md` | Tabla principal en test (n = 6421), F1 por clase, modelo recomendado | Cifras citables |
| `fix_prediccion_cinc2020.md` | Postmortem: esquema v1 incompleto, normalización, recorte | Contexto histórico de `schema`/`signal` |

Chequeo humano: si cambian los resultados, actualizar `resultados_finales.md` **y** `webapp/static/metrics/metrics.json`.
