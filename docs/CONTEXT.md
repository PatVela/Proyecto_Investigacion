# docs/ — documentación del proyecto

**Una casa por hecho:** el README raíz es la portada; el detalle vive aquí.

| Archivo | Contenido | Úsese para |
|---|---|---|
| `informe_cinc2020_12.md` | Informe técnico: etiquetas, exclusiones, preprocesamiento, modelo | Metodología |
| `experimentos_cinc2020.md` | Experimentos A/B, comparación justa, ablaciones | Reproducir corridas |
| `resultados_finales.md` | Tablas en test (n = 6421) con F1 común y F0.5, F1 por clase, calibración, Challenge, modelo recomendado | Cifras del sistema anterior (reparto por registro); ver el aviso al inicio del archivo |
| `fix_prediccion_cinc2020.md` | Postmortem (2026-09-10): esquema v1 incompleto, normalización, recorte. Documento **histórico**: describe el estado de ese día | Contexto de `schema`/`signal` |
| `REPRODUCIBILIDAD.md` | Qué se comprobó y con qué nivel de prueba, qué no, y el procedimiento registrado del sistema del artículo | Reproducir el sistema vigente |

Chequeo humano: si cambian los resultados, actualizar `resultados_finales.md` **y** regenerar el bundle de la webapp (`deploy/results/`, `deploy/reports/`; ver `webapp/README.md`, Ejecución local). Las cifras de esta carpeta (salvo `REPRODUCIBILIDAD.md`) corresponden al sistema anterior, con reparto aleatorio por registro; el sistema vigente (reparto por origen y por paciente, evaluación externa) se describe en `deploy/CONTEXT.md`, `examples/cinc2020/README.md` y `REPRODUCIBILIDAD.md`, y sus resultados completos están en el artículo.
