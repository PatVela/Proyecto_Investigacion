# tests/ — suite pytest

**Trabajo:** red de seguridad (~100 tests según el último commit; requiere `pytest` instalado). Fixtures compartidas en `conftest.py`. Ejecutar: `pytest` desde la raíz.

| Área | Tests |
|---|---|
| `ecg/` | `test_schema`, `test_augment`, `test_network`, `test_calibration`, `test_train_monitor`, `test_accuracy`, `test_curves` |
| Scripts del pipeline | `test_cli_smoke`, `test_bootstrap`, `test_robustness`, `test_benchmark`, `test_sex_breakdown`, `test_make_figures`, `test_tune_optuna` (nuevo, sin versionar) |
| `tools/` | `test_tools_audit`, `test_tools_check`, `test_tools_missing` |
| `webapp/` | `test_webapp`, `test_webapp_ensemble`, `test_report` |

Chequeo humano: todo verde antes de mover o renombrar cualquier ruta del pipeline.
