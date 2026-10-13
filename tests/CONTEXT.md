# tests/ — suite pytest

**Trabajo:** red de seguridad (requiere `pip install -r requirements-dev.txt`). Fixtures compartidas en `conftest.py`. Ejecutar: `pytest` desde la raíz. Algunos tests se saltan solos si faltan datos locales (bundle, `runs/`).

| Área | Tests |
|---|---|
| `ecg/` | `test_schema`, `test_signal_schema_core`, `test_signal_sampling_rate`, `test_augment`, `test_network`, `test_calibration`, `test_train_monitor`, `test_accuracy`, `test_curves`, `test_shared_helpers`, `test_decision_logic` |
| Scripts del pipeline | `test_cli_smoke`, `test_bootstrap`, `test_robustness`, `test_benchmark`, `test_sex_breakdown`, `test_make_figures`, `test_tune_optuna`, `test_challenge_score`, `test_audit_dx_codes`, `test_compare_models`, `test_error_analysis`, `test_calibrate_cli`, `test_platt_evaluation`, `test_predict_hdf5_logits` |
| `tools/` | `test_tools_audit`, `test_tools_check`, `test_tools_missing`, `test_inference_bundle`, `test_examples` |
| `webapp/` (lógica) | `test_webapp`, `test_webapp_ensemble`, `test_webapp_loaders`, `test_webapp_security`, `test_report`, `test_plot_grid`, `test_model_resolution`, `test_model_cache`, `test_error_handling`, `test_payload_size`, `test_webapp_platt` (PNG, PDF y gzip: más pequeños sin cambiar el contenido) |
| Ejecución local | `test_dev_server` (`python webapp/app.py` con el ensemble), `test_deploy_config`, `test_deploy_entrypoint` (scripts de Windows; se omiten fuera de Windows), `test_serving_device`, `test_public_limits`, `test_docs_env_vars` |
| Interfaz | `test_ui_accessibility`, `test_ui_loading_theming`, `test_ui_content` (comprobaciones estáticas sobre HTML/CSS/JS; el comportamiento real se verifica además en un navegador) |
| Higiene del repo | `test_no_stale_paths` (rutas `results/` obsoletas), `test_requirements` (dependencias declaradas), `test_docs_links` (enlaces de los .md y repo/rama vigentes) |

Chequeo humano: todo verde antes de mover o renombrar cualquier ruta del pipeline; si un cambio toca seguridad (subidas, rutas, carga de checkpoints), añadir primero el test que falla con el código viejo.
