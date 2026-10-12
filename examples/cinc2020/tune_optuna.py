"""Tuning de hiperparámetros con Optuna (objetivo: val_f1_macro).

Lee un config base (p. ej. ``config_resnet_v2.json``), sugiere
hiperparámetros por trial, entrena con ``ecg.train`` y escribe el
config ganador (p. ej. ``config_resnet_v3.json``)::

    python examples/cinc2020/tune_optuna.py --n-trials 30 --epochs 40

Notas:
    * Un trial a la vez (n_jobs=1): la VRAM no da para más.
    * El estudio vive en sqlite: reanudar es repetir el mismo comando
      con igual ``--study-name`` y ``--storage``.
    * ``train()`` no cambia de comportamiento: solo recibe un
      ``epoch_callback`` que reporta métricas para podar trials malos.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ecg.train import load_config, train  # noqa: E402


def suggest_params(trial, base: dict) -> dict:
    """Sugiere hiperparámetros y los fusiona sobre una copia de ``base``."""
    suggested = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "conv_dropout": trial.suggest_float("conv_dropout", 0.1, 0.5),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
    }
    params = copy.deepcopy(base)
    params.update(suggested)
    return params


def run_trial(params: dict, experiment: str, trial) -> float:
    """Entrena un trial y devuelve su mejor val_f1_macro (poda si va mal)."""
    import optuna  # import tardío: --help funciona sin optuna instalado

    def on_epoch(epoch: int, row: dict) -> None:
        trial.report(row["val_f1_macro"], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    args = SimpleNamespace(
        seed=None, device=None, no_amp=False,
        experiment=experiment, resume=None, epochs=None,
    )
    save_dir = train(args, params, epoch_callback=on_epoch)
    summary = json.loads(
        (save_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    return float(summary["best_monitored_value"])


def objective(trial, *, base: dict, trial_epochs: int, save_root: str, study_tag: str) -> float:
    """Objetivo Optuna: entrena y devuelve mejor val_f1_macro del trial."""
    params = suggest_params(trial, base)
    params["max_epochs"] = trial_epochs
    params["save_dir"] = save_root
    params["seed"] = int(base.get("seed", 42)) + trial.number
    experiment = f"{study_tag}-trial-{trial.number:03d}"
    return run_trial(params, experiment, trial)


def write_winning_config(base: dict, best_params: dict, path: str | Path) -> Path:
    """Fusiona ganadores sobre ``base`` y restaura las épocas completas."""
    config = copy.deepcopy(base)
    config.update(best_params)
    config["max_epochs"] = base.get("max_epochs", 150)
    out = Path(path)
    out.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Tuning Optuna hacia config v3 (maximiza val_f1_macro)"
    )
    parser.add_argument("--config", default="examples/cinc2020/config_resnet_v2.json",
                        help="Config base con early_stopping_metric=val_f1_macro")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=40,
                        help="Épocas por trial (el podado corta los malos antes)")
    parser.add_argument("--study-name", default="cinc2020_resnet_v3")
    parser.add_argument("--storage", default="sqlite:///runs/models/optuna_cinc2020.db",
                        help="Estudio persistente (repite comando para reanudar)")
    parser.add_argument("--save-dir", default="saved/tune_v3",
                        help="Raíz de artefactos por trial")
    parser.add_argument("--output-config", default="examples/cinc2020/config_resnet_v3.json",
                        help="Dónde escribir el config ganador")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    if args.storage.startswith("sqlite:///"):
        Path(args.storage[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)

    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    from optuna.trial import TrialState

    base = load_config(args.config)
    if base.get("early_stopping_metric", "val_loss") != "val_f1_macro":
        raise SystemExit(
            "tune_optuna requiere early_stopping_metric=val_f1_macro en el config base"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: objective(
            trial, base=base, trial_epochs=args.epochs,
            save_root=args.save_dir, study_tag=args.study_name,
        ),
        n_trials=args.n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    completed = [t for t in study.trials if t.state is TrialState.COMPLETE]
    if not completed:
        raise SystemExit("Ningún trial completó: revisa los logs de saved/")
    best = study.best_trial
    print(f"\nMejor val_f1_macro: {best.value:.4f} (trial {best.number})")
    for key, value in best.params.items():
        print(f"  {key} = {value}")
    out = write_winning_config(base, best.params, args.output_config)
    print(f"Config ganador: {out}")
    print(f"Siguiente: python -m ecg.train {out} --experiment cinc2020_resnet_v3")


if __name__ == "__main__":
    main()
