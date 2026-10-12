"""tune_optuna: fusión de params y config ganador (sin entrenar)."""

import inspect
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples" / "cinc2020"))

import tune_optuna as tune
from ecg.train import train

BASE = {
    "train": "data/x/train.h5",
    "dev": "data/x/val.h5",
    "save_dir": "saved/x",
    "max_epochs": 150,
    "early_stopping_metric": "val_f1_macro",
    "seed": 43,
    "learning_rate": 0.0005,
}

SUGGESTED = {
    "learning_rate": 0.001,
    "weight_decay": 0.0002,
    "conv_dropout": 0.3,
    "label_smoothing": 0.05,
}


class FakeTrial:
    def __init__(self, values):
        self.values = values
        self.names = []

    def suggest_float(self, name, *args, **kwargs):
        self.names.append(name)
        return self.values[name]


def test_suggest_merges_over_copy():
    trial = FakeTrial(SUGGESTED)
    params = tune.suggest_params(trial, BASE)
    assert set(trial.names) == set(SUGGESTED)
    for key, value in SUGGESTED.items():
        assert params[key] == value
    assert params["max_epochs"] == 150
    assert params["seed"] == 43
    assert BASE["learning_rate"] == 0.0005  # base intacta


def test_write_winning_config_restores_epochs(tmp_path):
    out = tune.write_winning_config(BASE, SUGGESTED, tmp_path / "v3.json")
    config = json.loads(out.read_text(encoding="utf-8"))
    for key, value in SUGGESTED.items():
        assert config[key] == value
    assert config["max_epochs"] == 150
    assert config["seed"] == 43


def test_train_epoch_callback_defaults_to_none():
    param = inspect.signature(train).parameters["epoch_callback"]
    assert param.default is None


def test_main_rejects_non_f1_monitor(tmp_path):
    optuna = pytest.importorskip("optuna")
    cfg = tmp_path / "base.json"
    cfg.write_text(json.dumps({"max_epochs": 2}))
    with pytest.raises(SystemExit, match="val_f1_macro"):
        tune.main(["--config", str(cfg)])
