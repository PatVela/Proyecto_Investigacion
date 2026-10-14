"""Tests del monitor de early stopping (val_loss vs val_f1_macro)."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from ecg.train import evaluate_val_f1, is_monitor_improvement


def test_is_monitor_improvement():
    assert is_monitor_improvement(0.6, 0.5, "val_f1_macro") is True
    assert is_monitor_improvement(0.4, 0.5, "val_f1_macro") is False
    assert is_monitor_improvement(0.5, 0.5, "val_f1_macro") is False
    assert is_monitor_improvement(0.4, 0.5, "val_loss") is True
    assert is_monitor_improvement(0.6, 0.5, "val_loss") is False


def test_evaluate_val_f1_known_case():
    xs = torch.randn(4, 3)
    ys = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=torch.float32)
    loader = DataLoader(TensorDataset(xs, ys), batch_size=2)

    class Fixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            if self.calls == 1:  # muestras 0-1: clase 0 perfecta, clase 1 positiva
                return torch.tensor([[10.0, 10.0], [10.0, 10.0]])
            return torch.tensor([[-10.0, 10.0], [-10.0, 10.0]])  # muestras 2-3

    f1 = evaluate_val_f1(Fixed(), loader, torch.device("cpu"), amp=False)
    # Clase 0: F1=1.0. Clase 1: predice todo 1 con y=[1,0,1,0] → P=0.5,R=1,F1=2/3.
    assert abs(f1 - (1.0 + 2.0 / 3.0) / 2.0) < 1e-4


# --- claves de configuración que nada lee (erratas) ------------------------------------------------------------------

def _keys_read_from(path, names):
    """Literales usados como params.get("k") / params["k"] (o config.get / cfg.get) en un archivo."""
    import ast
    from pathlib import Path

    keys = set()
    for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
            if ast.unparse(node.func.value) in names and isinstance(node.args[0], ast.Constant):
                keys.add(node.args[0].value)
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) in names and isinstance(node.slice, ast.Constant):
            keys.add(node.slice.value)
    return keys


def test_known_config_keys_cover_everything_the_code_reads():
    """Si alguien añade params.get("nueva") sin ponerla en KNOWN_CONFIG_KEYS, la corrida avisaría de una clave 'ignorada'."""
    from ecg.train import KNOWN_CONFIG_KEYS

    read = (_keys_read_from("ecg/train.py", {"params"}) | _keys_read_from("ecg/network.py", {"config"})
            | _keys_read_from("ecg/hdf5_dataset.py", {"self.augment_cfg"}) | _keys_read_from("ecg/augment.py", {"cfg"})
            | _keys_read_from("examples/cinc2020/evaluate.py", {"config"}))
    assert read, "el AST no encontró ninguna clave: la prueba dejó de medir algo"
    assert read <= KNOWN_CONFIG_KEYS, sorted(read - KNOWN_CONFIG_KEYS)


def test_shipped_configs_have_no_unknown_keys():
    import json
    from pathlib import Path

    from ecg.train import unknown_config_keys

    configs = sorted(Path("examples/cinc2020").glob("config*.json"))
    assert len(configs) >= 4
    for path in configs:
        assert unknown_config_keys(json.loads(path.read_text(encoding="utf-8"))) == [], path.name


def test_a_misspelled_hyperparameter_is_reported(capsys, tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from ecg import train as train_mod

    assert train_mod.unknown_config_keys({"conv_dropuot": 0.5, "conv_dropout": 0.2, "seed": 1}) == ["conv_dropuot"]
    config = json.loads(open("examples/cinc2020/config_synthetic.json", encoding="utf-8").read())
    config["conv_dropuot"] = 0.5                                       # errata de "conv_dropout"
    monkeypatch.setattr(train_mod, "make_dataloaders", lambda params: (_ for _ in ()).throw(RuntimeError("alto")))
    args = SimpleNamespace(seed=None, device="cpu", no_amp=True, experiment="x", resume=None, epochs=1)
    try:
        train_mod.train(args, config)
    except RuntimeError:
        pass
    out = capsys.readouterr().out
    assert "AVISO" in out and "conv_dropuot" in out
