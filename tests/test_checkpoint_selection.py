"""Elegir el mejor checkpoint de una carpeta sin volver a cargarlos todos en cada petición.

Medido: con 12 checkpoints (663 MB) en saved/, la webapp en modo desarrollo tardaba 3,5-4,7 s por página y 5,7 s por
análisis porque ordenar los checkpoints hacía torch.load de cada uno varias veces por petición.
"""

import os

import pytest
import torch

from ecg import util


def _ckpt(path, val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"val_loss": val_loss, "model_state_dict": {}}, str(path))
    return path


@pytest.fixture()
def loads(monkeypatch):
    """Cuenta cuántos checkpoints se leen de verdad."""
    util._VAL_LOSS_CACHE.clear()
    calls = []
    original = util.load_checkpoint

    def counting(path, *args, **kwargs):
        calls.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(util, "load_checkpoint", counting)
    return calls


def test_best_checkpoint_prefers_real_best_pt_with_the_lowest_val_loss(tmp_path, loads):
    _ckpt(tmp_path / "synth" / "best.pt", 0.01)                   # sintético: nunca debe ganar aunque su pérdida sea baja
    _ckpt(tmp_path / "run1" / "best.pt", 0.40)
    _ckpt(tmp_path / "run2" / "best.pt", 0.30)
    _ckpt(tmp_path / "run2" / "latest.pt", 0.10)                  # best.pt va antes que latest.pt
    assert util.best_checkpoint(tmp_path) == tmp_path / "run2" / "best.pt"


def test_val_loss_is_read_once_per_file_not_once_per_call(tmp_path, loads):
    for i, loss in enumerate((0.5, 0.3, 0.4)):
        _ckpt(tmp_path / f"r{i}" / "best.pt", loss)
    util.best_checkpoint(tmp_path)
    assert len(loads) == 3
    for _ in range(5):                                            # 5 páginas / análisis más
        util.best_checkpoint(tmp_path)
    assert len(loads) == 3


def test_a_rewritten_checkpoint_is_read_again(tmp_path, loads):
    old = _ckpt(tmp_path / "a" / "best.pt", 0.50)
    other = _ckpt(tmp_path / "b" / "best.pt", 0.40)
    assert util.best_checkpoint(tmp_path) == other
    _ckpt(old, 0.10)
    stat = old.stat()
    os.utime(old, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))   # mismo tamaño y fecha casi igual: el caché debe notarlo
    assert util.best_checkpoint(tmp_path) == old


def test_unreadable_or_loss_free_checkpoints_sort_last_without_crashing(tmp_path, loads):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "best.pt").write_bytes(b"no es un checkpoint")
    good = _ckpt(tmp_path / "y" / "best.pt", 0.9)
    assert util.checkpoint_val_loss(tmp_path / "x" / "best.pt") is None
    assert util.checkpoint_val_loss(tmp_path / "no_existe.pt") is None
    assert util.best_checkpoint(tmp_path) == good


def test_webapp_checkpoint_listing_reuses_the_same_cache(tmp_path, loads):
    pytest.importorskip("flask")
    from webapp.app import _list_checkpoints

    for i, loss in enumerate((0.5, 0.3)):
        _ckpt(tmp_path / f"r{i}" / "best.pt", loss)
    first = _list_checkpoints(tmp_path)
    assert first[0].parent.name == "r1" and len(loads) == 2
    _list_checkpoints(tmp_path)
    util.best_checkpoint(tmp_path)
    assert len(loads) == 2
