"""Caché de modelos, inferencias simultáneas acotadas y matplotlib protegido entre hilos."""

import threading
import time

import numpy as np
import pytest
import torch

pytest.importorskip("flask")

from ecg import load, network
from webapp import prediction as pred


def _ckpt(path, seed):
    torch.manual_seed(seed)
    model = network.build_network(num_leads=12, num_classes=12)
    torch.save({"model_state_dict": model.state_dict(), "class_names": load.CLASS_NAMES, "label_schema": load.LABEL_SCHEMA,
                "optimizer_state_dict": {"state": {}, "param_groups": []}, "epoch": 3, "val_loss": 0.4}, str(path))
    return path


def _csv(path):
    sig = np.random.default_rng(4).normal(0, 0.3, (load.WINDOW_LENGTH, load.NUM_LEADS))
    np.savetxt(str(path), sig, delimiter=",", header=",".join(load.STANDARD_LEAD_ORDER), comments="")
    return path


@pytest.fixture(autouse=True)
def fresh_cache():
    pred.clear_model_cache()
    yield
    pred.clear_model_cache()


@pytest.fixture()
def load_counter(monkeypatch):
    calls = []
    original = pred.load_model

    def counting(path, device=None, **kw):
        calls.append(str(path))
        return original(path, device=device, **kw)

    monkeypatch.setattr(pred, "load_model", counting)
    return calls


def test_models_are_loaded_once_across_predictions(tmp_path, load_counter):
    a, b, csv = _ckpt(tmp_path / "a.pt", 1), _ckpt(tmp_path / "b.pt", 2), _csv(tmp_path / "r.csv")
    first = pred.run_prediction_from_saved_paths([csv], str(a), None, tmp_path / "o1", model_path_b=str(b))
    second = pred.run_prediction_from_saved_paths([csv], str(a), None, tmp_path / "o2", model_path_b=str(b))
    assert len(load_counter) == 2                                     # A y B una sola vez (antes: 4 cargas)
    assert first["per_window_probabilities"] == second["per_window_probabilities"]
    assert first["checkpoint_epoch"] == 3 and first["checkpoint_val_loss"] == 0.4


def test_cached_checkpoint_drops_weights_and_optimizer_state(tmp_path):
    _, light, classes = pred.get_cached_model(_ckpt(tmp_path / "a.pt", 1), torch.device("cpu"))
    assert classes == load.CLASS_NAMES and light["epoch"] == 3
    assert not ({"model_state_dict", "optimizer_state_dict"} & set(light))


def test_cache_reloads_when_the_file_changes(tmp_path, load_counter):
    path = _ckpt(tmp_path / "a.pt", 1)
    cpu = torch.device("cpu")
    m1 = pred.get_cached_model(path, cpu)[0]
    assert pred.get_cached_model(path, cpu)[0] is m1 and len(load_counter) == 1
    time.sleep(0.02)
    _ckpt(path, 2)                                                    # mismo nombre, otros pesos
    m2 = pred.get_cached_model(path, cpu)[0]
    assert m2 is not m1 and len(load_counter) == 2
    assert len(pred._MODEL_CACHE) == 1                                # la versión anterior se descarta


def test_cache_is_bounded_and_evicts_least_recently_used(tmp_path, load_counter, monkeypatch):
    monkeypatch.setattr(pred, "MODEL_CACHE_SIZE", 2)
    cpu = torch.device("cpu")
    paths = [_ckpt(tmp_path / f"{n}.pt", i) for i, n in enumerate("abc")]
    for p in paths:
        pred.get_cached_model(p, cpu)
    assert len(pred._MODEL_CACHE) == 2 and len(load_counter) == 3
    pred.get_cached_model(paths[2], cpu)                              # sigue en caché
    assert len(load_counter) == 3
    pred.get_cached_model(paths[0], cpu)                              # fue expulsado
    assert len(load_counter) == 4


def test_concurrent_requests_respect_the_inference_limit(tmp_path, monkeypatch):
    ckpt, csv = _ckpt(tmp_path / "a.pt", 1), _csv(tmp_path / "r.csv")
    monkeypatch.setattr(pred, "_INFERENCE_SLOTS", threading.BoundedSemaphore(2))
    state = {"now": 0, "peak": 0}
    guard = threading.Lock()
    original = pred._infer_windows_unlocked

    def slow(model, windows, device):
        with guard:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.15)
        try:
            return original(model, windows, device)
        finally:
            with guard:
                state["now"] -= 1

    monkeypatch.setattr(pred, "_infer_windows_unlocked", slow)
    results, errors = {}, []

    def worker(i):
        try:
            results[i] = pred.run_prediction_from_saved_paths([csv], str(ckpt), None, tmp_path / f"o{i}")
        except Exception as exc:                                      # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join(timeout=180) for t in threads]
    assert not errors and len(results) == 6
    assert state["peak"] <= 2 and state["peak"] >= 2                  # se usó la concurrencia permitida, sin pasarla
    assert len({tuple(r["per_window_probabilities"][0]) for r in results.values()}) == 1


def test_plots_from_concurrent_threads_are_all_valid(tmp_path):
    from PIL import Image

    signal = np.random.default_rng(0).normal(0, 0.3, (load.WINDOW_LENGTH, load.NUM_LEADS)).astype(np.float32)
    paths, errors = {}, []

    def worker(i):
        try:
            paths[i] = pred.plot_ecg_array(signal, tmp_path, f"rec{i}")
        except Exception as exc:                                      # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join(timeout=180) for t in threads]
    assert not errors and len(paths) == 4
    sizes = set()
    for p in paths.values():
        with Image.open(p) as img:
            sizes.add(img.size)
    assert len(sizes) == 1                                            # ninguna figura quedó mezclada o truncada
