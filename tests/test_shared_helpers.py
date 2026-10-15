"""Helpers compartidos (una sola definición) y dataset HDF5 unificado."""

import h5py
import numpy as np
import pytest
import torch

from ecg import hdf5_dataset, load, network, predict, train, util


# --------------------------------------------------------------------------- #
# una sola definición: train y predict reexportan, no copian
# --------------------------------------------------------------------------- #
def test_train_and_predict_share_single_implementations():
    assert train.get_device is util.get_device and predict.get_device is util.get_device
    assert train.model_params is network.params_from_config
    assert predict.model_params_from_config is network.params_from_config
    assert train.decode_h5_string is util.decode_h5_string
    assert predict.HDF5PredictionDataset is hdf5_dataset.HDF5ECGDataset
    assert issubclass(train.HDF5ECGDataset, hdf5_dataset.HDF5ECGDataset)


def test_decode_h5_string_variants():
    assert util.decode_h5_string(b"hola") == "hola"
    assert util.decode_h5_string("hola") == "hola"
    assert util.decode_h5_string(b"\xff") == "�"
    assert util.decode_h5_string(np.str_("x")) == "x"
    assert util.decode_h5_strings([b"a", "b", 3]) == ["a", "b", "3"]


def test_get_device():
    assert util.get_device("cpu").type == "cpu"
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    for requested in (None, "auto", "cuda"):
        assert util.get_device(requested).type == expected
    assert util.get_device("otro").type == "cpu"


def test_dataloader_kwargs_only_adds_worker_options_with_workers():
    plain = util.dataloader_kwargs(16, 0, True, shuffle=True)
    assert plain == {"batch_size": 16, "shuffle": True, "num_workers": 0, "pin_memory": True}
    workers = util.dataloader_kwargs("8", "3", False, prefetch_factor=5)
    assert workers["persistent_workers"] is True and workers["prefetch_factor"] == 5 and workers["num_workers"] == 3
    assert train._loader_kwargs({"batch": 4, "num_workers": 0}, shuffle=False)["batch_size"] == 4


def test_params_from_config_defaults_and_legacy_names():
    defaults = network.params_from_config({})
    assert defaults["num_classes"] == load.NUM_CLASSES and defaults["num_leads"] == 12
    assert defaults["input_length"] == load.WINDOW_LENGTH and defaults["conv_subsample_lengths"] == [1, 2] * 8
    legacy = network.params_from_config({"kernel_size": 15, "filter_length": 64, "drop_rate": 0.3})
    assert (legacy["conv_filter_length"], legacy["conv_num_filters_start"], legacy["conv_dropout"]) == (15, 64, 0.3)
    modern = network.params_from_config({"conv_filter_length": 9, "kernel_size": 15, "is_regular_conv": 1})
    assert modern["conv_filter_length"] == 9 and modern["is_regular_conv"] is True
    model = network.build_network(**network.params_from_config({}))
    assert model(torch.zeros(1, 12, 5000)).shape == (1, 12)


# --------------------------------------------------------------------------- #
# HDF5ECGDataset
# --------------------------------------------------------------------------- #
def _write_h5(path, n=6, labels=True, classes=12, names=True, length=5000):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("signals", data=rng.normal(size=(n, length, 12)).astype(np.float32))
        if labels:
            h5.create_dataset("labels", data=(rng.random((n, classes)) < 0.3).astype(np.float32))
        if names:
            h5.create_dataset("record_names", data=np.array([f"r{i}" for i in range(n)], dtype=object),
                              dtype=h5py.string_dtype())
        h5.attrs["label_schema"] = load.LABEL_SCHEMA
        h5.attrs["norm_mode"] = "physical"
        h5.attrs.create("classes", np.asarray(load.CLASS_NAMES, dtype=h5py.string_dtype()))
    return path


def test_in_ram_and_disk_modes_return_identical_items(tmp_path):
    p = _write_h5(tmp_path / "d.h5")
    ram, disk = hdf5_dataset.HDF5ECGDataset(p, preload=True), hdf5_dataset.HDF5ECGDataset(p, preload=False)
    assert len(ram) == len(disk) == 6 and ram._in_ram and not disk._in_ram
    for i in range(6):
        (xa, ya), (xb, yb) = ram[i], disk[i]
        assert xa.shape == (12, 5000) and ya.shape == (12,)
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_metadata_is_exposed_for_training_and_inference(tmp_path):
    ds = hdf5_dataset.HDF5ECGDataset(_write_h5(tmp_path / "d.h5"))
    assert ds.classes == load.CLASS_NAMES and ds.record_names[:2] == ["r0", "r1"]
    assert ds.label_schema == load.LABEL_SCHEMA and ds.norm_mode == "physical"
    assert ds.label_shape == (6, 12) and ds.signal_shape == (6, 5000, 12)


def test_missing_labels_allowed_for_inference_but_required_for_training(tmp_path):
    p = _write_h5(tmp_path / "nolab.h5", labels=False, names=False)
    ds = predict.HDF5PredictionDataset(p)
    x, y = ds[0]
    assert y.shape == (12,) and not y.any() and ds.label_shape is None
    assert ds.record_names[0] == "record_0"
    with pytest.raises(ValueError, match="labels"):
        train.HDF5ECGDataset(p)


def test_wrong_class_count_and_wrong_shape_fail_before_loading(tmp_path, capsys):
    with pytest.raises(ValueError, match="se esperaban 12"):
        train.HDF5ECGDataset(_write_h5(tmp_path / "c.h5", classes=7))
    assert "Precargando" not in capsys.readouterr().out         # valida antes de precargar
    with pytest.raises(ValueError, match="signals debe ser"):
        hdf5_dataset.HDF5ECGDataset(_write_h5(tmp_path / "s.h5", length=4000))
    with pytest.raises(FileNotFoundError):
        hdf5_dataset.HDF5ECGDataset(tmp_path / "nope.h5")


def test_training_dataset_augments_without_touching_stored_signals(tmp_path):
    p = _write_h5(tmp_path / "a.h5")
    plain = train.HDF5ECGDataset(p)
    aug = train.HDF5ECGDataset(p, augment_cfg={"aug_prob": 1.0, "aug_noise_std": 0.5, "aug_seed": 1})
    before = aug._signals.copy()
    assert not torch.equal(plain[0][0], aug[0][0])
    assert np.array_equal(aug._signals, before)
    assert torch.equal(plain[0][1], aug[0][1])               # las etiquetas no cambian
