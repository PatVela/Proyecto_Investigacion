"""Núcleo de señal y esquema: ventanas, recorte, cabecera WFDB y mapeo SNOMED."""

import numpy as np
import pytest

from ecg import load, schema, signal


# --------------------------------------------------------------------------- #
# ventanas y longitud
# --------------------------------------------------------------------------- #
def test_window_starts_short_signal_is_single_window_at_zero():
    assert signal.compute_window_starts(3000, 5000, 4) == [0]
    assert signal.compute_window_starts(5000, 5000, 4) == [0]


def test_window_starts_single_window_is_centered():
    assert signal.compute_window_starts(8000, 5000, 1) == [1500]
    assert signal.compute_window_starts(5001, 5000, 1) == [0]


def test_window_starts_multi_window_cover_both_ends_and_respect_max():
    starts = signal.compute_window_starts(20000, 5000, 4)
    assert starts[0] == 0 and starts[-1] == 15000 and len(starts) == 4
    assert starts == sorted(set(starts))
    assert len(signal.compute_window_starts(20000, 5000, 2)) == 2
    assert len(signal.compute_window_starts(7000, 5000, 4)) <= 4


def test_window_starts_never_exceed_signal():
    for n in (5200, 9999, 12345, 30000):
        for k in (1, 2, 3, 4):
            assert all(0 <= s <= n - 5000 for s in signal.compute_window_starts(n, 5000, k))


def test_fix_length_center_crop_and_zero_pad():
    x = np.arange(8000 * 12, dtype=np.float32).reshape(8000, 12)
    cropped = signal.fix_length(x, 5000)
    assert cropped.shape == (5000, 12) and np.array_equal(cropped, x[1500:6500])
    short = signal.fix_length(x[:3000], 5000)
    assert short.shape == (5000, 12) and np.array_equal(short[:3000], x[:3000]) and not short[3000:].any()


# --------------------------------------------------------------------------- #
# cabecera WFDB: ganancia / baseline / unidades
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("line,expected", [
    ("A0001.mat 16+24 1000/mV 16 0 28 -1716 0 I", (1000.0, 0.0, "mV")),
    ("A0001.mat 16+24 1000.0(5)/mV 16 0 28 -1716 0 I", (1000.0, 5.0, "mV")),       # baseline explícito
    ("A0001.mat 16+24 200(-3)/uV 16 0 28 -1716 0 I", (200.0, -3.0, "uV")),
    ("A0001.mat 16+24 1000/mV 16 7 28 -1716 0 I", (1000.0, 7.0, "mV")),            # adc_zero si no hay baseline
    ("A0001.mat 16+24 1000.0(5)/mV 16 7 28 -1716 0 I", (1000.0, 5.0, "mV")),       # el explícito prevalece
    ("A0001.mat 16+24 0/mV 16 0 28 -1716 0 I", (1000.0, 0.0, "mV")),               # ganancia 0 -> defecto
    ("A0001.mat", (1000.0, 0.0, "mV")),                                           # línea truncada
])
def test_parse_adc_spec(line, expected):
    assert signal.parse_adc_spec(line) == expected


def test_to_physical_units_applies_baseline_and_gain():
    digital = np.array([[1000.0, 2000.0], [0.0, -1000.0]])
    mv = signal.to_physical_units(digital, adc_gains=[1000.0, 500.0], adc_baselines=[0.0, 1000.0])
    assert np.allclose(mv, [[1.0, 2.0], [0.0, -4.0]])
    with pytest.raises(ValueError):
        signal.to_physical_units(digital, adc_gains=[1000.0], adc_baselines=[0.0])


def test_reorder_leads_puts_standard_order_and_detects_missing():
    names = list(reversed(signal.STANDARD_LEAD_ORDER))
    x = np.tile(np.arange(12), (4, 1)).astype(np.float32)          # columna j = j
    out = signal.reorder_leads(x, names)
    assert np.array_equal(out[0], np.arange(12)[::-1])
    with pytest.raises(ValueError):
        signal.reorder_leads(x[:, :11], names[:11])


# --------------------------------------------------------------------------- #
# esquema SNOMED
# --------------------------------------------------------------------------- #
def test_every_scored_challenge_code_belongs_to_a_class():
    missing = [c for c, _ in schema.SCORED_27_CODES if c not in schema.CODE_TO_CLASS_INDEX]
    assert missing == []


def test_snomed_codes_are_unique_across_groups():
    all_codes = [c for g in schema.CLASS_GROUPS_12 for c in g["codes"]]
    assert len(all_codes) == len(set(all_codes))


def test_parse_dx_field_cleans_dedupes_and_ignores_garbage():
    assert schema.parse_dx_field("426783006, 164934002,426783006") == ["426783006", "164934002"]
    assert schema.parse_dx_field("426783006.0") == ["426783006"]
    assert schema.parse_dx_field("nan, ,None,") == []
    assert schema.parse_dx_field(None) == []
    assert schema.parse_dx_field(["59118001", "59118001"]) == ["59118001"]


def test_codes_to_vector_multilabel_and_unknown_codes():
    v = schema.codes_to_vector("426783006,164934002,123456789")
    assert v.shape == (12,) and v.dtype == np.float32
    assert v[schema.CLASS_NAMES.index("NSR")] == 1 and v[schema.CLASS_NAMES.index("TAb")] == 1
    assert v.sum() == 2                                              # el código desconocido no activa nada
    assert schema.codes_to_vector("").sum() == 0
    assert schema.matched_class_names("59118001,713427006") == ["BBB"]


def test_preprocessing_matches_checkpoint_schema():
    assert schema.preprocessing_for_schema(schema.LABEL_SCHEMA) == {"norm_mode": "physical", "bandpass": True}
    assert schema.preprocessing_for_schema(schema.LEGACY_LABEL_SCHEMA) == {"norm_mode": "per_lead_zscore", "bandpass": False}
    assert schema.is_legacy_schema(schema.LEGACY_LABEL_SCHEMA) and not schema.is_legacy_schema(schema.LABEL_SCHEMA)


def test_load_facade_exposes_the_same_schema():
    assert load.CLASS_NAMES == schema.CLASS_NAMES and load.LABEL_SCHEMA == schema.LABEL_SCHEMA


# --------------------------------------------------------------------------- #
# preprocesamiento según el esquema del checkpoint: sin fallbacks silenciosos
# --------------------------------------------------------------------------- #
def test_missing_schema_warns_and_assumes_current_preprocessing():
    for missing in (None, "", "   "):
        with pytest.warns(UserWarning, match="no declara label_schema"):
            assert schema.preprocessing_for_schema(missing) == {"norm_mode": "physical", "bandpass": True}


def test_unknown_schema_is_rejected_instead_of_guessed():
    with pytest.raises(ValueError, match="desconocido"):
        schema.preprocessing_for_schema("cinc2020_12_grouped_snomed_v3")


def test_known_schemas_do_not_warn(recwarn):
    schema.preprocessing_for_schema(schema.LABEL_SCHEMA)
    schema.preprocessing_for_schema(schema.LEGACY_LABEL_SCHEMA)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_save_checkpoint_always_stamps_the_real_schema(tmp_path):
    import torch

    from ecg import network, util

    path = tmp_path / "ckpt.pt"
    util.save_checkpoint(path, network.build_network(num_leads=12, num_classes=12), class_names=schema.CLASS_NAMES)
    assert util.load_checkpoint(path)["label_schema"] == schema.LABEL_SCHEMA


def test_checkpoint_without_schema_warns_through_predict_helper():
    from ecg import predict

    with pytest.warns(UserWarning):
        assert predict.preprocessing_for_checkpoint({"class_names": schema.CLASS_NAMES})["norm_mode"] == "physical"
    with pytest.raises(ValueError):
        predict.preprocessing_for_checkpoint({"label_schema": "otra_cosa"})
