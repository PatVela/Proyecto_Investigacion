"""Tests del esquema de etiquetas v2 (ecg/load.py)."""

from ecg import load


def test_twelve_groups_unique_names():
    assert load.NUM_CLASSES == 12
    assert len(load.CLASS_GROUPS_12) == 12
    names = [g["name"] for g in load.CLASS_GROUPS_12]
    assert len(set(names)) == 12
    assert names == list(load.CLASS_NAMES)


def test_groups_have_codes_and_display_names():
    for group in load.CLASS_GROUPS_12:
        assert group["display_name"], group["name"]
        assert len(group["codes"]) >= 1, group["name"]


def test_nsr_is_first_and_snomed_covered():
    assert load.CLASS_GROUPS_12[0]["name"] == "NSR"
    assert "426783006" in load.CLASS_GROUPS_12[0]["codes"]
    assert load.LABEL_SCHEMA == "cinc2020_12_grouped_snomed_v2"


def test_input_constants():
    assert load.NUM_LEADS == 12
    assert load.TARGET_FS == 500
    assert load.WINDOW_LENGTH == 5000
    assert load.WINDOW_SECONDS == 10
