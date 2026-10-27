"""Reparto por origen y por grupo de paciente (``ecg.datasets.grouped_origin_split``) y su lector de grupos."""

from __future__ import annotations

import numpy as np
import pytest

from ecg.datasets import grouped_origin_split, load_groups_csv
from ecg.schema import NUM_CLASSES


def _records(rng: np.random.Generator) -> tuple[list[dict], dict[str, str]]:
    """Tres orígenes: ``alfa`` con pacientes de varios registros, ``beta`` sin identificadores y ``gamma`` para retener."""
    records, groups = [], {}
    for origin, n in (("alfa", 240), ("beta", 160), ("gamma", 90)):
        for i in range(n):
            label = (rng.random(NUM_CLASSES) < 0.25).astype(np.int32)
            label[i % NUM_CLASSES] = 1                                     # todas las clases tienen positivos en cada origen
            name = f"{origin}{i:04d}"
            records.append({"record_name": name, "source_db": origin, "label": label})
            if origin == "alfa":
                groups[name] = f"alfa:{i // 3}"                           # 80 pacientes de 3 registros
    return records, groups


def _split(holdout=None, seed=42):
    records, groups = _records(np.random.default_rng(0))
    return records, groups, grouped_origin_split(records, groups, holdout, random_state=seed)


def test_ningun_grupo_queda_en_dos_particiones():
    records, groups, splits = _split("gamma")
    where = {}
    for name, idx in splits.items():
        for i in idx:
            where.setdefault(groups.get(records[i]["record_name"], records[i]["record_name"]), set()).add(name)
    assert all(len(v) == 1 for v in where.values())


def test_el_origen_retenido_va_entero_a_test_b_y_solo_el():
    records, _, splits = _split("gamma")
    assert {records[i]["source_db"] for i in splits["test_b"]} == {"gamma"}
    assert len(splits["test_b"]) == 90
    for name in ("train", "val", "test"):
        assert all(records[i]["source_db"] != "gamma" for i in splits[name])


def test_cubre_todos_los_registros_sin_repetir():
    records, _, splits = _split("gamma")
    todos = np.concatenate(list(splits.values()))
    assert sorted(todos) == list(range(len(records)))


def test_proporciones_cercanas_a_70_15_15_en_los_origenes_repartidos():
    records, _, splits = _split("gamma")
    total = sum(len(splits[k]) for k in ("train", "val", "test"))
    assert total == 400
    assert 0.62 <= len(splits["train"]) / total <= 0.78
    assert 0.08 <= len(splits["val"]) / total <= 0.22
    assert 0.08 <= len(splits["test"]) / total <= 0.22


def test_es_determinista_y_cambia_con_la_semilla():
    _, _, a = _split("gamma", 42)
    _, _, b = _split("gamma", 42)
    _, _, c = _split("gamma", 7)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["test"], c["test"])


def test_sin_origen_retenido_no_hay_test_b():
    _, _, splits = _split(None)
    assert "test_b" not in splits and len(np.concatenate(list(splits.values()))) == 490


def test_origen_retenido_inexistente_falla():
    records, groups = _records(np.random.default_rng(0))
    with pytest.raises(ValueError, match="no existe"):
        grouped_origin_split(records, groups, "delta")


def test_lector_de_grupos(tmp_path):
    ok = tmp_path / "g.csv"
    ok.write_text("record_name,source_db,group_id\nA1,alfa,alfa:1\nA2,alfa,alfa:1\n", encoding="utf-8")
    assert load_groups_csv(ok) == {"A1": "alfa:1", "A2": "alfa:1"}
    repetido = tmp_path / "r.csv"
    repetido.write_text("record_name,group_id\nA1,g\nA1,h\n", encoding="utf-8")
    with pytest.raises(ValueError, match="repetidos"):
        load_groups_csv(repetido)
    sin_columna = tmp_path / "c.csv"
    sin_columna.write_text("record_name\nA1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="faltan"):
        load_groups_csv(sin_columna)


def test_make_patient_groups_solo_incluye_ptbxl_y_ptb_con_paciente_conocido():
    from tools import make_patient_groups as mpg

    asignaciones = [{"record": "HR00001", "split": "train", "origin": "ptb-xl"}, {"record": "S0001", "split": "test", "origin": "ptb"},
                    {"record": "HR00002", "split": "val", "origin": "ptb-xl"}, {"record": "A0001", "split": "train", "origin": "cpsc_2018"}]
    pacientes = {"HR00001": 10, "S0001": "patient001"}
    filas = mpg.build_groups(asignaciones, lambda registro, origen: pacientes.get(registro))
    assert filas == [("HR00001", "ptb-xl", "ptb-xl:10"), ("S0001", "ptb", "ptb:patient001")]
