"""ecg/datasets.py::build_external_test: HDF5 de solo prueba con todos los registros de un origen.

Se comprueba lo que importa para que una evaluación externa sea válida: sale el mismo preprocesamiento que en el reparto
habitual, una clase sin positivos no falla, los registros sin clase se conservan, las exclusiones se cuentan y un nombre
que no existe es un error.
"""

import h5py
import numpy as np
import pytest

from ecg import datasets, load
from ecg.util import decode_h5_strings

from test_dataset_build import _write_record


def _codes(*classes):
    return [load.CLASS_GROUPS_12[c]["codes"][0] for c in classes]


@pytest.fixture(scope="module")
def external(tmp_path_factory):
    """Dos orígenes: 'ext_a' con 6 registros (una clase sin positivos, uno sin clase) y 'ext_b' con 2."""
    base = tmp_path_factory.mktemp("externo")
    root = base / "dataset"
    for i, classes in enumerate([(0,), (1,), (2, 3), (0, 4), (5,), ()]):
        _write_record(root / "training" / "ext_a" / "g1", f"JS{i:05d}", _codes(*classes) or ["999999"])
    for i, classes in enumerate([(6,), (7,)]):
        _write_record(root / "training" / "ext_b" / "g1", f"JS{100 + i:05d}", _codes(*classes))
    return {"base": base, "root": root}


def _read(path):
    with h5py.File(path, "r") as f:
        return {"names": decode_h5_strings(f["record_names"][:]), "labels": f["labels"][:], "signals": f["signals"][:]}


def test_all_records_of_the_origin_are_written_without_requiring_positives_in_every_class(external):
    out = external["base"] / "a.h5"
    summary = datasets.build_external_test(external["root"], out, origin="ext_a", workers=1)
    data = _read(out)
    assert summary["records"] == 6 and len(data["names"]) == 6
    assert data["signals"].shape[1:] == (datasets.WINDOW_LENGTH, datasets.NUM_LEADS)
    assert summary["positives"][load.CLASS_NAMES[11]] == 0                       # clase sin positivos: no falla
    assert summary["records_without_any_class"] == 1                              # el registro sin clase se conserva
    assert (data["labels"].sum(axis=1) == 0).sum() == 1


def test_the_origin_filter_leaves_the_other_origin_out(external):
    out = external["base"] / "b.h5"
    summary = datasets.build_external_test(external["root"], out, origin="ext_b", workers=1)
    assert summary["records"] == 2 and set(decode_h5_strings(_read(out)["names"])) == {"JS00100", "JS00101"}


def test_the_signal_is_the_same_as_in_the_usual_split_hdf5(external, tmp_path):
    """Con la misma señal y las mismas opciones, la ventana es la que habría salido como 'test' en build_datasets."""
    out = external["base"] / "c.h5"
    datasets.build_external_test(external["root"], out, origin="ext_a", workers=1)
    records, _ = datasets.scan_records(external["root"])
    records = [r for r in records if r["source_db"] == "ext_a"]
    reference = tmp_path / "ref.h5"
    datasets.process_split_to_hdf5(records, range(len(records)), reference, num_workers=1, windows_max=1)
    a, b = _read(out), _read(reference)
    assert a["names"] == b["names"]
    np.testing.assert_array_equal(a["signals"], b["signals"])
    np.testing.assert_array_equal(a["labels"], b["labels"])


def test_excluded_records_are_left_out_counted_and_listed(external):
    out = external["base"] / "d.h5"
    summary = datasets.build_external_test(external["root"], out, origin="ext_a", exclude_records=["JS00003"], workers=1)
    assert summary["records"] == 5 and summary["excluded"] == ["JS00003"]
    assert "JS00003" not in set(decode_h5_strings(_read(out)["names"]))
    assert "JS00003" not in (external["base"] / "d.h5.records.csv").read_text()


def test_an_excluded_name_that_does_not_exist_is_an_error(external):
    with pytest.raises(ValueError, match="NOEXISTE"):
        datasets.build_external_test(external["root"], external["base"] / "e.h5", origin="ext_a", exclude_records=["NOEXISTE"], workers=1)


def test_an_unknown_origin_is_an_error(external):
    with pytest.raises(RuntimeError, match="origen"):
        datasets.build_external_test(external["root"], external["base"] / "f.h5", origin="no_existe", workers=1)


def test_the_cli_script_runs_and_writes_the_manifest(external):
    import subprocess, sys
    from pathlib import Path
    script = Path(datasets.__file__).resolve().parents[1] / "examples" / "cinc2020" / "build_external_test.py"
    out = external["base"] / "g.h5"
    proc = subprocess.run([sys.executable, "-X", "utf8", str(script), "--data-dir", str(external["root"]), "--origin", "ext_b",
                           "--output", str(out), "--workers", "1"], capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr[-800:]
    assert out.exists() and (external["base"] / "g.h5.records.csv").exists() and (external["base"] / "g.h5.summary.json").exists()
