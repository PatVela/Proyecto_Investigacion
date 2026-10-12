"""audit_dx_codes: cada .hea debe contarse una sola vez (regresión del doble conteo)."""

import audit_dx_codes as audit


def _hea(path, dx):
    path.write_text(f"{path.stem} 12 500 5000\n#Dx: {dx}\n", encoding="utf-8")


def test_each_header_counted_once(tmp_path):
    sub = tmp_path / "g1"
    sub.mkdir()
    _hea(tmp_path / "A0001.hea", "426783006,164934002")
    _hea(sub / "A0002.hea", "426783006")
    n_hea, n_with_dx, counter = audit.scan_headers(tmp_path)
    assert n_hea == 2
    assert n_with_dx == 2
    assert counter["426783006"] == 2
    assert counter["164934002"] == 1


def test_uppercase_extension_counted_once_and_not_duplicated(tmp_path):
    _hea(tmp_path / "B0001.HEA", "426783006")
    _hea(tmp_path / "B0002.hea", "426783006")
    n_hea, _, counter = audit.scan_headers(tmp_path)
    assert n_hea == 2
    assert counter["426783006"] == 2


def test_header_without_dx_is_listed_but_not_counted_as_labelled(tmp_path):
    (tmp_path / "C0001.hea").write_text("C0001 12 500 5000\n", encoding="utf-8")
    n_hea, n_with_dx, counter = audit.scan_headers(tmp_path)
    assert (n_hea, n_with_dx, sum(counter.values())) == (1, 0, 0)
