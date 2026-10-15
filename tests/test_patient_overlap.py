"""tools/check_patient_overlap.py: solape de pacientes entre train/val y test (T11), sobre datos sintéticos."""

import csv
import json

import numpy as np
import pytest

from tools import check_patient_overlap as cpo

PTBXL = [(1, 10, 50, 0), (2, 10, 50, 0), (3, 11, 60, 1), (4, 12, 70, 0), (5, 12, 70, 0), (6, 13, 40, 1)]   # ecg_id, paciente, edad, sexo
PTB_RECORDS = ["patient001/s0001_re", "patient001/s0001lre", "patient002/s0002_re", "patient003/s0003_re"]
SPLITS = {"HR00001": "train", "HR00002": "test", "HR00003": "test", "HR00004": "train", "HR00005": "val", "HR00006": "test", "HR00007": "test",
          "S0001": "train", "S0002": "test", "S0003": "test", "S0004": "train"}


def _header(path, age, sex):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{path.stem} 12 500 5000\n# Age: {age}\n# Sex: {sex}\n# Dx: 426783006\n", encoding="utf-8")


@pytest.fixture()
def data(tmp_path):
    database = tmp_path / "ptbxl_database.csv"
    database.write_text("ecg_id,patient_id,age,sex\n" + "".join(f"{e},{p}.0,{a},{s}\n" for e, p, a, s in PTBXL), encoding="utf-8")
    records = tmp_path / "RECORDS"
    records.write_text("\n".join(PTB_RECORDS) + "\n", encoding="utf-8")
    challenge = tmp_path / "training"
    for e, _, a, s in PTBXL:
        _header(challenge / "ptb-xl" / "g1" / f"HR{e:05d}.hea", a, "Male" if s == 0 else "Female")
    for k, (age, sex) in enumerate([(70, "Male"), (70, "Male"), (55, "Female"), (40, "Male")], start=1):
        _header(challenge / "ptb" / "g1" / f"S{k:04d}.hea", age, sex)
    split_csv = tmp_path / "split_assignments.csv"
    split_csv.write_text("split,index,record_name,source_db\n" + "".join(
        f"{split},{i},{name},{'ptb-xl' if name.startswith('HR') else 'ptb'}\n" for i, (name, split) in enumerate(SPLITS.items())), encoding="utf-8")
    return {"database": database, "records": records, "challenge": challenge, "split": split_csv, "tmp": tmp_path}


def _patient_of(data):
    table, patients = cpo.ptbxl_table(data["database"]), cpo.ptb_patients(data["records"])
    def patient_of(record, origin):
        if origin == "ptb-xl":
            n = cpo.record_number(record, "HR")
            return table[n][0] if n in table else None
        return patients[cpo.record_number(record, "S")]

    return patient_of


def test_overlap_counts_the_test_records_whose_patient_is_in_train(data):
    result = cpo.overlap(cpo.read_assignments(data["split"]), _patient_of(data))
    xl, ptb = result["ptb-xl"], result["ptb"]
    assert (xl["test"], xl["test_con_paciente_en_train"], xl["test_con_paciente_en_train_pct"]) == (3, 1, 33.33)
    assert xl["val_con_paciente_en_train"] == 1 and xl["pacientes"] == 4          # pacientes 10, 11, 12 y 13; el 12 está en train y en val
    assert (ptb["test"], ptb["test_con_paciente_en_train"], ptb["test_con_paciente_en_train_pct"]) == (2, 1, 50.0)


def test_test_records_with_an_unknown_patient_are_not_counted_as_overlapped(data):
    known, seen = cpo.overlapped_test_records(cpo.read_assignments(data["split"]), _patient_of(data), "ptb-xl")
    assert known == {"HR00002", "HR00003", "HR00006"}                    # HR00007 no está en la base: no es "paciente None" compartido
    assert seen == {"HR00002"}


def test_the_name_mapping_is_accepted_when_age_and_sex_agree_and_rejected_when_they_do_not(data):
    headers = {p.stem: p for p in (data["challenge"] / "ptb-xl").rglob("*.hea")}
    table = cpo.ptbxl_table(data["database"])
    assert cpo.check_ptbxl_mapping(headers, table)["rate"] == 1.0
    shuffled = {f"HR{n:05d}": headers[f"HR{(n % 6) + 1:05d}"] for n in range(1, 7)}      # cada nombre apunta al encabezado de otro registro
    assert cpo.check_ptbxl_mapping(shuffled, table)["rate"] < cpo.MIN_AGREEMENT
    ptb = cpo.check_ptb_mapping({p.stem: p for p in (data["challenge"] / "ptb").rglob("*.hea")}, cpo.ptb_patients(data["records"]))
    assert ptb == {"pairs": 1, "sex_rate": 1.0, "age_rate": 1.0}


def test_main_measures_and_writes_the_result_when_the_mapping_holds(data, capsys):
    out = data["tmp"] / "solape.json"
    code = cpo.main(["--ptbxl-database", str(data["database"]), "--ptb-records", str(data["records"]), "--split-csv", str(data["split"]),
                     "--challenge-dir", str(data["challenge"]), "--output", str(out)])
    assert code == 0 and "1/3 registros de test con paciente en train (33.33 %)" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["solape"]["ptb"]["test_con_paciente_en_train"] == 1


def test_main_refuses_to_measure_when_the_name_mapping_is_not_verified(data, capsys):
    _header(data["challenge"] / "ptb-xl" / "g1" / "HR00001.hea", 20, "Female")        # edad y sexo que no son los de ecg_id 1
    _header(data["challenge"] / "ptb-xl" / "g1" / "HR00002.hea", 21, "Female")
    code = cpo.main(["--ptbxl-database", str(data["database"]), "--ptb-records", str(data["records"]), "--split-csv", str(data["split"]),
                     "--challenge-dir", str(data["challenge"])])
    assert code == 1 and "NO VERIFICADA" in capsys.readouterr().err


def test_f1_is_computed_separately_for_records_with_and_without_overlap(data):
    path = data["tmp"] / "predictions_test.csv"
    rows = [("HR00002", (1, 0), (1, 0)), ("HR00003", (0, 1), (0, 0)), ("HR00006", (1, 1), (1, 1))]    # (verdad, decisión) de 2 clases
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_name", "true_A", "prob_A", "pred_A", "true_B", "prob_B", "pred_B"])
        for name, true, pred in rows:
            writer.writerow([name, true[0], 0.5, pred[0], true[1], 0.5, pred[1]])
    result = cpo.f1_with_and_without_overlap(path, overlapped={"HR00002"}, origin_records={"HR00002", "HR00003", "HR00006"}, bootstrap=50)
    assert (result["n_con_solape"], result["n_sin_solape"]) == (1, 2)
    assert result["f1_micro_con_solape"] == 1.0                                       # el único registro con solape se acertó entero
    assert result["f1_micro_sin_solape"] == pytest.approx(round(2 * 2 / (2 * 2 + 0 + 1), 4))   # TP=2, FP=0, FN=1
    assert len(result["f1_macro_sin_menos_con_ic95"]) == 2


def test_f1_helper_ignores_classes_without_positives():
    true = np.array([[1, 0], [0, 0]])
    pred = np.array([[1, 1], [0, 0]])
    macro, micro = cpo.macro_micro_f1(true, pred)
    assert macro == 1.0 and micro == pytest.approx(2 / 3)                              # la clase 2 no tiene positivos: no entra en el macro
