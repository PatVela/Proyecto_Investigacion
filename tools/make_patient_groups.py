"""Escribe el CSV de grupos de paciente (``record_name``, ``source_db``, ``group_id``) de PTB-XL y PTB para un reparto por paciente.

Reutiliza la lógica de ``tools/check_patient_overlap.py``, incluida su comprobación de que los nombres del Challenge corresponden a los
identificadores de PhysioNet (se niega a escribir si no se cumple). Los demás orígenes no traen identificador de paciente y no aparecen
en el CSV: ``build_datasets.py --groups-csv`` los trata como un grupo por registro.

Entradas que no se versionan (de PhysioNet): ``ptbxl_database.csv`` de PTB-XL 1.0.3 y ``RECORDS`` de PTB 1.0.0.

Uso (desde la raíz)::

    python tools/make_patient_groups.py --ptbxl-database data/externo/ptbxl_database.csv --ptb-records data/externo/RECORDS
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_patient_overlap as cpo  # noqa: E402

DEFAULT_OUTPUT = cpo.REPO_ROOT / "data" / "externo" / "patient_groups.csv"


def build_groups(assignments: list[dict], patient_of) -> list[tuple[str, str, str]]:
    """``(record_name, source_db, group_id)`` de cada registro de PTB-XL o PTB cuyo paciente se conoce."""
    rows = []
    for item in assignments:
        if item["origin"] not in cpo.ORIGINS:
            continue
        patient = patient_of(item["record"], item["origin"])
        if patient is not None:
            rows.append((item["record"], item["origin"], f"{item['origin']}:{patient}"))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSV de grupos de paciente de PTB-XL y PTB")
    parser.add_argument("--ptbxl-database", required=True, help="ptbxl_database.csv de PTB-XL 1.0.3")
    parser.add_argument("--ptb-records", required=True, help="RECORDS de PTB 1.0.0")
    parser.add_argument("--split-csv", default=str(cpo.DEFAULT_SPLIT), help="split_assignments.csv con la lista de registros y su origen")
    parser.add_argument("--challenge-dir", default=str(cpo.DEFAULT_CHALLENGE), help="carpeta training/ del Challenge 2020 (encabezados .hea)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    table, patients = cpo.ptbxl_table(args.ptbxl_database), cpo.ptb_patients(args.ptb_records)
    headers = {origin: {p.stem: p for p in (Path(args.challenge_dir) / origin).rglob("*.hea")} for origin in cpo.ORIGINS}
    mapping = {"ptb-xl": cpo.check_ptbxl_mapping(headers["ptb-xl"], table), "ptb": cpo.check_ptb_mapping(headers["ptb"], patients)}
    ptbxl_ok = mapping["ptb-xl"]["rate"] >= cpo.MIN_AGREEMENT
    ptb_ok = mapping["ptb"]["pairs"] > 0 and mapping["ptb"]["sex_rate"] >= cpo.MIN_AGREEMENT and mapping["ptb"]["age_rate"] >= 0.9
    if not (ptbxl_ok and ptb_ok):
        print("NO VERIFICADA: la correspondencia de nombres con los identificadores de paciente no se cumple; no se escribe el CSV.", file=sys.stderr)
        return 1

    def patient_of(record: str, origin: str):
        if origin == "ptb-xl":
            n = cpo.record_number(record, "HR")
            return table[n][0] if n in table else None
        return patients.get(cpo.record_number(record, "S"))

    rows = build_groups(cpo.read_assignments(Path(args.split_csv)), patient_of)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["record_name", "source_db", "group_id"])
        writer.writerows(rows)
    print(f"{len(rows)} registros con grupo de paciente ({len({r[2] for r in rows})} pacientes) -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
