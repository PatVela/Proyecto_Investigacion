"""Mide el solape de pacientes entre train/val y test en PTB-XL y PTB (tarea T11 de tasks/analisis_mejoras.md).

El reparto de ``data/cinc2020_12`` es por registro, y PTB-XL y PTB tienen varios registros por paciente: un paciente puede quedar
a ambos lados. Este script cuenta cuántos registros de test comparten paciente con train o val y, con las predicciones guardadas,
compara el F1 de los registros con y sin solape.

Entradas que no se versionan (de PhysioNet): ``ptbxl_database.csv`` de PTB-XL 1.0.3 y ``RECORDS`` de PTB 1.0.0. La correspondencia
con los nombres del Challenge es una hipótesis (``HR<n>`` = ``ecg_id`` n; ``S<k>`` = línea k de ``RECORDS``): el script la
comprueba con la edad y el sexo de los encabezados y se niega a medir si no se cumple.

Uso (desde la raíz)::

    python tools/check_patient_overlap.py --ptbxl-database data/externo/ptbxl_database.csv --ptb-records data/externo/RECORDS
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = REPO_ROOT / "data" / "cinc2020_12" / "split_assignments.csv"
DEFAULT_CHALLENGE = REPO_ROOT / "dataset2020" / "physionet.org" / "files" / "challenge-2020" / "1.0.2" / "training"
ORIGINS = {"ptb-xl": "HR", "ptb": "S"}          # origen -> prefijo del nombre de registro del Challenge
MIN_AGREEMENT = 0.99                             # por debajo, la correspondencia de nombres no se da por verificada


def record_number(name: str, prefix: str) -> int | None:
    match = re.fullmatch(prefix + r"(\d+)", name)
    return int(match.group(1)) if match else None


def ptbxl_table(database_csv: Path) -> dict[int, tuple[int, float | None, str | None]]:
    """``ecg_id -> (patient_id, edad, sexo)``; PTB-XL codifica el sexo como 0 = hombre y 1 = mujer."""
    table = {}
    with Path(database_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            age = float(row["age"]) if row.get("age") not in (None, "") else None
            sex = {"0": "male", "1": "female"}.get(str(row.get("sex", "")).strip().split(".")[0])
            table[int(float(row["ecg_id"]))] = (int(float(row["patient_id"])), age, sex)
    return table


def ptb_patients(records_file: Path) -> dict[int, str]:
    """``k -> paciente`` con k = número de línea (desde 1) de RECORDS, p. ej. ``patient001/s0010_re``."""
    lines = [line.strip() for line in Path(records_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {k: line.split("/")[0] for k, line in enumerate(lines, start=1)}


def header_age_sex(hea_path: Path) -> tuple[float | None, str | None]:
    age, sex = None, None
    for line in Path(hea_path).read_text(encoding="utf-8-sig").splitlines():
        text = line.strip().lstrip("#").strip().lower()
        if text.startswith("age:"):
            try:
                age = float(text.split(":", 1)[1])
            except ValueError:
                age = None
        elif text.startswith("sex:"):
            sex = text.split(":", 1)[1].strip() or None
    return age, sex


def read_assignments(split_csv: Path) -> list[dict]:
    with Path(split_csv).open(newline="", encoding="utf-8") as handle:
        return [{"record": r["record_name"], "split": r["split"], "origin": r["source_db"]} for r in csv.DictReader(handle)]


def check_ptbxl_mapping(headers: dict[str, Path], table: dict) -> dict:
    """Fracción de registros HR<n> cuya edad y sexo coinciden con los de ``ecg_id`` n (se omiten edades de 90 o más)."""
    compared = agree = 0
    for name, path in headers.items():
        n = record_number(name, "HR")
        if n not in table:
            continue
        _, age_db, sex_db = table[n]
        age, sex = header_age_sex(path)
        if age is None or sex is None or age_db is None or sex_db is None or age_db >= 90:
            continue
        compared += 1
        agree += int(abs(age - age_db) < 0.5 and sex == sex_db)
    return {"compared": compared, "agree": agree, "rate": agree / compared if compared else 0.0}


def check_ptb_mapping(headers: dict[str, Path], patients: dict[int, str]) -> dict:
    """Coherencia dentro de cada paciente de RECORDS: sus registros del Challenge deben compartir sexo (y casi siempre edad)."""
    groups = defaultdict(list)
    for name, path in headers.items():
        k = record_number(name, "S")
        if k in patients:
            groups[patients[k]].append(header_age_sex(path))
    pairs = same_sex = same_age = 0
    for members in groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                (a1, s1), (a2, s2) = members[i], members[j]
                if None in (a1, a2, s1, s2):
                    continue
                pairs += 1
                same_sex += int(s1 == s2)
                same_age += int(abs(a1 - a2) < 0.5)
    return {"pairs": pairs, "sex_rate": same_sex / pairs if pairs else 0.0, "age_rate": same_age / pairs if pairs else 0.0}


def overlap(assignments: list[dict], patient_of) -> dict:
    """Por origen: registros de test (y de val) cuyo paciente aparece en train (y en train o val)."""
    out = {}
    for origin in ORIGINS:
        rows = [(a["split"], patient_of(a["record"], origin)) for a in assignments if a["origin"] == origin]
        rows = [(split, p) for split, p in rows if p is not None]
        by_split = {s: {p for sp, p in rows if sp == s} for s in ("train", "val", "test")}
        test = [p for s, p in rows if s == "test"]
        val = [p for s, p in rows if s == "val"]
        n_test, n_val = len(test), len(val)
        out[origin] = {
            "registros": len(rows), "pacientes": len({p for _, p in rows}),
            "test": n_test, "test_con_paciente_en_train": sum(p in by_split["train"] for p in test),
            "test_con_paciente_en_train_o_val": sum(p in by_split["train"] | by_split["val"] for p in test),
            "val": n_val, "val_con_paciente_en_train": sum(p in by_split["train"] for p in val),
        }
        for key in ("test_con_paciente_en_train", "test_con_paciente_en_train_o_val"):
            out[origin][key + "_pct"] = round(100 * out[origin][key] / n_test, 2) if n_test else None
    return out


def overlapped_test_records(assignments: list[dict], patient_of, origin: str) -> tuple[set[str], set[str]]:
    """``(registros de test con paciente conocido, los que comparten paciente con train o val)``; los de paciente desconocido no cuentan."""
    patients = {a["record"]: patient_of(a["record"], origin) for a in assignments if a["origin"] == origin}
    split = {a["record"]: a["split"] for a in assignments if a["origin"] == origin}
    train_val = {p for r, p in patients.items() if split[r] in ("train", "val") and p is not None}
    known = {r for r, p in patients.items() if split[r] == "test" and p is not None}
    return known, {r for r in known if patients[r] in train_val}


def macro_micro_f1(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """F1-macro (solo clases con algún positivo en el subconjunto) y F1-micro."""
    tp, fp, fn = (true * pred).sum(0), ((1 - true) * pred).sum(0), (true * (1 - pred)).sum(0)
    present = true.sum(0) > 0
    macro = float(np.mean(2 * tp[present] / np.maximum(2 * tp[present] + fp[present] + fn[present], 1)))
    micro = float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1))
    return macro, micro


def f1_with_and_without_overlap(predictions_csv: Path, overlapped: set[str], origin_records: set[str], bootstrap: int = 1000, seed: int = 0) -> dict:
    """F1 de los registros de test de un origen con y sin paciente en train/val, y diferencia con IC95 bootstrap (macro)."""
    with Path(predictions_csv).open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["record_name"] in origin_records]
    if not rows:
        if not origin_records:                      # el origen no tiene registros de test conocidos: nada que comparar
            return {"n_con_solape": 0, "n_sin_solape": 0}
        raise ValueError(f"{predictions_csv} no trae ninguno de los {len(origin_records)} registros de test de este origen; "
                         "¿es el predictions_test.csv de esta partición?")
    classes = [c[5:] for c in rows[0] if c.startswith("true_")]
    true = np.array([[int(r[f"true_{c}"]) for c in classes] for r in rows])
    pred = np.array([[int(r[f"pred_{c}"]) for c in classes] for r in rows])
    mask = np.array([r["record_name"] in overlapped for r in rows])
    out = {"n_con_solape": int(mask.sum()), "n_sin_solape": int((~mask).sum())}
    for label, sel in (("con_solape", mask), ("sin_solape", ~mask)):
        if sel.sum():
            out["f1_macro_" + label], out["f1_micro_" + label] = (round(v, 4) for v in macro_micro_f1(true[sel], pred[sel]))
    if mask.sum() and (~mask).sum():
        rng, diffs = np.random.default_rng(seed), []
        idx_with, idx_without = np.flatnonzero(mask), np.flatnonzero(~mask)
        for _ in range(bootstrap):
            a, b = rng.choice(idx_with, idx_with.size), rng.choice(idx_without, idx_without.size)
            diffs.append(macro_micro_f1(true[b], pred[b])[0] - macro_micro_f1(true[a], pred[a])[0])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out["f1_macro_sin_menos_con_ic95"] = [round(float(lo), 4), round(float(hi), 4)]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Solape de pacientes entre train/val y test en PTB-XL y PTB")
    parser.add_argument("--ptbxl-database", required=True, help="ptbxl_database.csv de PTB-XL 1.0.3")
    parser.add_argument("--ptb-records", required=True, help="RECORDS de PTB 1.0.0")
    parser.add_argument("--split-csv", default=str(DEFAULT_SPLIT))
    parser.add_argument("--challenge-dir", default=str(DEFAULT_CHALLENGE), help="carpeta training/ del Challenge (encabezados .hea)")
    parser.add_argument("--predictions", default=None, help="predictions_test.csv de una evaluación, para comparar F1 con y sin solape")
    parser.add_argument("--output", default=None, help="JSON de salida")
    args = parser.parse_args(argv)

    table, patients = ptbxl_table(args.ptbxl_database), ptb_patients(args.ptb_records)
    assignments = read_assignments(args.split_csv)
    headers = {origin: {p.stem: p for p in (Path(args.challenge_dir) / origin).rglob("*.hea")} for origin in ORIGINS}
    mapping = {"ptb-xl": check_ptbxl_mapping(headers["ptb-xl"], table), "ptb": check_ptb_mapping(headers["ptb"], patients)}
    print("Correspondencia de identificadores:", json.dumps(mapping, ensure_ascii=False))
    ptbxl_ok = mapping["ptb-xl"]["rate"] >= MIN_AGREEMENT
    ptb_ok = mapping["ptb"]["pairs"] > 0 and mapping["ptb"]["sex_rate"] >= MIN_AGREEMENT and mapping["ptb"]["age_rate"] >= 0.9
    if not (ptbxl_ok and ptb_ok):
        print("NO VERIFICADA: no se mide el solape (la hipótesis de nombres no se cumple para "
              f"{', '.join(o for o, ok in (('PTB-XL', ptbxl_ok), ('PTB', ptb_ok)) if not ok)}).", file=sys.stderr)
        return 1

    def patient_of(record: str, origin: str):
        if origin == "ptb-xl":
            n = record_number(record, "HR")
            return table[n][0] if n in table else None
        k = record_number(record, "S")
        return patients.get(k)

    result = {"correspondencia": mapping, "solape": overlap(assignments, patient_of)}
    for origin, stats in result["solape"].items():
        print(f"{origin}: {stats['test_con_paciente_en_train']}/{stats['test']} registros de test con paciente en train "
              f"({stats['test_con_paciente_en_train_pct']} %); en train o val: {stats['test_con_paciente_en_train_o_val_pct']} %; "
              f"{stats['pacientes']} pacientes en {stats['registros']} registros")
    if args.predictions:
        result["f1"] = {}
        for origin in ORIGINS:
            known, seen = overlapped_test_records(assignments, patient_of, origin)
            try:
                result["f1"][origin] = f1_with_and_without_overlap(args.predictions, seen, known)
            except ValueError as exc:
                print(f"{origin}: {exc}", file=sys.stderr)
                return 1
            print(origin, json.dumps(result["f1"][origin], ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
