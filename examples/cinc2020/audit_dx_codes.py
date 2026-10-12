"""Auditoría de cobertura Dx→clase para CINC2020-12.

Recorre todos los .hea, cuenta en cuántos registros aparece cada código
Dx y lo cruza con (a) el esquema de 12 clases (ecg/load.py) y (b) el
mapeo oficial de 27 clases puntuadas (official/dx_mapping_scored.csv).

Sirve para verificar que cada código del esquema existe realmente en
los encabezados y para detectar códigos Dx sin clase asignada.

Uso:
    python examples/cinc2020/audit_dx_codes.py --data_dir dataset2020 --output dx_code_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from collections import Counter

from ecg.load import CLASS_GROUPS_12

DX_RE = re.compile(r"^#\s*Dx\s*:\s*(.*?)\s*$")


def load_official(path: Path) -> dict[str, str]:
    """Código SNOMED → abreviatura oficial (27 puntuados)."""
    mapping: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("SNOMED CT Code") or "").strip()
            if code:
                mapping[code] = (row.get("Abbreviation") or "").strip()
    return mapping


def scan_headers(data_dir: Path) -> tuple[int, int, Counter]:
    """Devuelve (n_hea, n_con_dx, contador código→n_registros)."""
    hea_files: list[Path] = sorted(data_dir.rglob("*.hea")) + sorted(data_dir.rglob("*.HEA"))
    counter: Counter = Counter()
    n_with_dx = 0
    for hea in hea_files:
        try:
            text = hea.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        codes: set[str] = set()
        for line in text.splitlines():
            match = DX_RE.match(line)
            if match:
                codes.update(c.strip() for c in match.group(1).split(",") if c.strip())
        if codes:
            n_with_dx += 1
            counter.update(codes)
    return len(hea_files), n_with_dx, counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita códigos Dx vs esquema de 12 clases.")
    parser.add_argument("--data-dir", default="dataset2020", help="Carpeta raíz con los .hea")
    parser.add_argument("--output", default="dx_code_audit.csv", help="CSV de salida")
    parser.add_argument("--official", default=str(Path(__file__).parent / "official" / "dx_mapping_scored.csv"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"No existe --data_dir: {data_dir}")

    official = load_official(Path(args.official))
    schema_class: dict[str, str] = {}
    for group in CLASS_GROUPS_12:
        for code in group["codes"]:
            schema_class[str(code)] = group["name"]

    n_hea, n_with_dx, counter = scan_headers(data_dir)
    if n_hea == 0:
        raise SystemExit(f"Sin .hea en {data_dir}")

    out_path = Path(args.output)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["snomed_code", "n_records", "frac_records", "official_abbrev", "assigned_class"])
        for code, n in counter.most_common():
            writer.writerow([code, n, f"{n / n_hea:.4f}", official.get(code, "EXTRA"),
                             schema_class.get(code, "SIN_COBERTURA")])

    n_official_seen = sum(1 for c in official if counter.get(c, 0) > 0)
    zero_schema = sorted((c for c in schema_class if counter.get(c, 0) == 0),
                         key=lambda c: (schema_class[c], c))
    uncovered = [(c, n) for c, n in counter.most_common() if c not in schema_class]

    print(f".hea escaneados: {n_hea} ({n_with_dx} con Dx válido)")
    print(f"Códigos Dx distintos: {len(counter)}")
    print(f"Oficiales puntuados presentes en datos: {n_official_seen}/{len(official)}")
    missing_official = [f"{c} ({official[c]})" for c in official if counter.get(c, 0) == 0]
    if missing_official:
        print(f"  Ausentes: {', '.join(missing_official)}")
    print(f"Códigos del esquema con frecuencia 0: {len(zero_schema)}")
    for code in zero_schema:
        tag = official.get(code) or "EXTRA"
        print(f"  {code} (clase {schema_class[code]}, {tag})")
    print(f"Códigos Dx sin clase asignada: {len(uncovered)}")
    for code, n in uncovered[:10]:
        print(f"  {code}: {n} registros")
    print(f"CSV: {out_path}")


if __name__ == "__main__":
    main()
