"""
audit_signal_scaling.py

Auditoría de escala y metadatos de señales CINC2020.

NO modifica los datos.
NO genera HDF5.
NO normaliza señales.

Analiza:
- Frecuencia de muestreo
- Número de muestras
- Gain por derivación
- Baseline por derivación
- Unidades
- ADC / resolución
- Rangos digitales
- Rangos físicos reconstruidos
- Posibles saturaciones
- Señales constantes
- Estadísticas por base de datos

Uso desde la raíz del proyecto:

    python audit_signal_scaling.py

Opcional:

    python audit_signal_scaling.py --sample-per-db 20
    python audit_signal_scaling.py --sample-per-db 100
"""

from __future__ import annotations

import argparse
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.io
import wfdb


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset2020"

STANDARD_LEADS = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

DEFAULT_SAMPLE_PER_DB = 20


# ============================================================
# UTILIDADES
# ============================================================

def infer_source_db(header_path: Path) -> str:
    """
    Detecta la base CINC2020 a partir de la ruta.

    Ejemplo:
        dataset2020/training/ptb-xl/g1/A0001.hea
        -> ptb-xl
    """

    parts = [p.lower() for p in header_path.parts]

    if "training" in parts:
        idx = parts.index("training")

        if idx + 1 < len(parts):
            return header_path.parts[idx + 1]

    # Fallback
    if len(header_path.parts) >= 2:
        return header_path.parent.name

    return "unknown"


def normalize_lead_name(name: str) -> str | None:
    """
    Normaliza nombres de derivaciones WFDB.
    """

    if not name:
        return None

    x = name.strip().upper()

    aliases = {
        "I": "I",
        "II": "II",
        "III": "III",
        "AVR": "aVR",
        "AVL": "aVL",
        "AVF": "aVF",
        "V1": "V1",
        "V2": "V2",
        "V3": "V3",
        "V4": "V4",
        "V5": "V5",
        "V6": "V6",
    }

    return aliases.get(x)


def load_mat_signal(mat_path: Path) -> np.ndarray:
    """
    Carga la matriz 'val' del .mat.

    Devuelve:
        shape = (leads, samples)
    """

    mat = scipy.io.loadmat(mat_path)

    if "val" not in mat:
        raise ValueError("No existe matriz 'val'")

    signal = np.asarray(mat["val"])

    if signal.ndim != 2:
        raise ValueError(
            f"Forma inesperada: {signal.shape}"
        )

    return signal


def physical_from_digital(
    digital: np.ndarray,
    gain: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    """
    Reconstruye aproximadamente la señal física:

        physical = (digital - baseline) / gain

    digital:
        (leads, samples)

    gain:
        (leads,)

    baseline:
        (leads,)
    """

    gain_safe = gain.astype(np.float64).copy()

    gain_safe[gain_safe == 0] = np.nan

    return (
        digital.astype(np.float64)
        - baseline[:, None]
    ) / gain_safe[:, None]


def percentile_stats(x: np.ndarray) -> dict:
    """
    Estadísticas robustas de una señal.
    """

    finite = x[np.isfinite(x)]

    if finite.size == 0:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "p01": np.nan,
            "p99": np.nan,
        }

    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p99": float(np.percentile(finite, 99)),
    }


def is_constant(signal: np.ndarray) -> bool:
    """
    Detecta señal completamente constante.
    """

    return bool(
        np.all(signal == signal.flat[0])
    )


def saturation_fraction(
    signal: np.ndarray,
    low: int = -32768,
    high: int = 32767,
) -> float:
    """
    Porcentaje de muestras en los límites típicos int16.
    """

    total = signal.size

    if total == 0:
        return 0.0

    saturated = np.count_nonzero(
        (signal <= low) | (signal >= high)
    )

    return saturated / total


# ============================================================
# HEADER
# ============================================================

def read_header(header_path: Path):
    """
    Lee header WFDB.
    """

    record_without_ext = header_path.with_suffix("")

    record = wfdb.rdheader(
        str(record_without_ext)
    )

    return record


# ============================================================
# ANÁLISIS DE UN REGISTRO
# ============================================================

class _RecordAnalyzer:
    """Analiza un registro WFDB (.hea + .mat), una fase por método."""

    def __init__(self, header_path):
        self.header_path = header_path
        self.result = None
        self.record = None
        self.mat_path = None
        self.digital = None
        self.fs = None
        self.n_sig = None
        self.sig_len = None
        self.signal_names = None
        self.gains = None
        self.baselines = None
        self.units = None
        self.adc_res = None
        self.adc_zero = None
        self.init_value = None
        self.physical = None
        self.digital_stats = None
        self.physical_stats = None
        self.sat_frac = None
        self.constant = None
        self.leads_info = None

    def run(self):
        """Ejecuta todas las fases. Devuelve dict con metadatos y estadísticas."""
        self.result = {
            "header": self.header_path,
            "db": infer_source_db(self.header_path),
            "ok": False,
            "error": None,
        }

        try:
            self.load()
            self.parse_header()
            self.validate_dimensions()
            self.compute_physical()
            self.compute_global_stats()
            self.compute_saturation()
            self.check_constant()
            self.build_leads()
            self.finalize()

        except Exception as exc:

            self.result["error"] = str(exc)

        return self.result


    def load(self):
        """Lee header y señal digital."""

        self.record = read_header(self.header_path)

        self.mat_path = self.header_path.with_suffix(".mat")

        if not self.mat_path.exists():
            raise FileNotFoundError(
                f"No existe {self.mat_path.name}"
            )

        self.digital = load_mat_signal(self.mat_path)


    def parse_header(self):
        """Header."""

        self.fs = float(self.record.fs)
        self.n_sig = int(self.record.n_sig)
        self.sig_len = int(self.record.sig_len)

        self.signal_names = list(self.record.sig_name)

        self.gains = np.asarray(
            self.record.adc_gain,
            dtype=np.float64,
        )

        self.baselines = np.asarray(
            self.record.baseline,
            dtype=np.float64,
        )

        self.units = list(self.record.units)

        self.adc_res = getattr(
            self.record,
            "adc_res",
            None,
        )

        self.adc_zero = getattr(
            self.record,
            "adc_zero",
            None,
        )

        self.init_value = getattr(
            self.record,
            "init_value",
            None,
        )


    def validate_dimensions(self):
        """Comprobaciones."""

        if self.digital.shape[0] != self.n_sig:
            raise ValueError(
                f"MAT tiene {self.digital.shape[0]} leads, "
                f"header indica {self.n_sig}"
            )

        if self.digital.shape[1] != self.sig_len:
            raise ValueError(
                f"MAT tiene {self.digital.shape[1]} muestras, "
                f"header indica {self.sig_len}"
            )

        if len(self.gains) != self.n_sig:
            raise ValueError(
                "Número de gains incompatible"
            )

        if len(self.baselines) != self.n_sig:
            raise ValueError(
                "Número de baselines incompatible"
            )


    def compute_physical(self):
        """Señal física."""

        self.physical = physical_from_digital(
            self.digital,
            self.gains,
            self.baselines,
        )


    def compute_global_stats(self):
        """Estadísticas globales."""

        self.digital_stats = percentile_stats(
            self.digital
        )

        self.physical_stats = percentile_stats(
            self.physical
        )


    def compute_saturation(self):
        """Saturación."""

        self.sat_frac = saturation_fraction(
            self.digital
        )


    def check_constant(self):
        """Señales constantes."""

        self.constant = is_constant(
            self.digital
        )


    def build_leads(self):
        """Derivaciones."""

        self.leads_info = []

        for i in range(self.n_sig):

            d = self.digital[i]
            p = self.physical[i]

            ds = percentile_stats(d)
            ps = percentile_stats(p)

            lead_info = {
                "index": i,
                "name": self.signal_names[i]
                if i < len(self.signal_names)
                else f"lead_{i}",

                "normalized_name":
                    normalize_lead_name(
                        self.signal_names[i]
                    )
                    if i < len(self.signal_names)
                    else None,

                "gain": float(self.gains[i]),

                "baseline": float(
                    self.baselines[i]
                ),

                "units":
                    self.units[i]
                    if i < len(self.units)
                    else None,

                "adc_res":
                    self.adc_res[i]
                    if self.adc_res is not None
                    else None,

                "adc_zero":
                    self.adc_zero[i]
                    if self.adc_zero is not None
                    else None,

                "init_value":
                    self.init_value[i]
                    if self.init_value is not None
                    else None,

                "digital": ds,
                "physical": ps,

                "constant": bool(
                    np.all(d == d[0])
                ),

                "saturation_fraction":
                    saturation_fraction(d),
            }

            self.leads_info.append(
                lead_info
            )


    def finalize(self):
        """Agrega resultados al dict."""

        self.result.update(
            {
                "ok": True,
                "fs": self.fs,
                "n_sig": self.n_sig,
                "sig_len": self.sig_len,
                "signal_names": self.signal_names,
                "gains": self.gains,
                "baselines": self.baselines,
                "units": self.units,
                "adc_res": self.adc_res,
                "adc_zero": self.adc_zero,
                "init_value": self.init_value,
                "digital": self.digital_stats,
                "physical": self.physical_stats,
                "constant": self.constant,
                "saturation_fraction": self.sat_frac,
                "leads": self.leads_info,
            }
        )



def analyze_record(header_path: Path) -> dict:
    """
    Analiza un registro completo.

    Devuelve metadatos del header y estadísticas
    de la señal digital/física.
    """

    return _RecordAnalyzer(
        header_path
    ).run()




# ============================================================
# FORMATO DE SALIDA
# ============================================================

def print_separator(char="=", width=78):
    print(char * width)


def print_record_summary(result: dict):
    """
    Imprime resumen de un registro.
    """

    print()
    print_separator("-")

    print(
        f"Registro : "
        f"{result['header'].stem}"
    )

    print(
        f"Base     : "
        f"{result['db']}"
    )

    print(
        f"Fs       : "
        f"{result['fs']} Hz"
    )

    print(
        f"Muestras : "
        f"{result['sig_len']}"
    )

    print(
        f"Leads    : "
        f"{result['n_sig']}"
    )

    print(
        f"Digital  : "
        f"[{result['digital']['min']:.2f}, "
        f"{result['digital']['max']:.2f}]"
    )

    print(
        f"Físico   : "
        f"[{result['physical']['min']:.6g}, "
        f"{result['physical']['max']:.6g}]"
    )

    print(
        f"Std dig. : "
        f"{result['digital']['std']:.4f}"
    )

    print(
        f"Std fís. : "
        f"{result['physical']['std']:.6g}"
    )

    print(
        f"Constante: "
        f"{'SÍ' if result['constant'] else 'NO'}"
    )

    print(
        f"Saturación int16: "
        f"{result['saturation_fraction'] * 100:.4f}%"
    )

    print()

    print(
        f"{'Lead':<8}"
        f"{'Gain':>12}"
        f"{'Baseline':>12}"
        f"{'Units':>10}"
        f"{'Dig min':>12}"
        f"{'Dig max':>12}"
        f"{'Phys min':>14}"
        f"{'Phys max':>14}"
    )

    print_separator("-")

    for lead in result["leads"]:

        d = lead["digital"]
        p = lead["physical"]

        print(
            f"{lead['name']:<8}"
            f"{lead['gain']:>12.4f}"
            f"{lead['baseline']:>12.2f}"
            f"{str(lead['units']):>10}"
            f"{d['min']:>12.2f}"
            f"{d['max']:>12.2f}"
            f"{p['min']:>14.6g}"
            f"{p['max']:>14.6g}"
        )


# ============================================================
# RESUMEN POR BASE
# ============================================================

def print_database_summary(results_by_db):

    print()
    print_separator()

    print("RESUMEN POR BASE DE DATOS")

    print_separator()

    print(
        f"{'Base':<22}"
        f"{'N':>7}"
        f"{'Fs':>15}"
        f"{'Gain':>18}"
        f"{'Units':>15}"
        f"{'Const.':>9}"
        f"{'Sat.':>10}"
    )

    print_separator("-")

    for db in sorted(results_by_db):

        results = results_by_db[db]

        valid = [
            r for r in results
            if r["ok"]
        ]

        if not valid:
            continue

        fs_values = Counter(
            round(r["fs"], 3)
            for r in valid
        )

        gain_values = []

        units_values = []

        constant_count = 0

        saturation_values = []

        for r in valid:

            gain_values.extend(
                [
                    round(float(g), 6)
                    for g in r["gains"]
                    if np.isfinite(g)
                ]
            )

            units_values.extend(
                [
                    str(u)
                    for u in r["units"]
                ]
            )

            if r["constant"]:
                constant_count += 1

            saturation_values.append(
                r["saturation_fraction"]
            )

        common_fs = fs_values.most_common(1)[0][0]

        unique_gains = sorted(
            set(gain_values)
        )

        unique_units = sorted(
            set(units_values)
        )

        if len(unique_gains) > 6:
            gain_text = (
                f"{len(unique_gains)} valores"
            )
        else:
            gain_text = ",".join(
                f"{x:g}"
                for x in unique_gains
            )

        if len(unique_units) > 4:
            units_text = (
                f"{len(unique_units)} valores"
            )
        else:
            units_text = ",".join(
                unique_units
            )

        mean_sat = (
            statistics.mean(
                saturation_values
            )
            * 100
        )

        print(
            f"{db:<22}"
            f"{len(valid):>7}"
            f"{common_fs:>15g}"
            f"{gain_text:>18}"
            f"{units_text:>15}"
            f"{constant_count:>9}"
            f"{mean_sat:>9.4f}%"
        )


# ============================================================
# RESUMEN DE GAINS
# ============================================================

def print_gain_summary(results_by_db):

    print()
    print_separator()

    print("DISTRIBUCIÓN DE GAIN POR BASE")

    print_separator()

    for db in sorted(results_by_db):

        valid = [
            r for r in results_by_db[db]
            if r["ok"]
        ]

        if not valid:
            continue

        gains = []

        for r in valid:

            gains.extend(
                [
                    float(x)
                    for x in r["gains"]
                    if np.isfinite(x)
                ]
            )

        counter = Counter(
            round(g, 6)
            for g in gains
        )

        print()
        print(f"{db}:")

        for gain, count in counter.most_common(15):

            print(
                f"  gain={gain:<12g}"
                f" → {count} canales"
            )


# ============================================================
# UNIDADES
# ============================================================

def print_units_summary(results_by_db):

    print()
    print_separator()

    print("UNIDADES POR BASE")

    print_separator()

    for db in sorted(results_by_db):

        valid = [
            r for r in results_by_db[db]
            if r["ok"]
        ]

        if not valid:
            continue

        counter = Counter()

        for r in valid:
            counter.update(
                str(u)
                for u in r["units"]
            )

        print(
            f"{db}: "
            + ", ".join(
                f"{u} ({n})"
                for u, n in counter.items()
            )
        )


# ============================================================
# ALERTAS
# ============================================================

def print_warnings(results):

    print()
    print_separator()

    print("ALERTAS")

    print_separator()

    warnings = 0

    for r in results:

        if not r["ok"]:

            print(
                f"[ERROR] "
                f"{r['header'].stem}: "
                f"{r['error']}"
            )

            warnings += 1

            continue

        name = r["header"].stem

        # Señal constante
        if r["constant"]:

            print(
                f"[CONSTANTE] "
                f"{name} "
                f"({r['db']})"
            )

            warnings += 1

        # Saturación
        if r["saturation_fraction"] > 0:

            print(
                f"[SATURACIÓN] "
                f"{name} "
                f"({r['db']}): "
                f"{r['saturation_fraction'] * 100:.4f}%"
            )

            warnings += 1

        # Gain cero
        zero_gains = [
            i
            for i, g in enumerate(r["gains"])
            if g == 0
        ]

        if zero_gains:

            print(
                f"[GAIN=0] "
                f"{name}: "
                f"canales {zero_gains}"
            )

            warnings += 1

        # Unidades desconocidas
        unknown_units = [
            u
            for u in r["units"]
            if u is None or str(u).strip() == ""
        ]

        if unknown_units:

            print(
                f"[UNIDADES VACÍAS] "
                f"{name}"
            )

            warnings += 1

    if warnings == 0:

        print(
            "No se encontraron alertas "
            "en la muestra analizada."
        )

    print()
    print(
        f"Total de alertas: {warnings}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Audita gain, baseline, unidades "
            "y escala de señales CINC2020."
        )
    )

    parser.add_argument(
        "--sample-per-db",
        type=int,
        default=DEFAULT_SAMPLE_PER_DB,
        help=(
            "Número de registros a analizar "
            "por base de datos."
        ),
    )

    args = parser.parse_args()

    print_separator()

    print("AUDITORÍA DE ESCALA — CINC2020")

    print_separator()

    print(
        f"Proyecto : {PROJECT_ROOT}"
    )

    print(
        f"Dataset  : {DATASET_DIR}"
    )

    print(
        f"Muestra  : "
        f"{args.sample_per_db} registros/base"
    )

    print()

    if not DATASET_DIR.exists():

        print(
            f"ERROR: no existe:\n"
            f"{DATASET_DIR}"
        )

        return 1

    # --------------------------------------------------------
    # Buscar headers
    # --------------------------------------------------------

    print("Buscando archivos .hea...")

    headers = sorted(
        DATASET_DIR.rglob("*.hea")
    )

    print(
        f"Encontrados: {len(headers)}"
    )

    if not headers:

        print(
            "No se encontraron headers."
        )

        return 1

    # --------------------------------------------------------
    # Agrupar por base
    # --------------------------------------------------------

    by_db = defaultdict(list)

    for header in headers:

        db = infer_source_db(header)

        by_db[db].append(header)

    print()

    print(
        f"Bases encontradas: "
        f"{len(by_db)}"
    )

    for db in sorted(by_db):

        print(
            f"  {db:<25}"
            f"{len(by_db[db]):>8}"
        )

    # --------------------------------------------------------
    # Seleccionar muestra
    # --------------------------------------------------------

    selected = []

    rng = np.random.default_rng(2020)

    for db in sorted(by_db):

        db_headers = by_db[db]

        if len(db_headers) <= args.sample_per_db:

            chosen = db_headers

        else:

            indices = rng.choice(
                len(db_headers),
                size=args.sample_per_db,
                replace=False,
            )

            chosen = [
                db_headers[int(i)]
                for i in indices
            ]

        selected.extend(chosen)

    print()
    print(
        f"Registros seleccionados: "
        f"{len(selected)}"
    )

    # --------------------------------------------------------
    # Analizar
    # --------------------------------------------------------

    results = []

    print()
    print_separator()

    print("ANALIZANDO")

    print_separator()

    for idx, header in enumerate(
        selected,
        start=1,
    ):

        print(
            f"[{idx:>4}/{len(selected)}] "
            f"{infer_source_db(header):<20} "
            f"{header.stem}"
        )

        result = analyze_record(
            header
        )

        results.append(result)

    # --------------------------------------------------------
    # Agrupar resultados
    # --------------------------------------------------------

    results_by_db = defaultdict(list)

    for result in results:

        results_by_db[
            result["db"]
        ].append(result)

    # --------------------------------------------------------
    # Mostrar detalles
    # --------------------------------------------------------

    for result in results:

        if result["ok"]:

            print_record_summary(
                result
            )

    # --------------------------------------------------------
    # Resúmenes
    # --------------------------------------------------------

    print_database_summary(
        results_by_db
    )

    print_gain_summary(
        results_by_db
    )

    print_units_summary(
        results_by_db
    )

    print_warnings(
        results
    )

    # --------------------------------------------------------
    # Conclusión
    # --------------------------------------------------------

    print()
    print_separator()

    print("AUDITORÍA TERMINADA")

    print_separator()

    valid = sum(
        r["ok"]
        for r in results
    )

    errors = len(results) - valid

    print(
        f"Analizados : {len(results)}"
    )

    print(
        f"Correctos  : {valid}"
    )

    print(
        f"Errores    : {errors}"
    )

    print()

    print(
        "IMPORTANTE:"
    )

    print(
        "Este script NO modifica los datos "
        "ni aplica normalización."
    )

    print(
        "Los resultados deben utilizarse para "
        "decidir cómo modificar ecg/load.py."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )