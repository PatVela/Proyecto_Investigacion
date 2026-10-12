"""Esquema SNOMED agrupado a 12 clases (v2) y mapeo a vectores multilabel.

Extraído de ``ecg/load.py`` (split de depuración): tablas de códigos,
constantes de normalización y funciones de etiquetas. Sin dependencias
de PyTorch.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


# Esquema de etiquetas: 12 clases agrupadas por SNOMED-CT (v2).
#
# La v1 mapeaba ~40 códigos y dejaba ~21% de registros CINC2020 como vectores
# all-zero (LBBB, TInv, IRBBB, Brady genérica, NSSTTA, STD/STE, LQT, OldMI,
# isquemias, etc. quedaban sin clase), además de ignorar 13 de las 27 clases
# puntuadas oficiales del Challenge 2020. Eso introduce ruido de etiquetas
# masivo: ECG claramente anormales etiquetados como "nada", y clases como TAb
# entrenadas con TAb=0 para la mayoría de las anomalías ST-T reales.
#
# La v2 agrupa códigos clínicamente relacionados para cubrir las 27 clases
# puntuadas oficiales
# (https://github.com/physionetchallenges/evaluation-2020/blob/master/dx_mapping_scored.csv)
# manteniendo 12 salidas. All-zero esperado: <1% (solo diagnósticos
# no-ECG como HF/HVD/CHD o WPW aislado).
# ---------------------------------------------------------------------------
LABEL_SCHEMA = "cinc2020_12_grouped_snomed_v2"
LEGACY_LABEL_SCHEMA = "cinc2020_12_grouped_snomed"

CLASS_GROUPS_12 = [
    {
        "name": "NSR",
        "display_name": "Sinus rhythm (normal / arrhythmia)",
        "description": "Ritmo sinusal normal o arritmia sinusal benigna.",
        "codes": ["426783006", "427393009"],
    },
    {
        "name": "AxisDev",
        "display_name": "Cardiac axis deviation group",
        "description": "Desviación del eje (LAD/RAD/indeterminado) y bloqueos fasciculares.",
        "codes": ["39732003", "445118002", "47665007", "445211001", "251200008"],
    },
    {
        "name": "MI",
        "display_name": "Myocardial infarction group",
        "description": "Infarto (incluye antiguo/anterior/agudo), isquemia y Q anormal.",
        "codes": [
            "164865005", "57054005", "164867002", "54329005",
            "164861001", "413444003", "413844008", "426434006",
            "425419005", "425623009", "164917005",
        ],
    },
    {
        "name": "TAb",
        "display_name": "ST-T / repolarization abnormality group",
        "description": "Anomalías ST-T y de repolarización (TAb/TInv/NSSTTA/STD/STE/QT).",
        "codes": [
            "164934002", "59931005", "428750005", "429622005",
            "164931005", "164930006", "55930002", "704997005",
            "111975006", "77867006", "164937009", "251259000",
            "428417006",
        ],
    },
    {
        "name": "AF",
        "display_name": "Atrial arrhythmia group",
        "description": "Fibrilación/flutter auricular y variantes relacionadas.",
        "codes": [
            "164889003", "164890007", "195080001", "282825002",
            "426749004", "314208002",
        ],
    },
    {
        "name": "LVH",
        "display_name": "Ventricular hypertrophy / enlargement group",
        "description": "Hipertrofia/crecimiento ventricular y auricular, strain.",
        "codes": [
            "164873001", "370365005", "446813000", "67741000119109",
            "253352002", "195126007", "89792004", "266249003",
            "253339007", "446358003",
        ],
    },
    {
        "name": "VEctopy",
        "display_name": "Ventricular ectopy / tachycardia group",
        "description": "Ectopia ventricular y taquiarritmias ventriculares (TV/FV incluidas para no etiquetarlas como sanas).",
        "codes": [
            "427172004", "17338001", "164884008", "11157007",
            "251180001", "251182009", "75532003", "81898007",
            "164895002", "425856008", "164896001", "111288001",
            "49260003", "13640000",
        ],
    },
    {
        "name": "AVBlock",
        "display_name": "Atrioventricular / sinoatrial block group",
        "description": "Bloqueos AV (incluye PR prolongado) y sinoauriculares.",
        "codes": [
            "270492004", "195042002", "233917008", "27885002",
            "54016002", "204384007", "164947007", "65778007",
        ],
    },
    {
        "name": "STach",
        "display_name": "Sinus tachycardia",
        "description": "Taquicardia sinusal.",
        "codes": ["427084000"],
    },
    {
        "name": "BBB",
        "display_name": "Intraventricular conduction block group",
        "description": "Bloqueos de rama (der/izq, completos/incompletos), conducción inespecífica y QRS anormal/bajo voltaje.",
        "codes": [
            "59118001", "713427006", "713426002", "164909002",
            "251120003", "6374002", "698252002", "82226007",
            "251146004", "164951009",
        ],
    },
    {
        "name": "SB",
        "display_name": "Bradycardia group",
        "description": "Bradicardia sinusal/genérica, disfunción sinusal y síndrome bradi-taqui.",
        "codes": ["426177001", "426627000", "60423000", "74615001"],
    },
    {
        "name": "AEctopy_Junctional",
        "display_name": "Supraventricular rhythm group",
        "description": "Ectopia auricular/supraventricular, ritmos de unión, TSV y ritmos de marcapasos.",
        "codes": [
            "284470004", "63593006", "713422000", "426664006",
            "29320008", "426995002", "251164006", "426648003",
            "195101003", "251268003", "251170000", "251168009",
            "251173003", "426761007", "67198005", "10370003",
            "251266004",
        ],
    },
]

# Esquema v1 conservado solo para auditoría/migración de checkpoints antiguos.
CLASS_GROUPS_12_V1_LEGACY = [
    {"name": "NSR", "codes": ["426783006"]},
    {"name": "LAD", "codes": ["39732003"]},
    {"name": "MI", "codes": ["164865005"]},
    {"name": "TAb", "codes": ["164934002"]},
    {"name": "AF", "codes": ["164889003", "164890007", "195080001", "282825002", "426749004", "314208002"]},
    {"name": "LVH", "codes": ["164873001"]},
    {"name": "VEctopy", "codes": ["427172004", "17338001", "164884008", "11157007", "251180001", "251182009", "75532003", "81898007"]},
    {"name": "AVBlock", "codes": ["270492004", "195042002", "233917008", "27885002", "54016002", "204384007"]},
    {"name": "STach", "codes": ["427084000"]},
    {"name": "RBBB", "codes": ["59118001", "713427006"]},
    {"name": "SB", "codes": ["426177001"]},
    {"name": "AEctopy_Junctional", "codes": ["284470004", "63593006", "713422000", "426664006", "29320008", "426995002", "251164006", "426648003", "195101003", "251268003", "251170000", "251168009", "251173003"]},
]
LEGACY_CLASS_NAMES = [g["name"] for g in CLASS_GROUPS_12_V1_LEGACY]

# Las 27 clases puntuadas oficiales del Challenge 2020 (dx_mapping_scored.csv).
# Cada entrada: (código SNOMED, abreviatura). Útil para mapear grupos -> códigos
# oficiales en examples/cinc2020/challenge_score.py.
SCORED_27_CODES = [
    ("270492004", "IAVB"), ("164889003", "AF"), ("164890007", "AFL"),
    ("426627000", "Brady"), ("713427006", "CRBBB"), ("713426002", "IRBBB"),
    ("445118002", "LAnFB"), ("39732003", "LAD"), ("164909002", "LBBB"),
    ("251146004", "LQRSV"), ("698252002", "NSIVCB"), ("10370003", "PR"),
    ("284470004", "PAC"), ("427172004", "PVC"), ("164947007", "LPR"),
    ("111975006", "LQT"), ("164917005", "QAb"), ("47665007", "RAD"),
    ("59118001", "RBBB"), ("427393009", "SA"), ("426177001", "SB"),
    ("426783006", "NSR"), ("427084000", "STach"), ("63593006", "SVPB"),
    ("164934002", "TAb"), ("59931005", "TInv"), ("17338001", "VPB"),
]

CLASS_NAMES = [group["name"] for group in CLASS_GROUPS_12]
CLASS_DISPLAY_NAMES = [group["display_name"] for group in CLASS_GROUPS_12]
CLASS_DESCRIPTIONS = [group["description"] for group in CLASS_GROUPS_12]
NUM_CLASSES = len(CLASS_GROUPS_12)

CODE_TO_CLASS_INDEX: dict[str, int] = {}
for class_index, group in enumerate(CLASS_GROUPS_12):
    for code in group["codes"]:
        code = str(code).strip()
        if code in CODE_TO_CLASS_INDEX:
            raise ValueError(f"Código SNOMED duplicado en CLASS_GROUPS_12: {code}")
        CODE_TO_CLASS_INDEX[code] = class_index

CODE_TO_CLASS_NAME = {
    code: CLASS_NAMES[index]
    for code, index in CODE_TO_CLASS_INDEX.items()
}

EXCLUDED_GROUPS = [
    {
        "name": "WPW",
        "reason": "WPW/preexcitación (74390002, 195060002) es muy infrecuente en CINC2020 y morfológicamente singular; queda fuera del esquema v2.",
    },
    {
        "name": "NonECG_Diagnoses",
        "reason": "Diagnósticos clínicos no-ECG (p.ej. HF 84114007, HVD 368009, CHD 53741008, TIA 266257000) no describen morfología del trazado.",
    },
    {
        "name": "Noise",
        "reason": "Ruido no tiene código SNOMED-CT diagnóstico equivalente en el Challenge 2020.",
    },
]

# Normalización por defecto del esquema v2.
DEFAULT_NORM_MODE = "physical"
PHYSICAL_CLIP_MV = 5.0
DEFAULT_ADC_GAIN = 1000.0
BANDPASS_LO_HZ = 0.5
BANDPASS_HI_HZ = 50.0


# ---------------------------------------------------------------------------
# Etiquetas SNOMED -> vector multilabel.
# ---------------------------------------------------------------------------
def normalize_snomed_code(code) -> str | None:
    if code is None:
        return None
    text = str(code).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    text = text.replace(" ", "")
    return text if text else None


def parse_dx_field(raw_dx: str | Iterable[str] | None) -> list[str]:
    if raw_dx is None:
        return []
    parts = raw_dx.split(",") if isinstance(raw_dx, str) else list(raw_dx)
    codes, seen = [], set()
    for part in parts:
        code = normalize_snomed_code(part)
        if code is None or code in seen:
            continue
        seen.add(code)
        codes.append(code)
    return codes


def codes_to_vector(dx_codes: str | Iterable[str] | None) -> np.ndarray:
    """Mapea códigos SNOMED a vector multilabel binario de 12 clases."""
    y = np.zeros(NUM_CLASSES, dtype=np.float32)
    for code in parse_dx_field(dx_codes):
        index = CODE_TO_CLASS_INDEX.get(code)
        if index is not None:
            y[index] = 1.0
    return y


def matched_class_names(dx_codes: str | Iterable[str] | None) -> list[str]:
    vector = codes_to_vector(dx_codes)
    return [CLASS_NAMES[i] for i, active in enumerate(vector) if active > 0]


def class_codes_as_strings() -> list[str]:
    return [",".join(group["codes"]) for group in CLASS_GROUPS_12]


def class_groups_as_records() -> list[dict]:
    return [
        {
            "index": index,
            "class": group["name"],
            "display_name": group["display_name"],
            "description": group["description"],
            "snomed_codes": ",".join(group["codes"]),
        }
        for index, group in enumerate(CLASS_GROUPS_12)
    ]


# Backwards-friendly alias used by earlier drafts.
codes_to_12_vector = codes_to_vector


def preprocessing_for_schema(label_schema: str | None) -> dict:
    """Devuelve parámetros de preprocesamiento compatibles con un checkpoint.

    - Esquema v2 (o desconocido/nuevo): ``physical`` + bandpass.
    - Esquema v1 legacy: ``per_lead_zscore`` sin bandpass.

    Usar SIEMPRE esta función en inferencia (webapp/predict) para no mezclar
    un checkpoint v1 con preprocesamiento v2 (o viceversa), lo que produce
    predicciones basura silenciosas.
    """
    if label_schema == LEGACY_LABEL_SCHEMA:
        return {"norm_mode": "per_lead_zscore", "bandpass": False}
    return {"norm_mode": DEFAULT_NORM_MODE, "bandpass": True}


def is_legacy_schema(label_schema: str | None) -> bool:
    return str(label_schema or "") == LEGACY_LABEL_SCHEMA


# ---------------------------------------------------------------------------
# Lectores ECG / preprocesamiento de señal.
# ---------------------------------------------------------------------------
