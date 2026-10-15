"""Etiquetas reales: nombres de clase v1/v2, texto escrito a mano y comparación multilabel con las predicciones."""

from __future__ import annotations

import re
from typing import Iterable

from ecg import load

# Traducción de nombres v2 -> v1 para comparar contra checkpoints legacy.
V2_TO_V1_CLASS = {"AxisDev": "LAD", "BBB": "RBBB"}


def group_for_class(name: str) -> dict:
    for group in load.CLASS_GROUPS_12:
        if group["name"] == name:
            return group
    for group in load.CLASS_GROUPS_12_V1_LEGACY:
        if group["name"] == name:
            return {"name": name, "display_name": name, "description": "", "codes": group["codes"]}
    return {"name": name, "display_name": name, "description": "", "codes": []}


def translate_classes_to_checkpoint(names: Iterable[str], class_names: list[str]) -> list[str]:
    """Traduce clases v2 a nombres v1 si el checkpoint es legacy."""
    names = list(names)
    if "AxisDev" in class_names or "BBB" in class_names or "LAD" not in class_names:
        return names
    return [V2_TO_V1_CLASS.get(n, n) for n in names]


# ---------------------------------------------------------------------------
# Etiquetas reales / comparación multilabel
# ---------------------------------------------------------------------------
def split_reference_label_text(value: str | Iterable[str] | None) -> tuple[list[str], list[str]]:
    """``(clases, no_reconocidos)`` de un texto manual con clases o SNOMED (nombres del esquema de 12 clases).

    Lo que no se reconoce se devuelve en vez de descartarse: con ``AFF`` por ``AF`` la comparación seguía como si el
    registro no tuviera etiqueta, el AF acertado salía como falso positivo y el F1 como 0.0, sin ningún aviso.
    """
    if value is None:
        return [], []
    if isinstance(value, str):
        parts = re.split(r"[,;\n]+", value)
    else:
        parts = list(value)

    upper_to_name = {name.upper(): name for name in load.CLASS_NAMES}
    upper_to_name.update({"LAD": "AxisDev", "RBBB": "BBB"})  # alias legacy v1
    display_to_name = {group["display_name"].upper(): group["name"] for group in load.CLASS_GROUPS_12}
    found: list[str] = []
    unrecognized: list[str] = []
    seen = set()
    for part in parts:
        raw = str(part).strip()
        if not raw:
            continue
        code = load.normalize_snomed_code(raw)
        name = None
        if code in load.CODE_TO_CLASS_INDEX:
            name = load.CLASS_NAMES[load.CODE_TO_CLASS_INDEX[code]]
        else:
            key = raw.upper().replace(" ", "_")
            name = upper_to_name.get(key) or display_to_name.get(raw.upper())
        if not name:
            if raw not in unrecognized:
                unrecognized.append(raw)
        elif name not in seen:
            found.append(name)
            seen.add(name)
    return found, unrecognized


def _class_order(names: Iterable[str]) -> list[str]:
    order = {name: i for i, name in enumerate(load.CLASS_NAMES)}
    return sorted(set(names), key=lambda n: order.get(n, 10_000))


def build_label_comparison(
    true_classes: Iterable[str] | None,
    predicted_rows: list[dict],
    source: str | None,
    dx_codes: Iterable[str] | None = None,
    manual_raw: str | None = None,
) -> dict:
    """Compara predicciones finales positivas vs etiquetas reales dentro del esquema."""
    predicted = _class_order(row["class"] for row in predicted_rows if int(row.get("prediction", 0)) == 1)
    true = _class_order(true_classes or [])
    has_truth = source is not None
    if not has_truth:
        return {
            "available": False,
            "source": None,
            "reason": "No hay etiquetas reales disponibles. En .hea se leen desde el campo Dx; para CSV puede escribir clases o SNOMED manualmente.",
            "true_classes": [],
            "predicted_classes": predicted,
        }

    true_set = set(true)
    pred_set = set(predicted)
    tp = _class_order(true_set & pred_set)
    fp = _class_order(pred_set - true_set)
    fn = _class_order(true_set - pred_set)
    union = true_set | pred_set

    if not true_set and not pred_set:
        precision = recall = f1 = jaccard = 1.0
    elif not pred_set:
        precision = 0.0
        recall = 0.0 if true_set else 1.0
        f1 = 0.0
        jaccard = 0.0 if union else 1.0
    elif not true_set:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        jaccard = 0.0
    else:
        precision = len(tp) / len(pred_set)
        recall = len(tp) / len(true_set)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        jaccard = len(tp) / len(union) if union else 1.0

    return {
        "available": True,
        "source": source,
        "manual_raw": manual_raw or "",
        "dx_codes": list(dx_codes or []),
        "true_classes": true,
        "predicted_classes": predicted,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "exact_match": pred_set == true_set,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
    }
