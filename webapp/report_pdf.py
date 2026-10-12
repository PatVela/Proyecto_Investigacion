"""Generación de PDF profesional para resultados de la webapp CINC2020-12."""

from __future__ import annotations

import html
import os
from datetime import datetime
from pathlib import Path


TITLE = "Informe de análisis automatizado de ECG de 12 derivaciones"
SUBTITLE = "Clasificación multilabel con red neuronal convolucional profunda (PhysioNet/CinC Challenge 2020)"
DISCLAIMER = (
    "Documento de apoyo generado automáticamente con fines académicos. "
    "No constituye diagnóstico médico ni sustituye el criterio de un profesional de la salud."
)
DISCLAIMER_SHORT = "Uso académico. No constituye diagnóstico médico."

try:
    from webapp.project_info import PROJECT_INFO as _PROJECT_INFO
except Exception:
    _PROJECT_INFO = {}


def _institution_lines() -> tuple[str, str]:
    if _PROJECT_INFO:
        line1 = str(_PROJECT_INFO.get("institucion", ""))
        line2 = " · ".join(p for p in [
            str(_PROJECT_INFO.get("facultad", "")),
            str(_PROJECT_INFO.get("escuela", "")),
        ] if p)
        return line1, line2
    return "ECG CINC2020-12", ""


def build_pdf_report(prediction: dict, output_path: str | Path) -> str:
    """Genera PDF profesional con veredicto, gráfico, tablas y trazado ECG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        return _build_reportlab_pdf(prediction, output_path)
    except Exception:
        return _build_minimal_pdf(prediction, output_path)


# ---------------------------------------------------------------------------
# Helpers de formato
# ---------------------------------------------------------------------------

def _esc(value) -> str:
    if value is None:
        return "—"
    return html.escape(str(value), quote=False)


def _basename(value) -> str:
    text = str(value or "").strip()
    if not text or text in {"—", "fallback_global_0.5"}:
        return text or "—"
    return os.path.basename(text.replace("\\", "/")) or text


def _fmt_float(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(value, digits: int = 2) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _human_input(value) -> str:
    mapping = {
        "wfdb_hea_mat": "Archivos WFDB (.hea + .mat)",
        "csv": "Archivo CSV de 12 derivaciones",
    }
    return mapping.get(str(value), str(value or "—"))


def _human_norm(value) -> str:
    mapping = {
        "physical": "Conversión a milivoltios + filtro pasa-banda 0.5–50 Hz + recorte ±5 mV",
        "global_zscore": "Normalización z global del registro",
        "per_lead_zscore": "Normalización z por derivación (esquema previo)",
    }
    return mapping.get(str(value), str(value or "—"))


def _human_schema(value) -> str:
    text = str(value or "")
    if "v2" in text:
        return "CINC2020-12 v2 (27 diagnósticos puntuados agrupados en 12 clases)"
    if "v1" in text or text == "cinc2020_12_grouped_snomed":
        return "CINC2020-12 v1 (esquema previo)"
    return text or "—"


def _human_device(value) -> str:
    return {"cuda": "GPU (CUDA)", "cpu": "CPU"}.get(str(value), str(value or "—"))


def _confidence_level(prob: float) -> str:
    if prob >= 0.80:
        return "ALTA"
    if prob >= 0.50:
        return "MODERADA"
    return "BAJA"


def _fmt_windows(prediction: dict) -> str:
    n = int(prediction.get("num_windows") or 1)
    secs = prediction.get("window_seconds", 10)
    text = f"{n} ventana de {secs} s" if n == 1 else f"{n} ventanas de {secs} s"
    if n > 1:
        text += f" · agregación {prediction.get('window_aggregation', 'max')}"
    return text


# ---------------------------------------------------------------------------
# PDF principal (reportlab)
# ---------------------------------------------------------------------------

def _build_reportlab_pdf(prediction: dict, output_path: Path) -> str:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfgen.canvas import Canvas as _Canvas
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    # Paleta institucional UNSA: granate + oro (verdes/rojos semánticos intactos)
    NAVY = colors.HexColor("#84030c")
    NAVY_DARK = colors.HexColor("#5e0208")
    GOLD = colors.HexColor("#cea962")
    RED = colors.HexColor("#b91c1c")
    GREEN = colors.HexColor("#15803d")
    GREEN_BG = colors.HexColor("#dcfce7")
    RED_BG = colors.HexColor("#fee2e2")
    GRAY_BG = colors.HexColor("#f1f5f9")
    AMBER_BG = colors.HexColor("#fef3c7")
    SLATE = colors.HexColor("#475569")
    GRID = colors.HexColor("#cbd5e1")
    BAR_BG = colors.HexColor("#e2e8f0")
    BAR_NEG = colors.HexColor("#94a3b8")
    BAR_POS = colors.HexColor("#dc2626")
    BAR_NSR = colors.HexColor("#16a34a")

    record_name = str(prediction.get("record_name") or "—")
    job_id = str(prediction.get("job_id") or "")
    folio = f"ECG-{job_id[:8].upper()}-{record_name}" if job_id else f"ECG-{record_name}"
    emission = datetime.now().strftime("%Y-%m-%d %H:%M")
    inst1, inst2 = _institution_lines()
    predictions = list(prediction.get("predictions") or [])
    positives = list(prediction.get("positive_predictions") or [])
    comparison = prediction.get("label_comparison") or {}
    details = prediction.get("technical_details") or {}

    # ---- estilos ----
    styles = getSampleStyleSheet()
    s_title = ParagraphStyle("rTitle", parent=styles["Title"], fontSize=17, leading=20,
                             textColor=NAVY, alignment=1, spaceAfter=2)
    s_sub = ParagraphStyle("rSub", parent=styles["Normal"], fontSize=9, leading=12,
                           textColor=SLATE, alignment=1, spaceAfter=6)
    s_h1 = ParagraphStyle("rH1", parent=styles["Heading2"], fontSize=12, leading=14,
                          textColor=NAVY, spaceBefore=12, spaceAfter=6,
                          borderPadding=(0, 0, 4), borderWidth=0)
    s_body = ParagraphStyle("rBody", parent=styles["BodyText"], fontSize=9, leading=12.5)
    s_small = ParagraphStyle("rSmall", parent=styles["BodyText"], fontSize=7.5, leading=10,
                             textColor=SLATE)
    s_cell = ParagraphStyle("rCell", parent=styles["BodyText"], fontSize=8.5, leading=11)
    s_verdict = ParagraphStyle("rVerdict", parent=styles["BodyText"], fontSize=10, leading=14)
    s_sign = ParagraphStyle("rSign", parent=styles["Normal"], fontSize=8.5, leading=12, alignment=1)

    content_w = A4[0] - 30 * mm
    story: list = []

    # ---- portada / título ----
    story.append(Paragraph(_esc(TITLE), s_title))
    story.append(Paragraph(_esc(SUBTITLE), s_sub))
    story.append(Table([[""]], colWidths=[content_w],
                       style=TableStyle([("LINEBELOW", (0, 0), (-1, 0), 1.5, GOLD)])))
    story.append(Spacer(1, 4))
    head_rows = [
        [Paragraph(f"<b>Folio:</b> {_esc(folio)}", s_cell),
         Paragraph(f"<b>Fecha de emisión:</b> {_esc(emission)}", s_cell)],
        [Paragraph(f"<b>Registro:</b> {_esc(record_name)}", s_cell),
         Paragraph(f"<b>Paciente:</b> {_esc(prediction.get('patient_name') or 'No registrado')}", s_cell)],
    ]
    story.append(Table(head_rows, colWidths=[content_w / 2, content_w / 2], hAlign="LEFT",
                       style=TableStyle([
                           ("BACKGROUND", (0, 0), (-1, -1), GRAY_BG),
                           ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                           ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID),
                           ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6),
                       ])))

    # ---- 1. Datos del estudio ----
    story.append(Paragraph("1. Datos del estudio", s_h1))
    dx_codes = prediction.get("dx_codes") or []
    matched = prediction.get("matched_classes") or []
    leads = prediction.get("lead_names") or []
    info_rows = [
        ("Edad / Sexo",
         f"{prediction.get('patient_age') or 'No registrada'} / {prediction.get('patient_sex') or 'No registrado'}"),
        ("Tipo de entrada", _human_input(prediction.get("input_type"))),
        ("Derivaciones", ", ".join(leads) if leads else "—"),
        ("Frecuencia de muestreo", f"{prediction.get('original_sampling_rate', '—')} Hz (origen) → "
                                   f"{prediction.get('target_sampling_rate', '—')} Hz (análisis)"),
        ("Ventanas analizadas", _fmt_windows(prediction)),
        ("Códigos Dx del header", ", ".join(dx_codes) if dx_codes else "No disponibles"),
        ("Clases reales (esquema)", ", ".join(matched) if matched else "Ninguna del esquema / no disponibles"),
    ]
    story.append(_kv_table(info_rows, content_w, NAVY, GRID, s_cell))

    # ---- 2. Impresión diagnóstica ----
    story.append(Paragraph("2. Impresión diagnóstica del modelo", s_h1))
    story.append(_verdict_box(prediction, positives, s_verdict, s_small,
                              GREEN, GREEN_BG, RED, RED_BG, AMBER_BG, GRAY_BG, GRID, SLATE, content_w))

    # ---- 3. Probabilidades ----
    story.append(Paragraph("3. Probabilidades por clase", s_h1))
    story.append(Paragraph(
        "Barras ordenadas de mayor a menor probabilidad. La marca vertical indica el umbral de decisión "
        "de cada clase; la barra se resalta cuando la probabilidad lo supera.", s_small))
    story.append(Spacer(1, 4))
    story.append(_probability_chart(predictions, content_w, BAR_BG, BAR_NEG, BAR_POS, BAR_NSR, SLATE))
    story.append(Spacer(1, 6))
    det_rows = [[Paragraph("<b><font color=\"white\">Clase</font></b>", s_cell), Paragraph("<b><font color=\"white\">Prob.</font></b>", s_cell),
                 Paragraph("<b><font color=\"white\">Umbral</font></b>", s_cell), Paragraph("<b><font color=\"white\">Margen</font></b>", s_cell),
                 Paragraph("<b><font color=\"white\">Decisión</font></b>", s_cell), Paragraph("<b><font color=\"white\">Obs.</font></b>", s_cell)]]
    for item in predictions:
        prob = float(item.get("probability", 0.0) or 0.0)
        thr = float(item.get("threshold", prediction.get("threshold", 0.5)) or 0.0)
        is_pos = bool(item.get("prediction"))
        decision = ('<font color="#b91c1c"><b>POSITIVO</b></font>' if is_pos
                    else '<font color="#64748b">negativo</font>')
        obs = "Cerca del umbral" if item.get("near_threshold") else ("Post hoc NSR" if item.get("postprocessed") else "")
        det_rows.append([
            Paragraph(f"<b>{_esc(item.get('class', ''))}</b>", s_cell),
            Paragraph(_fmt_pct(prob), s_cell),
            Paragraph(_fmt_pct(thr), s_cell),
            Paragraph(f"{(prob - thr) * 100:+.2f}%", s_cell),
            Paragraph(decision, s_cell),
            Paragraph(_esc(obs), s_cell),
        ])
    story.append(_data_table(det_rows, content_w, NAVY, GRID,
                             widths=[0.24, 0.13, 0.13, 0.13, 0.17, 0.20]))

    # ---- 4. Comparación con etiquetas reales ----
    story.append(Paragraph("4. Comparación con etiquetas reales", s_h1))
    if comparison.get("available"):
        source = {"wfdb_header_dx": "Campo Dx del header WFDB", "manual": "Ingreso manual"}.get(
            comparison.get("source"), str(comparison.get("source") or "—"))
        pred_list = comparison.get("predicted_classes") or []
        fp = comparison.get("false_positive") or []
        fn = comparison.get("false_negative") or []
        if comparison.get("exact_match"):
            interp = "Las clases predichas <b>coinciden exactamente</b> con las etiquetas reales disponibles."
        elif not pred_list and fn and not fp:
            interp = ("El modelo no activó ninguna clase existiendo etiqueta real: "
                      "se considera <b>falso negativo</b>.")
        elif fp and not fn:
            interp = ("El modelo activó clases ausentes en la etiqueta real: "
                      "se considera <b>falso positivo</b>.")
        else:
            interp = "Existen <b>diferencias parciales</b> (multilabel) frente a la etiqueta real."
        story.append(Paragraph(interp, s_body))
        story.append(Spacer(1, 4))
        comp_rows = [
            ("Fuente de etiquetas", source),
            ("Etiquetas reales", ", ".join(comparison.get("true_classes") or []) or "Ninguna"),
            ("Clases predichas", ", ".join(pred_list) or "Ninguna"),
            ("Aciertos", ", ".join(comparison.get("true_positive") or []) or "Ninguno"),
            ("Falsos positivos", ", ".join(fp) or "Ninguno"),
            ("Falsos negativos", ", ".join(fn) or "Ninguno"),
            ("Coincidencia exacta", "Sí" if comparison.get("exact_match") else "No"),
            ("F1 / Jaccard",
             f"{_fmt_float(comparison.get('f1'), 3)} / {_fmt_float(comparison.get('jaccard'), 3)}"),
        ]
        story.append(_kv_table(comp_rows, content_w, NAVY, GRID, s_cell))
    else:
        story.append(Paragraph(
            "No se proporcionaron etiquetas reales para este registro, por lo que no es posible "
            "contrastar la predicción. En archivos WFDB las etiquetas se leen automáticamente del "
            "campo <b>Dx</b>; en CSV pueden ingresarse manualmente.", s_body))

    # ---- 5. Trazado ----
    story.append(Paragraph("5. Trazado ECG analizado", s_h1))
    plot_path = prediction.get("plot_path")
    if plot_path and Path(plot_path).exists():
        img = Image(str(plot_path))
        max_w = content_w
        max_h = 118 * mm
        scale = min(max_w / img.imageWidth, max_h / img.imageHeight, 1.0)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        story.append(img)
        story.append(Paragraph(
            "Trazado de 12 derivaciones con retícula ECG (0.04 s por cuadro pequeño). "
            "Amplitud en milivoltios tras el preprocesamiento.", s_small))
    else:
        story.append(Paragraph("Trazado no disponible para este informe.", s_body))

    # ---- 6. Nota metodológica ----
    story.append(Paragraph("6. Nota metodológica", s_h1))
    ckpt_name = _basename(prediction.get("model_path"))
    thr_src = prediction.get("threshold_source") or "fallback_global_0.5"
    thr_src = "Umbral global 0.5 (sin calibración)" if thr_src == "fallback_global_0.5" else _basename(thr_src)
    fb = prediction.get("normal_fallback") or {}
    method_rows = [
        ("Modelo", f"{prediction.get('model_type', '—')} · {prediction.get('num_classes', 12)} clases (sigmoid)"),
        ("Esquema de etiquetas", _human_schema(prediction.get("label_schema"))),
        ("Preprocesamiento",
         f"12 derivaciones → orden estándar → 500 Hz → {_human_norm(prediction.get('norm_mode'))} → "
         f"{prediction.get('window_seconds', 10)} s / {prediction.get('input_length', 5000)} muestras"),
        ("Regla de decisión", str(details.get("decision_rule", "—"))),
        ("Umbrales", thr_src),
        ("Regla post hoc NSR",
         f"Aplicada (P(NSR) ≥ {fb.get('min_nsr_probability')})" if fb.get("applied")
         else "No aplicada" if fb.get("enabled", True) else "Desactivada"),
        ("Checkpoint", f"{ckpt_name} · época {prediction.get('checkpoint_epoch', '—')} · "
                       f"val_loss {_fmt_float(prediction.get('checkpoint_val_loss'))}"),
        ("Cómputo", _human_device(details.get("device"))),
    ]
    story.append(_kv_table(method_rows, content_w, NAVY, GRID, s_cell))

    # ---- Anexo ----
    story.append(Paragraph("Anexo A. Correspondencia de códigos SNOMED-CT", s_h1))
    anx_rows = [[Paragraph("<b><font color=\"white\">Clase</font></b>", s_cell), Paragraph("<b><font color=\"white\">Descripción</font></b>", s_cell),
                 Paragraph("<b><font color=\"white\">Códigos SNOMED-CT</font></b>", s_cell)]]
    for item in predictions:
        codes = ", ".join(item.get("snomed_codes", []) or [])
        anx_rows.append([
            Paragraph(f"<b>{_esc(item.get('class', ''))}</b>", s_cell),
            Paragraph(_esc(item.get("display_name", "")), s_cell),
            Paragraph(_esc(codes) if codes else "—", s_small),
        ])
    story.append(_data_table(anx_rows, content_w, NAVY, GRID, widths=[0.22, 0.33, 0.45]))

    # ---- Firmas ----
    story.append(Paragraph("7. Validación", s_h1))
    story.append(Paragraph(
        "Este informe fue generado automáticamente. La validación clínica corresponde a un profesional.",
        s_small))
    story.append(Spacer(1, 6))
    sig_style = ParagraphStyle("rSig", parent=s_sign, spaceBefore=18)
    sig_rows = [[
        Paragraph(f"Procesamiento automático<br/><br/>_________________________<br/>Folio { _esc(folio)}<br/>{_esc(emission)}", sig_style),
        Paragraph("Validador clínico<br/><br/>_________________________<br/>Nombre / Firma / Sello<br/>CMP: __________", sig_style),
    ]]
    story.append(Table(sig_rows, colWidths=[content_w / 2, content_w / 2], hAlign="CENTER"))

    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<i>{_esc(DISCLAIMER)}</i>", s_small))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=24 * mm,
        bottomMargin=16 * mm,
        title=f"{TITLE} — {record_name}",
        author=inst1 or "ECG CINC2020-12",
        subject=f"Folio {folio}",
    )
    doc.build(story, canvasmaker=_canvas_factory(folio, emission, record_name, inst1, inst2))
    return str(output_path)


def _kv_table(rows, content_w, navy, grid, cell_style):
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Table, TableStyle
    body = [[
        Paragraph(f"<b><font color=\"white\">{html.escape(str(k), quote=False)}</font></b>", cell_style),
        Paragraph(_esc(v), cell_style),
    ] for k, v in rows]
    table = Table(body, colWidths=[42 * mm, content_w - 42 * mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), navy),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, grid),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    return table


def _data_table(rows, content_w, navy, grid, widths):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle
    table = Table(rows, colWidths=[content_w * w for w in widths], hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, grid),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    return table


def _verdict_box(prediction, positives, s_verdict, s_small,
                 green, green_bg, red, red_bg, amber_bg, gray_bg, grid, slate, content_w):
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Table, TableStyle

    fb = prediction.get("normal_fallback") or {}
    if positives:
        names = [p.get("class", "") for p in positives]
        probs = [float(p.get("probability", 0.0) or 0.0) for p in positives]
        top = max(probs) if probs else 0.0
        level = _confidence_level(top)
        if set(names) == {"NSR"}:
            bg, border = green_bg, green
            title = '<font color="#15803d"><b>RITMO SINUSAL DENTRO DE LÍMITES NORMALES</b></font>'
            detail = (f"El modelo asigna <b>NSR {_fmt_pct(top)}</b> (confianza {level}). "
                      f"No se activó ninguna clase patológica sobre su umbral.")
        else:
            bg, border = (amber_bg, red) if any(n == "NSR" for n in names) else (red_bg, red)
            items = "<br/>".join(
                f"• <b>{html.escape(p.get('class', ''), quote=False)}</b> — "
                f"{html.escape(p.get('display_name', ''), quote=False)}: "
                f"<b>{_fmt_pct(p.get('probability', 0))}</b>"
                + (" <i>(regla post hoc)</i>" if p.get("postprocessed") else "")
                for p in positives
            )
            title = '<font color="#b91c1c"><b>HALLAZGOS COMPATIBLES CON LAS SIGUIENTES CLASES</b></font>'
            detail = f"{items}<br/><br/>Confianza máxima del modelo: <b>{level} ({_fmt_pct(top)})</b>."
        if fb.get("applied"):
            detail += (f"<br/><i>Nota: NSR fue añadida por regla post hoc (ninguna clase superó su umbral "
                       f"y P(NSR) ≥ {fb.get('min_nsr_probability')}).</i>")
    else:
        bg, border = gray_bg, slate
        title = "<b>SIN HALLAZGOS SOBRE EL UMBRAL DE DECISIÓN</b>"
        detail = ("Ninguna clase superó su umbral de decisión. Esto no descarta patología: "
                  "se recomienda revisión por un especialista.")

    box = Table([[Paragraph(f"{title}<br/><br/>{detail}", s_verdict)]], colWidths=[content_w])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 1.2, border),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return box


def _probability_chart(predictions, content_w, bar_bg, bar_neg, bar_pos, bar_nsr, slate):
    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.lib.units import mm

    rows = list(predictions)[:12]
    row_h = 14
    top_pad = 4
    height = top_pad + row_h * max(len(rows), 1) + 4
    drawing = Drawing(content_w, height)

    name_w = 118
    bar_x = name_w + 6
    bar_w = content_w - bar_x - 62
    pct_x = bar_x + bar_w + 6

    for i, item in enumerate(rows):
        y = height - top_pad - (i + 1) * row_h + 3
        prob = min(max(float(item.get("probability", 0.0) or 0.0), 0.0), 1.0)
        thr = min(max(float(item.get("threshold", 0.5) or 0.0), 0.0), 1.0)
        is_pos = bool(item.get("prediction"))
        name = str(item.get("class", ""))
        fill = (bar_nsr if name == "NSR" else bar_pos) if is_pos else bar_neg

        drawing.add(String(0, y, name, fontName="Helvetica-Bold", fontSize=7.5, fillColor=slate))
        drawing.add(Rect(bar_x, y - 1, bar_w, 9, strokeColor=bar_bg, fillColor=bar_bg,
                         strokeWidth=0.4))
        if prob > 0.003:
            drawing.add(Rect(bar_x, y - 1, max(bar_w * prob, 2), 9, strokeColor=fill,
                             fillColor=fill, strokeWidth=0))
        tick_x = bar_x + bar_w * thr
        drawing.add(Line(tick_x, y - 2.5, tick_x, y + 9.5, strokeColor=slate, strokeWidth=1.4))
        drawing.add(String(pct_x, y, f"{prob * 100:.1f}%", fontName="Helvetica", fontSize=7.5,
                           fillColor=slate))
    return drawing


def _canvas_factory(folio, emission, record, inst1, inst2):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen.canvas import Canvas

    navy = colors.HexColor("#84030c")
    gold = colors.HexColor("#cea962")
    escudo_path = Path(__file__).resolve().parent / "static" / "escudo-unsa.png"
    escudo_ok = escudo_path.exists()

    class DecoratedCanvas(Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_pages: list = []

        def showPage(self):
            self._saved_pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._saved_pages)
            for number, state in enumerate(self._saved_pages, start=1):
                self.__dict__.update(state)
                self._draw_decor(number, total)
                super().showPage()
            super().save()

        def _draw_decor(self, number: int, total: int):
            self.saveState()
            page_w, page_h = A4
            # Banda superior institucional
            mm = 2.83465
            self.setFillColor(navy)
            self.rect(0, page_h - 20 * mm, page_w, 20 * mm, stroke=0, fill=1)
            self.setFillColor(gold)
            self.rect(0, page_h - 20 * mm, page_w, 1 * mm, stroke=0, fill=1)
            text_x = 15 * mm
            if escudo_ok:
                try:
                    ew, eh = 12.1 * mm, 15 * mm
                    ex, ey = 15 * mm, page_h - 17.5 * mm
                    self.setFillColor(colors.white)
                    self.roundRect(ex - 1.2 * mm, ey - 1.2 * mm, ew + 2.4 * mm, eh + 2.4 * mm,
                                   2 * mm, stroke=0, fill=1)
                    self.drawImage(str(escudo_path), ex, ey, ew, eh, mask="auto")
                    text_x = 32 * mm
                except Exception:
                    text_x = 15 * mm
            self.setFillColor(colors.white)
            self.setFont("Helvetica-Bold", 8)
            self.drawString(text_x, page_h - 11 * mm, (inst1 or "ECG CINC2020-12")[:90])
            self.setFont("Helvetica", 7)
            self.drawString(text_x, page_h - 15.5 * mm, (inst2 or "")[:110])
            right = f"Folio {folio}  ·  {emission}"
            self.setFont("Helvetica", 7)
            self.drawRightString(page_w - 15 * 2.83465, page_h - 11 * 2.83465, right[:90])
            self.drawRightString(page_w - 15 * 2.83465, page_h - 15.5 * 2.83465, f"Registro {record}"[:60])
            # Pie de página
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(15 * 2.83465, 13 * 2.83465, page_w - 15 * 2.83465, 13 * 2.83465)
            self.setFillColor(colors.HexColor("#64748b"))
            self.setFont("Helvetica-Oblique", 6.5)
            self.drawString(15 * 2.83465, 10 * 2.83465, DISCLAIMER_SHORT)
            self.setFont("Helvetica", 7)
            self.drawRightString(page_w - 15 * 2.83465, 10 * 2.83465, f"Página {number} de {total}")
            self.restoreState()

    def factory(*args, **kwargs):
        return DecoratedCanvas(*args, **kwargs)

    return factory


def _escape_pdf_text(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_minimal_pdf(prediction: dict, output_path: Path) -> str:
    """Fallback sin reportlab: PDF mínimo de una página con texto."""
    comparison = prediction.get("label_comparison") or {}
    lines = [
        TITLE,
        DISCLAIMER,
        f"Registro: {prediction.get('record_name', '—')}",
        f"Paciente: {prediction.get('patient_name') or '—'}",
        f"Modelo: {prediction.get('model_type', '—')}",
        f"Checkpoint: {prediction.get('model_path', '—')}",
        "",
        "Comparación real vs predicho:",
        f"Disponible: {'sí' if comparison.get('available') else 'no'}",
        f"Reales: {', '.join(comparison.get('true_classes') or []) or '—'}",
        f"Predichas: {', '.join(comparison.get('predicted_classes') or []) or '—'}",
        f"Exact match: {'sí' if comparison.get('exact_match') else 'no'}",
        "",
        "Probabilidades:",
    ]
    for item in prediction.get("predictions", [])[:20]:
        lines.append(f"{item.get('class')}: {item.get('probability', 0) * 100:.2f}% pred={item.get('prediction')}")

    content = "BT /F1 9 Tf 40 790 Td "
    for i, line in enumerate(lines[:55]):
        if i > 0:
            content += "0 -13 Td "
        content += f"({_escape_pdf_text(line[:115])}) Tj "
    content += "ET"
    content_bytes = content.encode("latin-1", errors="replace")

    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        b"5 0 obj << /Length " + str(len(content_bytes)).encode() + b" >> stream\n" + content_bytes + b"\nendstream endobj\n",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj
    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    output_path.write_bytes(pdf)
    return str(output_path)
