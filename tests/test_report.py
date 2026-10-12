"""Tests del informe PDF (webapp/report_pdf.py)."""

from webapp.report_pdf import build_pdf_report


def _prediction():
    preds = []
    for i, name in enumerate(["NSR", "SB", "AF", "MI", "TAb", "BBB"]):
        prob = 0.94 if name == "NSR" else round(0.30 - i * 0.03, 3)
        preds.append({
            "class": name, "display_name": name, "probability": prob,
            "threshold": 0.5, "prediction": prob >= 0.5,
            "postprocessed": False, "near_threshold": False,
            "snomed_codes": ["426783006"] if name == "NSR" else [],
        })
    return {
        "job_id": "test-job-0001", "record_name": "E00001",
        "input_type": "wfdb_hea_mat",
        "lead_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "original_sampling_rate": 500, "target_sampling_rate": 500,
        "num_windows": 1, "window_seconds": 10,
        "dx_codes": ["426783006"], "matched_classes": ["NSR"],
        "predictions": preds,
        "positive_predictions": [p for p in preds if p["prediction"]],
        "model_type": "resnet1d", "num_classes": 12,
        "label_schema": "cinc2020_12_grouped_snomed_v2",
        "norm_mode": "physical", "input_length": 5000,
        "model_path": "saved/best.pt", "checkpoint_epoch": 27,
        "checkpoint_val_loss": 0.4651, "threshold": 0.5,
        "threshold_source": "thresholds.csv",
        "normal_fallback": {"enabled": True, "applied": False, "min_nsr_probability": 0.5},
        "label_comparison": {
            "available": True, "source": "wfdb_header_dx",
            "true_classes": ["NSR"], "predicted_classes": ["NSR"],
            "true_positive": ["NSR"], "false_positive": [], "false_negative": [],
            "exact_match": True, "f1": 1.0, "jaccard": 1.0,
        },
        "technical_details": {"device": "cpu", "decision_rule": "probabilidad >= umbral."},
        "plot_path": None,
    }


def test_pdf_builds_with_verdict(tmp_path):
    out = tmp_path / "reporte.pdf"
    path = build_pdf_report(_prediction(), out)
    assert out.exists() and out.stat().st_size > 5000
    from pypdf import PdfReader
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    assert "RITMO SINUSAL" in text
    assert "Folio" in text


def test_pdf_builds_empty_predictions(tmp_path):
    prediction = _prediction()
    prediction["predictions"] = []
    prediction["positive_predictions"] = []
    prediction["label_comparison"] = {"available": False}
    out = tmp_path / "vacio.pdf"
    build_pdf_report(prediction, out)
    assert out.exists() and out.stat().st_size > 1000
