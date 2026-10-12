"use strict";

// ============================================================ i18n =========
const I18N = {
  es: {
    "app.title": "Clasificador de ECG",
    "nav.upload": "Cargar ECG",
    "nav.result": "Resultado",
    "nav.detail": "Detalle Técnico",
    "nav.experiments": "Experimentos",
    "model.ready": "Modelo listo",
    "model.pending": "Modelo pendiente",
    "upload.title": "Cargar electrocardiograma",
    "upload.intro": "Sube un ECG de 12 derivaciones en CSV o el par WFDB .hea + .mat. El análisis se ejecuta solo al pulsar Analizar ECG.",
    "upload.drop": "Arrastra el archivo aquí",
    "upload.dropSub": "o haz clic para seleccionarlo. Para WFDB selecciona ambos archivos del mismo registro.",
    "upload.formats": "Formatos:",
    "upload.help": "¿Qué formato usar?",
    "upload.helpCsv": "CSV: columnas de derivaciones estándar I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6. Puede iniciar con # Sampling Rate: 500 Hz.",
    "upload.helpWfdb": "WFDB: sube juntos registro.hea y registro.mat. La app leerá el header, preprocesará la señal y generará un CSV convertido descargable.",
    "upload.patient": "Paciente / ID (opcional)",
    "upload.patientPh": "Nombre o código",
    "upload.age": "Edad (opcional)",
    "upload.threshold": "Threshold multilabel",
    "upload.trueLabels": "Etiquetas reales opcionales para CSV",
    "upload.trueLabelsPh": "NSR, AF o 426783006,164889003",
    "upload.trueLabelsHint": "Si subes .hea + .mat, la comparación usa automáticamente el campo Dx del header. Para CSV puedes escribir clases o códigos SNOMED separados por coma.",
    "upload.autoModel": "La app selecciona automáticamente el mejor checkpoint disponible en la carpeta saved/ configurada.",
    "btn.analyze": "Analizar ECG",
    "btn.clear": "Limpiar",
    "btn.pdf": "Descargar informe (PDF)",
    "btn.csv": "Descargar CSV convertido",
    "btn.png": "Abrir trazado PNG",
    "result.title": "Resultado",
    "result.sub": "Predicción multilabel, trazado en papel milimetrado, probabilidades por clase y descarga de reporte.",
    "result.paper": "El trazado se muestra con cuadrícula tipo papel milimetrado ECG.",
    "empty.title": "Aún no hay resultados",
    "empty.body": "Sube un ECG y pulsa Analizar ECG.",
    "detail.title": "Detalle Técnico",
    "detail.sub": "Ficha técnica del modelo, preprocesamiento, umbrales, comparación con etiquetas reales y métricas exportadas.",
    "detail.emptyTitle": "Aún no hay detalle de predicción",
    "detail.emptyBody": "Después de analizar un ECG se completará la ficha del registro.",
    "detail.method": "Enfoque: la señal se ordena a 12 derivaciones estándar, se convierte a milivoltios con la ganancia del header, se filtra (pasa-banda 0.5–50 Hz), se recorta a ±5 mV, se remuestrea a 500 Hz, se ajusta a 5000 muestras y se clasifica con salidas sigmoid independientes y umbrales por clase.",
    "detail.recordSheet": "Ficha del registro analizado",
    "detail.probabilities": "Probabilidades por clase",
    "detail.modelSheet": "Ficha técnica del modelo cargado automáticamente",
    "detail.schema": "Esquema de clases SNOMED-CT",
    "detail.metrics": "Métricas de evaluación",
    "detail.metricsNote": "Métricas calculadas por el script de evaluación sobre validación y test; la tabla por clase muestra el split test.",
    "detail.metricsPending": "Métricas pendientes.",
    "detail.metricsCmd": "Después de entrenar, ejecute evaluación para llenar esta sección:",
    "detail.metricsHint": "Ese comando también calibra los umbrales por clase que la app detecta automáticamente.",
    "detail.bestClass": "Mejor clase (F1 test)",
    "detail.worstClass": "Peor clase (F1 test)",
    "tech.arch": "Arquitectura",
    "tech.input": "Entrada",
    "tech.classes": "Clases",
    "tech.checkpoint": "Checkpoint",
    "tech.trained": "Entrenado",
    "tech.epoch": "Época / val_loss",
    "tech.params": "Parámetros",
    "tech.loss": "Pérdida",
    "tech.activation": "Activación",
    "tech.schema": "Esquema de etiquetas",
    "tech.norm": "Normalización",
    "tech.normPhysical": "physical: mV + pasa-banda 0.5–50 Hz + recorte ±5 mV",
    "tech.normLegacy": "z-score por derivación (esquema previo)",
    "tech.thresholds": "Umbrales",
    "tech.thresholdsCal": "calibrados por clase",
    "tech.thresholdsGlobal": "global 0.5 (respaldo)",
    "table.class": "Clase",
    "table.description": "Descripción",
    "table.codes": "Códigos",
    "table.model": "Modelo",
    "metric.support": "Soporte +",
    "metric.sensitivity": "Sensibilidad",
    "metric.specificity": "Especificidad",
    "metric.precisionMacro": "Precisión macro",
    "metric.sensitivityMacro": "Sensibilidad macro",
    "exp.title": "Experimentos",
    "exp.sub": "Comparación de evaluaciones y robustez con perturbaciones controladas.",
    "exp.method": "Objetivo experimental: comparar de forma justa dos evaluaciones (por ejemplo, ResNet v1 frente a ResNet v2) usando el mismo dataset, split y métricas; y medir robustez ante degradaciones controladas de la señal.",
    "exp.compare": "Comparación de modelos",
    "exp.pending": "Resultado pendiente.",
    "exp.compareCmd": "Genere primero métricas para ambos modelos y luego compare:",
    "exp.robust": "Robustez frente a perturbaciones ECG",
    "exp.robustCmd": "Ejecute el experimento reproducible de robustez:",
    "exp.perturbation": "Perturbación",
    "exp.level": "Nivel",
    "exp.compareNote": "Comparación justa: mismo dataset, split, preprocesamiento y métricas para ambas evaluaciones. La estrella marca el mejor F1 macro en test.",
    "exp.robustNote": "Cada fila aplica una degradación controlada al test y mide cuánto cae el F1 macro frente a la señal limpia.",
    "exp.deltaCol": "ΔF1 vs limpio",
    "exp.biggestDrop": "Mayor caída vs señal limpia",
    "footer.author": "Autor:",
    "footer.advisor": "Asesor:",
    "footer.model": "Modelo",
    "footer.disclaimer": "Herramienta de investigación — no reemplaza la lectura de un cardiólogo. Trabajo académico · UNSA.",
    "status.select": "Selecciona un CSV o el par .hea + .mat.",
    "status.ready": "listo para analizar",
    "status.loading": "Analizando la señal y ejecutando inferencia…",
    "status.done": "Análisis completado.",
    "status.error": "No se pudo analizar: ",
    "status.invalidCombo": "Sube un único CSV o exactamente el par .hea + .mat del mismo registro.",
    "status.stemMismatch": "El .hea y el .mat deben tener el mismo nombre base.",
    "notice.wfdb": "Entrada WFDB leída correctamente; se generó un CSV convertido.",
    "notice.csv": "Entrada CSV preprocesada a la forma esperada por el modelo.",
    "notice.truth": "Etiquetas reales disponibles para comparación.",
    "diag.top": "probabilidad top",
    "diag.noResult": "Sin resultado",
    "diag.noProb": "No se recibieron probabilidades.",
    "diag.noPositiveTitle": "Sin predicción positiva",
    "diag.topCandidate": "Mayor probabilidad observada",
    "diag.positives": "Predicciones positivas",
    "diag.nonePositive": "Ninguna clase superó el threshold definido",
    "diag.nonePositiveClassThresholds": "Ninguna clase superó su umbral calibrado",
    "diag.borderline": "Predicción cercana al umbral; interpretar como baja confianza.",
    "diag.fallbackApplied": "NSR añadido por regla de fallback normal porque ninguna clase superó su umbral.",
    "traffic.positive": "positiva",
    "traffic.negative": "bajo threshold",
    "summary.title": "Resumen multilabel:",
    "summary.none": "Sin positivos a threshold",
    "summary.noneClassThresholds": "Sin positivos con umbrales calibrados",
    "summary.model": "Modelo usado para este resultado",
    "summary.rule": "Regla de decisión",
    "summary.ruleClassThresholds": "probabilidad ≥ umbral calibrado por clase",
    "summary.ruleGlobalThreshold": "probabilidad ≥ threshold global",
    "gt.title": "Comparación con etiquetas reales",
    "gt.interpretation": "Interpretación",
    "gt.falseNegativeOnly": "El modelo no activó ninguna clase; como la etiqueta real sí existe, se considera falso negativo.",
    "gt.falsePositiveOnly": "El modelo activó clases que no están en la etiqueta real; se considera falso positivo.",
    "gt.mixedErrors": "Hay diferencias multilabel: algunas clases faltan y/o sobran frente al header.",
    "gt.correct": "Las clases predichas coinciden exactamente con las etiquetas reales disponibles.",
    "gt.notAvailable": "No hay etiquetas reales disponibles. En .hea + .mat se leen automáticamente desde Dx; para CSV puedes escribir clases o SNOMED antes de analizar.",
    "gt.sourceHeader": "Header WFDB Dx",
    "gt.sourceManual": "Ingreso manual",
    "gt.exact": "Coincidencia exacta multilabel",
    "gt.notExact": "Diferencias frente a las etiquetas reales",
    "gt.true": "Reales",
    "gt.pred": "Predichas",
    "gt.tp": "Aciertos",
    "gt.fp": "Falsos positivos",
    "gt.fn": "Falsos negativos",
    "gt.none": "Ninguna",
    "prob.index": "#",
    "prob.prob": "Probabilidad",
    "prob.threshold": "Umbral",
    "prob.margin": "Margen",
    "prob.state": "Estado",
    "prob.yes": "Positiva",
    "prob.no": "Negativa",
    "detail.barsTitle": "Métricas por clase (interactivo)",
    "curves.title": "Curvas ROC/PR interactivas",
    "curves.hint": "Seleccione una clase para ver sus curvas ROC y Precision-Recall en test.",
    "curves.selectClass": "Clase",
    "curves.pending": "Las curvas se generan automáticamente al ejecutar la evaluación (ver Detalle Técnico › Métricas).",
    "curves.roc": "Curva ROC",
    "curves.pr": "Curva Precision-Recall",
    "curves.fpr": "Tasa falsos positivos",
    "curves.tpr": "Tasa verdaderos positivos",
    "curves.precision": "Precisión",
    "curves.recall": "Recall",
    "curves.noData": "Sin curvas disponibles.",
    "bars.f1": "F1",
    "bars.auroc": "AUROC",
    "bars.title": "F1 y AUROC por clase (test)",
    "detail.groupRecord": "Registro y adquisición",
    "detail.groupDecision": "Decisión del modelo",
    "detail.groupPreproc": "Preprocesamiento y esquema",
    "detail.schemaNote": "Referencia completa de códigos por clase (la tabla de probabilidades los omite por legibilidad).",
    "detail.varsNote": "Con las variables $resnet, $resnet2, $mejor y $evalMejor definidas en el README principal.",
    "tech.modelB": "Modelo B (ensemble)",
    "tech.objective": "objetivo",
    "tech.advanced": "Hiperparámetros de entrenamiento (avanzado)",
    "prob.marginNote": "Margen = probabilidad − umbral. El marcador ~ indica cercanía al umbral (|margen| < 0.05): baja confianza. Pase el cursor sobre la clase para ver sus códigos SNOMED.",
    "bars.noData": "No se pudieron cargar los datos del gráfico interactivo.",
    "exp.verdictLead": "Ganador en F1-macro (test)",
    "exp.levelNote": "Niveles: ruido σ en mV, deriva amplitud en mV (seno 0.5 Hz), escalado factor ×, apagado número de derivaciones.",
    "history.title": "Historial de sesión",
    "history.empty": "Aún no hay análisis en esta sesión.",
    "history.clear": "Limpiar historial",
    "history.positives": "Positivos",
    "history.none": "Sin positivos",
    "history.pdf": "PDF",
    "history.csv": "CSV",
  },
  en: {
    "app.title": "ECG Classifier",
    "nav.upload": "Upload ECG",
    "nav.result": "Result",
    "nav.detail": "Technical Detail",
    "nav.experiments": "Experiments",
    "model.ready": "Model ready",
    "model.pending": "Model pending",
    "upload.title": "Upload electrocardiogram",
    "upload.intro": "Upload a 12-lead ECG as CSV or the WFDB .hea + .mat pair. Analysis starts only when you press Analyze ECG.",
    "upload.drop": "Drag the file here",
    "upload.dropSub": "or click to select it. For WFDB, select both files from the same record.",
    "upload.formats": "Formats:",
    "upload.help": "Which format should I use?",
    "upload.helpCsv": "CSV: standard lead columns I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6. It may start with # Sampling Rate: 500 Hz.",
    "upload.helpWfdb": "WFDB: upload record.hea and record.mat together. The app reads the header, preprocesses the signal and generates a downloadable converted CSV.",
    "upload.patient": "Patient / ID (optional)",
    "upload.patientPh": "Name or code",
    "upload.age": "Age (optional)",
    "upload.threshold": "Multilabel threshold",
    "upload.trueLabels": "Optional true labels for CSV",
    "upload.trueLabelsPh": "NSR, AF or 426783006,164889003",
    "upload.trueLabelsHint": "If you upload .hea + .mat, comparison uses the header Dx field automatically. For CSV, write class names or SNOMED codes separated by commas.",
    "upload.autoModel": "The app automatically selects the best available checkpoint in the configured saved/ folder.",
    "btn.analyze": "Analyze ECG",
    "btn.clear": "Clear",
    "btn.pdf": "Download report (PDF)",
    "btn.csv": "Download converted CSV",
    "btn.png": "Open PNG trace",
    "result.title": "Result",
    "result.sub": "Multilabel prediction, ECG-paper grid trace, per-class probabilities and report download.",
    "result.paper": "The trace is displayed with an ECG-paper grid.",
    "empty.title": "No results yet",
    "empty.body": "Upload an ECG and press Analyze ECG.",
    "detail.title": "Technical Detail",
    "detail.sub": "Model sheet, preprocessing, thresholds, true-label comparison and exported metrics.",
    "detail.emptyTitle": "No prediction detail yet",
    "detail.emptyBody": "After analyzing an ECG, the record sheet will be filled in.",
    "detail.method": "Method: the signal is ordered into 12 standard leads, converted to millivolts with the header gain, filtered (0.5–50 Hz bandpass), clipped to ±5 mV, resampled to 500 Hz, fixed to 5000 samples and classified with independent sigmoid outputs and per-class thresholds.",
    "detail.recordSheet": "Analyzed record sheet",
    "detail.probabilities": "Per-class probabilities",
    "detail.modelSheet": "Automatically loaded model sheet",
    "detail.schema": "SNOMED-CT class schema",
    "detail.metrics": "Evaluation metrics",
    "detail.metricsNote": "Metrics computed by the evaluation script on validation and test; the per-class table shows the test split.",
    "detail.metricsPending": "Metrics pending.",
    "detail.metricsCmd": "After training, run evaluation to fill this section:",
    "detail.metricsHint": "That command also calibrates the per-class thresholds that the app detects automatically.",
    "detail.bestClass": "Best class (test F1)",
    "detail.worstClass": "Worst class (test F1)",
    "tech.arch": "Architecture",
    "tech.input": "Input",
    "tech.classes": "Classes",
    "tech.checkpoint": "Checkpoint",
    "tech.trained": "Trained",
    "tech.epoch": "Epoch / val_loss",
    "tech.params": "Parameters",
    "tech.loss": "Loss",
    "tech.activation": "Activation",
    "tech.schema": "Label schema",
    "tech.norm": "Normalization",
    "tech.normPhysical": "physical: mV + 0.5–50 Hz bandpass + ±5 mV clip",
    "tech.normLegacy": "per-lead z-score (legacy schema)",
    "tech.thresholds": "Thresholds",
    "tech.thresholdsCal": "class-calibrated",
    "tech.thresholdsGlobal": "global 0.5 (fallback)",
    "table.class": "Class",
    "table.description": "Description",
    "table.codes": "Codes",
    "table.model": "Model",
    "metric.support": "Support +",
    "metric.sensitivity": "Sensitivity",
    "metric.specificity": "Specificity",
    "metric.precisionMacro": "Macro precision",
    "metric.sensitivityMacro": "Macro sensitivity",
    "exp.title": "Experiments",
    "exp.sub": "Evaluation comparison and robustness under controlled perturbations.",
    "exp.method": "Experimental goal: fairly compare two evaluations (e.g. ResNet v1 vs ResNet v2) using the same dataset, split and metrics; and measure robustness under controlled signal degradations.",
    "exp.compare": "Model comparison",
    "exp.pending": "Result pending.",
    "exp.compareCmd": "Generate metrics for both models first, then compare:",
    "exp.robust": "Robustness to ECG perturbations",
    "exp.robustCmd": "Run the reproducible robustness experiment:",
    "exp.perturbation": "Perturbation",
    "exp.level": "Level",
    "exp.compareNote": "Fair comparison: same dataset, split, preprocessing and metrics for both evaluations. The star marks the best test macro F1.",
    "exp.robustNote": "Each row applies a controlled degradation to the test set and measures how much macro F1 drops versus the clean signal.",
    "exp.deltaCol": "ΔF1 vs clean",
    "exp.biggestDrop": "Largest drop vs clean signal",
    "footer.author": "Author:",
    "footer.advisor": "Advisor:",
    "footer.model": "Model",
    "footer.disclaimer": "Research tool — it does not replace a cardiologist's reading. Academic work · UNSA.",
    "status.select": "Select a CSV or the .hea + .mat pair.",
    "status.ready": "ready to analyze",
    "status.loading": "Analyzing the signal and running inference…",
    "status.done": "Analysis completed.",
    "status.error": "Could not analyze: ",
    "status.invalidCombo": "Upload one CSV or exactly the .hea + .mat pair from the same record.",
    "status.stemMismatch": "The .hea and .mat files must have the same base name.",
    "notice.wfdb": "WFDB input read successfully; a converted CSV was generated.",
    "notice.csv": "CSV input preprocessed into the model's expected shape.",
    "notice.truth": "True labels are available for comparison.",
    "diag.top": "top probability",
    "diag.noResult": "No result",
    "diag.noProb": "No probabilities were received.",
    "diag.noPositiveTitle": "No positive prediction",
    "diag.topCandidate": "Highest observed probability",
    "diag.positives": "Positive predictions",
    "diag.nonePositive": "No class exceeded the selected threshold",
    "diag.nonePositiveClassThresholds": "No class exceeded its calibrated threshold",
    "diag.borderline": "Prediction is close to the threshold; interpret as low confidence.",
    "diag.fallbackApplied": "NSR was added by the normal-fallback rule because no class exceeded its threshold.",
    "traffic.positive": "positive",
    "traffic.negative": "below threshold",
    "summary.title": "Multilabel summary:",
    "summary.none": "No positives at threshold",
    "summary.noneClassThresholds": "No positives with calibrated thresholds",
    "summary.model": "Model used for this result",
    "summary.rule": "Decision rule",
    "summary.ruleClassThresholds": "probability ≥ class-calibrated threshold",
    "summary.ruleGlobalThreshold": "probability ≥ global threshold",
    "gt.title": "True-label comparison",
    "gt.interpretation": "Interpretation",
    "gt.falseNegativeOnly": "The model did not activate any class; because a true label exists, this is a false negative.",
    "gt.falsePositiveOnly": "The model activated classes that are not in the true label; this is a false positive.",
    "gt.mixedErrors": "There are multilabel differences: some classes are missing and/or extra relative to the header.",
    "gt.correct": "The predicted classes exactly match the available true labels.",
    "gt.notAvailable": "No true labels are available. With .hea + .mat they are read automatically from Dx; for CSV you can enter classes or SNOMED codes before analysis.",
    "gt.sourceHeader": "WFDB header Dx",
    "gt.sourceManual": "Manual input",
    "gt.exact": "Exact multilabel match",
    "gt.notExact": "Differences against true labels",
    "gt.true": "True",
    "gt.pred": "Predicted",
    "gt.tp": "True positives",
    "gt.fp": "False positives",
    "gt.fn": "False negatives",
    "gt.none": "None",
    "prob.index": "#",
    "prob.prob": "Probability",
    "prob.threshold": "Threshold",
    "prob.margin": "Margin",
    "prob.state": "State",
    "prob.yes": "Positive",
    "prob.no": "Negative",
    "detail.barsTitle": "Per-class metrics (interactive)",
    "curves.title": "Interactive ROC/PR curves",
    "curves.hint": "Select a class to view its ROC and Precision-Recall curves on test.",
    "curves.selectClass": "Class",
    "curves.pending": "Curves are generated automatically when running evaluation (see Technical Detail › Metrics).",
    "curves.roc": "ROC curve",
    "curves.pr": "Precision-Recall curve",
    "curves.fpr": "False positive rate",
    "curves.tpr": "True positive rate",
    "curves.precision": "Precision",
    "curves.recall": "Recall",
    "curves.noData": "No curves available.",
    "bars.f1": "F1",
    "bars.auroc": "AUROC",
    "bars.title": "F1 and AUROC per class (test)",
    "detail.groupRecord": "Record and acquisition",
    "detail.groupDecision": "Model decision",
    "detail.groupPreproc": "Preprocessing and schema",
    "detail.schemaNote": "Full per-class code reference (the probabilities table omits them for readability).",
    "detail.varsNote": "Using the $resnet, $resnet2, $mejor and $evalMejor variables defined in the main README.",
    "tech.modelB": "Model B (ensemble)",
    "tech.objective": "objective",
    "tech.advanced": "Training hyperparameters (advanced)",
    "prob.marginNote": "Margin = probability − threshold. The ~ marker means near-threshold (|margin| < 0.05): low confidence. Hover the class to see its SNOMED codes.",
    "bars.noData": "Could not load the interactive chart data.",
    "exp.verdictLead": "Winner on test F1-macro",
    "exp.levelNote": "Levels: noise σ in mV, wander amplitude in mV (0.5 Hz sine), scaling factor ×, dropout lead count.",
    "history.title": "Session history",
    "history.empty": "No analyses yet in this session.",
    "history.clear": "Clear history",
    "history.positives": "Positives",
    "history.none": "No positives",
    "history.pdf": "PDF",
    "history.csv": "CSV",
  },
};

let LANG = "es";
try {
  const savedLang = localStorage.getItem("ecg-lang");
  if (savedLang === "en" || savedLang === "es") LANG = savedLang;
} catch (e) {}

function tr(key) {
  return (I18N[LANG] && I18N[LANG][key]) || (I18N.es && I18N.es[key]) || key;
}

function applyI18n() {
  document.documentElement.lang = LANG;
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = tr(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach(el => { el.placeholder = tr(el.dataset.i18nPlaceholder); });
  const es = document.getElementById("langEs");
  const en = document.getElementById("langEn");
  if (es) es.classList.toggle("active", LANG === "es");
  if (en) en.classList.toggle("active", LANG === "en");
  refreshFileCard();
  if (typeof renderHistory === "function") renderHistory();
  if (lastResult) render(lastResult, false);
}

// ============================================================ state =======
const $ = (id) => document.getElementById(id);
const dropzone = $("dropzone");
const fileInput = $("file");
const btnClassify = $("btnClassify");
const btnClear = $("btnClear");
const statusEl = $("status");
const fileCard = $("fileCard");
const visContent = $("visContent");
const visEmpty = $("visEmpty");
const detContent = $("detContent");
const detEmpty = $("detEmpty");
const skel = $("skel");
const diagnosisBox = $("diagnosisBox");
const trafficBox = $("trafficBox");
const summaryBox = $("summaryBox");
const gtBox = $("gtBox");
const plotDiv = $("plotDiv");
const technicalRows = $("technicalRows");
const probTable = $("probTable");
const btnPrint = $("btnPrint");
const btnCsv = $("btnCsv");
const btnPng = $("btnPng");
const threshold = $("threshold");
const patName = $("patName");
const patAge = $("patAge");
const trueLabels = $("trueLabels");

let currentFiles = [];
let lastResult = null;

const PALETTE = [
  "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#d62728", "#17becf",
  "#8c564b", "#e377c2", "#bcbd22", "#0d9488", "#64748b", "#7c3aed",
];

// ============================================================ utils =======
function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = String(s ?? "");
  return d.innerHTML;
}

function setStatus(msg, cls) {
  statusEl.textContent = msg;
  statusEl.className = "status " + (cls || "");
}

function setLoading(on) {
  btnClassify.disabled = on;
  btnClassify.classList.toggle("btn-loading", on);
  btnClassify.setAttribute("aria-busy", on ? "true" : "false");
  if (skel) skel.hidden = !on;
}

function fmtPct(value) {
  const n = Number(value || 0);
  return `${(n * 100).toFixed(2)}%`;
}

function shortPath(path) {
  const text = String(path || "").replace(/\\/g, "/");
  if (!text) return "—";
  const parts = text.split("/");
  return parts.slice(-3).join("/");
}

function baseName(path) {
  const text = String(path || "").replace(/\\/g, "/");
  if (!text) return "—";
  const parts = text.split("/");
  return parts[parts.length - 1] || text;
}

function listOrNone(values) {
  return values && values.length ? values.join(", ") : tr("gt.none");
}

function comparisonInterpretation(cmp) {
  if (!cmp || !cmp.available) return "";
  if (cmp.exact_match) return tr("gt.correct");
  const pred = cmp.predicted_classes || [];
  const fp = cmp.false_positive || [];
  const fn = cmp.false_negative || [];
  if (!pred.length && fn.length && !fp.length) return tr("gt.falseNegativeOnly");
  if (fp.length && !fn.length) return tr("gt.falsePositiveOnly");
  return tr("gt.mixedErrors");
}

function kv(label, value) {
  return `<div class="gt-kv"><span>${escapeHtml(label)}:</span> <b>${escapeHtml(value)}</b></div>`;
}

function fileExt(name) {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

function fileStem(name) {
  const clean = name.split(/[\\/]/).pop() || name;
  const i = clean.lastIndexOf(".");
  return (i >= 0 ? clean.slice(0, i) : clean).toLowerCase();
}

function validateFiles(files) {
  if (!files.length) return tr("status.select");
  const exts = files.map(f => fileExt(f.name));
  if (files.length === 1 && exts[0] === ".csv") return "";
  if (files.length === 2 && exts.includes(".hea") && exts.includes(".mat")) {
    const hea = files.find(f => fileExt(f.name) === ".hea");
    const mat = files.find(f => fileExt(f.name) === ".mat");
    if (fileStem(hea.name) !== fileStem(mat.name)) return tr("status.stemMismatch");
    return "";
  }
  return tr("status.invalidCombo");
}

// ============================================================ navigation ===
function goTo(sec) {
  document.querySelectorAll(".nav-link").forEach(b => b.classList.toggle("active", b.dataset.sec === sec));
  ["upload", "vis", "det", "exp"].forEach(s => {
    const el = $("sec-" + s);
    if (el) el.hidden = (s !== sec);
  });
  if (sec === "det" && typeof loadMetricBars === "function") loadMetricBars();
  if (sec === "exp" && typeof loadCurves === "function") loadCurves();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.querySelectorAll(".nav-link").forEach(b => b.addEventListener("click", () => goTo(b.dataset.sec)));

// ============================================================ theme/lang ===
function updateThemeIcon() {
  const el = $("themeToggle");
  if (!el) return;
  const dark = document.documentElement.dataset.theme === "dark";
  el.innerHTML = dark
    ? '<svg class="ico"><use href="#i-sun"/></svg>'
    : '<svg class="ico"><use href="#i-moon"/></svg>';
}

try {
  const saved = localStorage.getItem("ecg-theme");
  if (saved) document.documentElement.dataset.theme = saved;
} catch (e) {}

$("themeToggle").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme;
  const nxt = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = nxt;
  try { localStorage.setItem("ecg-theme", nxt); } catch (e) {}
  updateThemeIcon();
});
$("langEs").addEventListener("click", () => { LANG = "es"; try { localStorage.setItem("ecg-lang", LANG); } catch (e) {} applyI18n(); });
$("langEn").addEventListener("click", () => { LANG = "en"; try { localStorage.setItem("ecg-lang", LANG); } catch (e) {} applyI18n(); });
updateThemeIcon();

// ============================================================ file upload ==
function refreshFileCard() {
  if (!currentFiles.length) {
    fileCard.hidden = true;
    return;
  }
  fileCard.hidden = false;
  const total = currentFiles.reduce((acc, f) => acc + (f.size || 0), 0);
  const size = total > 1048576 ? `${(total / 1048576).toFixed(2)} MB` : `${Math.max(1, Math.round(total / 1024))} KB`;
  const names = currentFiles.map(f => `<div class="fname">${escapeHtml(f.name)}</div>`).join("");
  const validation = validateFiles(currentFiles);
  const validationHtml = validation ? `<div class="hint error-text">${escapeHtml(validation)}</div>` : "";
  fileCard.innerHTML = `
    <svg class="ico"><use href="#i-file"/></svg>
    <div class="file-list">${names}<div class="hint muted">${currentFiles.length} archivo(s) · ${size} · ${tr("status.ready")}</div>${validationHtml}</div>`;
}

function handleFiles(files) {
  if (!files || !files.length) return;
  currentFiles = Array.from(files);
  refreshFileCard();
  setStatus("", "");
}

dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", e => { e.preventDefault(); dropzone.classList.remove("drag"); handleFiles(e.dataTransfer.files); });
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => handleFiles(fileInput.files));

const fmtHelp = $("fmtHelp");
const fmtHelpBox = $("fmtHelpBox");
fmtHelp.addEventListener("click", () => { fmtHelpBox.hidden = !fmtHelpBox.hidden; });

btnClear.addEventListener("click", () => {
  currentFiles = [];
  lastResult = null;
  fileInput.value = "";
  fileCard.hidden = true;
  setStatus("", "");
  visContent.hidden = true;
  visEmpty.hidden = false;
  detContent.hidden = true;
  detEmpty.hidden = false;
  goTo("upload");
});

// ============================================================ submit ======
async function postFiles() {
  const validation = validateFiles(currentFiles);
  if (validation) {
    setStatus(validation, "err");
    refreshFileCard();
    return;
  }

  const fd = new FormData();
  currentFiles.forEach(file => fd.append("files", file, file.name));
  fd.append("patient_name", patName.value || "");
  fd.append("patient_age", patAge.value || "");
  fd.append("threshold", threshold.value || "0.5");
  fd.append("true_labels", trueLabels.value || "");

  setLoading(true);
  setStatus(tr("status.loading"), "loading");
  visEmpty.hidden = true;
  visContent.hidden = false;

  try {
    const resp = await fetch("/predict", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok || data.ok === false || data.status === "error") {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    render(data);
    setStatus(tr("status.done"), "ok");
    goTo("vis");
  } catch (err) {
    console.error(err);
    setStatus(tr("status.error") + err.message, "err");
  } finally {
    setLoading(false);
  }
}

btnClassify.addEventListener("click", postFiles);

// ============================================================ render ======
function setModelNameFromResult(result) {
  const text = shortPath(result.model_path);
  const sidebar = $("modelName");
  const footer = $("modelNameFooter");
  if (sidebar) {
    sidebar.textContent = text;
    sidebar.title = result.model_path || "";
  }
  if (footer) footer.textContent = text;
}

function renderNotice(result) {
  const notice = $("notice");
  const messages = [];
  if (result.input_type === "wfdb_hea_mat") messages.push(tr("notice.wfdb"));
  if (result.input_type === "csv") messages.push(tr("notice.csv"));
  if (result.label_comparison && result.label_comparison.available) messages.push(tr("notice.truth"));
  notice.textContent = messages.join(" · ");
  notice.style.display = messages.length ? "block" : "none";
}

function renderDiagnosis(result) {
  const top = (result.top_predictions || [])[0];
  const positives = result.positive_predictions || [];
  const primary = positives[0] || null;
  const hasPositive = positives.length > 0;
  const posText = hasPositive
    ? positives.map(r => r.postprocessed
        ? `${r.class} (${fmtPct(r.probability)}; fallback NSR, threshold ${fmtPct(r.threshold ?? result.threshold)})`
        : `${r.class} (${fmtPct(r.probability)} ≥ ${fmtPct(r.threshold ?? result.threshold)})`
      ).join(", ")
    : (result.using_class_thresholds ? tr("diag.nonePositiveClassThresholds") : tr("diag.nonePositive"));
  const titleRow = primary || top;
  const cardColor = hasPositive && primary ? PALETTE[primary.index % PALETTE.length] : "#64748b";
  const title = hasPositive && primary ? primary.class : tr("diag.noPositiveTitle");
  let desc = tr("diag.noProb");
  if (hasPositive && primary) {
    const thresholdText = primary.postprocessed
      ? `fallback NSR · threshold original ${fmtPct(primary.threshold ?? result.threshold)}`
      : `threshold ${fmtPct(primary.threshold ?? result.threshold)}`;
    desc = `${escapeHtml(primary.display_name)} · ${tr("diag.top")} ${fmtPct(primary.probability)} · ${thresholdText}`;
  } else if (top) {
    desc = `${tr("diag.topCandidate")}: ${escapeHtml(top.class)} · ${fmtPct(top.probability)} < threshold ${fmtPct(top.threshold ?? result.threshold)}`;
  }
  const borderline = titleRow && (titleRow.near_threshold || Math.abs(Number(titleRow.margin || 0)) < 0.05);
  diagnosisBox.innerHTML = `
    <div class="diagnosis-card" style="border-left-color:${cardColor}">
      <div class="diagnosis-name" style="color:${cardColor}">${escapeHtml(title)}</div>
      <div class="diagnosis-desc">${desc}</div>
      <div class="diagnosis-conf">${tr("diag.positives")}: <strong>${escapeHtml(posText)}</strong></div>
      ${result.normal_fallback && result.normal_fallback.applied ? `<div class="fallback-note">${tr("diag.fallbackApplied")}</div>` : ""}
      ${borderline ? `<div class="borderline-warn">⚠ ${tr("diag.borderline")}</div>` : ""}
    </div>`;
}

function renderTraffic(result) {
  const rows = (result.predictions || []).slice(0, 8);
  trafficBox.innerHTML = rows.map(row => {
    const color = PALETTE[row.index % PALETTE.length];
    const positive = Number(row.prediction) === 1;
    const label = positive ? tr("traffic.positive") : tr("traffic.negative");
    return `<span class="traffic-item"><span class="traffic-dot" style="background:${color}"></span>${escapeHtml(row.class)} · ${label} (${fmtPct(row.probability)})</span>`;
  }).join("");
}

function renderSummary(result) {
  const positives = result.positive_predictions || [];
  summaryBox.innerHTML = `<div class="summary-label">${tr("summary.title")}</div>`;
  if (positives.length) {
    positives.forEach(row => {
      const color = PALETTE[row.index % PALETTE.length];
      const post = row.postprocessed ? " · fallback" : "";
      summaryBox.innerHTML += `<span class="chip" style="background:${color}" title="${escapeHtml(row.display_name)}">${escapeHtml(row.class)} · ${fmtPct(row.probability)}${post}</span>`;
    });
  } else {
    const noneText = result.using_class_thresholds
      ? tr("summary.noneClassThresholds")
      : `${tr("summary.none")} ${escapeHtml(result.threshold)}`;
    summaryBox.innerHTML += `<span class="chip" style="background:#64748b">${noneText}</span>`;
  }
  const rule = result.using_class_thresholds
    ? `${tr("summary.ruleClassThresholds")} (${escapeHtml(baseName(result.threshold_source))})`
    : `${tr("summary.ruleGlobalThreshold")} ${escapeHtml(result.threshold)}`;
  summaryBox.innerHTML += `<p class="hint muted summary-hint">${tr("summary.model")}: <strong>${escapeHtml(result.model_type || "—")}</strong><br>${tr("summary.rule")}: ${rule}</p>`;
}

function renderComparison(result) {
  const cmp = result.label_comparison || null;
  if (!cmp) {
    gtBox.hidden = true;
    return;
  }
  gtBox.hidden = false;
  if (!cmp.available) {
    gtBox.className = "gt gt-warn";
    gtBox.innerHTML = `<strong>${tr("gt.title")}</strong><br>${tr("gt.notAvailable")}`;
    return;
  }
  const ok = Boolean(cmp.exact_match);
  gtBox.className = `gt ${ok ? "gt-ok" : "gt-bad"}`;
  const source = cmp.source === "wfdb_header_dx" ? tr("gt.sourceHeader") : tr("gt.sourceManual");
  const interp = comparisonInterpretation(cmp);
  gtBox.innerHTML = `
    <strong>${tr("gt.title")}: ${ok ? tr("gt.exact") : tr("gt.notExact")}</strong>
    <p class="gt-interpretation"><b>${tr("gt.interpretation")}:</b> ${escapeHtml(interp)}</p>
    <div class="gt-grid">
      ${kv(tr("gt.true"), listOrNone(cmp.true_classes))}
      ${kv(tr("gt.pred"), listOrNone(cmp.predicted_classes))}
      ${kv(tr("gt.tp"), listOrNone(cmp.true_positive))}
      ${kv(tr("gt.fp"), listOrNone(cmp.false_positive))}
      ${kv(tr("gt.fn"), listOrNone(cmp.false_negative))}
      ${kv("F1 / Jaccard", `${Number(cmp.f1).toFixed(3)} / ${Number(cmp.jaccard).toFixed(3)}`)}
    </div>
    <div class="hint muted">${source}${cmp.dx_codes && cmp.dx_codes.length ? ` · DX: ${escapeHtml(cmp.dx_codes.join(", "))}` : ""}</div>`;
}

function renderPlot(result) {
  if (result.plot_url) {
    plotDiv.innerHTML = `<a href="${escapeHtml(result.plot_url)}" target="_blank" rel="noopener"><img class="plot" src="${escapeHtml(result.plot_url)}" alt="ECG paper grid 12-lead trace"></a>`;
    btnPng.href = result.plot_url;
    btnPng.hidden = false;
  } else {
    plotDiv.innerHTML = `<p class="hint muted">No se generó imagen del trazado.</p>`;
    btnPng.hidden = true;
  }
}

function renderButtons(result) {
  if (result.pdf_url) {
    btnPrint.disabled = false;
    btnPrint.onclick = () => {
      const a = document.createElement("a");
      a.href = result.pdf_url;
      a.download = `${result.record_name || "ecg"}_reporte.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    };
  } else {
    btnPrint.disabled = true;
    btnPrint.onclick = null;
  }
  if (result.converted_csv_url) {
    btnCsv.href = result.converted_csv_url;
    btnCsv.download = `${result.record_name || "ecg"}_convertido.csv`;
    btnCsv.hidden = false;
  } else {
    btnCsv.hidden = true;
  }
}

function row(label, value) {
  return `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(value ?? "—")}</td></tr>`;
}

function rowHtml(label, html) {
  return `<tr><th>${escapeHtml(label)}</th><td>${html}</td></tr>`;
}

function detSub(title) {
  return `<tr class="det-sub"><td colspan="2">${escapeHtml(title)}</td></tr>`;
}

function fmtNum(value, digits = 3) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function renderTechnical(result) {
  const details = result.technical_details || {};
  const cmp = result.label_comparison || {};
  const exactBadge = !cmp.available ? "—"
    : cmp.exact_match ? `<span class="badge ok">${LANG === "es" ? "Sí" : "Yes"}</span>` : `<span class="badge no">No</span>`;
  technicalRows.innerHTML = [
    detSub(tr("detail.groupRecord")),
    row(LANG === "es" ? "Registro" : "Record", result.record_name),
    row(LANG === "es" ? "Paciente / ID" : "Patient / ID", result.patient_name || "—"),
    row(LANG === "es" ? "Edad" : "Age", result.patient_age || "—"),
    row(LANG === "es" ? "Tipo de entrada" : "Input type", result.input_type),
    row(LANG === "es" ? "Archivos fuente" : "Source files", (result.source_files || []).map(shortPath).join(" | ")),
    row(LANG === "es" ? "Shape procesado" : "Processed shape", Array.isArray(result.processed_shape) ? result.processed_shape.join(" × ") : result.processed_shape),
    row(LANG === "es" ? "Frecuencia original" : "Original sampling rate", result.original_sampling_rate ? `${result.original_sampling_rate} Hz` : "—"),
    row(LANG === "es" ? "Frecuencia objetivo" : "Target sampling rate", result.target_sampling_rate ? `${result.target_sampling_rate} Hz` : "—"),
    row(LANG === "es" ? "Ventana" : "Window", `${result.window_seconds} s · ${result.input_length} samples`),
    row(LANG === "es" ? "Ventanas analizadas" : "Analyzed windows", result.num_windows ? `${result.num_windows} × ${result.window_seconds} s · ${result.window_aggregation || "max"}` : "—"),
    row(LANG === "es" ? "Derivaciones" : "Leads", Array.isArray(result.lead_names) ? result.lead_names.join(", ") : result.lead_names),
    row("DX", result.dx_codes && result.dx_codes.length ? result.dx_codes.join(", ") : "—"),
    detSub(tr("detail.groupDecision")),
    row(LANG === "es" ? "Modelo" : "Model", result.model_type),
    row("Checkpoint", shortPath(result.model_path)),
    row(LANG === "es" ? "Época checkpoint" : "Checkpoint epoch", result.checkpoint_epoch),
    row("Val loss", fmtNum(result.checkpoint_val_loss)),
    row("Threshold", result.using_class_thresholds ? (LANG === "es" ? "calibrado por clase" : "class-calibrated") : result.threshold),
    row(LANG === "es" ? "Fuente de umbrales" : "Threshold source", baseName(result.threshold_source)),
    row(LANG === "es" ? "Calibración" : "Calibration", result.calibrated ? `${LANG === "es" ? "temperature scaling por clase" : "per-class temperature scaling"} (${baseName(result.temperature_source)})` : (LANG === "es" ? "sin calibrar" : "uncalibrated")),
    row(LANG === "es" ? "Clases reales" : "True classes", cmp.available ? listOrNone(cmp.true_classes) : "—"),
    row(LANG === "es" ? "Clases predichas" : "Predicted classes", cmp.predicted_classes ? listOrNone(cmp.predicted_classes) : "—"),
    rowHtml(LANG === "es" ? "Coincidencia exacta" : "Exact match", exactBadge),
    row("Fallback NSR", result.normal_fallback && result.normal_fallback.applied ? `${LANG === "es" ? "aplicado" : "applied"} · min=${result.normal_fallback.min_nsr_probability}` : (LANG === "es" ? "no aplicado" : "not applied")),
    detSub(tr("detail.groupPreproc")),
    row(LANG === "es" ? "Preprocesamiento" : "Preprocessing", details.preprocessing),
    row(LANG === "es" ? "Normalización" : "Normalization", details.norm_mode || "—"),
    row(LANG === "es" ? "Estilo de trazado" : "Trace style", details.plot_style),
    row(LANG === "es" ? "Esquema" : "Schema", details.label_schema),
    row(LANG === "es" ? "Activación" : "Activation", details.activation),
    row(LANG === "es" ? "Pérdida" : "Loss", details.loss),
    row(LANG === "es" ? "Regla de decisión" : "Decision rule", details.decision_rule),
    row(LANG === "es" ? "Dispositivo" : "Device", details.device),
  ].join("");

  const rows = result.predictions || [];
  probTable.innerHTML = `
    <thead><tr><th>${tr("prob.index")}</th><th>${tr("table.class")}</th><th>${tr("table.description")}</th><th>${tr("prob.prob")}</th><th>${tr("prob.threshold")}</th><th>${tr("prob.margin")}</th><th>${tr("prob.state")}</th></tr></thead>
    <tbody>
      ${rows.map(r => {
        const color = PALETTE[r.index % PALETTE.length];
        const width = Math.max(0, Math.min(100, Number(r.probability || 0) * 100));
        const pos = Number(r.prediction) === 1;
        return `<tr>
          <td class="iv-idx">${escapeHtml(r.index)}</td>
          <td class="prob-name" title="SNOMED: ${escapeHtml(Array.isArray(r.snomed_codes) ? r.snomed_codes.join(", ") : (r.snomed_codes || ""))}">${escapeHtml(r.class)}</td>
          <td>${escapeHtml(r.display_name || r.description || "")}</td>
          <td><span class="prob-bar"><span class="prob-fill" style="width:${width}%;background:${color}"></span></span>${fmtPct(r.probability)}</td>
          <td>${fmtPct(r.threshold ?? result.threshold)}</td>
          <td>${Number(r.margin ?? 0).toFixed(3)}${r.near_threshold ? ' · ~' : ''}</td>
          <td><span class="badge ${pos ? "ok" : "no"}">${pos ? tr("prob.yes") : tr("prob.no")}</span>${r.postprocessed ? '<span class="badge post">fallback</span>' : ''}</td>
        </tr>`;
      }).join("")}
    </tbody>`;
  const probNote = $("probNote");
  if (probNote) {
    probNote.textContent = tr("prob.marginNote");
    probNote.hidden = false;
  }
}

function render(result, updateModel = true) {
  lastResult = result;
  if (updateModel) setModelNameFromResult(result);
  visEmpty.hidden = true;
  visContent.hidden = false;
  detEmpty.hidden = true;
  detContent.hidden = false;
  if (skel) skel.hidden = true;

  renderNotice(result);
  renderDiagnosis(result);
  renderTraffic(result);
  renderSummary(result);
  renderComparison(result);
  renderPlot(result);
  renderButtons(result);
  renderTechnical(result);
  saveHistory(result);
  renderHistory();
}

// ============================================================ charts ======
let barsLoaded = false;
let curvesCache = null;

function plotFontColor() {
  return document.documentElement.dataset.theme === "dark" ? "#e2e8f0" : "#1e293b";
}

function plotlyAvailable() {
  return typeof Plotly !== "undefined";
}

async function loadMetricBars() {
  const box = $("metricBars");
  if (!box || barsLoaded) return;
  barsLoaded = true;
  if (!plotlyAvailable()) {
    box.innerHTML = `<p class="hint muted">Plotly no disponible.</p>`;
    return;
  }
  try {
    const resp = await fetch("/metrics");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const rows = (data && data.per_class_test) || [];
    if (!rows.length) {
      box.innerHTML = `<p class="hint muted">${escapeHtml(tr("bars.noData"))}</p>`;
      return;
    }
    const x = rows.map(r => r.class);
    const f1 = rows.map(r => Number(r.f1));
    const auroc = rows.map(r => (r.auroc === null || r.auroc === undefined) ? NaN : Number(r.auroc));
    const font = { color: plotFontColor(), size: 11 };
    Plotly.newPlot(box, [
      { x, y: f1, name: tr("bars.f1"), type: "bar", marker: { color: "#2563eb" } },
      { x, y: auroc, name: tr("bars.auroc"), type: "bar", marker: { color: "#16a34a" } },
    ], {
      title: { text: tr("bars.title"), font: { ...font, size: 13 } },
      barmode: "group", height: 340, margin: { t: 50, b: 110 },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font, xaxis: { tickangle: -30 }, yaxis: { range: [0, 1] },
    }, { responsive: true, displaylogo: false });
  } catch (err) {
    box.innerHTML = `<p class="hint muted">${escapeHtml(tr("bars.noData"))}</p>`;
  }
}

async function loadCurves() {
  const sel = $("curveClass");
  if (!sel) return;
  if (curvesCache) {
    if (sel.value) drawClassCurves(sel.value);
    return;
  }
  if (!plotlyAvailable()) return;
  try {
    const resp = await fetch("/curves");
    const data = await resp.json();
    if (!data || !data.found) return;
    curvesCache = data;
    try {
      const mresp = await fetch("/metrics");
      if (mresp.ok) {
        const mdata = await mresp.json();
        curvesCache.stats = {};
        ((mdata && mdata.per_class_test) || []).forEach(r => {
          curvesCache.stats[r.class] = r;
        });
      }
    } catch (err) { /* sin stats: las curvas igual se dibujan */ }
    sel.innerHTML = data.classes.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
    sel.onchange = () => drawClassCurves(sel.value);
    if (data.classes.length) drawClassCurves(data.classes[0]);
  } catch (err) { /* la sección queda con placeholders */ }
}

function drawClassCurves(cls) {
  if (!curvesCache || !plotlyAvailable()) return;
  const curve = curvesCache.curves[cls];
  const rocBox = $("rocPlot");
  const prBox = $("prPlot");
  if (!curve || !rocBox || !prBox) return;
  const statsBox = $("curveStats");
  const stats = curvesCache.stats && curvesCache.stats[cls];
  if (statsBox) {
    if (stats && stats.auroc !== null && stats.auroc !== undefined) {
      const fmt3 = v => (v === null || v === undefined || Number.isNaN(Number(v))) ? "—" : Number(v).toFixed(3);
      statsBox.textContent = `${cls} · AUROC ${fmt3(stats.auroc)} · AUPRC ${fmt3(stats.auprc)} · F1 ${fmt3(stats.f1)} (test)`;
      statsBox.hidden = false;
    } else {
      statsBox.hidden = true;
    }
  }
  const font = { color: plotFontColor(), size: 11 };
  const base = { height: 320, margin: { t: 50, r: 20 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font, showlegend: false };
  if (curve.fpr && curve.tpr) {
    Plotly.newPlot(rocBox, [
      { x: curve.fpr, y: curve.tpr, type: "scatter", mode: "lines", line: { color: "#2563eb", width: 2.5 } },
      { x: [0, 1], y: [0, 1], type: "scatter", mode: "lines", line: { color: "#94a3b8", dash: "dash" } },
    ], { ...base, title: { text: `${tr("curves.roc")} · ${cls}`, font: { ...font, size: 13 } },
        xaxis: { title: tr("curves.fpr"), range: [0, 1] }, yaxis: { title: tr("curves.tpr"), range: [0, 1] } },
      { responsive: true, displaylogo: false });
  }
  if (curve.precision && curve.recall) {
    Plotly.newPlot(prBox, [
      { x: curve.recall, y: curve.precision, type: "scatter", mode: "lines", line: { color: "#dc2626", width: 2.5 } },
    ], { ...base, title: { text: `${tr("curves.pr")} · ${cls}`, font: { ...font, size: 13 } },
        xaxis: { title: tr("curves.recall"), range: [0, 1] }, yaxis: { title: tr("curves.precision"), range: [0, 1] } },
      { responsive: true, displaylogo: false });
  }
}

// ============================================================ history =====
const HISTORY_KEY = "ecg-history";
const HISTORY_MAX = 20;

function getHistory() {
  try {
    const raw = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(raw) ? raw : [];
  } catch (e) { return []; }
}

function saveHistory(result) {
  if (!result || !result.record_name) return;
  const positives = (result.positive_predictions || []).map(p => ({ c: p.class, p: Number(Number(p.probability).toFixed(4)) }));
  const sig = JSON.stringify({ r: result.record_name, p: positives });
  const entry = { ts: Date.now(), record: result.record_name, positives,
    pdf: result.pdf_url || null, csv: result.converted_csv_url || null };
  const rest = getHistory().filter(e => JSON.stringify({ r: e.record, p: e.positives }) !== sig);
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify([entry, ...rest].slice(0, HISTORY_MAX))); } catch (e) {}
}

function renderHistory() {
  const box = $("historyBox");
  if (!box) return;
  const items = getHistory();
  if (!items.length) {
    box.innerHTML = `<p class="hint muted">${escapeHtml(tr("history.empty"))}</p>`;
    return;
  }
  box.innerHTML = items.map(e => {
    const when = new Date(e.ts).toLocaleString(LANG === "es" ? "es-ES" : "en-US");
    const pos = (e.positives && e.positives.length)
      ? e.positives.map(p => `${escapeHtml(p.c)} ${(Number(p.p) * 100).toFixed(1)}%`).join(", ")
      : escapeHtml(tr("history.none"));
    const links = `${e.pdf ? `<a href="${escapeHtml(e.pdf)}" download>${escapeHtml(tr("history.pdf"))}</a>` : ""}${e.csv ? ` <a href="${escapeHtml(e.csv)}" download>${escapeHtml(tr("history.csv"))}</a>` : ""}`;
    return `<div class="history-item"><strong>${escapeHtml(e.record)}</strong><span>${escapeHtml(tr("history.positives"))}: ${pos}</span><time>${escapeHtml(when)}</time><span class="h-links">${links}</span></div>`;
  }).join("");
}

(function initHistory() {
  const btn = $("btnClearHistory");
  if (btn) btn.addEventListener("click", () => {
    try { localStorage.removeItem(HISTORY_KEY); } catch (e) {}
    renderHistory();
  });
  renderHistory();
})();

applyI18n();
