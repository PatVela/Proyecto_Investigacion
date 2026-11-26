# -*- coding: utf-8 -*-
"""R11 (MM-01): comparación EJECUTADA del modelo publicado de Ribeiro et al. (pesos de Zenodo, reimplementado en PyTorch y validado en validar_port_ribeiro.py) con el sistema principal P3 de este trabajo
sobre los MISMOS registros de Chapman-Shaoxing (C) y Ningbo (D) que evaluó P3 en la prueba externa (records de runs/t19/eval/p3_C y p3_D).
Diseño fijado antes de ver resultados de C y D (ver PROTOCOLO_COMPARACION_RIBEIRO_R11.md): el modelo de Ribeiro NO se ajusta ni se calibra con C o D; entrada en mV (decisión tomada con la amplitud, sin etiquetas);
remuestreo 500→400 Hz, relleno simétrico con ceros a 4096 muestras (como su conjunto de prueba), sin filtrado; derivaciones en el orden I, II, III, aVR, aVL, aVF, V1–V6 (el que reproduce sus predicciones publicadas).
Esta comparación es POSTERIOR a los resultados de C y D de P3 y por tanto EXPLORATORIA: no forma parte del protocolo sellado y no reemplaza la evaluación primaria.
Métricas: AUROC y AUPRC por clase (sin umbral; métrica principal) con IC del 95 % por bootstrap de registros (1 000 remuestreos, semilla 0, los mismos para ambos sistemas, diferencia pareada); F1 informativo con el umbral de validación de P3
y 0.5 para Ribeiro (sus umbrales no se publican con el modelo) y, como cota, el mejor umbral posible en esos mismos datos para ambos.
Dos definiciones de etiqueta: (R) las de Ribeiro, derivadas de los códigos SNOMED de la cabecera de cada registro; (G) las agrupadas de este trabajo (las etiquetas de predictions_test.csv).
Solo ST↔STach tiene el mismo código en ambos; AF y SB son clases de P3 con más códigos; 1dAVb, RBBB y LBBB son subconjuntos de AVBlock y BBB: esas filas son descriptivas, no una comparación justa.
Dos variantes de entrada del modelo de Ribeiro, ambas fijadas de antemano: A = su entrada propia (mV sin filtrar, 400 Hz, relleno simétrico); B = el preprocesamiento de este trabajo
(ecg.signal.load_wfdb_record_windows: mV, 500 Hz, pasa-banda 0.5–50 Hz, recorte ±5 mV) seguido de remuestreo a 400 Hz y relleno a 4096 (es lo que el informe del profesor pide: «under the authors' preprocessing»).
Uso: python comparar_ribeiro_cd.py <datos_ribeiro> <datos_challenge2021> <eval P3 (runs/t19/eval)> <salida> <raíz del repositorio (con ecg/)> [max_registros_por_origen]"""
import csv, hashlib, json, os, re, sys, time
from math import gcd
from pathlib import Path
import numpy as np, torch
from scipy.io import loadmat
from scipy.signal import resample_poly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ribeiro_pytorch import RibeiroResNet, CLASES

RIB, CH, EVAL, OUT, REPO = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
MAXN = int(sys.argv[6]) if len(sys.argv) > 6 else None
ORIG = os.environ.get("R11_ORIGENES", "CD")                  # «C», «D» o «CD»: permite ejecutar un origen mientras el otro se descarga; el remuestreo (semilla 0) es por origen, de modo que el resultado no cambia
SUF = "" if ORIG == "CD" else "_" + ORIG
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
from ecg import signal as _sg
from ecg.signal import load_wfdb_record_windows          # preprocesamiento EXACTO de este trabajo (reordena derivaciones, mV, 500 Hz, pasa-banda 0.5-50 Hz, recorte ±5 mV)
torch.set_num_threads(8)
BASE = CH / "physionet.org" / "files" / "challenge-2021" / "1.0.3" / "training"
CODIGOS = {"1dAVb": {"270492004"}, "RBBB": {"59118001", "713427006"}, "LBBB": {"164909002", "733534002"}, "SB": {"426177001"}, "AF": {"164889003"}, "ST": {"427084000"}}
CLASE_NUESTRA = {"1dAVb": "AVBlock", "RBBB": "BBB", "LBBB": "BBB", "SB": "SB", "AF": "AF", "ST": "STach"}
EQUIV = {"ST": "equivalente (mismo código SNOMED)", "AF": "P3 agrupa más códigos (flutter y otros)", "SB": "P3 agrupa más códigos (otros de bradicardia y nodo sinusal)",
         "1dAVb": "no comparable: AVBlock de P3 incluye otros bloqueos", "RBBB": "no comparable: BBB de P3 agrupa derecha, izquierda e incompletos", "LBBB": "no comparable: BBB de P3 agrupa derecha, izquierda e incompletos"}
m = RibeiroResNet(RIB / "extraido" / "model" / "model.hdf5")

def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest().upper()
def archivo_senal(hea):
    """Archivo de señal que nombra la cabecera (casi siempre <registro>.mat; en Ningbo el registro S23074 nombra JS23074.mat)."""
    return hea.parent / hea.read_text().splitlines()[1].split()[0]
def leer(hea):
    lin = hea.read_text().splitlines()
    cab = lin[0].split(); fs = float(cab[2])
    gan = float(re.search(r"([\d.]+)\(", lin[1]).group(1))
    dx = set()
    for l in lin:
        if l.startswith("# Dx:") or l.startswith("#Dx:"):
            dx = {c.strip() for c in l.split(":", 1)[1].split(",") if c.strip()}
    v = loadmat(str(archivo_senal(hea)))["val"].astype(np.float32) / gan        # mV
    return v, fs, dx
PLANAS = []
def ventana_b(mat, hea):
    """Ventana de este trabajo para el registro. load_wfdb_record_windows es la ruta de INFERENCIA y rechaza (ValueError) los registros con más de la mitad de las derivaciones planas; la evaluación de P3 se hizo con los HDF5 de
    build_external_test.py, que NO descartan registros por eso (ecg/signal.py, require_usable_signal), y P3 sí predijo esos registros. Para la entrada B se replica entonces el mismo preprocesamiento sin la comprobación de utilidad."""
    try:
        v, _ = load_wfdb_record_windows(mat, hea, windows_max=1); return v
    except ValueError as e:
        if "plana" not in str(e): raise
        PLANAS.append(Path(hea).stem)
        meta = _sg.parse_header(hea); raw = _sg.load_mat_signal(mat, meta["num_leads"])
        full = _sg.preprocess_ecg_array(raw, sampling_rate=meta["sampling_freq"], lead_names=meta["lead_names"], target_fs=_sg.TARGET_FS, target_length=None, norm_mode=_sg.DEFAULT_NORM_MODE,
                                        adc_gains=meta.get("adc_gains"), adc_baselines=meta.get("adc_baselines"), units="digital", bandpass=True)
        ini = _sg.compute_window_starts(full.shape[0], _sg.WINDOW_LENGTH, 1)[0]
        pieza = full[ini:ini + _sg.WINDOW_LENGTH, :]
        if pieza.shape[0] < _sg.WINDOW_LENGTH: pieza = np.pad(pieza, ((0, _sg.WINDOW_LENGTH - pieza.shape[0]), (0, 0)), mode="constant")
        return pieza.astype(np.float32)[None]
def a_ribeiro(v, fs):
    if abs(fs - 400) > 1e-6:
        a, b = int(round(400)), int(round(fs)); g = gcd(a, b)
        v = resample_poly(v, a // g, b // g, axis=1).astype(np.float32)
    L = v.shape[1]
    if L >= 4096:
        i = (L - 4096) // 2; v = v[:, i:i + 4096]
    else:
        izq = (4096 - L) // 2; v = np.pad(v, ((0, 0), (izq, 4096 - L - izq)))
    return v.T                                                                       # (4096, 12)

def rangos_auroc(y, p):
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    o = np.argsort(p, kind="stable"); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    r = (np.bincount(inv, weights=r) / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def ap(y, p):
    o = np.argsort(-p, kind="stable"); ys = y[o]
    tp = np.cumsum(ys); prec = tp / np.arange(1, len(ys) + 1)
    return np.nan if ys.sum() == 0 else float((prec * ys).sum() / ys.sum())
def f1(y, pred):
    tp = int((y & pred).sum()); fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
def mejor_f1(y, p):
    o = np.argsort(-p, kind="stable"); ys = y[o]; tp = np.cumsum(ys); k = np.arange(1, len(ys) + 1); fn = ys.sum() - tp
    return float(np.max(2 * tp / (2 * tp + (k - tp) + fn)))

ARR = {}
t0 = time.time(); meta = {"modelo_md5": hashlib.md5((RIB / "extraido" / "model" / "model.hdf5").read_bytes()).hexdigest(), "pesos_sha256": sha(RIB / "extraido" / "model" / "model.hdf5"), "origenes": {}}
filas = [["variante_de_entrada_de_Ribeiro", "origen", "definicion_de_etiqueta", "clase_Ribeiro", "clase_P3", "equivalencia", "n", "positivos", "AUROC_P3", "AUROC_Ribeiro", "dif_AUROC_R_menos_P3", "IC95_dif_AUROC", "AUPRC_P3", "AUPRC_Ribeiro", "dif_AUPRC_R_menos_P3", "IC95_dif_AUPRC",
          "F1_P3_umbral_validacion", "F1_Ribeiro_umbral_0.5", "F1_P3_mejor_umbral_en_estos_datos", "F1_Ribeiro_mejor_umbral_en_estos_datos"]]
for origen, (carpeta, p3) in {"C": ("chapman_shaoxing", "p3_C"), "D": ("ningbo", "p3_D")}.items():
    if origen not in ORIG: continue
    pr = EVAL / p3 / "predictions_test.csv"
    cab = next(csv.reader(open(pr, encoding="utf-8"))); filas_p3 = list(csv.DictReader(open(pr, encoding="utf-8")))
    nombres = [r["record_name"] for r in filas_p3][:MAXN]
    idx = {h.stem: h for g in sorted((BASE / carpeta).glob("g*")) for h in g.glob("*.hea")}
    ok, faltan, probs, dxs = [], [], {"A": [], "B": []}, []
    lote = {"A": [], "B": []}
    def vaciar():
        if lote["A"]:
            with torch.no_grad():
                for v_ in ("A", "B"):
                    probs[v_].append(m(torch.from_numpy(np.stack(lote[v_]))).numpy()); lote[v_] = []
    for k, nom in enumerate(nombres):
        if nom not in idx:
            faltan.append(nom); continue
        v, fs, dx = leer(idx[nom])
        lote["A"].append(a_ribeiro(v, fs))                                                   # variante A: entrada propia del modelo (sin filtro)
        ventana = ventana_b(archivo_senal(idx[nom]), idx[nom])   # variante B: preprocesamiento de este trabajo
        lote["B"].append(a_ribeiro(ventana[0].T.copy(), 500.0))
        ok.append(nom); dxs.append(dx)
        if len(lote["A"]) == 64: vaciar()
        if (k + 1) % 2000 == 0: print(f"[{time.time() - t0:6.0f} s] {origen}: {k + 1}/{len(nombres)} registros", flush=True)
    vaciar()
    if not ok:
        print(f"{origen}: ningún registro encontrado en {carpeta}; se omite (descarga incompleta)"); continue
    PV = {v_: np.concatenate(probs[v_]) for v_ in ("A", "B")}; pos = {n: i for i, n in enumerate(ok)}
    P = PV["A"]
    sel = [r for r in filas_p3 if r["record_name"] in pos]
    assert len(sel) == len(ok) == len(P), (len(sel), len(ok), len(P))
    for v_, P_ in PV.items():
        with open(OUT / f"probabilidades_ribeiro_{origen}_variante{v_}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n"); w.writerow(["record_name"] + CLASES); [w.writerow([n] + [f"{x:.7g}" for x in p]) for n, p in zip(ok, P_)]
    meta["origenes"][origen] = {"registros_con_senal_plana_preprocesados_sin_comprobacion_de_utilidad": list(PLANAS), "predicciones_P3_sha256": sha(pr), "registros_en_P3": len(filas_p3), "registros_evaluados": len(ok), "registros_sin_archivo": len(faltan), "primeros_sin_archivo": faltan[:10]}
    rng = np.random.default_rng(0); B = 1000; n = len(ok); ib = [rng.integers(0, n, n) for _ in range(B)]
    for var, cls_r in [(a_, b_) for a_ in ("A", "B") for b_ in CLASES]:
        P = PV[var]
        c = CLASES.index(cls_r); cn = CLASE_NUESTRA[cls_r]
        p_r = P[:, c].astype(np.float64); p_n = np.array([float(r[f"prob_{cn}"]) for r in sel])
        umbral_n = float(next(r for r in csv.DictReader(open(EVAL / p3 / "thresholds_validation.csv", encoding="utf-8")) if r["class"] == cn)["threshold"])
        for defn in ("R", "G"):
            y = np.array([(len(dx & CODIGOS[cls_r]) > 0) for dx in dxs]) if defn == "R" else np.array([int(r[f"true_{cn}"]) == 1 for r in sel])
            if defn == "G" and cls_r in ("RBBB", "LBBB"): continue                      # la etiqueta agrupada BBB no separa derecha de izquierda
            au_n, au_r, ap_n, ap_r = rangos_auroc(y, p_n), rangos_auroc(y, p_r), ap(y, p_n), ap(y, p_r)
            d_au = [rangos_auroc(y[i], p_r[i]) - rangos_auroc(y[i], p_n[i]) for i in ib]; d_ap = [ap(y[i], p_r[i]) - ap(y[i], p_n[i]) for i in ib]
            q = lambda v: f"{np.nanpercentile(v, 2.5):.4f} a {np.nanpercentile(v, 97.5):.4f}"
            ARR[(var, origen, defn, cls_r)] = dict(au_n=au_n, au_r=au_r, ap_n=ap_n, ap_r=ap_r, d_au=np.array(d_au), d_ap=np.array(d_ap), f1_n=f1(y, p_n >= umbral_n), f1_r=f1(y, p_r >= 0.5), n=n, pos=int(y.sum()))
            filas.append([var, origen, defn, cls_r, cn, EQUIV[cls_r], n, int(y.sum()), f"{au_n:.4f}", f"{au_r:.4f}", f"{au_r - au_n:+.4f}", q(d_au), f"{ap_n:.4f}", f"{ap_r:.4f}", f"{ap_r - ap_n:+.4f}", q(d_ap),
                          f"{f1(y, p_n >= umbral_n):.4f}", f"{f1(y, p_r >= 0.5):.4f}", f"{mejor_f1(y, p_n):.4f}", f"{mejor_f1(y, p_r):.4f}"])
    for var in ("A", "B"):                                                                      # macro sobre las tres clases comparables o aproximadas (ST, AF, SB)
        for defn in ("R", "G"):
            g = [ARR[(var, origen, defn, c_)] for c_ in ("ST", "AF", "SB")]
            q = lambda v: f"{np.nanpercentile(v, 2.5):.4f} a {np.nanpercentile(v, 97.5):.4f}"
            m_au_n, m_au_r = np.mean([x["au_n"] for x in g]), np.mean([x["au_r"] for x in g]); m_ap_n, m_ap_r = np.mean([x["ap_n"] for x in g]), np.mean([x["ap_r"] for x in g])
            d_au = np.mean([x["d_au"] for x in g], axis=0); d_ap = np.mean([x["d_ap"] for x in g], axis=0)
            filas.append([var, origen, defn, "macro(ST,AF,SB)", "macro(STach,AF,SB)", "tres clases: una equivalente y dos aproximadas", n, "", f"{m_au_n:.4f}", f"{m_au_r:.4f}", f"{m_au_r - m_au_n:+.4f}", q(d_au),
                          f"{m_ap_n:.4f}", f"{m_ap_r:.4f}", f"{m_ap_r - m_ap_n:+.4f}", q(d_ap), f"{np.mean([x['f1_n'] for x in g]):.4f}", f"{np.mean([x['f1_r'] for x in g]):.4f}", "", ""])
    print(f"[{time.time() - t0:6.0f} s] {origen}: listo ({len(ok)} registros evaluados, {len(faltan)} sin archivo)", flush=True)
with open(OUT / f"comparacion_ribeiro_vs_P3{SUF}.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f, lineterminator="\n").writerows(filas)
meta["tiempo_s"] = round(time.time() - t0); meta["registros_maximos_por_origen"] = MAXN
(OUT / f"proveniencia{SUF}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
for r in filas[1:]:
    print(" | ".join(str(x) for x in (r[0], r[1], r[2], r[3], "n=" + str(r[6]), "pos=" + str(r[7]), "AUROC P3 " + r[8], "Rib " + r[9], "dif " + r[10] + " [" + r[11] + "]", "AP P3 " + r[12], "Rib " + r[13])))
