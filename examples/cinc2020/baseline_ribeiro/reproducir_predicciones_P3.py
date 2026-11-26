# -*- coding: utf-8 -*-
"""R11 (MM-10, MM-11, MM-44): REPRODUCE EN CPU, con los pesos publicados del bundle (models/resnet34_v1.pt y v2.pt) y la calibración publicada (deploy/platt_a.csv, platt_b.csv, platt_ensemble.csv),
las probabilidades del sistema principal P3 que la evaluación guardó en runs/t19/eval/p3_C y p3_D (predictions_test.csv), a partir de los registros públicos del Challenge 2021 descargados de nuevo.
Es una reproducción de la INFERENCIA (nivel 4 para esos registros): reejecuta la red; no reentrena ni vuelve a calcular métricas del artículo. Se hace sobre una muestra de los primeros N registros de cada origen.
Uso: python reproducir_predicciones_P3.py <raíz del repositorio (con models/ y deploy/)> <datos_challenge2021> <eval P3 (runs/t19/eval)> <salida.csv> [N por origen]"""
import csv, sys, time
from pathlib import Path
import numpy as np, torch
REPO, CH, EVAL, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
N = int(sys.argv[5]) if len(sys.argv) > 5 else 300
sys.path.insert(0, str(REPO))
from ecg import calibration, load, predict
from ecg.signal import load_wfdb_record_windows
torch.set_num_threads(8)
dev = torch.device("cpu")
mA, _, names = predict.load_model(REPO / "models" / "resnet34_v1.pt", device=dev)
mB, _, names_b = predict.load_model(REPO / "models" / "resnet34_v2.pt", device=dev)
assert names == names_b == load.CLASS_NAMES
a1, b1 = calibration.load_platt_csv(REPO / "deploy" / "platt_a.csv", names); a2, b2 = calibration.load_platt_csv(REPO / "deploy" / "platt_b.csv", names)
aE, bE = calibration.load_platt_csv(REPO / "deploy" / "platt_ensemble.csv", names)
BASE = CH / "physionet.org" / "files" / "challenge-2021" / "1.0.3" / "training"
filas = [["origen", "registros_comparados", "dif_max_abs_probabilidad", "dif_media_abs_probabilidad", "celdas_clase_registro_con_decision_distinta", "celdas_totales", "tiempo_s"]]
detalle = []
for origen, (carpeta, p3) in {"C": ("chapman_shaoxing", "p3_C"), "D": ("ningbo", "p3_D")}.items():
    guardadas = {r["record_name"]: r for r in csv.DictReader(open(EVAL / p3 / "predictions_test.csv", encoding="utf-8"))}
    idx = {h.stem: h for g in sorted((BASE / carpeta).glob("g*")) for h in g.glob("*.hea")}
    nombres = [n for n in guardadas if n in idx][:N]
    if not nombres:
        print(origen, ": sin registros descargados"); continue
    t0 = time.time(); difs = []; cambios = 0
    for n in nombres:
        v, _ = load_wfdb_record_windows(idx[n].parent / idx[n].read_text().splitlines()[1].split()[0], idx[n], windows_max=1)   # el archivo de señal que nombra la cabecera (Ningbo S23074 nombra JS23074.mat)
        p1 = predict.predict_array(mA, v[0], dev)[None]; p2 = predict.predict_array(mB, v[0], dev)[None]
        pe = calibration.apply_platt(calibration.ensemble_probabilities(calibration.apply_platt(p1, a1, b1), calibration.apply_platt(p2, a2, b2), 0.5), aE, bE)[0]
        g = guardadas[n]
        for c, nombre in enumerate(names):
            d = abs(float(pe[c]) - float(g[f"prob_{nombre}"])); difs.append(d)
            if int(float(pe[c]) >= float(next(r for r in csv.DictReader(open(EVAL / p3 / "thresholds_validation.csv", encoding="utf-8")) if r["class"] == nombre)["threshold"])) != int(g[f"pred_{nombre}"]): cambios += 1
    filas.append([origen, len(nombres), f"{max(difs):.3e}", f"{np.mean(difs):.3e}", cambios, len(difs), f"{time.time() - t0:.0f}"])
    print(filas[-1], flush=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f, lineterminator="\n").writerows(filas)
