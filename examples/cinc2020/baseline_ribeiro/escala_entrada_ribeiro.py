# -*- coding: utf-8 -*-
"""R11 (MM-01): decide la unidad de entrada del modelo de Ribeiro SIN mirar etiquetas ni resultados de C y D: compara la distribución de amplitudes de su conjunto de prueba (CODE-test, unidades de los autores)
con la de registros de Chapman-Shaoxing convertidos a milivoltios con la ganancia de su cabecera. Si las medianas de la amplitud pico por registro son del mismo orden, se alimenta en mV.
Uso: python escala_entrada_ribeiro.py <datos_ribeiro> <carpeta g1 de chapman_shaoxing> <salida.json>"""
import json, re, sys
from pathlib import Path
import h5py, numpy as np
from scipy.io import loadmat
D, G1, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
with h5py.File(D / "extraido" / "data" / "ecg_tracings.hdf5", "r") as f:
    X = np.array(f["tracings"], dtype=np.float32)
pico_code = np.percentile(np.abs(X), 99, axis=(1, 2))
heas = sorted(G1.glob("*.hea"))[:400]
pico_ch = []
for h in heas:
    gan = float(re.search(r"([\d.]+)\(", h.read_text().splitlines()[1]).group(1))
    v = loadmat(str(h.with_suffix(".mat")))["val"].astype(np.float32) / gan      # mV
    pico_ch.append(np.percentile(np.abs(v), 99))
pico_ch = np.array(pico_ch)
res = {"CODE-test_mediana_pico_p99": float(np.median(pico_code)), "CODE-test_p25_p75": [float(np.percentile(pico_code, 25)), float(np.percentile(pico_code, 75))],
       "Chapman_g1_mV_mediana_pico_p99": float(np.median(pico_ch)), "Chapman_g1_p25_p75": [float(np.percentile(pico_ch, 25)), float(np.percentile(pico_ch, 75))], "registros_chapman": len(pico_ch),
       "razon_CODE_sobre_Chapman_mV": float(np.median(pico_code) / np.median(pico_ch))}
print(json.dumps(res, indent=1))
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
