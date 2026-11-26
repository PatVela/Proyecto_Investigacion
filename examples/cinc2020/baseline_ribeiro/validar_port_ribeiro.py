# -*- coding: utf-8 -*-
"""R11 (MM-01): valida la reimplementación en PyTorch (ribeiro_pytorch.py) contra las predicciones PUBLICADAS por los autores (dnn_predicts/model.npy) sobre su conjunto de prueba
(CODE-test, 827 trazados, Zenodo 10.5281/zenodo.3625006). Cuenta como evidencia de que el port reproduce el modelo publicado: diferencia máxima y media entre probabilidades, y coincidencia de AUROC
con la anotación de referencia (gold_standard.csv). Se prueban los dos órdenes de derivaciones que los README de los autores dan (AVR,AVL,AVF y AVL,AVF,AVR).
Uso:  python validar_port_ribeiro.py <carpeta datos_ribeiro> <carpeta de salida>"""
import csv, sys
from pathlib import Path
import h5py, numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ribeiro_pytorch import RibeiroResNet   # R11B: se quitó «CLASES» (sin usar); pyflakes, que tests/test_static_analysis.py exige sobre los .py versionados, lo señalaba

D, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
with h5py.File(D / "extraido" / "data" / "ecg_tracings.hdf5", "r") as f:
    X = np.array(f["tracings"], dtype=np.float32)
pub = np.load(D / "model_published_predictions.npy")
oro = np.array([[int(v) for v in r] for r in list(csv.reader(open(D / "extraido" / "data" / "annotations" / "gold_standard.csv")))[1:]])
print("tracings:", X.shape, "| publicadas:", pub.shape, "| oro:", oro.shape, "| amplitud |x| p99 =", float(np.percentile(np.abs(X), 99)))
def auroc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    # rangos con empates promediados
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    sumas = np.bincount(inv, weights=r); r = (sumas / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
m = RibeiroResNet(D / "extraido" / "model" / "model.hdf5")
ordenes = {"DI,DII,DIII,AVR,AVL,AVF,V1-V6 (README principal)": list(range(12)),
           "DI,DII,DIII,AVL,AVF,AVR,V1-V6 (README de datos y pesos)": [0, 1, 2, 4, 5, 3, 6, 7, 8, 9, 10, 11]}
filas = [["orden_de_derivaciones_alimentado", "diferencia_maxima_abs", "diferencia_media_abs", "correlacion", "auroc_port_vs_oro_macro", "auroc_publicadas_vs_oro_macro"]]
for nombre, perm in ordenes.items():
    P = np.concatenate([m(torch.from_numpy(X[i:i + 64][:, :, perm])).numpy() for i in range(0, len(X), 64)])
    dif = np.abs(P - pub)
    aur = [auroc(oro[:, c], P[:, c]) for c in range(6)]; aup = [auroc(oro[:, c], pub[:, c]) for c in range(6)]
    filas.append([nombre, f"{dif.max():.3e}", f"{dif.mean():.3e}", f"{np.corrcoef(P.ravel(), pub.ravel())[0, 1]:.6f}", f"{np.mean(aur):.4f}", f"{np.mean(aup):.4f}"])
    print(nombre, "| dif máx", f"{dif.max():.3e}", "| dif media", f"{dif.mean():.3e}", "| AUROC macro port", f"{np.mean(aur):.4f}", "publicadas", f"{np.mean(aup):.4f}")
    if nombre.startswith("DI,DII,DIII,AVR") : np.save(OUT / "predicciones_port_orden_README_principal.npy", P)
with open(OUT / "validacion_port_ribeiro.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f, lineterminator="\n").writerows(filas)
