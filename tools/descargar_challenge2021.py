# -*- coding: utf-8 -*-
"""R11 (MM-01, MM-44): descarga reanudable de Chapman-Shaoxing y Ningbo del PhysioNet/CinC Challenge 2021 v1.0.3 (datos públicos, CC BY 4.0; ≈ 5.4 GB; 46 subcarpetas gN).
Equivale a descargar_challenge2021.sh del repositorio de los autores (wget -r), pero con la biblioteca estándar de Python, hilos y reanudación por archivo (no vuelve a bajar lo que ya existe).
Conserva la estructura physionet.org/files/challenge-2021/1.0.3/training/<origen>/<gN>/ para que tools/verificar_challenge2021.py y build_external_test.py funcionen igual.
Uso:  python descargar_challenge2021.py <carpeta destino> [hilos] [grupos...]   (grupos, p. ej. chapman_shaoxing/g1; sin grupos = los 46)"""
import re, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://physionet.org/files/challenge-2021/1.0.3/training"
DEST = Path(sys.argv[1]); HILOS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
GRUPOS = sys.argv[3:] or [f"chapman_shaoxing/g{i}" for i in range(1, 12)] + [f"ningbo/g{i}" for i in range(1, 36)]
UA = {"User-Agent": "Mozilla/5.0 (descarga de datos publicos de PhysioNet; uso academico)"}

def get(url, intentos=6, binario=True):
    for k in range(intentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:
            time.sleep(2 * (k + 1))
            ultimo = e
    raise RuntimeError(f"{url}: {ultimo}")

def listar(grupo):
    html = get(f"{BASE}/{grupo}/").decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="([A-Za-z0-9_\-\.]+\.(?:hea|mat))"', html)))

def bajar(grupo, nombre):
    destino = DEST / "physionet.org" / "files" / "challenge-2021" / "1.0.3" / "training" / grupo / nombre
    if destino.exists() and destino.stat().st_size > 0:
        return 0
    destino.parent.mkdir(parents=True, exist_ok=True)
    datos = get(f"{BASE}/{grupo}/{nombre}")
    tmp = destino.with_suffix(destino.suffix + ".part")
    tmp.write_bytes(datos); tmp.replace(destino)
    return len(datos)

t0 = time.time(); total_bytes = 0; total_archivos = 0; errores = []
for g in GRUPOS:
    archivos = listar(g)
    nuevos = 0
    with ThreadPoolExecutor(HILOS) as ex:
        fut = {ex.submit(bajar, g, a): a for a in archivos}
        for f in as_completed(fut):
            try:
                b = f.result(); total_bytes += b; nuevos += 1 if b else 0
            except Exception as e:
                errores.append((g, fut[f], str(e)[:120]))
    total_archivos += len(archivos)
    print(f"[{time.time() - t0:7.0f} s] {g}: {len(archivos)} archivos en el índice, {nuevos} descargados ahora; acumulado {total_bytes / 1e9:.2f} GB; errores {len(errores)}", flush=True)
print(f"FIN: {total_archivos} archivos indexados; {total_bytes / 1e9:.2f} GB descargados en esta ejecución; errores: {len(errores)}")
for e in errores[:20]:
    print("ERROR", e)
