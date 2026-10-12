"""Descarga archivos .hea/.mat faltantes del Challenge 2020 desde PhysioNet.

Fusiona los antiguos ``download_missing_heas.py``, ``download_missing_mats.py``
y ``download_missing_pairs.py`` (idénticos salvo extensión/fuente)::

    python tools/download_missing.py --kind mats --data-dir dataset2020

Modos (``--kind``):
    heas  - rastrea .mat locales sin .hea y descarga el .hea correspondiente.
    mats  - rastrea .hea locales sin .mat y descarga el .mat correspondiente.
    pairs - compara cada carpeta local contra el índice de PhysioNet y
            descarga los pares incompletos (.hea y/o .mat).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import urllib.request

BASE_URL = "https://physionet.org/files/challenge-2020/1.0.2/"
BASE_TRAINING_URL = BASE_URL + "training/"


def _remote_subpath(local_path: str) -> str:
    """Ruta relativa para PhysioNet respetando la jerarquía de carpetas."""
    partes = os.path.normpath(local_path).split(os.sep)
    if "1.0.2" in partes:
        idx_base = partes.index("1.0.2") + 1
    else:
        idx_base = partes.index("training") if "training" in partes else 0
    return "/".join(partes[idx_base:])


def descargar_faltantes(data_dir: str = "dataset2020", want_ext: str = ".mat") -> None:
    """Descarga los archivos ``want_ext`` cuyo par existe pero ellos faltan."""
    have_ext = ".hea" if want_ext == ".mat" else ".mat"
    print(f"Buscando archivos {want_ext} faltantes en la estructura local...")
    have_files = glob.glob(os.path.join(data_dir, "**", f"*{have_ext}"), recursive=True)

    faltantes = []
    for have in have_files:
        want = have[: -len(have_ext)] + want_ext
        if not os.path.exists(want):
            faltantes.append((have, want))

    if not faltantes:
        print(f"¡No faltan archivos {want_ext}! Todo está completo.")
        return

    print(f"Se encontraron {len(faltantes)} archivos {want_ext} faltantes. Descargando...\n")
    exitos = errores = 0
    for i, (_have_path, want_path) in enumerate(faltantes, 1):
        url = BASE_URL + _remote_subpath(want_path)
        rel = os.path.join(os.path.dirname(want_path), os.path.basename(want_path))
        print(f"[{i}/{len(faltantes)}] Guardando en {rel} ... ", end="", flush=True)
        try:
            os.makedirs(os.path.dirname(want_path), exist_ok=True)
            urllib.request.urlretrieve(url, want_path)
            print("OK")
            exitos += 1
        except Exception as exc:  # noqa: BLE001 - se reporta y se sigue
            print(f"ERROR ({exc})")
            errores += 1
    print(f"\nFinalizado. Exitosos: {exitos} | Errores: {errores}")


def obtener_archivos_remotos(url_subcarpeta: str) -> set[str]:
    """Lee el índice HTML de una carpeta en PhysioNet y extrae los registros."""
    try:
        req = urllib.request.Request(url_subcarpeta, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            html = response.read().decode("utf-8")
            heas = re.findall(r'href="([^"]+\.hea)"', html)
            return set(f[:-4] for f in heas)
    except Exception as exc:  # noqa: BLE001 - se reporta y se sigue
        print(f" Error al conectar con {url_subcarpeta}: {exc}")
        return set()


def descargar_pares_faltantes(data_dir: str = "dataset2020") -> None:
    """Descarga pares incompletos comparando contra el índice de PhysioNet."""
    base_path = os.path.join(data_dir, "physionet.org", "files", "challenge-2020", "1.0.2", "training")

    subcarpetas_rel = [
        "cpsc_2018/g1", "cpsc_2018/g2", "cpsc_2018/g3", "cpsc_2018/g4", "cpsc_2018/g5", "cpsc_2018/g6", "cpsc_2018/g7",
        "cpsc_2018_extra/g1", "cpsc_2018_extra/g2", "cpsc_2018_extra/g3", "cpsc_2018_extra/g4",
        "georgia/g1", "georgia/g2", "georgia/g3", "georgia/g4", "georgia/g5", "georgia/g6", "georgia/g7", "georgia/g8",
        "georgia/g9", "georgia/g10", "georgia/g11",
        "ptb/g1",
        "ptb-xl/g1", "ptb-xl/g2", "ptb-xl/g3", "ptb-xl/g4", "ptb-xl/g5", "ptb-xl/g6", "ptb-xl/g7", "ptb-xl/g8",
        "ptb-xl/g9", "ptb-xl/g10", "ptb-xl/g11", "ptb-xl/g12", "ptb-xl/g13", "ptb-xl/g14", "ptb-xl/g15", "ptb-xl/g16",
        "ptb-xl/g17", "ptb-xl/g18", "ptb-xl/g19", "ptb-xl/g20", "ptb-xl/g21", "ptb-xl/g22",
        "st_petersburg_incart/g1",
    ]

    descargas_pendientes: list[tuple[str, str]] = []
    print("--- FASE 1: Identificando archivos faltantes ---")
    for i, rel_folder in enumerate(subcarpetas_rel, 1):
        local_folder = os.path.normpath(os.path.join(base_path, rel_folder))
        os.makedirs(local_folder, exist_ok=True)
        print(f"[{i}/{len(subcarpetas_rel)}] Verificando en servidor: {rel_folder}...", end="", flush=True)
        archivos_locales = set(os.listdir(local_folder)) if os.path.exists(local_folder) else set()
        url_folder = BASE_TRAINING_URL + rel_folder + "/"
        registros_remotos = obtener_archivos_remotos(url_folder)
        faltantes_carpeta = 0
        for reg in registros_remotos:
            if (reg + ".hea") not in archivos_locales:
                descargas_pendientes.append((url_folder + reg + ".hea", os.path.join(local_folder, reg + ".hea")))
                faltantes_carpeta += 1
            if (reg + ".mat") not in archivos_locales:
                descargas_pendientes.append((url_folder + reg + ".mat", os.path.join(local_folder, reg + ".mat")))
                faltantes_carpeta += 1
        print(f" OK (Faltan {faltantes_carpeta} archivos)")

    print("\n--- FASE 2: Descargando archivos faltantes ---")
    if not descargas_pendientes:
        print("¡No hay nada pendiente! Tu dataset ya está 100% completo.")
        return
    print(f"Iniciando la descarga de {len(descargas_pendientes)} archivos...\n")
    exitos, errores = 0, 0
    total = len(descargas_pendientes)
    for i, (url, destino) in enumerate(descargas_pendientes, 1):
        nombre = os.path.basename(destino)
        carpeta_padre = os.path.basename(os.path.dirname(destino))
        print(f"[{i}/{total}] Descargando {carpeta_padre}/{nombre} ... ", end="", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(destino, "wb") as out_file:
                out_file.write(resp.read())
            print("OK")
            exitos += 1
        except Exception as exc:  # noqa: BLE001 - se reporta y se sigue
            print(f"ERROR ({exc})")
            errores += 1
    print(f"\nProceso finalizado. Descargados con éxito: {exitos} | Errores: {errores}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Descarga .hea/.mat faltantes del Challenge 2020")
    parser.add_argument("--kind", choices=["heas", "mats", "pairs"], default="mats",
                        help="Qué descargar: heas, mats o pares incompletos")
    parser.add_argument("--data-dir", default="dataset2020", help="Raíz del dataset local")
    args = parser.parse_args(argv)
    if args.kind == "pairs":
        descargar_pares_faltantes(args.data_dir)
    else:
        descargar_faltantes(args.data_dir, want_ext=".hea" if args.kind == "heas" else ".mat")


if __name__ == "__main__":
    main()
