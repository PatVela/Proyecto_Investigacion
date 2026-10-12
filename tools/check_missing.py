"""Verifica integridad .hea/.mat del dataset CINC2020 local.

Fusiona los antiguos ``check_missing_heas.py``, ``check_missing_mats.py`` y
``check_missing_pairs.py`` (idénticos salvo extensión/fuente)::

    python tools/check_missing.py --kind mats --data-dir dataset2020

Modos (``--kind``):
    heas  - reporta .mat locales sin su .hea.
    mats  - reporta .hea locales sin su .mat.
    pairs - compara carpetas locales contra PhysioNet y reporta registros
            donde faltan ambos archivos.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import urllib.request
from collections import defaultdict

BASE_URL = "https://physionet.org/files/challenge-2020/1.0.2/training/"


def verificar_faltantes(data_dir: str = "dataset2020", want_ext: str = ".mat") -> None:
    """Reporta archivos ``want_ext`` cuyo par existe pero ellos faltan."""
    have_ext = ".hea" if want_ext == ".mat" else ".mat"
    print(f"Escaneando carpetas en '{data_dir}'...\n")
    have_files = glob.glob(os.path.join(data_dir, "**", f"*{have_ext}"), recursive=True)
    if not have_files:
        print(f"No se encontraron archivos {have_ext} en '{data_dir}'. Revisa la ruta.")
        return

    faltantes_por_carpeta: dict[str, list[str]] = defaultdict(list)
    total_ok = total_faltantes = 0
    for have_path in have_files:
        want_path = have_path[: -len(have_ext)] + want_ext
        if not os.path.exists(want_path):
            faltantes_por_carpeta[os.path.dirname(have_path)].append(os.path.basename(have_path))
            total_faltantes += 1
        else:
            total_ok += 1

    print("==== RESUMEN DEL ESCANEO ====")
    print(f"Total de registros ({have_ext}) encontrados: {len(have_files)}")
    print(f"Registros OK con su {want_ext}:               {total_ok}")
    print(f"Registros incompletos (falta {want_ext}):      {total_faltantes}\n")
    if total_faltantes == 0:
        print(f"¡Todo perfecto! Todos los archivos {have_ext} tienen su respectivo {want_ext}.")
        return
    print("==== DETALLE DE CARPETAS AFECTADAS ====")
    for carpeta, archivos in faltantes_por_carpeta.items():
        print(f"\nCarpeta: {carpeta}")
        print(f"  Faltan {len(archivos)} archivos {want_ext}")
        detalle = archivos[:5] if len(archivos) > 5 else archivos
        print(f"  Ejemplos faltantes: {detalle}")


def obtener_archivos_remotos(url_subcarpeta: str) -> set[str]:
    """Lee el índice HTML de una carpeta en PhysioNet y extrae los registros."""
    try:
        req = urllib.request.Request(url_subcarpeta, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            html = response.read().decode("utf-8")
            heas = re.findall(r'href="([^"]+\.hea)"', html)
            return set(f[:-4] for f in heas)
    except Exception as exc:  # noqa: BLE001 - se reporta y se sigue
        print(f"Error al leer índice remoto {url_subcarpeta}: {exc}")
        return set()


def verificar_parejas_faltantes(data_dir: str = "dataset2020") -> None:
    """Reporta registros donde faltan ambos archivos contra PhysioNet."""
    base_path = os.path.join(data_dir, "physionet.org", "files", "challenge-2020", "1.0.2", "training")
    if not os.path.exists(base_path):
        print(f"Error: No se encontró la carpeta base '{base_path}'.")
        return

    carpetas_locales = []
    for root, dirs, _files in os.walk(base_path):
        if not dirs:  # Carpetas finales que contienen los archivos
            carpetas_locales.append(root)

    print(f"Escaneando {len(carpetas_locales)} subcarpetas contra el servidor de PhysioNet...\n")
    faltan_ambos: list[str] = []
    total_remotos = 0
    for i, carpeta in enumerate(carpetas_locales, 1):
        rel_folder = os.path.relpath(carpeta, base_path).replace("\\", "/")
        url_folder = BASE_URL + rel_folder + "/"
        print(f"[{i}/{len(carpetas_locales)}] Verificando remoto: {rel_folder} ... ", end="", flush=True)
        registros_remotos = obtener_archivos_remotos(url_folder)
        total_remotos += len(registros_remotos)
        archivos_locales = set(os.listdir(carpeta))
        locales_incompletos = 0
        for reg in registros_remotos:
            if (reg + ".hea") not in archivos_locales and (reg + ".mat") not in archivos_locales:
                faltan_ambos.append(f"{rel_folder}/{reg}")
                locales_incompletos += 1
        print(f"OK ({len(registros_remotos)} remotos | Faltan ambos: {locales_incompletos})")

    print("\n==== RESUMEN DE INTEGRIDAD CONTRA PHYSIONET ====")
    print(f"Total de registros evaluados en servidor: {total_remotos}")
    print(f"Registros donde FALTA AMBOS (.hea y .mat): {len(faltan_ambos)}\n")
    if faltan_ambos:
        print("==== LISTA DE REGISTROS QUE FALTAN POR COMPLETO (AMBOS ARCHIVOS) ====")
        for rec in faltan_ambos:
            print(f" - {rec}")
    else:
        print("¡Excelente! No falta ningún par (.hea + .mat) en ninguna carpeta.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verifica integridad .hea/.mat del dataset local")
    parser.add_argument("--kind", choices=["heas", "mats", "pairs"], default="mats",
                        help="Qué verificar: heas, mats o pares contra PhysioNet")
    parser.add_argument("--data-dir", default="dataset2020", help="Raíz del dataset local")
    args = parser.parse_args(argv)
    if args.kind == "pairs":
        verificar_parejas_faltantes(args.data_dir)
    else:
        verificar_faltantes(args.data_dir, want_ext=".hea" if args.kind == "heas" else ".mat")


if __name__ == "__main__":
    main()
