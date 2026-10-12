#!/bin/bash
# Descarga de los subconjuntos públicos de training/ del PhysioNet/CinC Challenge 2020.
#
# Se descarga la versión 1.0.2 desde PhysioNet y se conserva la estructura
# generada por wget dentro de dataset2020/. No descarga sources/.

set -euo pipefail

DEST="dataset2020"
BASE_URL="https://physionet.org/files/challenge-2020/1.0.2/training"

SUBSETS=(
  "cpsc_2018"
  "cpsc_2018_extra"
  "georgia"
  "ptb"
  "ptb-xl"
  "st_petersburg_incart"
)

mkdir -p "$DEST"
cd "$DEST"

echo "======================================================"
echo " Descargando PhysioNet/CinC Challenge 2020 v1.0.2"
echo "======================================================"
echo "Destino: $(pwd)"
echo "Subconjuntos training/: ${SUBSETS[*]}"
echo "NO se descargará: sources/"
echo ""

for subset in "${SUBSETS[@]}"; do
  echo "------------------------------------------------------"
  echo "Descargando: ${subset}"
  echo "------------------------------------------------------"
  wget -r -N -c -np -R "index.html*" "${BASE_URL}/${subset}/"
done

echo ""
echo "======================================================"
echo " Descarga completa"
echo "======================================================"
echo "Estructura descargada:"
find . -maxdepth 6 -type d | sort
