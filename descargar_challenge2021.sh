#!/bin/bash
# Descarga paralela de Chapman-Shaoxing y Ningbo.
# PhysioNet/CinC Challenge 2021 v1.0.3.
#
# Cada proceso descarga una subcarpeta gN independiente.
# Muestra el progreso de descarga en tiempo real.
# Permite reanudar archivos incompletos.
# Conserva la estructura original.
# No descarga sources/ ni otros subconjuntos.

set -uo pipefail

DEST="dataset2021"
BASE_URL="https://physionet.org/files/challenge-2021/1.0.3/training"

# Número máximo de descargas simultáneas.
JOBS=4

mkdir -p "$DEST"
cd "$DEST"

echo "======================================================"
echo " PhysioNet/CinC Challenge 2021 v1.0.3"
echo " Descarga paralela por subcarpetas"
echo "======================================================"
echo "Destino: $(pwd)"
echo "Procesos simultáneos: $JOBS"
echo ""

# Generar lista de subcarpetas.
generate_tasks() {
    for ((i=1; i<=11; i++)); do
        echo "chapman_shaoxing/g${i}"
    done

    for ((i=1; i<=35; i++)); do
        echo "ningbo/g${i}"
    done
}

# Descargar cada subcarpeta.
download_group() {
    local path="$1"
    local subset="${path%%/*}"
    local group="${path#*/}"

    local url="${BASE_URL}/${subset}/${group}/"
    local log="wget_${subset}_${group}.log"

    echo ""
    echo "======================================================"
    echo "[INICIO] ${path}"
    echo "======================================================"

    if wget \
        -r -N -c -np \
        -R "index.html*" \
        --tries=10 \
        --timeout=30 \
        --waitretry=5 \
        --show-progress \
        --progress=bar:force:noscroll \
        "$url" 2>&1 | tee "$log"; then

        echo ""
        echo "[OK] ${path}"
        return 0
    else
        echo ""
        echo "[ERROR] ${path} - Revisar ${log}"
        return 1
    fi
}

export -f download_group
export BASE_URL

# Ejecutar hasta JOBS descargas simultáneas.
STATUS=0

if ! generate_tasks | xargs -r -P "$JOBS" -I {} \
    bash -c 'download_group "$1"' _ "{}"; then
    STATUS=1
fi

echo ""
echo "======================================================"

if [ "$STATUS" -eq 0 ]; then
    echo " Todas las subcarpetas finalizaron correctamente."
else
    echo " Algunas descargas presentaron errores."
    echo " Revisa los archivos wget_*.log"
fi

echo "======================================================"

echo "Directorios descargados:"
find ./physionet.org/files/challenge-2021/1.0.3/training \
    -mindepth 1 -maxdepth 2 -type d | sort

exit "$STATUS"