#!/bin/bash

# Salir si ocurre un error
set -e

echo "Actualizando pip e instalando dependencias desde requirements.txt..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Verificando instalación de PyTorch y CUDA..."
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Ninguno\"}')"