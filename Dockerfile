# Webapp ECG CINC2020-12 (CPU).
# Construir:  docker build -t ecg-cinc2020 .
# Ejecutar:   docker run --rm -p 5002:5002 -v ./saved:/app/saved:ro ecg-cinc2020
# (monte su carpeta local `saved/` con el checkpoint entrenado).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# PyTorch CPU primero para que requirements.txt no traiga la rueda CUDA.
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5002

CMD ["python", "webapp/app.py", "--saved", "saved", "--host", "0.0.0.0", "--port", "5002"]
