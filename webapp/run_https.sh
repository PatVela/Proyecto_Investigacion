#!/usr/bin/env bash
# Run the ECG web app over HTTPS via waitress, using a TLS certificate.
#
#   ./run_https.sh                 # generates a self-signed cert in ./certs
#   CERT=cert.pem KEY=key.pem ./run_https.sh   # use your own cert
#
# A self-signed cert works for internal / demo use (the browser will warn you to
# trust it). For a production HTTPS hostname, either supply a real cert from
# Let's Encrypt / a CA, or terminate TLS in a reverse proxy (nginx/caddy/traefik)
# or a platform such as Render/Heroku (see README).
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-5000}"
SAVED="${ECG_SAVED:-saved}"
CERT="${CERT:-certs/ecg-cert.pem}"
KEY="${KEY:-certs/ecg-key.pem}"

mkdir -p certs
if [ ! -f "$CERT" ] || [ ! -f "$KEY" ]; then
  echo "Generando certificado autofirmado en $CERT / $KEY ..."
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout "$KEY" -out "$CERT" \
    -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
fi

echo "Sirviendo en https://0.0.0.0:$PORT (modelo: $SAVED)"
ECG_SAVED="$SAVED" python -m waitress \
  --listen="*:$PORT" \
  --ssl-certfile="$CERT" \
  --ssl-keyfile="$KEY" \
  "webapp.wsgi:app"
