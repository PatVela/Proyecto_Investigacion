#!/usr/bin/env bash
# Expose the local ECG app to the internet with an HTTPS tunnel, so a device on
# ANY network can open it (no port-forwarding, no public IP). Uses Cloudflare
# Tunnel if available, otherwise ngrok. You must install one of them:
#
#   cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
#   ngrok:       https://ngrok.com/ (requires an auth token: ngrok config add-authtoken <TOKEN>)
#
# Usage:
#   ./run_public.sh            # starts tunnel + the app (short URL printed)
#   PORT=5000 TUNNEL=cloudflare ./run_public.sh
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-5000}"
SAVED="${ECG_SAVED:-saved}"
TUNNEL="${TUNNEL:-auto}"

# 1. Start the app (HTTP is fine; the tunnel provides HTTPS).
ECG_SAVED="$SAVED" python -m waitress --listen="127.0.0.1:$PORT" "webapp.wsgi:app" &
APP_PID=$!
trap 'kill $APP_PID 2>/dev/null || true' EXIT
sleep 1
echo "App local en http://127.0.0.1:$PORT"

# 2. Open the tunnel.
if command -v cloudflared >/dev/null 2>&1 && [ "$TUNNEL" != "ngrok" ]; then
  echo "Abriendo túnel Cloudflare ..."
  cloudflared tunnel --url "http://127.0.0.1:$PORT"
elif command -v ngrok >/dev/null 2>&1; then
  echo "Abriendo túnel ngrok ..."
  ngrok http "$PORT"
else
  echo "No se encontró cloudflared ni ngrok."
  echo "Instala uno (ver cabecera del script) y vuelve a ejecutar."
  kill "$APP_PID" 2>/dev/null || true
  exit 1
fi
