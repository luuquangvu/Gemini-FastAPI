#!/bin/sh
# Probe the API as an unprivileged user rather than as container root.
set -eu

APP_UID=10000
APP_GID=10000
HEALTHCHECK_CODE='import urllib.request; from app.utils import g_config; urllib.request.urlopen(f"http://127.0.0.1:{g_config.server.port}/health", timeout=5).close()'

cd /app

if [ "$(id -u)" -eq 0 ]; then
    exec /usr/bin/setpriv \
        --reuid="$APP_UID" \
        --regid="$APP_GID" \
        --clear-groups \
        python -c "$HEALTHCHECK_CODE"
fi

exec python -c "$HEALTHCHECK_CODE"
