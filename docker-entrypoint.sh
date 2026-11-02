#!/bin/sh
# Prepare writable data directories and drop root before starting the app.
set -eu

# Fixed unprivileged identity. Never adopt the owner of a host bind mount: that
# owner can be a privileged host account (often root) and reusing it would keep,
# rather than drop, privileges inside the container.
APP_UID=10000
APP_GID=10000
DATA_DIR=/app/data
CACHE_DIR=/app/cache

# An explicit unprivileged container user (docker run --user) is honored as-is.
# It cannot chown the mounts, but it can still refuse setuid/file-capability
# escalation on the next exec.
if [ "$(id -u)" -ne 0 ]; then
    exec /usr/bin/setpriv --no-new-privs "$@"
fi

# Make image-managed directories and bind mounts writable by the fixed runtime
# user. The ownership check is O(1), so steady-state starts skip the recursive
# chown entirely; only the first start against a mismatched mount pays for it.
for dir in "$DATA_DIR" "$CACHE_DIR"; do
    mkdir -p "$dir"
    if [ "$(stat -c '%u:%g' "$dir")" != "$APP_UID:$APP_GID" ]; then
        chown -R "$APP_UID:$APP_GID" "$dir"
    fi
done

exec /usr/bin/setpriv \
    --reuid="$APP_UID" \
    --regid="$APP_GID" \
    --clear-groups \
    --bounding-set=-all \
    --no-new-privs \
    "$@"
