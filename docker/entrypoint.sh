#!/bin/sh
# Prepare writable data directories and drop root before starting the app.
set -eu

# Fixed unprivileged identity. Never adopt the owner of a host bind mount: that
# owner can be a privileged host account (often root) and reusing it would keep,
# rather than drop, privileges inside the container.
APP_UID=10000
APP_GID=10000
export APP_UID APP_GID

# The storage and cache locations are configurable, so resolve the effective paths
# instead of assuming they all live under /app/data. When this runs as root it also
# fixes ownership; when the container is already unprivileged it reports any path
# that is not writable.
python /usr/local/bin/docker-prepare-dirs.py

# An explicit unprivileged container user (docker run --user) is honored as-is.
# It cannot chown the mounts, but it can still refuse setuid/file-capability
# escalation on the next exec.
if [ "$(id -u)" -ne 0 ]; then
    exec /usr/bin/setpriv --no-new-privs "$@"
fi

exec /usr/bin/setpriv \
    --reuid="$APP_UID" \
    --regid="$APP_GID" \
    --clear-groups \
    --bounding-set=-all \
    --no-new-privs \
    "$@"
