FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/var/cache/apt,id=apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,id=aptlib,sharing=locked \
    <<EOF
set -eux
apt-get update
apt-get install -y --no-install-recommends ca-certificates git
EOF
RUN --mount=type=cache,target=/root/.cache/uv <<EOF
set -eux
uv sync --locked --no-install-project --no-default-groups
EOF

FROM python:3.14-slim-trixie AS runtime

# The entrypoint starts as root only to fix ownership of data/cache bind mounts,
# then always drops to the fixed unprivileged account (10000:10000) before
# exec'ing the application. It never adopts the host mount owner.
# tally ignore=hadolint/DL3002
USER root

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,id=apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,id=aptlib,sharing=locked \
    <<EOF
set -eux
apt-get update
apt-get install -y --no-install-recommends ca-certificates tini
groupadd --gid 10000 app
useradd --uid 10000 --gid app --home-dir /app --no-create-home \
	--shell /usr/sbin/nologin app
mkdir -p /app/cache /app/data
chown app:app /app/cache /app/data
EOF

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /app/.venv .venv/
COPY app/ app/
COPY config/ config/
COPY run.py run.py
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY --chmod=0755 docker/healthcheck.sh /usr/local/bin/docker-healthcheck.sh
COPY --chmod=0755 docker/prepare-dirs.py /usr/local/bin/docker-prepare-dirs.py

EXPOSE 8000

HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=10m \
    --retries=3 \
    CMD ["/usr/local/bin/docker-healthcheck.sh"]

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]

CMD ["python", "run.py"]
