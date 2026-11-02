#!/usr/bin/env python3
"""Prepare the writable directories used by the container runtime.

The application's storage locations are configurable, so the entrypoint cannot assume
they live under ``/app/data``. This resolves the same paths the application will use
(``CONFIG_*`` environment overrides first, then the YAML file, then built-in defaults),
creates them, and hands them to the runtime account. It is deliberately best-effort: a
path that cannot be prepared is reported, but the application is still allowed to start
and expose the failure through ``/health``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

DEFAULT_ROOT = Path("/app")
DEFAULT_DB_PATH = "data/lmdb"
DEFAULT_MEDIA_PATH = "data/media"
DEFAULT_CACHE_PATH = "/app/cache"


def _env(name: str) -> str | None:
    """Read an environment variable, honoring the case-insensitive settings convention."""
    if (value := os.environ.get(name)) is not None:
        return value
    folded = name.upper()
    return next(
        (value for key, value in os.environ.items() if key.upper() == folded),
        None,
    )


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default."""
    value = _env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _warn(message: str) -> None:
    """Report a preparation problem without aborting container startup."""
    print(f"docker-entrypoint: warning: {message}", file=sys.stderr)


def _load_config(root: Path) -> dict[str, object]:
    """Load the YAML config file the application reads, if it exists."""
    config_path = Path(_env("CONFIG_PATH") or "config/config.yaml")
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        with config_path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        _warn(f"cannot read {config_path}: {exc}")
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(key): value for key, value in loaded.items()}


def _nested(config: dict[str, object], *keys: str) -> object:
    """Return a nested mapping value, or None when the key is absent."""
    current: object = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _resolve(value: object, default: str, root: Path) -> Path:
    """Resolve a configured path against the application directory."""
    raw = value.strip() if isinstance(value, str) and value.strip() else default
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _targets(root: Path) -> list[Path]:
    """Return the effective storage, media, and cookie directories."""
    config = _load_config(root)
    db_path = _env("CONFIG_STORAGE__PATH")
    media_path = _env("CONFIG_STORAGE__MEDIA_PATH")
    cache_path = _env("GEMINI_COOKIE_PATH")

    paths = [
        _resolve(
            db_path if db_path is not None else _nested(config, "storage", "path"),
            DEFAULT_DB_PATH,
            root,
        ),
        _resolve(
            media_path if media_path is not None else _nested(config, "storage", "media_path"),
            DEFAULT_MEDIA_PATH,
            root,
        ),
        _resolve(cache_path, DEFAULT_CACHE_PATH, root),
    ]

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _overlaps_application(path: Path, root: Path) -> bool:
    """Whether preparing this path would grant the app user control over app code."""
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    return resolved == Path(resolved.anchor) or resolved == root or root.is_relative_to(resolved)


def _chown_tree(path: Path, uid: int, gid: int) -> None:
    """Assign a directory and its contents to the runtime account without following links."""
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        os.chown(dirpath, uid, gid, follow_symlinks=False)
        for name in dirnames + filenames:
            os.chown(os.path.join(dirpath, name), uid, gid, follow_symlinks=False)


def main() -> int:
    """Create the configured directories and make them writable by the runtime user."""
    root = Path(_env("APP_ROOT") or DEFAULT_ROOT)
    uid = _env_int("APP_UID", 10000)
    gid = _env_int("APP_GID", 10000)
    running_as_root = os.geteuid() == 0

    for target in _targets(root):
        if _overlaps_application(target, root):
            _warn(f"refusing to prepare {target}; it overlaps the application directory")
            continue

        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _warn(f"cannot create {target}: {exc}")
            continue
        if not target.is_dir():
            _warn(f"{target} is not a directory")
            continue

        if running_as_root:
            try:
                _chown_tree(target, uid, gid)
            except OSError as exc:
                _warn(f"cannot assign {target} to {uid}:{gid}: {exc}")
                continue
        elif not os.access(target, os.W_OK | os.X_OK):
            _warn(
                f"{target} is not writable by uid {os.geteuid()}; run the container as root "
                "or make the mount writable by that user"
            )
            continue

        print(f"docker-entrypoint: prepared {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
