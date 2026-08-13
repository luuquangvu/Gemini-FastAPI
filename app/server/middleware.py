import hashlib
import hmac
import tempfile
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

from app.utils import g_config

# Persistent directory for storing generated media
MEDIA_STORE_DIR = Path(g_config.storage.media_path)
MEDIA_STORE_DIR.mkdir(parents=True, exist_ok=True)


def get_media_store_dir() -> Path:
    """Returns a persistent directory for storing media."""
    return MEDIA_STORE_DIR


def get_media_token(filename: str) -> str:
    """Generate a HMAC-SHA256 token for a filename using the API key."""
    secret = g_config.server.api_key
    if not secret:
        return ""

    msg = filename.encode("utf-8")
    secret_bytes = secret.encode("utf-8")
    return hmac.new(secret_bytes, msg, hashlib.sha256).hexdigest()


def verify_media_token(filename: str, token: str | None) -> bool:
    """Verify the provided token against the filename."""
    if expected := get_media_token(filename):
        return hmac.compare_digest(token, expected) if token else False
    return True  # No auth required


def cleanup_expired_media(retention_days: int) -> int:
    """Delete media files in MEDIA_STORE_DIR older than retention_days."""
    if retention_days <= 0:
        return 0

    now = time.time()
    retention_seconds = retention_days * 24 * 60 * 60
    cutoff = now - retention_seconds

    count = 0
    for file_path in MEDIA_STORE_DIR.iterdir():
        if not file_path.is_file():
            continue
        try:
            if file_path.stat().st_mtime < cutoff:
                file_path.unlink()
                count += 1
        except Exception as e:
            logger.warning(f"Failed to delete expired media {file_path}: {e}")

    if count > 0:
        logger.info(f"Cleaned up {count} expired media files.")
    return count


def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail}},
        )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"message": str(exc)}},
    )


def get_temp_dir():
    temp_dir = tempfile.TemporaryDirectory()
    try:
        yield Path(temp_dir.name)
    finally:
        temp_dir.cleanup()


def verify_gemini_api_key(request: Request):
    """Gemini-style auth: x-goog-api-key header, key= query param, or Bearer fallback."""
    if not g_config.server.api_key:
        return ""

    # 1) x-goog-api-key header
    api_key = request.headers.get("x-goog-api-key")
    if api_key:
        if api_key != g_config.server.api_key:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Wrong API key")
        return api_key

    # 2) key= query parameter
    api_key = request.query_params.get("key")
    if api_key:
        if api_key != g_config.server.api_key:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Wrong API key")
        return api_key

    # 3) fall back to Authorization: Bearer
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        api_key = auth_header[7:]
        if api_key != g_config.server.api_key:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Wrong API key")
        return api_key

    raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    if not g_config.server.api_key:
        return ""

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")

    api_key = credentials.credentials
    if api_key != g_config.server.api_key:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Wrong API key")

    return api_key


def add_exception_handler(app: FastAPI):
    app.add_exception_handler(Exception, global_exception_handler)


def add_cors_middleware(app: FastAPI):
    if g_config.cors.enabled:
        cors = g_config.cors
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors.allow_origins,
            allow_credentials=cors.allow_credentials,
            allow_methods=cors.allow_methods,
            allow_headers=cors.allow_headers,
        )
