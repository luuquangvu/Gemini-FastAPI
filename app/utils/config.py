import ast
import os
import sys
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast, get_args

import orjson
from curl_cffi import BrowserTypeLiteral
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

CONFIG_PATH = "config/config.yaml"


class HTTPSConfig(BaseModel):
    """HTTPS configuration"""

    enabled: bool = Field(default=False, description="Enable HTTPS")
    key_file: str = Field(default="certs/privkey.pem", description="SSL private key file path")
    cert_file: str = Field(default="certs/fullchain.pem", description="SSL certificate file path")


class ServerConfig(BaseModel):
    """Server configuration"""

    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port number")
    api_key: str | None = Field(
        default=None,
        description="API key for authentication, if set, will enable API key validation",
    )
    https: HTTPSConfig = Field(default=HTTPSConfig(), description="HTTPS configuration")


class GeminiClientSettings(BaseModel):
    """Credential set for one Gemini client."""

    id: str = Field(..., description="Unique identifier for the client")
    secure_1psid: str | None = Field(default=None, description="Gemini Secure 1PSID")
    secure_1psidts: str | None = Field(default=None, description="Gemini Secure 1PSIDTS")
    proxy: str | None = Field(default=None, description="Proxy URL for this Gemini client")
    impersonate: str | None = Field(
        default=None,
        description="Browser impersonation target for curl_cffi. None uses library default",
    )

    @field_validator("proxy", "impersonate", mode="before")
    @classmethod
    def _blank_string_to_none(cls, value: str | None) -> str | None:
        """Normalize empty or whitespace-only strings to None."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("impersonate")
    @classmethod
    def _validate_impersonate(cls, value: str | None) -> str | None:
        """Validate that impersonate is a supported curl_cffi BrowserTypeLiteral value."""
        if value is None:
            return None
        allowed = get_args(BrowserTypeLiteral)
        if value not in allowed:
            raise ValueError(
                f"impersonate={value!r} is not supported. Allowed values: {', '.join(allowed)}"
            )
        return value


class GeminiModelConfig(BaseModel):
    """Configuration for a custom Gemini model."""

    model_name: str | None = Field(default=None, description="Name of the model")
    model_header: dict[str, str | None] | None = Field(
        default=None, description="Header for the model"
    )

    @field_validator("model_header", mode="before")
    @classmethod
    def _parse_json_string(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                return orjson.loads(v)
            except orjson.JSONDecodeError:
                try:
                    return ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    return v
        return v


class ChatMode(StrEnum):
    """Chat mode options for Gemini conversation handling."""

    NORMAL = "normal"
    TEMPORARY = "temporary"


class OversizedContextStrategy(StrEnum):
    """Strategy for handling oversized context."""

    COMPACTION = "compaction"
    FILE = "file"


class GeminiConfig(BaseModel):
    """Gemini API configuration, including session behavior and generation options."""

    clients: list[GeminiClientSettings] = Field(
        ..., description="List of Gemini client credential pairs"
    )
    models: list[GeminiModelConfig] = Field(default=[], description="List of custom Gemini models")
    model_strategy: Literal["append", "overwrite"] = Field(
        default="append",
        description="Strategy for loading models: 'append' merges custom with default, 'overwrite' uses only custom",
    )
    timeout: int = Field(default=450, ge=30, description="Init timeout in seconds")
    watchdog_timeout: int = Field(default=120, ge=30, description="Watchdog timeout in seconds")
    recovery_timeout: int = Field(
        default=300,
        ge=30,
        description="Timeout in seconds for conversation-turns recovery after a truncated stream",
    )
    auto_refresh: bool = Field(True, description="Enable auto-refresh for Gemini sessions")
    refresh_interval: int = Field(
        default=600,
        ge=60,
        description="Interval in seconds to refresh Gemini sessions (Not less than 60s)",
    )
    auto_close: bool = Field(
        default=True, description="Enable auto-close for Gemini sessions after inactivity"
    )
    close_delay: int = Field(
        default=900, ge=30, description="Inactivity delay in seconds before auto-closing"
    )
    verbose: bool = Field(True, description="Enable verbose logging for Gemini API requests")
    extended_thinking: bool = Field(
        default=False,
        description="Enable Gemini extended thinking mode for message generation",
    )
    max_chars_per_request: int = Field(
        default=1_000_000,
        ge=1,
        description="Maximum characters Gemini Web can accept per request",
    )
    oversized_context_strategy: OversizedContextStrategy = Field(
        default=OversizedContextStrategy.FILE,
        description=(
            "Strategy for oversized context: 'compaction' summarizes older turns into a "
            "bounded summary block (last 8 kept verbatim) for temporary/fallback replay; "
            "'file' sends oversized context as an attachment (default, behavior-preserving)"
        ),
    )
    default_model: str | None = Field(
        default=None,
        description="Fallback model name used when a requested model is unknown (null disables fallback)",
    )
    validate_session_on_init: bool = Field(
        default=True,
        description="Run a temporary-chat session validation probe when clients initialize",
    )
    allow_private_url_fetch: bool = Field(
        default=False,
        description="Allow server-side fetching of private/loopback image URLs (SSRF risk; default blocks them)",
    )
    url_fetch_timeout: int = Field(
        default=15,
        ge=1,
        le=120,
        description="Timeout in seconds for server-side URL image fetches",
    )
    chat_mode: ChatMode = Field(
        default=ChatMode.NORMAL,
        description=(
            "Chat mode: 'normal' uses standard chats; 'temporary' sends with Google's temporary "
            "mode (not saved to the account) and applies a tighter effective input limit. "
            "Warning: Google may close a temporary window at any time mid-conversation, and the "
            "reply can then come back without the earlier context instead of erroring"
        ),
    )

    @field_validator("models", mode="before")
    @classmethod
    def _parse_models_json(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().startswith("["):
            try:
                return orjson.loads(v)
            except orjson.JSONDecodeError:
                try:
                    return ast.literal_eval(v)
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Failed to parse models JSON or Python literal: {e}")
                    return v
        return v

    @field_validator("models")
    @classmethod
    def _filter_valid_models(cls, v: list[GeminiModelConfig]) -> list[GeminiModelConfig]:
        """Filter out models that don't have all required fields set."""
        return filter_valid_models(v)

    @model_validator(mode="after")
    def _bound_recovery_timeout(self) -> "GeminiConfig":
        if self.recovery_timeout > self.timeout:
            logger.warning(
                f"recovery_timeout ({self.recovery_timeout}) exceeds timeout "
                f"({self.timeout}); clamping recovery_timeout to timeout."
            )
            self.recovery_timeout = self.timeout
        return self


class CORSConfig(BaseModel):
    """CORS configuration"""

    enabled: bool = Field(default=True, description="Enable CORS support")
    allow_origins: list[str] = Field(
        default=["*"], description="List of allowed origins for CORS requests"
    )
    allow_credentials: bool = Field(default=True, description="Allow credentials in CORS requests")
    allow_methods: list[str] = Field(
        default=["*"], description="List of allowed HTTP methods for CORS requests"
    )
    allow_headers: list[str] = Field(
        default=["*"], description="List of allowed headers for CORS requests"
    )


class StorageConfig(BaseModel):
    """LMDB Storage configuration"""

    path: str = Field(
        default="data/lmdb",
        description="Path to the storage directory where data will be saved",
    )
    media_path: str = Field(
        default="data/media",
        description="Path to the directory where generated media will be stored",
    )
    max_size: int = Field(
        default=1024**3,  # 1 GB
        ge=1,
        description="Maximum size of the storage in bytes",
    )
    retention_days: int = Field(
        default=7,
        ge=0,
        description="Number of days to retain conversations before automatic cleanup (0 disables cleanup)",
    )


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="DEBUG",
        description="Logging level",
    )


class Config(BaseSettings):
    """Application configuration"""

    # Server configuration
    server: ServerConfig = Field(
        default=ServerConfig(),
        description="Server configuration, including host, port, and API key",
    )

    # CORS configuration
    cors: CORSConfig = Field(
        default=CORSConfig(),
        description="CORS configuration, allows cross-origin requests",
    )

    # Gemini API configuration
    gemini: GeminiConfig = Field(..., description="Gemini API configuration, must be set")

    storage: StorageConfig = Field(
        default=StorageConfig(),
        description="Storage configuration, defines where and how data will be stored",
    )

    # Logging configuration
    logging: LoggingConfig = Field(
        default=LoggingConfig(),
        description="Logging configuration",
    )

    model_config = SettingsConfigDict(
        env_prefix="CONFIG_",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
        yaml_file=os.getenv("CONFIG_PATH", CONFIG_PATH),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Read settings: env -> yaml -> default"""
        return (
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )


def extract_gemini_clients_env() -> dict[int, dict[str, Any]]:
    """Extract and remove all Gemini clients related environment variables, return a mapping from index to field dict."""
    prefix = "CONFIG_GEMINI__CLIENTS__"
    env_overrides: dict[int, dict[str, Any]] = {}
    to_delete = []
    for k, v in os.environ.items():
        if k.startswith(prefix):
            parts = k.split("__")
            if len(parts) < 4:
                continue
            index_str, field = parts[2], parts[3].lower()
            if not index_str.isdigit():
                continue
            idx = int(index_str)
            env_overrides.setdefault(idx, {})[field] = v
            to_delete.append(k)

    for k in to_delete:
        del os.environ[k]
    return env_overrides


def _merge_clients_with_env(
    base_clients: list[GeminiClientSettings] | None,
    env_overrides: dict[int, dict[str, Any]],
) -> list[GeminiClientSettings]:
    """Return Gemini clients with environment overrides applied to the base list."""
    if not env_overrides:
        return base_clients or []
    result_clients: list[GeminiClientSettings] = []
    if base_clients:
        result_clients = [client.model_copy() for client in base_clients]
    for idx in sorted(env_overrides):
        overrides = env_overrides[idx]
        if idx < len(result_clients):
            client_dict = result_clients[idx].model_dump()
            client_dict.update(overrides)
            result_clients[idx] = GeminiClientSettings(**client_dict)
        elif idx == len(result_clients):
            new_client = GeminiClientSettings(**overrides)
            result_clients.append(new_client)
        else:
            raise IndexError(
                f"Client index {idx} in env is out of range (current count: {len(result_clients)}). "
                "Client indices must be contiguous starting from 0."
            )
    return result_clients or base_clients or []


def extract_gemini_models_env() -> dict[int, dict[str, Any]]:
    """Extract and remove all Gemini models related environment variables, supporting nested fields."""
    root_key = "CONFIG_GEMINI__MODELS"
    env_overrides: dict[int, dict[str, Any]] = {}

    if root_key in os.environ:
        val = os.environ[root_key]
        models_list = None
        parsed_successfully = False

        try:
            models_list = orjson.loads(val)
            parsed_successfully = True
        except orjson.JSONDecodeError:
            try:
                models_list = ast.literal_eval(val)
                parsed_successfully = True
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Failed to parse {root_key} as JSON or Python literal: {e}")

        if parsed_successfully and isinstance(models_list, list):
            for idx, model_data in enumerate(models_list):
                if isinstance(model_data, dict):
                    env_overrides[idx] = cast(dict[str, Any], model_data)

            del os.environ[root_key]

    return env_overrides


def _merge_models_with_env(
    base_models: list[GeminiModelConfig] | None,
    env_overrides: dict[int, dict[str, Any]],
):
    """Override base_models with env_overrides using standard update (replace whole fields)."""
    if not env_overrides:
        return base_models or []
    result_models: list[GeminiModelConfig] = []
    if base_models:
        result_models = [model.model_copy() for model in base_models]

    for idx in sorted(env_overrides):
        overrides = env_overrides[idx]
        if idx < len(result_models):
            model_dict = result_models[idx].model_dump()
            model_dict.update(overrides)
            result_models[idx] = GeminiModelConfig(**model_dict)
        elif idx == len(result_models):
            new_model = GeminiModelConfig(**overrides)
            result_models.append(new_model)
        else:
            raise IndexError(
                f"Model index {idx} in env is out of range (current count: {len(result_models)}). "
                "Model indices must be contiguous starting from 0."
            )
    return result_models


def filter_valid_models(v: list[GeminiModelConfig]) -> list[GeminiModelConfig]:
    """Filter out models that don't have all required fields set."""
    valid_models = []
    for model in v:
        if model.model_name and model.model_header:
            valid_models.append(model)
        else:
            missing = []
            if not model.model_name:
                missing.append("model_name")
            if not model.model_header:
                missing.append("model_header")
            logger.warning(f"Discarding custom model due to missing {', '.join(missing)}: {model}")
    return valid_models


_RELOAD_STATE: dict[str, Any] = {
    "mtime": None,
    "models": None,
}


def register_boot_models(models: list[GeminiModelConfig]) -> None:
    """Record the boot-time merged model list and the config file mtime."""
    _RELOAD_STATE["models"] = [m.model_copy() for m in models]
    try:
        _RELOAD_STATE["mtime"] = Path(CONFIG_PATH).stat().st_mtime
    except OSError:
        _RELOAD_STATE["mtime"] = None


def reload_models_if_changed() -> bool:
    """Reload `gemini.models` from the config yaml when its mtime changes.

    Hot edits to config.yaml's model list apply on the next request without a
    restart. Boot-time env overrides (CONFIG_GEMINI__MODELS) stay in force until
    restart: the env vars are consumed at boot (pydantic-settings mechanics) and
    are not re-applied on hot reload. On any read/parse failure the previously
    known list is kept and a WARNING is logged — request paths never break.
    """
    path = Path(CONFIG_PATH)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    if mtime == _RELOAD_STATE["mtime"]:
        return False
    current = _RELOAD_STATE["models"]

    import yaml

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8") or "{}") or {}
        models_raw = (raw.get("gemini") or {}).get("models") or []
        if not isinstance(models_raw, list):
            raise ValueError("gemini.models must be a list")
        models: list[GeminiModelConfig] = []
        for entry in models_raw:
            try:
                models.append(GeminiModelConfig.model_validate(entry))
            except ValidationError as e:
                raise ValueError(f"invalid model entry: {e}") from e
        reloaded = filter_valid_models(models)
    except Exception as e:
        logger.warning(f"Hot reload of gemini.models failed ({e}); keeping previous model list.")
        _RELOAD_STATE["mtime"] = mtime
        return False

    _RELOAD_STATE["models"] = reloaded
    _RELOAD_STATE["mtime"] = mtime
    if reloaded != current:
        logger.info(
            f"Hot reloaded {len(reloaded)} custom model(s) from {CONFIG_PATH} "
            f"(was {len(current) if current else 0})."
        )
    return True


def get_reloaded_models() -> list[GeminiModelConfig]:
    """Return the current effective custom model list (boot list unless hot-reloaded)."""
    return list(_RELOAD_STATE["models"] or [])


def load_cached_1psidts(psid: str) -> str | None:
    """Return the cached rotated __Secure-1PSIDTS for the given 1PSID, if a cache file exists.

    Mirrors the pinned dependency's cookie cache layout
    (gemini_webapi/utils/rotate_1psidts.py): ${GEMINI_COOKIE_PATH:-$TMPDIR/gemini_webapi}/
    .cached_cookies_<psid>.json holds a cookie-name/value list written on rotation.
    """
    if not psid:
        return None
    cache_dir = os.getenv("GEMINI_COOKIE_PATH") or os.path.join(
        tempfile.gettempdir(), "gemini_webapi"
    )
    cache_path = Path(cache_dir) / f".cached_cookies_{psid}.json"
    if not cache_path.is_file():
        return None
    try:
        cookies = orjson.loads(cache_path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        logger.warning(f"Failed to read cookie cache {cache_path}: {e}")
        return None
    for cookie in cookies or []:
        if isinstance(cookie, dict) and cookie.get("name") == "__Secure-1PSIDTS":
            value = cookie.get("value")
            if isinstance(value, str) and value:
                return value
    return None


def initialize_config() -> Config:
    """
    Initialize configuration from environment variables and the YAML settings source.

    Returns:
        Config: Configuration object with Gemini client and model overrides merged
    """
    try:
        env_clients_overrides = extract_gemini_clients_env()
        env_models_overrides = extract_gemini_models_env()
        settings_cls: type[Any] = Config
        config = cast(Config, settings_cls())

        config.gemini.clients = _merge_clients_with_env(
            config.gemini.clients, env_clients_overrides
        )
        config.gemini.models = _merge_models_with_env(config.gemini.models, env_models_overrides)
        register_boot_models(config.gemini.models)

        return config
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e!s}")
        sys.exit(1)
