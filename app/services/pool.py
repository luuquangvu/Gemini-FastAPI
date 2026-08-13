import asyncio
import random
from collections import deque
from typing import Any

from loguru import logger

from app.utils import g_config
from app.utils.config import load_cached_1psidts
from app.utils.singleton import Singleton

from .client import GeminiClientWrapper


class GeminiClientPool(metaclass=Singleton):
    """Pool of GeminiClient instances identified by unique ids."""

    def __init__(self) -> None:
        self._clients: list[GeminiClientWrapper] = []
        self._id_map: dict[str, GeminiClientWrapper] = {}
        self._round_robin: deque[GeminiClientWrapper] = deque()
        self._restart_locks: dict[str, asyncio.Lock] = {}
        self._client_specs: dict[str, dict[str, Any]] = {}

        if len(g_config.gemini.clients) == 0:
            raise ValueError("No Gemini clients configured")

        for c in g_config.gemini.clients:
            kwargs = c.model_dump(exclude={"id"})
            client = GeminiClientWrapper(
                client_id=c.id,
                **kwargs,
            )
            self._clients.append(client)
            self._id_map[c.id] = client
            self._round_robin.append(client)
            self._restart_locks[c.id] = asyncio.Lock()
            self._client_specs[c.id] = kwargs

    async def _init_one(self, client: GeminiClientWrapper) -> bool:
        """Initialize a single client: primary credentials, cached-1PSIDTS retry, probe."""
        live_client = await self._init_with_fallback(client)
        if live_client is None:
            return False

        if g_config.gemini.validate_session_on_init and not await live_client.validate_session():
            logger.error(
                f"Session validation probe failed for client {live_client.id}; "
                f"marking it unavailable."
            )
            await live_client.close()
            return False
        return True

    async def _init_with_fallback(self, client: GeminiClientWrapper) -> GeminiClientWrapper | None:
        """Initialize a client, retrying once with a cached rotated 1PSIDTS on failure.

        Returns the live client on success (possibly a replacement object), or None."""
        if await self._init_attempt(client):
            return client

        spec = self._client_specs.get(client.id) or {}
        psid = spec.get("secure_1psid")
        cached_1psidts = load_cached_1psidts(psid) if psid else None
        if not cached_1psidts or cached_1psidts == spec.get("secure_1psidts"):
            logger.error(f"Failed to initialize client {client.id}")
            return None
        logger.warning(
            f"Client {client.id} init failed; retrying once with cached rotated 1PSIDTS."
        )
        retry_kwargs = {**spec, "secure_1psidts": cached_1psidts}
        replacement = GeminiClientWrapper(client_id=client.id, **retry_kwargs)
        if await self._init_attempt(replacement):
            self._replace_client(client, replacement, retry_kwargs)
            return replacement
        logger.error(f"Client {client.id} retry with cached 1PSIDTS also failed; giving up.")
        return None

    async def _init_attempt(self, client: GeminiClientWrapper) -> bool:
        """Run library init; returns True on success."""
        try:
            await client.init()
            return True
        except Exception:
            return False

    def _replace_client(
        self, old: GeminiClientWrapper, replacement: GeminiClientWrapper, kwargs: dict[str, Any]
    ) -> None:
        """Swap a client object for a fresh one (credential retry)."""
        for i, client in enumerate(self._clients):
            if client is old:
                self._clients[i] = replacement
                break
        self._id_map[replacement.id] = replacement
        self._round_robin = deque(self._clients)
        self._client_specs[replacement.id] = kwargs

    async def init(self) -> None:
        """Initialize all clients in the pool with staggered start times."""
        clients_to_init = [c for c in self._clients if not c.running()]
        for i, client in enumerate(clients_to_init):
            await self._init_one(client)

            if i < len(clients_to_init) - 1:
                delay = random.uniform(5, 30)
                logger.info(f"Staggering next initialization by {delay:.2f}s")
                await asyncio.sleep(delay)

        success_count = sum(bool(client.running()) for client in self._clients)
        if success_count == 0:
            raise RuntimeError("Failed to initialize any Gemini clients")

    async def acquire(self, client_id: str | None = None) -> GeminiClientWrapper:
        """Return a healthy client by id or using round-robin."""
        if not self._round_robin:
            raise RuntimeError("No Gemini clients configured")

        if client_id:
            client = self._id_map.get(client_id)
            if not client:
                raise ValueError(f"Client id {client_id} not found")
            if await self._ensure_client_ready(client):
                return client
            raise RuntimeError(
                f"Gemini client {client_id} is not running and could not be restarted"
            )

        for _ in range(len(self._round_robin)):
            client = self._round_robin[0]
            self._round_robin.rotate(-1)
            if await self._ensure_client_ready(client):
                return client

        raise RuntimeError("No Gemini clients are currently available")

    async def _ensure_client_ready(self, client: GeminiClientWrapper) -> bool:
        """Make sure the client is running, attempting a restart if needed."""
        if client.running():
            return True

        lock = self._restart_locks.get(client.id)
        if lock is None:
            return False

        async with lock:
            if client.running():
                return True

            live_client = await self._init_with_fallback(client)
            if live_client is None:
                return False
            if (
                g_config.gemini.validate_session_on_init
                and not await live_client.validate_session()
            ):
                logger.error(
                    f"Session validation probe failed for restarted client {live_client.id}."
                )
                await live_client.close()
                return False
            logger.info(f"Restarted Gemini client {live_client.id} after it stopped.")
            return True

    @property
    def clients(self) -> list[GeminiClientWrapper]:
        """Return managed clients."""
        return self._clients

    async def close(self) -> None:
        """Close all clients in the pool."""
        if not self._clients:
            return

        logger.info(f"Closing {len(self._clients)} Gemini clients...")
        await asyncio.gather(
            *(client.close() for client in self._clients if client.running()),
            return_exceptions=True,
        )
        logger.info("All Gemini clients closed.")

    def status(self) -> dict[str, bool]:
        """Return healthy status for each client."""
        return {client.id: client.is_healthy() for client in self._clients}
