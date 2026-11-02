"""Startup behavior when the conversation store cannot be opened."""

import asyncio

from app import main


def test_retention_cleanup_disables_itself_when_storage_is_unavailable(monkeypatch):
    """A storage permission error must not abort the whole server at startup."""

    def _unavailable():
        raise OSError("data/lmdb: Permission denied")

    monkeypatch.setattr(main, "LMDBConversationStore", _unavailable)

    asyncio.run(main._run_retention_cleanup(asyncio.Event()))
