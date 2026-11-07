"""Refresh an AI ingestion credential over a protected Flixer connection."""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
from pathlib import Path
from .flixer_client import FlixerClient
from .logger_export import _read_config

logger = logging.getLogger("flixer_ai.logger_enrollment")


async def refresh(client: FlixerClient, path: Path) -> None:
    try:
        result = await client.logger_configuration()
        endpoint, token, expiry = result.get("endpoint"), result.get("token"), result.get("expiresAt")
        if not isinstance(endpoint, str) or not isinstance(token, str) or not isinstance(expiry, str):
            raise ValueError("Invalid logger configuration.")
        expiration = datetime.datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        if expiration.tzinfo is None or expiration <= datetime.datetime.now(datetime.timezone.utc):
            raise ValueError("Expired logger credential.")
        temporary = path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump({"endpoint": endpoint, "token": token}, output)
            _read_config(temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    except (RuntimeError, ValueError, TypeError, OSError) as error:
        path.unlink(missing_ok=True)
        logger.warning("Logger enrollment unavailable (%s); export is suspended.", type(error).__name__)


async def run(client: FlixerClient, path: Path) -> None:
    while True:
        await refresh(client, path)
        await asyncio.sleep(60)
