"""Optional, privacy-preserving Sentry error reporting."""

import asyncio
import logging
import os
from importlib.metadata import version

import httpx
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from sentry_sdk.types import Event, Hint

logger = logging.getLogger("flixer_ai.monitoring")


def sanitize_event(event: Event, _hint: Hint) -> Event:
    # Recommendation requests and exception text can contain private library data.
    safe = {key: event[key] for key in ("event_id", "timestamp", "platform", "level", "release", "environment") if key in event}
    safe["exception"] = {"values": [{"type": "ApplicationError", "value": "Unhandled Flixer AI error"}]}
    return safe


def initialize(dsn: str | None = None) -> bool:
    try:
        sentry_sdk.init(
            dsn=dsn or "",
            release=f"flixer-ai@{version('flixer-ai')}",
            environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
            default_integrations=False,
            auto_enabling_integrations=False,
            integrations=[StarletteIntegration(), FastApiIntegration(), AsyncioIntegration()],
            before_send=sanitize_event,
            send_client_reports=False,
            auto_session_tracking=False,
            traces_sample_rate=0,
        )
    except ValueError:
        logger.error("Could not initialize Sentry; error reporting is disabled.")
        return False
    return bool(dsn)


async def load_configuration(base_url: str) -> None:
    async with httpx.AsyncClient(timeout=5, follow_redirects=False) as client:
        while True:
            try:
                response = await client.get(f"{base_url.rstrip('/')}/api/sentry/config/ai")
                response.raise_for_status()
                config = response.json()
                if not isinstance(config, dict) or not isinstance(config.get("enabled"), bool):
                    raise ValueError("Invalid Sentry configuration")
                dsn = config.get("dsn")
                if config["enabled"] and (not isinstance(dsn, str) or not dsn.strip()):
                    raise ValueError("Invalid Sentry configuration")
                enabled = initialize(dsn.strip() if config["enabled"] else None)
                logger.info("Sentry configuration loaded; reporting is %s.", "enabled" if enabled else "disabled")
                return
            except (httpx.HTTPError, ValueError):
                logger.warning("Could not load Sentry configuration from Flixer; retrying with reporting disabled.")
                await asyncio.sleep(5)


def flush() -> None:
    sentry_sdk.flush(timeout=2)
