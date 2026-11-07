from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import os
import secrets
import stat
import time
from pathlib import Path
from urllib.parse import quote, urlsplit
from uuid import UUID

import httpx

from .config import Settings
from .log_context import trace_flags, trace_id
from .schemas import RecommendationContext

logger = logging.getLogger("flixer_ai.flixer")


def secure_control_url(value: str) -> bool:
    try:
        url = urlsplit(value)
        if not url.hostname or url.username or url.password or url.query or url.fragment:
            return False
        _ = url.port
        if url.scheme == "https":
            return True
        return url.scheme == "http" and ipaddress.ip_address(url.hostname).is_loopback
    except ValueError:
        return False


def signature(secret: str, timestamp: str, method: str, path: str, body: bytes = b"") -> str:
    body_hash = hashlib.sha256(body).hexdigest()
    message = f"flixer-ai:v1:{timestamp}:{method.upper()}:{path}:{body_hash}"
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


class FlixerClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(timeout=settings.flixer_timeout_seconds, follow_redirects=False)
        self._logger_client = (
            httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(
                    uds=settings.logger_control_socket, limits=httpx.Limits(max_keepalive_connections=0),
                ),
                timeout=settings.flixer_timeout_seconds, follow_redirects=False,
            ) if settings.logger_control_socket else None
        )

    async def close(self) -> None:
        await self._client.aclose()
        if self._logger_client:
            await self._logger_client.aclose()

    async def recommendation_context(self, user_id: UUID) -> RecommendationContext:
        path = f"/api/internal/ai/users/{quote(str(user_id), safe='')}/recommendation-context"
        return await self._get_context(path)

    async def library_context(self) -> RecommendationContext:
        return await self._get_context("/api/internal/ai/library-context")

    async def runtime_configuration(self) -> dict[str, object]:
        return await self._get_json("/api/internal/ai/runtime-configuration")

    async def logger_configuration(self) -> dict[str, object]:
        if self._logger_client:
            path = Path(self.settings.logger_control_socket)
            try:
                parent = os.lstat(path.parent)
                socket = os.lstat(path)
                if (not path.is_absolute() or not stat.S_ISDIR(parent.st_mode)
                    or parent.st_mode & 0o022 or not stat.S_ISSOCK(socket.st_mode)
                    or parent.st_uid != socket.st_uid):
                    raise RuntimeError("Logger control socket is unsafe.")
            except OSError as error:
                raise RuntimeError("Logger control socket is unavailable.") from error
            return await self._request_json("GET", "/api/internal/ai/logger-configuration", client=self._logger_client, origin="http://localhost")
        if not secure_control_url(self.settings.flixer_url):
            raise RuntimeError("Logger enrollment requires a protected Flixer connection.")
        return await self._get_json("/api/internal/ai/logger-configuration")

    async def load_recommendation_cache(self, user_id: str) -> dict[str, object]:
        path = f"/api/internal/ai/users/{quote(user_id, safe='')}/recommendation-cache"
        return await self._get_json(path)

    async def save_recommendation_cache(self, user_id: str, state: dict[str, object]) -> None:
        path = f"/api/internal/ai/users/{quote(user_id, safe='')}/recommendation-cache"
        body = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        await self._request_json("PUT", path, body)

    async def clear_recommendation_cache(self) -> None:
        await self._request_json("DELETE", "/api/internal/ai/recommendation-cache")

    async def _get_context(self, path: str) -> RecommendationContext:
        return RecommendationContext.model_validate(await self._get_json(path))

    async def _get_json(self, path: str) -> dict[str, object]:
        return await self._request_json("GET", path)

    async def _request_json(
        self, method: str, path: str, body: bytes = b"", *,
        client: httpx.AsyncClient | None = None, origin: str | None = None,
    ) -> dict[str, object]:
        secret = self.settings.internal_token
        if not secret:
            raise RuntimeError("The Flixer internal token is not configured.")
        timestamp = str(int(time.time()))
        headers = {
            "x-flixer-ai-timestamp": timestamp,
            "x-flixer-ai-signature": signature(secret, timestamp, method, path, body),
        }
        current_trace = trace_id.get()
        if current_trace:
            span = secrets.token_hex(8)
            while span == "0" * 16:
                span = secrets.token_hex(8)
            headers["traceparent"] = f"00-{current_trace}-{span}-{trace_flags.get()}"
        if body:
            headers["content-type"] = "application/json"
        try:
            response = await (client or self._client).request(method, f"{origin or self.settings.flixer_url}{path}", headers=headers, content=body or None)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Flixer returned an invalid response.")
            return result
        except (httpx.HTTPError, ValueError) as error:
            logger.warning("Flixer recommendation data request failed: %s", error)
            raise RuntimeError("Flixer could not provide recommendation data.") from error
