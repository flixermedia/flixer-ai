from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from urllib.parse import quote
from uuid import UUID

import httpx

from .config import Settings
from .schemas import RecommendationContext

logger = logging.getLogger("flixer_ai.flixer")


def signature(secret: str, timestamp: str, method: str, path: str, body: bytes = b"") -> str:
    body_hash = hashlib.sha256(body).hexdigest()
    message = f"flixer-ai:v1:{timestamp}:{method.upper()}:{path}:{body_hash}"
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


class FlixerClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(timeout=settings.flixer_timeout_seconds, follow_redirects=False)

    async def close(self) -> None:
        await self._client.aclose()

    async def recommendation_context(self, user_id: UUID) -> RecommendationContext:
        path = f"/api/internal/ai/users/{quote(str(user_id), safe='')}/recommendation-context"
        return await self._get_context(path)

    async def library_context(self) -> RecommendationContext:
        return await self._get_context("/api/internal/ai/library-context")

    async def runtime_configuration(self) -> dict[str, object]:
        return await self._get_json("/api/internal/ai/runtime-configuration")

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

    async def _request_json(self, method: str, path: str, body: bytes = b"") -> dict[str, object]:
        secret = self.settings.internal_token
        if not secret:
            raise RuntimeError("The Flixer internal token is not configured.")
        timestamp = str(int(time.time()))
        headers = {
            "x-flixer-ai-timestamp": timestamp,
            "x-flixer-ai-signature": signature(secret, timestamp, method, path, body),
        }
        if body:
            headers["content-type"] = "application/json"
        try:
            response = await self._client.request(method, f"{self.settings.flixer_url}{path}", headers=headers, content=body or None)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Flixer returned an invalid response.")
            return result
        except (httpx.HTTPError, ValueError) as error:
            logger.warning("Flixer recommendation data request failed: %s", error)
            raise RuntimeError("Flixer could not provide recommendation data.") from error
