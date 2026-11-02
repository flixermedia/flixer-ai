import asyncio

import httpx

from flixer_ai.config import Settings
from flixer_ai.flixer_client import FlixerClient, signature


def test_library_context_uses_the_core_hmac_contract(tmp_path) -> None:
    settings = Settings(
        port=8094,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        recommender_profile="classic",
        allow_downloads=False,
        flixer_url="http://flixer:8081",
        internal_token="a-shared-secret-long-enough-for-tests",
    )
    async def exercise_client() -> None:
        client = FlixerClient(settings)
        await client.close()

        async def handler(request: httpx.Request) -> httpx.Response:
            timestamp = request.headers["x-flixer-ai-timestamp"]
            expected = signature(settings.internal_token or "", timestamp, "GET", "/api/internal/ai/library-context")
            assert request.url.path == "/api/internal/ai/library-context"
            assert request.headers["x-flixer-ai-signature"] == expected
            return httpx.Response(200, json={"candidates": [], "history": [], "favorites": []})

        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            context = await client.library_context()
        finally:
            await client.close()
        assert context.candidates == []

    asyncio.run(exercise_client())
