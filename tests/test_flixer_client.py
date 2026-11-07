import asyncio

import httpx

from flixer_ai.config import Settings
from flixer_ai.flixer_client import FlixerClient, signature
from flixer_ai.log_context import trace_flags, trace_id


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


def test_callbacks_forward_trace_context_without_changing_authentication(tmp_path) -> None:
    settings = Settings(
        port=8094, data_dir=tmp_path, models_dir=tmp_path / "models",
        recommender_profile="classic", allow_downloads=False, flixer_url="http://flixer:8081",
        internal_token="a-shared-secret-long-enough-for-tests",
    )
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        timestamp = request.headers["x-flixer-ai-timestamp"]
        assert request.headers["x-flixer-ai-signature"] == signature(
            settings.internal_token or "", timestamp, "GET", "/api/internal/ai/library-context"
        )
        seen.append(request.headers.get("traceparent"))
        return httpx.Response(200, json={"candidates": [], "history": [], "favorites": []})

    async def exercise_client() -> None:
        client = FlixerClient(settings)
        await client.close()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await client.library_context()
            token = trace_id.set("a" * 32)
            flags_token = trace_flags.set("00")
            try:
                await asyncio.gather(client.library_context(), client.library_context())
            finally:
                trace_flags.reset(flags_token)
                trace_id.reset(token)
        finally:
            await client.close()

    asyncio.run(exercise_client())
    assert seen[0] is None
    spans = []
    for header in seen[1:]:
        assert header is not None
        version, trace, span, flags = header.split("-")
        assert (version, trace, flags) == ("00", "a" * 32, "00")
        assert len(span) == 16 and int(span, 16) != 0
        spans.append(span)
    assert len(set(spans)) == 2
