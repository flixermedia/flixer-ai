import asyncio
import json

import httpx
import sentry_sdk
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sentry_sdk.transport import Transport

from flixer_ai import monitoring


class MemoryTransport(Transport):
    def __init__(self):
        self.envelopes = []

    def capture_envelope(self, envelope):
        self.envelopes.append(envelope)

    def flush(self, timeout, callback=None):
        pass

    def kill(self):
        pass


def test_disabled_and_invalid_dsn(monkeypatch, caplog):
    monkeypatch.setenv("SENTRY_DSN", "https://public@example.invalid/1")
    assert monitoring.initialize() is False
    assert sentry_sdk.get_client().transport is None
    assert monitoring.initialize("not-a-dsn") is False
    assert "Could not initialize Sentry" in caplog.text
    assert "not-a-dsn" not in caplog.text


def test_unhandled_http_exception_is_reported_without_private_data(monkeypatch):
    transport = MemoryTransport()
    real_init = sentry_sdk.init
    monkeypatch.setattr(monitoring.sentry_sdk, "init", lambda **options: real_init(transport=transport, **options))
    assert monitoring.initialize("https://public@example.invalid/1")
    app = FastAPI()

    @app.get("/private/{item}")
    def fail(item: str):
        raise RuntimeError(f"private exception {item}")

    try:
        response = TestClient(app, raise_server_exceptions=False).get(
            "/private/private-title", headers={"authorization": "Bearer private-token"}
        )
        assert response.status_code == 500
        sentry_sdk.flush()
        payload = json.dumps([item.payload.json for envelope in transport.envelopes for item in envelope.items])
        assert "Unhandled Flixer AI error" in payload
        assert "private" not in payload
        assert "authorization" not in payload.lower()
        assert "request" not in payload.lower()
    finally:
        sentry_sdk.get_client().close()


def test_load_configuration_retries_and_respects_disabled(monkeypatch):
    calls = []
    selected = []
    real_client = httpx.AsyncClient

    def handle(request):
        calls.append(request.url.path)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"enabled": False, "dsn": None})

    monkeypatch.setattr(monitoring.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))
    monkeypatch.setattr(monitoring, "initialize", lambda dsn=None: selected.append(dsn))

    async def immediate_sleep(_):
        pass

    monkeypatch.setattr(monitoring.asyncio, "sleep", immediate_sleep)
    asyncio.run(monitoring.load_configuration("http://flixer"))
    assert calls == ["/api/sentry/config/ai", "/api/sentry/config/ai"]
    assert selected == [None]


def test_load_configuration_uses_dashboard_dsn(monkeypatch):
    selected = []
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"enabled": True, "dsn": " https://public@example.invalid/1 "}))
    monkeypatch.setattr(monitoring.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    monkeypatch.setattr(monitoring, "initialize", lambda dsn=None: selected.append(dsn))
    asyncio.run(monitoring.load_configuration("http://flixer"))
    assert selected == ["https://public@example.invalid/1"]


def test_background_configuration_captures_exceptions_for_existing_app(monkeypatch):
    transport = MemoryTransport()
    real_init = sentry_sdk.init
    real_client = httpx.AsyncClient
    api_transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"enabled": True, "dsn": "https://public@example.invalid/1"}))
    monkeypatch.setattr(monitoring.httpx, "AsyncClient", lambda **kwargs: real_client(transport=api_transport, **kwargs))
    monkeypatch.setattr(monitoring.sentry_sdk, "init", lambda **options: real_init(transport=transport, **options))
    app = FastAPI()

    @app.get("/fail")
    def fail():
        raise RuntimeError("private")

    try:
        asyncio.run(monitoring.load_configuration("http://flixer"))
        assert TestClient(app, raise_server_exceptions=False).get("/fail").status_code == 500
        sentry_sdk.flush()
        assert len(transport.envelopes) == 1
        assert "private" not in json.dumps(transport.envelopes[0].items[0].payload.json)
    finally:
        sentry_sdk.get_client().close()
