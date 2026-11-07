import io
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from flixer_ai.logger_export import LoggerHandler, _read_config, start_logger, stop_logger
from flixer_ai.log_context import LogContextMiddleware, trace_id


@pytest.fixture
def logger_server():
    batches = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            batches.append((self.headers.get("Authorization"), payload))
            events = payload["events"]
            status = 409 if self.headers.get("X-Flixer-Log-Schema") != "1" else (
                400 if any(event["message"] == "invalid" for event in events) else 200
            )
            response = json.dumps({"accepted": len(events)}).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", batches
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert predicate()


def test_credential_file_requires_private_permissions_and_secure_origin(tmp_path):
    path = tmp_path / "ai.json"
    path.write_text(json.dumps({"endpoint": "https://logger.example", "token": "x" * 40}))
    path.chmod(0o644)
    with pytest.raises(ValueError, match="owned"):
        _read_config(path)
    path.chmod(0o600)
    assert _read_config(path) == ("https://logger.example", "x" * 40)
    path.write_text(json.dumps({"endpoint": "http://127.example", "token": "x" * 40}))
    with pytest.raises(ValueError, match="HTTPS"):
        _read_config(path)
    assert _read_config(tmp_path / "missing") is None


def test_schema_conflict_is_not_retried_or_split(monkeypatch):
    handler = LoggerHandler("http://127.0.0.1:8098", "x" * 40)
    calls = []

    def reject(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 409, "unsupported schema", {}, io.BytesIO())

    monkeypatch.setattr(handler.client, "open", reject)
    try:
        assert handler._send([b"{}", b"{}"]) == 409
        assert len(calls) == 1
    finally:
        handler.stop()


@pytest.mark.parametrize("replacement_endpoint,expected_calls,result", [
    ("http://127.0.0.1:8098", 1, 401),
    ("http://127.0.0.1:8099", 1, 401),
])
def test_unauthorized_batch_reloads_rotated_credential_without_crossing_endpoints(
    tmp_path, monkeypatch, replacement_endpoint, expected_calls, result,
):
    endpoint = "http://127.0.0.1:8098"
    first, replacement = "x" * 40, "y" * 40
    config = tmp_path / "ai.json"
    config.write_text(json.dumps({"endpoint": endpoint, "token": first}))
    config.chmod(0o600)
    handler = LoggerHandler(endpoint, first, config_path=config)
    calls = []

    def request(request, timeout):
        calls.append((request.full_url, request.get_header("Authorization"), request.data))
        if len(calls) == 1:
            config.write_text(json.dumps({"endpoint": replacement_endpoint, "token": replacement}))
            raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, io.BytesIO())
        return io.BytesIO(b'{"accepted":1}')

    monkeypatch.setattr(handler.client, "open", request)
    try:
        assert handler._send([b"{}"]) == result
        assert len(calls) == expected_calls
        assert calls[0][1] == "Bearer " + first
    finally:
        handler.stop()


def test_endpoint_change_during_batch_split_drops_remaining_siblings(tmp_path, monkeypatch):
    endpoint = "http://127.0.0.1:8098"
    config = tmp_path / "ai.json"
    config.write_text(json.dumps({"endpoint": endpoint, "token": "x" * 40}))
    config.chmod(0o600)
    handler = LoggerHandler(endpoint, "x" * 40, config_path=config)
    calls = []

    def request(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 400, "invalid", {}, io.BytesIO())
        config.write_text(json.dumps({"endpoint": "http://127.0.0.1:8099", "token": "y" * 40}))
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, io.BytesIO())

    monkeypatch.setattr(handler.client, "open", request)
    try:
        handler._deliver([b'{"event_id":"one"}', b'{"event_id":"two"}'], 0)
        assert calls == [endpoint + "/api/logs/batch"] * 2
        assert handler.dropped == 2
    finally:
        handler.stop()


def test_export_is_asynchronous_redacted_and_preserves_valid_siblings(logger_server):
    endpoint, batches = logger_server
    token = "x" * 40
    handler = LoggerHandler(endpoint, token)
    try:
        for message in (
            "failed token=verysecret https://example.com/a /home/user/movie user@example.com 192.0.2.10 00000000-0000-4000-8000-000000000001",
            "invalid",
            "valid",
        ):
            handler.handle(logging.LogRecord("flixer_ai.test", logging.ERROR, __file__, 1, message, (), None))
        _wait_for(lambda: handler.dropped == 1 and len(batches) >= 3)
        accepted = [item for _, batch in batches for item in batch["events"] if item["message"] != "invalid"]
        assert accepted
        assert all(auth == "Bearer " + token for auth, _ in batches)
        serialized = json.dumps(accepted)
        for secret in ("verysecret", "example.com", "/home/user", "user@example.com", "192.0.2.10", "00000000-0000-4000-8000-000000000001"):
            assert secret not in serialized
        assert any(event["message"] == "valid" for event in accepted)
        assert all(event["app"] == "flixer-ai" and event["schema_version"] == 1 for event in accepted)
        assert len({event["event_id"] for event in accepted}) < len(accepted)
    finally:
        handler.stop()


def test_access_logs_omit_client_and_request_path(logger_server):
    endpoint, batches = logger_server
    handler = LoggerHandler(endpoint, "x" * 40)
    try:
        record = logging.LogRecord(
            "uvicorn.access", logging.INFO, __file__, 1,
            '%s - "%s %s HTTP/%s" %d',
            ("192.0.2.1:1234", "POST", "/secret?token=private", "1.1", 200), None,
        )
        handler.handle(record)
        _wait_for(lambda: len(batches) == 1)
        message = batches[0][1]["events"][0]["message"]
        assert message == "HTTP POST 200"
        assert "192.0.2.1" not in message and "private" not in message
    finally:
        handler.stop()


def test_request_correlation_exports_only_safe_fields(logger_server):
    endpoint, batches = logger_server
    handler = LoggerHandler(endpoint, "x" * 40)
    token = trace_id.set("a" * 32)
    try:
        record = logging.LogRecord("flixer_ai.request", logging.INFO, __file__, 1, "request completed", (), None)
        record.duration_ms = 12.5
        record.status_code = 200
        record.model_profile = "classic"
        record.result_count = 5
        record.prompt = "private prompt"
        handler.handle(record)
        _wait_for(lambda: len(batches) == 1)
        event = batches[0][1]["events"][0]
        assert event["trace_id"] == "a" * 32
        assert event["duration_ms"] == 12.5
        assert event["fields"] == {"status_code": 200, "model_profile": "classic", "result_count": 5}
        assert "private prompt" not in json.dumps(event)
    finally:
        trace_id.reset(token)
        handler.stop()


def test_model_fields_reject_untrusted_values(logger_server):
    endpoint, batches = logger_server
    handler = LoggerHandler(endpoint, "x" * 40)
    try:
        record = logging.LogRecord("flixer_ai.test", logging.INFO, __file__, 1, "inference completed", (), None)
        record.model_profile = "private model details"
        record.result_count = True
        handler.handle(record)
        _wait_for(lambda: len(batches) == 1)
        event = batches[0][1]["events"][0]
        assert "fields" not in event
        assert "private model details" not in json.dumps(event)
    finally:
        handler.stop()


def test_http_request_logs_share_trace_and_do_not_export_request_data(logger_server):
    endpoint, batches = logger_server
    handler = LoggerHandler(endpoint, "x" * 40)
    target = logging.getLogger("flixer_ai")
    old_level = target.level
    target.setLevel(logging.INFO)
    target.addHandler(handler)
    app = FastAPI()
    app.add_middleware(LogContextMiddleware)

    @app.post("/private/path")
    async def route(request: Request):
        await request.body()
        logging.getLogger("flixer_ai.test").info("processing request")
        return {"ok": True}

    try:
        with TestClient(app) as client:
            response = client.post(
                "/private/path?token=sensitive",
                headers={"traceparent": "00-" + "a" * 32 + "-" + "1" * 16 + "-01"},
                json={"prompt": "private content"},
            )
        assert response.status_code == 200
        _wait_for(lambda: sum(len(batch["events"]) for _, batch in batches) >= 2)
        events = [event for _, batch in batches for event in batch["events"]]
        assert all(event.get("trace_id") == "a" * 32 for event in events)
        assert any(event.get("fields") == {"status_code": 200} and event["duration_ms"] >= 0 for event in events)
        serialized = json.dumps(events)
        assert all(value not in serialized for value in ("private/path", "sensitive", "private content"))
    finally:
        target.removeHandler(handler)
        target.setLevel(old_level)
        handler.stop()


def test_formatting_errors_never_escape_logging(logger_server):
    endpoint, _ = logger_server
    handler = LoggerHandler(endpoint, "x" * 40)
    class Broken:
        def __str__(self):
            raise RuntimeError("sensitive exception")

    try:
        handler.handle(logging.LogRecord("flixer_ai.test", logging.ERROR, __file__, 1, "%s", (Broken(),), None))
        assert handler.dropped == 1
    finally:
        handler.stop()


def test_retry_keeps_event_ids_and_honors_rate_limit(logger_server, monkeypatch):
    endpoint, _ = logger_server
    handler = LoggerHandler(endpoint, "x" * 40)
    bodies = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, _):
            return b'{"accepted":1}'

    def open_request(request, timeout):
        bodies.append(request.data)
        if len(bodies) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "limited", {"Retry-After": "0"}, None)
        return Response()

    monkeypatch.setattr(handler.client, "open", open_request)
    monkeypatch.setattr(handler.stopping, "wait", lambda _: False)
    try:
        assert handler._send([b'{"event_id":"same"}']) == 200
        assert len(bodies) == 2 and bodies[0] == bodies[1]
    finally:
        handler.stop()


def test_circuit_drops_batches_until_recovery(logger_server, monkeypatch):
    endpoint, _ = logger_server
    handler = LoggerHandler(endpoint, "x" * 40)
    calls = []

    def fail(_):
        calls.append(True)
        return 503

    monkeypatch.setattr(handler, "_send", fail)
    try:
        for _ in range(4):
            handler._deliver([b"{}"], 0)
        assert len(calls) == 3
        assert handler.dropped == 4
        handler._open_until = 0
        monkeypatch.setattr(handler, "_send", lambda _: 200)
        handler._deliver([b"{}"], 0)
        assert handler._failures == 0 and handler._open_until == 0
    finally:
        handler.stop()


def test_attach_and_detach_without_replacing_existing_handlers(tmp_path, logger_server):
    endpoint, _ = logger_server
    path = tmp_path / "ai.json"
    path.write_text(json.dumps({"endpoint": endpoint, "token": "x" * 40}))
    path.chmod(0o600)
    root = logging.getLogger()
    before = list(root.handlers)
    active = start_logger(path)
    assert active.active is not None
    assert root.handlers[:len(before)] == before
    assert active.active[0] in root.handlers
    stop_logger(active)
    assert root.handlers == before


def test_exporter_starts_when_configuration_arrives(tmp_path, logger_server, monkeypatch):
    endpoint, batches = logger_server
    path = tmp_path / "ai.json"
    monkeypatch.setattr("flixer_ai.logger_export._CONFIG_REFRESH_SECONDS", 0.01)
    root = logging.getLogger()
    before = list(root.handlers)
    supervisor = start_logger(path)
    try:
        assert supervisor.active is None
        path.write_text(json.dumps({"endpoint": endpoint, "token": "x" * 40}))
        path.chmod(0o600)
        _wait_for(lambda: supervisor.active is not None)
        assert supervisor.active is not None
        supervisor.active[0].handle(logging.LogRecord("flixer_ai.test", logging.INFO, __file__, 1, "started later", (), None))
        _wait_for(lambda: len(batches) == 1)
        assert batches[0][1]["events"][0]["message"] == "started later"
    finally:
        stop_logger(supervisor)
    assert root.handlers == before


def test_credential_rotation_and_suspension(tmp_path, logger_server, monkeypatch):
    endpoint, batches = logger_server
    path = tmp_path / "ai.json"
    first, second = "x" * 40, "y" * 40
    path.write_text(json.dumps({"endpoint": endpoint, "token": first}))
    path.chmod(0o600)
    monkeypatch.setattr("flixer_ai.logger_export._CONFIG_REFRESH_SECONDS", 0.01)
    handler = LoggerHandler(endpoint, first, config_path=path)

    def emit(message):
        handler.handle(logging.LogRecord("flixer_ai.test", logging.INFO, __file__, 1, message, (), None))

    try:
        emit("before")
        _wait_for(lambda: len(batches) == 1)
        path.write_text(json.dumps({"endpoint": endpoint, "token": second}))
        time.sleep(0.02)
        emit("during rotation")
        _wait_for(lambda: handler.token == second)
        emit("after")
        _wait_for(lambda: len(batches) == 2)
        assert batches[0][0] == "Bearer " + first
        assert batches[1][0] == "Bearer " + second

        path.chmod(0o644)
        time.sleep(0.02)
        emit("suspended")
        _wait_for(lambda: not handler._config_available)
        assert len(batches) == 2
        path.chmod(0o600)
        time.sleep(0.02)
        emit("resumed")
        _wait_for(lambda: len(batches) == 3)
        assert batches[2][1]["events"][0]["message"] == "resumed"
    finally:
        handler.stop()


def test_endpoint_change_drops_events_started_before_switch(tmp_path, logger_server):
    endpoint, batches = logger_server
    path = tmp_path / "ai.json"
    token = "x" * 40
    path.write_text(json.dumps({"endpoint": endpoint, "token": token}))
    path.chmod(0o600)
    handler = LoggerHandler(endpoint, token, config_path=path)
    entered, finish = threading.Event(), threading.Event()

    class SlowMessage:
        def __str__(self):
            entered.set()
            assert finish.wait(5)
            return "old endpoint event"

    thread = threading.Thread(target=lambda: handler.handle(
        logging.LogRecord("flixer_ai.test", logging.INFO, __file__, 1, "%s", (SlowMessage(),), None)
    ))
    try:
        thread.start()
        assert entered.wait(5)
        host, port = endpoint.rsplit(":", 1)
        path.write_text(json.dumps({"endpoint": f"{host}:0{port}", "token": token}))
        handler._next_config_check = 0
        assert handler._refresh_config() == "changed"
        finish.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        handler.handle(logging.LogRecord("flixer_ai.test", logging.INFO, __file__, 1, "new endpoint event", (), None))
        _wait_for(lambda: len(batches) == 1)
        assert [event["message"] for event in batches[0][1]["events"]] == ["new endpoint event"]
        assert handler.dropped == 1
    finally:
        finish.set()
        thread.join(timeout=5)
        handler.stop()


def test_credential_file_rejects_fifo_without_blocking(tmp_path):
    path = tmp_path / "ai.json"
    os.mkfifo(path, 0o600)
    with pytest.raises(ValueError, match="configuration file"):
        _read_config(path)


@pytest.mark.skipif(
    not all(os.getenv(name) for name in ("LOGGER_INTEGRATION_URL", "LOGGER_INTEGRATION_INGEST", "LOGGER_INTEGRATION_QUERY")),
    reason="requires an isolated real Flixer Logger",
)
def test_real_logger_ingestion_and_search():
    endpoint = os.environ["LOGGER_INTEGRATION_URL"]
    ingest = os.environ["LOGGER_INTEGRATION_INGEST"]
    query = os.environ["LOGGER_INTEGRATION_QUERY"]
    marker = "aiintegration" + uuid.uuid4().hex
    handler = LoggerHandler(endpoint, ingest)
    try:
        handler.handle(logging.LogRecord("flixer_ai.test", logging.ERROR, __file__, 1, marker + " token=secretvalue", (), None))

        def find():
            request = urllib.request.Request(
                f"{endpoint}/api/logs?field=text&value={marker}",
                headers={"Authorization": "Bearer " + query},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                events = json.load(response)["events"]
            return bool(events and events[0]["app"] == "flixer-ai" and "secretvalue" not in events[0]["message"])

        _wait_for(find)
        assert handler.dropped == 0
    finally:
        handler.stop()
