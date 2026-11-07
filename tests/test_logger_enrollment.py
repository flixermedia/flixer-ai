import asyncio
import datetime
import json
import os
import stat

import pytest

from flixer_ai.config import Settings
from flixer_ai.flixer_client import FlixerClient, secure_control_url, signature
from flixer_ai.logger_enrollment import refresh
from flixer_ai.logger_export import LoggerHandler, _logger_ssl_context, _read_config


@pytest.mark.parametrize(("url", "expected"), [
    ("https://flixer.example", True),
    ("http://127.0.0.1:8080", True),
    ("http://[::1]:8080", True),
    ("http://flixer:8080", False),
    ("http://localhost:8080", False),
    ("https://user:password@flixer.example", False),
    ("https://flixer.example?token=secret", False),
])
def test_control_transport(url, expected):
    assert secure_control_url(url) is expected


def test_refresh_writes_private_validated_configuration_and_suspends_on_failure(tmp_path):
    class Client:
        fail = False

        async def logger_configuration(self):
            if self.fail:
                raise RuntimeError("unavailable")
            return {
                "endpoint": "https://logger.example",
                "token": "x" * 40,
                "expiresAt": (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)).isoformat(),
            }

    client = Client()
    path = tmp_path / "config.json"
    asyncio.run(refresh(client, path))
    assert _read_config(path) == ("https://logger.example", "x" * 40)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    client.fail = True
    asyncio.run(refresh(client, path))
    assert not path.exists()


def test_credential_change_discards_old_generation_even_on_same_endpoint(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"endpoint":"https://logger.example","token":"' + "y" * 40 + '"}')
    path.chmod(0o600)
    handler = LoggerHandler("https://logger.example", "x" * 40, config_path=path)
    try:
        handler._next_config_check = 0
        assert handler._refresh_config() == "changed"
        assert handler._generation == 1
        handler._deliver([b"{}"], 0)
        assert handler.dropped == 1
    finally:
        handler.stop()


def test_logger_tls_uses_the_configured_ca_file(tmp_path, monkeypatch):
    ca_file = tmp_path / "logger-ca.pem"
    ca_file.write_text("invalid certificate")
    monkeypatch.setenv("FLIXER_LOGGER_CA_FILE", str(ca_file))
    with pytest.raises(OSError):
        _logger_ssl_context()


def test_client_refuses_plaintext_non_loopback_enrollment(tmp_path):
    settings = Settings(
        port=8094, data_dir=tmp_path, models_dir=tmp_path, recommender_profile="classic",
        allow_downloads=False, flixer_url="http://flixer:8080", internal_token="shared",
    )

    async def exercise():
        client = FlixerClient(settings)
        try:
            with pytest.raises(RuntimeError, match="protected"):
                await client.logger_configuration()
        finally:
            await client.close()

    asyncio.run(exercise())


def test_client_signs_local_socket_request_and_rejects_unsafe_directory(tmp_path):
    async def exercise():
        socket_path = tmp_path / "control.sock"
        secret = "s" * 32
        requests = []

        async def handler(reader, writer):
            request = (await reader.readuntil(b"\r\n\r\n")).decode("ascii")
            requests.append(request)
            lines = request.split("\r\n")
            path = lines[0].split()[1]
            headers = dict(line.split(": ", 1) for line in lines[1:] if ": " in line)
            assert path == "/api/internal/ai/logger-configuration"
            timestamp = headers["x-flixer-ai-timestamp"]
            assert headers["x-flixer-ai-signature"] == signature(secret, timestamp, "GET", path)
            body = json.dumps({"endpoint": "https://logger.example", "token": "x" * 40, "expiresAt": "2099-01-01T00:00:00Z"}).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(handler, path=str(socket_path))
        settings = Settings(
            port=8094, data_dir=tmp_path, models_dir=tmp_path, recommender_profile="classic",
            allow_downloads=False, flixer_url="http://flixer:8080", internal_token=secret,
            logger_control_socket=str(socket_path),
        )
        client = FlixerClient(settings)
        try:
            assert (await client.logger_configuration())["endpoint"] == "https://logger.example"
            tmp_path.chmod(0o777)
            with pytest.raises(RuntimeError, match="unsafe"):
                await client.logger_configuration()
            assert len(requests) == 1
        finally:
            tmp_path.chmod(0o700)
            await client.close()
            server.close()
            await server.wait_closed()

    asyncio.run(exercise())


def test_invalid_or_expired_response_removes_previous_configuration(tmp_path):
    class Client:
        async def logger_configuration(self):
            return {"endpoint": "https://logger.example", "token": "x" * 40, "expiresAt": "2000-01-01T00:00:00Z"}

    path = tmp_path / "config.json"
    path.write_text("old configuration")
    asyncio.run(refresh(Client(), path))
    assert not path.exists()
