import json
import logging
import os
import time
import urllib.request
import uuid

import pytest

from flixer_ai.logger_export import LoggerHandler


@pytest.mark.skipif(
    not all(os.getenv(name) for name in (
        "LOGGER_INTEGRATION_URL", "LOGGER_INTEGRATION_INGEST", "LOGGER_INTEGRATION_INGEST_NEXT",
        "LOGGER_INTEGRATION_INGEST_ID", "LOGGER_INTEGRATION_QUERY", "LOGGER_INTEGRATION_MANAGE",
    )),
    reason="requires an isolated real Flixer Logger with two ingestion credentials",
)
def test_real_logger_credential_rotation(tmp_path):
    endpoint = os.environ["LOGGER_INTEGRATION_URL"]
    first = os.environ["LOGGER_INTEGRATION_INGEST"]
    second = os.environ["LOGGER_INTEGRATION_INGEST_NEXT"]
    credential_id = os.environ["LOGGER_INTEGRATION_INGEST_ID"]
    query = os.environ["LOGGER_INTEGRATION_QUERY"]
    manage = os.environ["LOGGER_INTEGRATION_MANAGE"]
    path = tmp_path / "ai.json"
    path.write_text(json.dumps({"endpoint": endpoint, "token": first}))
    path.chmod(0o600)
    handler = LoggerHandler(endpoint, first, config_path=path)

    def emit_and_find(marker):
        handler.handle(logging.LogRecord("flixer_ai.rotation", logging.INFO, __file__, 1, marker, (), None))
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            request = urllib.request.Request(
                f"{endpoint}/api/logs?field=text&value={marker}",
                headers={"Authorization": "Bearer " + query},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                events = json.load(response)["events"]
            if any(event["message"] == marker and event["app"] == "flixer-ai" for event in events):
                return
            time.sleep(0.1)
        pytest.fail("rotation event was not stored")

    try:
        emit_and_find("first" + uuid.uuid4().hex)
        request = urllib.request.Request(
            f"{endpoint}/api/logger/credentials/{credential_id}",
            headers={"Authorization": "Bearer " + manage},
            method="DELETE",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            assert response.status == 204
        path.write_text(json.dumps({"endpoint": endpoint, "token": second}))
        handler._next_config_check = time.monotonic() + 60
        handler.handle(logging.LogRecord("flixer_ai.rotation", logging.INFO, __file__, 1, "rotation-trigger", (), None))
        deadline = time.monotonic() + 8
        while handler.token != second and time.monotonic() < deadline:
            time.sleep(0.1)
        assert handler.token == second
        emit_and_find("second" + uuid.uuid4().hex)
        assert handler.dropped >= 1
    finally:
        handler.stop()
