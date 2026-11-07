import json
import logging
import time

from flixer_ai.logger_export import start_logger, stop_logger


def test_stopped_supervisor_never_attaches_after_configuration_arrives(tmp_path, monkeypatch):
    monkeypatch.setattr("flixer_ai.logger_export._CONFIG_REFRESH_SECONDS", 0.01)
    path = tmp_path / "ai.json"
    root = logging.getLogger()
    before = list(root.handlers)
    supervisor = start_logger(path)
    stop_logger(supervisor)
    path.write_text(json.dumps({"endpoint": "http://127.0.0.1:8098", "token": "x" * 40}))
    path.chmod(0o600)
    time.sleep(0.05)
    assert supervisor.active is None
    assert root.handlers == before


def test_watcher_start_failure_does_not_break_service(tmp_path, monkeypatch, capsys):
    def fail(_thread):
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr("threading.Thread.start", fail)
    supervisor = start_logger(tmp_path / "ai.json")
    try:
        assert supervisor.active is None
        assert "configuration watcher could not start" in capsys.readouterr().err
    finally:
        stop_logger(supervisor)
