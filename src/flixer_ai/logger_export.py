"""Bounded, asynchronous Flixer Logger delivery for Python logging."""
from __future__ import annotations

import datetime
import email.utils
import http.client
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import queue
import random
import re
import ssl
import stat
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .log_context import trace_id

_REDACTIONS = (
    re.compile(r"(?i)\b(?:bearer|basic)\s+[a-z0-9._~+/-]+=*"),
    re.compile(r"""(?i)\b(?:set-cookie|cookie)["']?\s*[:=]\s*[^\r\n]*"""),
    re.compile(r"""(?i)\b[a-z0-9_-]*(?:password|passwd|secret|token|authorization|api[_-]?key|credential|private[_-]?key)[a-z0-9_-]*["']?\s*[:=]\s*("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;}]+)"""),
    re.compile(r"""(?i)\b(?:https?|wss?)://[^\s"'<>]+"""),
    re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"),
    re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"""(?:^|[\s("'=])(?:/[A-Za-z0-9._-]+){2,}[^\s"']*"""),
)
_CONFIG_PATH = Path("/run/flixer-logger/ai.json")
_MAX_BATCH_BYTES = 1024 * 1024
_CONFIG_REFRESH_SECONDS = 5.0


def _redact(value: str) -> str:
    for pattern in _REDACTIONS:
        value = pattern.sub("[REDACTED]", value)
    return value


def _report(message: str) -> None:
    sys.stderr.write(f"Flixer AI logger: {message}\n")


def _read_config(path: Path) -> tuple[str, str] | None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("configuration file must be owned by the service user and inaccessible to others")
        raw = source.read(4097)
    if len(raw) > 4096:
        raise ValueError("configuration file exceeds size limit")
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {"endpoint", "token"}:
        raise ValueError("invalid configuration fields")
    endpoint, token = data["endpoint"], data["token"]
    if not isinstance(endpoint, str) or not isinstance(token, str):
        raise ValueError("invalid configuration values")
    if any(char.isspace() or ord(char) < 32 for char in endpoint):
        raise ValueError("invalid logger endpoint")
    origin = urllib.parse.urlsplit(endpoint)
    _ = origin.port
    if origin.username or origin.password or not origin.hostname or origin.path not in ("", "/") or origin.query or origin.fragment:
        raise ValueError("logger endpoint must be an origin")
    if origin.scheme != "https":
        try:
            loopback = ipaddress.ip_address(origin.hostname).is_loopback
        except ValueError:
            loopback = False
        if origin.scheme != "http" or not loopback:
            raise ValueError("logger endpoint requires HTTPS except on loopback")
    if len(token) < 32 or len(token) > 1024 or not re.fullmatch(r"[a-zA-Z0-9._~+/-]+=*", token):
        raise ValueError("invalid ingestion credential")
    return endpoint.rstrip("/"), token


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _logger_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    ca_path = os.getenv("FLIXER_LOGGER_CA_FILE", "").strip()
    if ca_path:
        context.load_verify_locations(cafile=ca_path)
    return context


class LoggerHandler(logging.Handler):
    def __init__(self, endpoint: str, token: str, config_path: Path | None = None):
        super().__init__()
        self.endpoint = endpoint + "/api/logs/batch"
        self.token = token
        self._config_path = config_path
        self._next_config_check = time.monotonic() + _CONFIG_REFRESH_SECONDS
        self._config_available = True
        self._generation = 0
        self._generation_lock = threading.Lock()
        self.instance_id = str(uuid.uuid4())
        self.queue: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=1000)
        self.stopping = threading.Event()
        self.dropped = 0
        self._drop_lock = threading.Lock()
        self._reported_drops = 0
        self._failures = 0
        self._open_until = 0.0
        self.client = urllib.request.build_opener(
            _NoRedirect(), urllib.request.HTTPSHandler(context=_logger_ssl_context()),
        )
        self.worker = threading.Thread(target=self._run, name="flixer-logger-export", daemon=True)
        self.worker.start()

    def emit(self, record: logging.LogRecord) -> None:
        with self._generation_lock:
            generation = self._generation
        try:
            if record.name == "uvicorn.access" and isinstance(record.args, tuple) and len(record.args) == 5:
                method = record.args[1]
                status = record.args[4]
                safe_method = method if isinstance(method, str) and re.fullmatch(r"[A-Z]{1,16}", method) else "INVALID"
                safe_status = status if type(status) is int and 100 <= status <= 599 else "unknown"
                message = f"HTTP {safe_method} {safe_status}"
            else:
                message = _redact(record.getMessage())[:4000] or "(empty log message)"
            event = {
                "schema_version": 1,
                "event_id": uuid.uuid4().hex,
                "time": datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc).isoformat(),
                "level": "error" if record.levelno >= logging.ERROR else "warn" if record.levelno >= logging.WARNING else "info" if record.levelno >= logging.INFO else "debug",
                "instance_id": self.instance_id,
                "app": "flixer-ai",
                "source": _redact(record.name)[:256],
                "message": message,
            }
            current_trace = trace_id.get()
            if current_trace:
                event["trace_id"] = current_trace
            duration = getattr(record, "duration_ms", None)
            if type(duration) in (int, float) and math.isfinite(duration) and 0 <= duration <= 86_400_000:
                event["duration_ms"] = duration
            status = getattr(record, "status_code", None)
            fields = {}
            if type(status) is int and 100 <= status <= 599:
                fields["status_code"] = status
            profile = getattr(record, "model_profile", None)
            if isinstance(profile, str) and profile in {"classic", "micro", "mini"}:
                fields["model_profile"] = profile
            count = getattr(record, "result_count", None)
            if type(count) is int and 0 <= count <= 1000:
                fields["result_count"] = count
            if fields:
                event["fields"] = fields
            if record.exc_info:
                event["error"] = {
                    "type": _redact(record.exc_info[0].__name__)[:256],
                    "stack": _redact(logging.Formatter().formatException(record.exc_info))[:8000],
                }
            payload = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(payload) > 64 * 1024:
                raise ValueError("event exceeds size limit")
            self.queue.put_nowait((generation, payload))
        except Exception:
            # Logging must not propagate arbitrary user-defined formatting errors.
            self._drop(1)

    def _drop(self, count: int) -> None:
        with self._drop_lock:
            self.dropped += count

    def _report_drops(self) -> None:
        with self._drop_lock:
            dropped = self.dropped
        if dropped != self._reported_drops:
            self._reported_drops = dropped
            _report(f"events dropped={dropped}")

    def _run(self) -> None:
        pending: tuple[int, bytes] | None = None
        while not self.stopping.is_set() or not self.queue.empty() or pending is not None:
            try:
                first = pending if pending is not None else self.queue.get(timeout=0.25)
            except queue.Empty:
                self._report_drops()
                continue
            pending = None
            batch = [first]
            size = len(first[1]) + 16
            deadline = time.monotonic() + 0.2
            while len(batch) < 100:
                try:
                    timeout = 0 if self.stopping.is_set() else max(0, deadline - time.monotonic())
                    item = self.queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if size + len(item[1]) + 1 > _MAX_BATCH_BYTES:
                    pending = item
                    break
                batch.append(item)
                size += len(item[1]) + 1
            config_state = self._refresh_config()
            if config_state == "unavailable":
                self._drop(len(batch))
                self._report_drops()
                continue
            with self._generation_lock:
                generation = self._generation
            eligible = [payload for item_generation, payload in batch if item_generation == generation]
            self._drop(len(batch) - len(eligible))
            if eligible:
                self._deliver(eligible, generation)
            self._report_drops()
        self._report_drops()

    def _refresh_config(self) -> str:
        if self._config_path is None:
            return "ready"
        now = time.monotonic()
        if now < self._next_config_check:
            return "ready" if self._config_available else "unavailable"
        self._next_config_check = now + _CONFIG_REFRESH_SECONDS
        try:
            config = _read_config(self._config_path)
        except (OSError, ValueError, TypeError) as error:
            if self._config_available:
                _report(f"configuration is unavailable ({type(error).__name__}); export suspended")
            self._config_available = False
            return "unavailable"
        if config is None:
            if self._config_available:
                _report("configuration is missing; export suspended")
            self._config_available = False
            return "unavailable"
        endpoint, token = config
        changed = self.endpoint != endpoint + "/api/logs/batch" or self.token != token
        if changed:
            with self._generation_lock:
                self._generation += 1
        self.endpoint = endpoint + "/api/logs/batch"
        self.token = token
        if not self._config_available:
            _report("configuration restored; export resumed")
        self._config_available = True
        if changed:
            self._failures = 0
            self._open_until = 0
            _report("logger configuration changed; events queued for the previous credential will be discarded")
            return "changed"
        return "ready"

    def _deliver(self, batch: list[bytes], generation: int) -> None:
        with self._generation_lock:
            current_generation = self._generation
        if generation != current_generation:
            self._drop(len(batch))
            return
        if time.monotonic() < self._open_until:
            self._drop(len(batch))
            return
        result = self._send(batch)
        if result == 400 and len(batch) > 1:
            middle = len(batch) // 2
            self._deliver(batch[:middle], generation)
            self._deliver(batch[middle:], generation)
        elif result != 200:
            self._drop(len(batch))
            if result == 503:
                self._failures += 1
                if self._failures >= 3:
                    self._open_until = time.monotonic() + 30
            else:
                self._failures = 0
        else:
            self._failures = 0
            self._open_until = 0

    def _send(self, batch: list[bytes]) -> int:
        body = b'{"events":[' + b",".join(batch) + b"]}"
        for attempt in range(3):
            token = self.token
            request = urllib.request.Request(
                self.endpoint, body,
                headers={"Authorization": "Bearer " + token, "Content-Type": "application/json", "X-Flixer-Log-Schema": "1"},
                method="POST",
            )
            retry_delay = 0.0
            try:
                with self.client.open(request, timeout=5) as response:
                    payload = response.read(4097)
                    if len(payload) <= 4096:
                        result = json.loads(payload)
                        if isinstance(result, dict) and type(result.get("accepted")) is int and result["accepted"] == len(batch):
                            return 200
            except urllib.error.HTTPError as error:
                code = error.code
                retry_header = error.headers.get("Retry-After")
                error.close()
                if code == 401 and attempt < 2 and self._config_path is not None:
                    self._next_config_check = 0
                    if self._refresh_config() == "ready" and self.token != token:
                        continue
                if code == 400 or (code != 429 and code < 500):
                    return code
                if retry_header:
                    try:
                        retry_delay = float(retry_header)
                    except ValueError:
                        try:
                            retry_delay = email.utils.parsedate_to_datetime(retry_header).timestamp() - time.time()
                        except (TypeError, ValueError, OverflowError):
                            retry_delay = 0
            except (urllib.error.URLError, OSError, TimeoutError, ValueError, http.client.HTTPException):
                pass
            if attempt < 2:
                delay = min(30.0, max(0.25 * (2 ** attempt) + random.random() * 0.25, retry_delay))
                if self.stopping.wait(delay):
                    break
        return 503

    def stop(self) -> None:
        self.stopping.set()
        self.worker.join(timeout=1)
        if self.worker.is_alive():
            _report("export worker is still stopping; queued events may be lost")


class LoggerSupervisor:
    def __init__(self, path: Path):
        self.path = path
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.active: tuple[LoggerHandler, list[logging.Logger]] | None = None
        self.reported_error = False
        self.worker: threading.Thread | None = None
        if not self._start_if_available():
            self.worker = threading.Thread(target=self._wait_for_config, name="flixer-logger-config", daemon=True)
            try:
                self.worker.start()
            except RuntimeError as error:
                self.worker = None
                _report(f"configuration watcher could not start ({type(error).__name__})")

    def _start_if_available(self) -> bool:
        try:
            config = _read_config(self.path)
        except (OSError, ValueError, TypeError) as error:
            if not self.reported_error:
                _report(f"configuration is unavailable ({type(error).__name__}); export disabled")
                self.reported_error = True
            return False
        if config is None:
            return False
        with self.lock:
            if self.stopping.is_set():
                return False
            try:
                handler = LoggerHandler(*config, config_path=self.path)
            except RuntimeError as error:
                if not self.reported_error:
                    _report(f"export could not start ({type(error).__name__})")
                    self.reported_error = True
                return False
            loggers = [logging.getLogger()]
            for name in ("uvicorn", "uvicorn.access"):
                candidate = logging.getLogger(name)
                if not candidate.propagate:
                    loggers.append(candidate)
            for candidate in loggers:
                candidate.addHandler(handler)
            self.active = handler, loggers
            if self.reported_error:
                _report("configuration restored; export started")
            return True

    def _wait_for_config(self) -> None:
        while not self.stopping.wait(_CONFIG_REFRESH_SECONDS):
            if self._start_if_available():
                return

    def stop(self) -> None:
        self.stopping.set()
        if self.worker is not None:
            self.worker.join(timeout=1)
            if self.worker.is_alive():
                _report("configuration watcher is still stopping")
        with self.lock:
            active = self.active
            self.active = None
            if active is not None:
                handler, loggers = active
                for logger in loggers:
                    logger.removeHandler(handler)
                handler.stop()


def start_logger(path: Path = _CONFIG_PATH) -> LoggerSupervisor:
    return LoggerSupervisor(path)


def stop_logger(supervisor: LoggerSupervisor) -> None:
    supervisor.stop()
