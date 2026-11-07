"""Request-scoped, bounded correlation for Flixer AI log events."""

from __future__ import annotations

import contextvars
import logging
import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("flixer_ai_trace_id", default=None)
trace_flags: contextvars.ContextVar[str] = contextvars.ContextVar("flixer_ai_trace_flags", default="01")
_TRACEPARENT = re.compile(rb"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})")
_logger = logging.getLogger("flixer_ai.request")


def _trace_from_headers(scope: Scope) -> tuple[str, str]:
    headers = [value for name, value in scope.get("headers", []) if name.lower() == b"traceparent"]
    if len(headers) == 1:
        match = _TRACEPARENT.fullmatch(headers[0])
        if match and match.group(1) != b"0" * 32 and match.group(2) != b"0" * 16:
            return match.group(1).decode("ascii"), match.group(3).decode("ascii")
    return uuid.uuid4().hex, "01"


class LogContextMiddleware:
    """Bind trace IDs to HTTP request logs without recording request data."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming_id, incoming_flags = _trace_from_headers(scope)
        token = trace_id.set(incoming_id)
        flags_token = trace_flags.set(incoming_flags)
        started = time.monotonic()
        status = 500

        async def send_with_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            try:
                _logger.info(
                    "AI HTTP request completed.",
                    extra={"duration_ms": round((time.monotonic() - started) * 1000, 3), "status_code": status},
                )
            finally:
                trace_flags.reset(flags_token)
                trace_id.reset(token)
