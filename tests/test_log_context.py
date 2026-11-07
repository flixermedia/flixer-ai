import asyncio
import logging

from flixer_ai.log_context import LogContextMiddleware, trace_flags, trace_id


def test_request_context_is_isolated_and_reset(caplog):
    seen = []

    async def app(scope, receive, send):
        before = trace_id.get()
        await asyncio.sleep(0)
        seen.append((before, trace_id.get(), trace_flags.get()))
        await send({"type": "http.response.start", "status": 204, "headers": []})

    middleware = LogContextMiddleware(app)

    async def run():
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            pass

        headers = [
            [(b"traceparent", b"00-" + b"a" * 32 + b"-" + b"1" * 16 + b"-01")],
            [(b"traceparent", b"00-" + b"b" * 32 + b"-" + b"2" * 16 + b"-00")],
        ]
        await asyncio.gather(*(middleware({"type": "http", "headers": h}, receive, send) for h in headers))
        assert trace_id.get() is None
        assert trace_flags.get() == "01"

    with caplog.at_level(logging.INFO, logger="flixer_ai.request"):
        asyncio.run(run())
    assert set(seen) == {("a" * 32, "a" * 32, "01"), ("b" * 32, "b" * 32, "00")}
    assert len(caplog.records) == 2
    assert all(record.status_code == 204 and record.duration_ms >= 0 for record in caplog.records)


def test_invalid_or_repeated_traceparent_is_replaced():
    from flixer_ai.log_context import _trace_from_headers

    valid = b"00-" + b"a" * 32 + b"-" + b"1" * 16 + b"-01"
    for headers in (
        [(b"traceparent", b"00-" + b"0" * 32 + b"-" + b"1" * 16 + b"-01")],
        [(b"traceparent", b"00-" + b"a" * 32 + b"-" + b"0" * 16 + b"-01")],
        [(b"traceparent", b"invalid")],
        [(b"traceparent", valid), (b"traceparent", valid)],
    ):
        result, flags = _trace_from_headers({"type": "http", "headers": headers})
        assert len(result) == 32 and result != "a" * 32 and int(result, 16) != 0
        assert flags == "01"


def test_context_reset_after_exception():
    async def app(scope, receive, send):
        raise RuntimeError("request failed")

    async def run():
        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            pass

        try:
            await LogContextMiddleware(app)({"type": "http", "headers": []}, receive, send)
        except RuntimeError:
            pass
        else:
            raise AssertionError("exception was suppressed")
        assert trace_id.get() is None
        assert trace_flags.get() == "01"

    asyncio.run(run())
