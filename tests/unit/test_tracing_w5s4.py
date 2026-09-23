"""Unit tests for request tracing and context propagation."""

import logging

import pytest

from pg_mcp.observability.tracing import (
    TraceContext,
    clear_request_id,
    generate_request_id,
    get_request_id,
    get_tracing_logger,
    request_context,
    set_request_id,
    trace_async,
    trace_sync,
)


class TestRequestIdLifecycle:
    def test_generate_request_id_is_unique_uuid(self) -> None:
        first, second = generate_request_id(), generate_request_id()
        assert first != second
        assert len(first) == 36

    def test_set_and_clear_request_id(self) -> None:
        set_request_id("custom-id")
        assert get_request_id() == "custom-id"
        clear_request_id()
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_request_context_sets_and_resets(self) -> None:
        assert get_request_id() is None
        async with request_context("ctx-1") as request_id:
            assert request_id == "ctx-1"
            assert get_request_id() == "ctx-1"
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_request_context_generates_id_when_absent(self) -> None:
        async with request_context() as request_id:
            assert request_id is not None
            assert get_request_id() == request_id

    @pytest.mark.asyncio
    async def test_nested_request_context_restores_outer(self) -> None:
        async with request_context("outer"):
            async with request_context("inner"):
                assert get_request_id() == "inner"
            assert get_request_id() == "outer"

    @pytest.mark.asyncio
    async def test_request_context_resets_on_exception(self) -> None:
        with pytest.raises(RuntimeError):
            async with request_context("boom"):
                raise RuntimeError("fail")
        assert get_request_id() is None


class TestTraceDecorators:
    @pytest.mark.asyncio
    async def test_trace_async_injects_request_attributes(self) -> None:
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logger = logging.getLogger("trace-async-test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        @trace_async(operation="generate_sql")
        async def work() -> str:
            logging.getLogger("trace-async-test").info("working")
            return "done"

        async with request_context("req-42"):
            assert await work() == "done"

        assert records
        assert records[0].request_id == "req-42"  # type: ignore[attr-defined]
        assert records[0].operation == "generate_sql"  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_trace_async_without_context_still_runs(self) -> None:
        @trace_async()
        async def work(value: int) -> int:
            return value * 2

        assert await work(3) == 6

    def test_trace_sync_injects_request_attributes(self) -> None:
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logging.getLogger("trace-sync-test").addHandler(handler)
        logging.getLogger("trace-sync-test").setLevel(logging.INFO)

        @trace_sync(operation="validate_sql")
        def work() -> str:
            logging.getLogger("trace-sync-test").info("validating")
            return "ok"

        set_request_id("sync-req")
        try:
            assert work() == "ok"
        finally:
            clear_request_id()

        assert records
        assert records[0].request_id == "sync-req"  # type: ignore[attr-defined]

    def test_trace_sync_without_context_still_runs(self) -> None:
        @trace_sync()
        def work() -> str:
            return "ok"

        assert work() == "ok"

    @pytest.mark.asyncio
    async def test_trace_async_restores_log_factory_on_error(self) -> None:
        original = logging.getLogRecordFactory()

        @trace_async()
        async def failing() -> None:
            raise ValueError("nope")

        with pytest.raises(ValueError):
            async with request_context("err-req"):
                await failing()
        assert logging.getLogRecordFactory() is original


class TestTracingLogger:
    @pytest.mark.asyncio
    async def test_adds_request_id_to_extra(self) -> None:
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logger = get_tracing_logger("tracing-logger-test")
        logger._logger.addHandler(handler)
        logger._logger.setLevel(logging.INFO)

        async with request_context("log-req"):
            logger.info("hello", extra={"database": "db1"})

        assert records
        assert records[0].request_id == "log-req"  # type: ignore[attr-defined]
        assert records[0].database == "db1"  # type: ignore[attr-defined]

    def test_all_levels_log(self) -> None:
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        tracer = get_tracing_logger("tracing-levels-test")
        tracer._logger.addHandler(handler)
        tracer._logger.setLevel(logging.DEBUG)

        tracer.debug("d")
        tracer.info("i")
        tracer.warning("w")
        tracer.error("e")
        tracer.critical("c")
        assert len(records) == 5

    def test_exception_logs_traceback(self) -> None:
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        tracer = get_tracing_logger("tracing-exc-test")
        tracer._logger.addHandler(handler)

        try:
            raise ValueError("boom")
        except ValueError:
            tracer.exception("caught")
        assert records[0].exc_info is not None

    def test_trace_context_model(self) -> None:
        ctx = TraceContext(request_id="r1", operation="op", metadata={"k": "v"})
        assert ctx.parent_id is None
        assert ctx.metadata == {"k": "v"}
