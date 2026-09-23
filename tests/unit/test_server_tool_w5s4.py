"""Unit tests for the MCP query tool and server lifespan wiring.

The tool layer previously had no tests at all: these cover request shaping,
error mapping, and the full lifespan assembly with every external system
(PostgreSQL, OpenAI) mocked out.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import pg_mcp.server as server_module
from pg_mcp.models.query import QueryResponse


@pytest.fixture(autouse=True)
def reset_server_globals(monkeypatch: pytest.MonkeyPatch):
    """Give each test a pristine module-global state."""
    monkeypatch.setattr(server_module, "_orchestrator", None)
    yield


class TestQueryTool:
    @pytest.mark.asyncio
    async def test_not_initialized_returns_error(self) -> None:
        result = await server_module.query(question="hi", return_type="sql")
        assert result["success"] is False
        assert result["error"]["code"] == "SERVER_NOT_INITIALIZED"

    @pytest.mark.asyncio
    async def test_invalid_return_type_rejected(self) -> None:
        server_module._orchestrator = AsyncMock()  # must be initialized first
        result = await server_module.query(question="hi", return_type="rows")
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_PARAMETER"

    @pytest.mark.asyncio
    async def test_invalid_request_params_rejected(self) -> None:
        server_module._orchestrator = AsyncMock()
        result = await server_module.query(question="", return_type="sql")
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_REQUEST"

    @pytest.mark.asyncio
    async def test_successful_query_passes_response_through(self) -> None:
        orchestrator = AsyncMock()
        orchestrator.execute_query.return_value = QueryResponse(
            success=True,
            generated_sql="SELECT 1",
            confidence=100,
            tokens_used=12,
            request_id="req-x",
        )
        server_module._orchestrator = orchestrator

        result = await server_module.query(question="how many?", database="db1", return_type="sql")

        assert result["success"] is True
        assert result["generated_sql"] == "SELECT 1"
        assert result["tokens_used"] == 12
        assert result["request_id"] == "req-x"
        request = orchestrator.execute_query.call_args.args[0]
        assert request.question == "how many?"
        assert request.database == "db1"

    @pytest.mark.asyncio
    async def test_orchestrator_exception_maps_to_internal_error(self) -> None:
        orchestrator = AsyncMock()
        orchestrator.execute_query.side_effect = RuntimeError("kaboom")
        server_module._orchestrator = orchestrator

        result = await server_module.query(question="q", return_type="result")
        assert result["success"] is False
        assert result["error"]["code"] == "INTERNAL_ERROR"
        assert result["tokens_used"] == 0


class TestLifespanWiring:
    """Lifespan must assemble per-database executors, limiter, metrics."""

    @pytest.fixture
    def lifespan_patches(self, monkeypatch: pytest.MonkeyPatch):
        """Patch every external system the lifespan touches."""
        fake_pool = MagicMock()
        fake_pool.close = AsyncMock()

        async def fake_create_pool(config):
            fake_pool.name = config.name
            return fake_pool

        fake_schema = MagicMock()
        fake_schema.tables = []
        cache_instance = MagicMock()
        cache_instance.load = AsyncMock(return_value=fake_schema)
        cache_instance.stop_auto_refresh = AsyncMock()

        monkeypatch.setattr("pg_mcp.server.create_pool", fake_create_pool)
        monkeypatch.setattr("pg_mcp.server.SchemaCache", MagicMock(return_value=cache_instance))
        monkeypatch.setattr("pg_mcp.server.SQLGenerator", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr("pg_mcp.server.ResultValidator", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(
            "pg_mcp.server.MultiRateLimiter",
            MagicMock(return_value=MagicMock()),
        )
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OBSERVABILITY_METRICS_ENABLED", "false")
        return cache_instance, fake_pool

    @pytest.mark.asyncio
    async def test_lifespan_assembles_all_components(
        self, lifespan_patches, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_instance, fake_pool = lifespan_patches
        async with server_module.lifespan(server_module.mcp):
            assert server_module._orchestrator is not None
            assert server_module._pools is not None
            assert server_module._rate_limiter is not None
            assert server_module._metrics is not None
            # The executors map is bound per database
            assert server_module._orchestrator.executors
            assert server_module._orchestrator.rate_limiter is server_module._rate_limiter
            assert server_module._orchestrator.metrics is server_module._metrics
            assert server_module._orchestrator.default_database is not None

        # Shutdown closed pools and stopped the cache
        fake_pool.close.assert_awaited()
        cache_instance.stop_auto_refresh.assert_awaited()

    @pytest.mark.asyncio
    async def test_lifespan_closes_pools_on_startup_failure(
        self, lifespan_patches, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_instance, fake_pool = lifespan_patches
        # Schema load explodes during startup
        cache_instance.load = AsyncMock(side_effect=RuntimeError("db down"))
        with pytest.raises(RuntimeError, match="db down"):
            async with server_module.lifespan(server_module.mcp):
                pass
        # Shutdown path still ran and released the pools
        fake_pool.close.assert_awaited()


class TestMainEntryPoint:
    def test_main_runs_stdio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {}

        def fake_run(func, *args, **kwargs):
            calls["func"] = func
            return None

        import anyio

        monkeypatch.setattr(anyio, "run", fake_run)
        from pg_mcp.__main__ import main

        main()
        # anyio.run receives the bound run_stdio_async method of the mcp app
        assert calls["func"].__self__ is server_module.mcp
