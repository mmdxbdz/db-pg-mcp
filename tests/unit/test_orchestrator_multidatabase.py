"""Multi-database routing and resilience/observability integration tests.

Covers the W5-S4 codex review findings:
* review #1: multi-database support must be actually usable - each request is
  routed to the executor bound to its own database, unknown databases are
  rejected, and a default database can be configured;
* review #2: rate limiting, retry/backoff and metrics must be integrated into
  the real request pipeline, not merely exist as standalone modules.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import ResilienceConfig, SecurityConfig, ValidationConfig
from pg_mcp.models.query import QueryRequest, ReturnType
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.observability.tracing import get_request_id
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.sql_validator import SQLValidator


def build_orchestrator(
    executors: dict,
    *,
    default_database: str | None = None,
    resilience: ResilienceConfig | None = None,
    rate_limiter: MultiRateLimiter | None = None,
    metrics: MetricsCollector | None = None,
    schema_cache: MagicMock | None = None,
    generator: AsyncMock | None = None,
) -> QueryOrchestrator:
    """Build an orchestrator over the given fake executors and generators."""
    primary = next(iter(executors.values()))
    cache = schema_cache or MagicMock()
    cache.get.return_value = MagicMock(tables=[MagicMock()])
    return QueryOrchestrator(
        sql_generator=generator or AsyncMock(),
        sql_validator=SQLValidator.from_config(SecurityConfig()),
        sql_executor=primary,
        result_validator=AsyncMock(),
        schema_cache=cache,
        pools={name: MagicMock() for name in executors},
        resilience_config=resilience or ResilienceConfig(max_retries=0),
        validation_config=ValidationConfig(enabled=False),
        executors=executors,
        rate_limiter=rate_limiter,
        metrics=metrics,
        default_database=default_database,
    )


def make_executor(result_rows: list[dict]) -> AsyncMock:
    """Create an executor mock returning the given rows."""
    executor = AsyncMock()
    executor.execute.return_value = (result_rows, len(result_rows))
    return executor


def make_generator(responses: list) -> AsyncMock:
    """Create a generator mock with canned (sql, tokens) responses."""
    generator = AsyncMock()
    generator.generate_with_usage.side_effect = responses
    return generator


class TestPerDatabaseRouting:
    """Requests must execute against their own database's executor."""

    @pytest.mark.asyncio
    async def test_each_database_uses_its_own_executor(self) -> None:
        executor_a = make_executor([{"database": "analytics"}])
        executor_b = make_executor([{"database": "billing"}])
        generator = make_generator(
            [
                ("SELECT 'analytics' AS database", 10),
                ("SELECT 'billing' AS database", 20),
            ]
        )
        orchestrator = build_orchestrator(
            {"analytics": executor_a, "billing": executor_b},
            generator=generator,
        )

        response_a = await orchestrator.execute_query(
            QueryRequest(question="q", database="analytics", return_type=ReturnType.RESULT)
        )
        response_b = await orchestrator.execute_query(
            QueryRequest(question="q", database="billing", return_type=ReturnType.RESULT)
        )

        assert response_a.success and response_b.success
        assert response_a.data.rows[0]["database"] == "analytics"
        assert response_b.data.rows[0]["database"] == "billing"
        executor_a.execute.assert_called_once()
        executor_b.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_database_rejected_without_fallback(self) -> None:
        executor_a = make_executor([{"database": "analytics"}])
        generator = make_generator([("SELECT 1", 1)])
        orchestrator = build_orchestrator({"analytics": executor_a}, generator=generator)

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="billing", return_type=ReturnType.SQL)
        )

        assert not response.success
        assert response.error.code == "database_error"
        assert "billing" in response.error.message
        # No silent fallback: nothing generated, nothing executed
        generator.generate_with_usage.assert_not_called()
        executor_a.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_database_used_when_unspecified(self) -> None:
        executor_a = make_executor([{"database": "analytics"}])
        executor_b = make_executor([{"database": "billing"}])
        generator = make_generator([("SELECT 1", 5)])
        orchestrator = build_orchestrator(
            {"analytics": executor_a, "billing": executor_b},
            generator=generator,
            default_database="analytics",
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database=None, return_type=ReturnType.RESULT)
        )
        assert response.success
        assert response.data.rows[0]["database"] == "analytics"
        executor_a.execute.assert_called_once()
        executor_b.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_databases_require_explicit_choice(self) -> None:
        orchestrator = build_orchestrator(
            {
                "analytics": make_executor([{"database": "analytics"}]),
                "billing": make_executor([{"database": "billing"}]),
            }
        )
        response = await orchestrator.execute_query(
            QueryRequest(question="q", database=None, return_type=ReturnType.SQL)
        )
        assert not response.success
        assert response.error.code == "database_error"
        assert "analytics" in response.error.details["available_databases"]
        assert "billing" in response.error.details["available_databases"]


class TestRateLimitingIntegration:
    """Rate limiting must gate the real request pipeline."""

    def _build_with_limiter(self, resilience: ResilienceConfig) -> QueryOrchestrator:
        executor = make_executor([{"answer": 1}])
        generator = make_generator([("SELECT 1", 7)])
        return build_orchestrator(
            {"db1": executor},
            generator=generator,
            resilience=resilience,
            rate_limiter=MultiRateLimiter(query_limit=1, llm_limit=1),
        )

    @pytest.mark.asyncio
    async def test_query_rejected_when_rate_limit_exhausted(self) -> None:
        orchestrator = self._build_with_limiter(
            ResilienceConfig(max_retries=0, rate_limit_timeout=0.0)
        )
        limiter = orchestrator.rate_limiter
        # Exhaust the single query slot before issuing the request
        assert await limiter.query_limiter.acquire()

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db1", return_type=ReturnType.SQL)
        )
        assert not response.success
        assert response.error.code == "rate_limit_exceeded"
        # Rejected at the gate: no generation, no execution
        orchestrator.sql_generator.generate_with_usage.assert_not_called()
        orchestrator.executors["db1"].execute.assert_not_called()
        limiter.query_limiter.release()

    @pytest.mark.asyncio
    async def test_query_allowed_within_rate_limit(self) -> None:
        orchestrator = self._build_with_limiter(
            ResilienceConfig(max_retries=0, rate_limit_timeout=5.0)
        )
        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db1", return_type=ReturnType.SQL)
        )
        assert response.success

    @pytest.mark.asyncio
    async def test_llm_rate_limit_surfaces_as_rate_limit_error(self) -> None:
        orchestrator = self._build_with_limiter(
            ResilienceConfig(max_retries=0, rate_limit_timeout=0.0)
        )
        # Exhaust only the LLM slot; the query slot stays free
        assert await orchestrator.rate_limiter.llm_limiter.acquire()
        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db1", return_type=ReturnType.SQL)
        )
        assert not response.success
        assert response.error.code == "rate_limit_exceeded"
        orchestrator.rate_limiter.llm_limiter.release()


class TestRetryBackoffIntegration:
    """Retries must apply the configured exponential backoff delays."""

    @pytest.mark.asyncio
    async def test_backoff_delays_follow_exponential_schedule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        delays: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            delays.append(seconds)

        monkeypatch.setattr("pg_mcp.services.orchestrator.asyncio.sleep", fake_sleep)

        executor = make_executor([{"answer": 1}])
        # First two attempts violate policy, the third attempt is clean
        generator = make_generator(
            [
                ("DELETE FROM users", 1),
                ("DELETE FROM users", 1),
                ("SELECT 1", 1),
            ]
        )
        orchestrator = build_orchestrator(
            {"db1": executor},
            generator=generator,
            resilience=ResilienceConfig(max_retries=2, retry_delay=0.5, backoff_factor=2.0),
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db1", return_type=ReturnType.SQL)
        )

        assert response.success
        assert generator.generate_with_usage.call_count == 3
        # Exponential schedule: delay * factor ** attempt -> 0.5, 1.0
        assert delays == pytest.approx([0.5, 1.0])
        # Error feedback was passed to the retried generation calls
        second_call = generator.generate_with_usage.call_args_list[1]
        assert second_call.kwargs["previous_attempt"] == "DELETE FROM users"
        assert "not allowed" in second_call.kwargs["error_feedback"]

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises_security_error(self) -> None:
        executor = make_executor([{"answer": 1}])
        generator = AsyncMock()
        # Every attempt produces policy-violating SQL
        generator.generate_with_usage.return_value = ("DELETE FROM users", 1)
        orchestrator = build_orchestrator(
            {"db1": executor},
            generator=generator,
            resilience=ResilienceConfig(max_retries=1, retry_delay=0.1),
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db1", return_type=ReturnType.SQL)
        )
        assert not response.success
        assert response.error.code == "security_violation"
        assert generator.generate_with_usage.call_count == 2  # initial + 1 retry
        executor.execute.assert_not_called()


class TestObservabilityIntegration:
    """Metrics and tracing must be recorded by the real pipeline."""

    @pytest.mark.asyncio
    async def test_successful_request_recorded(self) -> None:
        from prometheus_client import REGISTRY

        metrics = MetricsCollector()
        executor = make_executor([{"answer": 1}])
        generator = make_generator([("SELECT 1", 123)])
        # Two requests below: a side_effect list would be exhausted and would
        # also shadow return_value, so clear it and use an infinite return.
        generator.generate_with_usage.side_effect = None
        generator.generate_with_usage.return_value = ("SELECT 1", 123)
        schema_cache = MagicMock()
        schema_cache.get.return_value = MagicMock(tables=[MagicMock()])
        schema_cache.get_cache_age.return_value = 12.5

        orchestrator = build_orchestrator(
            {"routedb": executor},
            generator=generator,
            metrics=metrics,
            schema_cache=schema_cache,
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="routedb", return_type=ReturnType.RESULT)
        )
        assert response.success

        before_requests = (
            REGISTRY.get_sample_value(
                "pg_mcp_query_requests_total",
                {"status": "success", "database": "routedb"},
            )
            or 0.0
        )
        before_tokens = (
            REGISTRY.get_sample_value("pg_mcp_llm_tokens_used_total", {"operation": "generate_sql"})
            or 0.0
        )
        before_llm_calls = (
            REGISTRY.get_sample_value("pg_mcp_llm_calls_total", {"operation": "generate_sql"})
            or 0.0
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="routedb", return_type=ReturnType.RESULT)
        )
        assert response.success

        # Global singleton counters accumulate across tests: assert deltas.
        requests = REGISTRY.get_sample_value(
            "pg_mcp_query_requests_total",
            {"status": "success", "database": "routedb"},
        )
        assert requests == before_requests + 1
        assert REGISTRY.get_sample_value("pg_mcp_query_duration_seconds_count") >= 1
        assert (
            REGISTRY.get_sample_value("pg_mcp_llm_calls_total", {"operation": "generate_sql"})
            == before_llm_calls + 1
        )
        # prometheus_client appends _total to Counter sample names
        tokens = REGISTRY.get_sample_value(
            "pg_mcp_llm_tokens_used_total", {"operation": "generate_sql"}
        )
        assert tokens == before_tokens + 123
        assert (
            REGISTRY.get_sample_value(
                "pg_mcp_schema_cache_age_seconds",
                {"database": "routedb"},
            )
            == 12.5
        )

    @pytest.mark.asyncio
    async def test_request_id_propagated_to_response_and_tracing(self) -> None:
        executor = make_executor([{"answer": 1}])
        generator = make_generator([("SELECT 1", 1)])
        orchestrator = build_orchestrator({"db1": executor}, generator=generator)

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db1", return_type=ReturnType.SQL)
        )
        assert response.success
        assert response.request_id is not None
        # The tracing context is closed when execute_query returns
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_tokens_used_in_response(self) -> None:
        executor = make_executor([{"answer": 1}])
        generator = make_generator([("SELECT 1", 321)])
        orchestrator = build_orchestrator({"db1": executor}, generator=generator)

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db1", return_type=ReturnType.RESULT)
        )
        assert response.tokens_used == 321
