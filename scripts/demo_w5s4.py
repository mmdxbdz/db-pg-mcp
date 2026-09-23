"""Feature demonstration for the W5-S4 codex-review fixes.

Shows, with real code paths (no live PostgreSQL/OpenAI required):
  1. Per-database routing with strict unknown-database rejection
  2. Access control: blocked tables/columns, EXPLAIN policy, write policy
  3. Query rate limiting under concurrent load
  4. LLM retry with exponential backoff
  5. Prometheus observability counters and request-id propagation
  6. Response model integrity (single to_dict, request_id field)

Run: uv run python scripts/demo_w5s4.py
"""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

from prometheus_client import REGISTRY

from pg_mcp.config.settings import (
    ResilienceConfig,
    SecurityConfig,
    ValidationConfig,
)
from pg_mcp.models.query import QueryRequest, QueryResponse, ReturnType
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.sql_validator import SQLValidator

RULE = "=" * 74


def header(title: str) -> None:
    print()
    print(RULE)
    print(f"  {title}")
    print(RULE)


def make_executor(rows: list[dict]) -> AsyncMock:
    executor = AsyncMock()
    executor.execute.return_value = (rows, len(rows))
    return executor


def make_generator(responses: list) -> AsyncMock:
    generator = AsyncMock()
    generator.generate_with_usage.side_effect = responses
    return generator


def build_orchestrator(
    executors,
    generator,
    *,
    security=None,
    resilience=None,
    rate_limiter=None,
    default_database=None,
) -> QueryOrchestrator:
    cache = MagicMock()
    cache.get.return_value = MagicMock(tables=[MagicMock()])
    primary = executors[default_database or next(iter(executors))]
    return QueryOrchestrator(
        sql_generator=generator,
        sql_validator=SQLValidator.from_config(security or SecurityConfig()),
        sql_executor=primary,
        result_validator=AsyncMock(),
        schema_cache=cache,
        pools={name: MagicMock() for name in executors},
        resilience_config=resilience or ResilienceConfig(max_retries=0),
        validation_config=ValidationConfig(enabled=False),
        executors=executors,
        rate_limiter=rate_limiter,
        metrics=MetricsCollector(),
        default_database=default_database,
    )


async def demo_routing() -> None:
    header("1. MULTI-DATABASE ROUTING (strict, no silent fallback)")
    analytics = make_executor([{"top_post": "Designing MCP tools"}])
    billing = make_executor([{"invoice_total": 4200}])
    generator = make_generator([("SELECT 1", 41), ("SELECT 2", 43), ("SELECT 3", 47)])
    orch = build_orchestrator(
        {"analytics": analytics, "billing": billing},
        generator,
        default_database="analytics",
    )

    r1 = await orch.execute_query(
        QueryRequest(
            question="Show the most-read post", database="analytics", return_type=ReturnType.RESULT
        )
    )
    r2 = await orch.execute_query(
        QueryRequest(question="Sum the invoices", database="billing", return_type=ReturnType.RESULT)
    )
    r3 = await orch.execute_query(
        QueryRequest(
            question="Query a database that is not configured",
            database="warehouse",
            return_type=ReturnType.RESULT,
        )
    )

    print(f"  analytics -> {r1.data.rows}   (executor calls: {analytics.execute.await_count})")
    print(f"  billing   -> {r2.data.rows}   (executor calls: {billing.execute.await_count})")
    print(f"  warehouse -> success={r3.success} error.code={r3.error.code!r}")
    assert r1.success and analytics.execute.await_count == 1
    assert r2.success and billing.execute.await_count == 1
    assert not r3.success
    print("  OK: each database served its own request; unknown database rejected")


async def demo_security() -> None:
    header("2. ACCESS CONTROL (blocked tables/columns, EXPLAIN and write policy)")
    security = SecurityConfig(
        blocked_tables="salaries,audit_log",
        blocked_columns="password_hash,ssn",
        allow_explain=False,
        allow_write_operations=False,
    )
    executor = make_executor([{"QUERY PLAN": "Seq Scan on t (cost=0..10)"}])
    orch = build_orchestrator(
        {"postgres": executor},
        make_generator(
            [
                ("SELECT name, password_hash FROM employees", 10),
                ("SELECT * FROM salaries", 11),
                ("EXPLAIN SELECT 1", 12),
                ("DELETE FROM posts WHERE id = 1", 13),
            ]
        ),
        security=security,
    )

    cases = [
        ("blocked column", "List names and hashes"),
        ("blocked table", "Show me the salaries table"),
        ("EXPLAIN denied", "Explain this query plan"),
        ("write denied", "Delete post 1"),
    ]
    for label, question in cases:
        r = await orch.execute_query(
            QueryRequest(question=question, database="postgres", return_type=ReturnType.RESULT)
        )
        print(f"  {label:<15} -> success={r.success} error.code={r.error.code!r}")
        assert not r.success

    rejected = REGISTRY.get_sample_value(
        "pg_mcp_sql_rejected_total", {"reason": "security_violation"}
    )
    print(f"  metric pg_mcp_sql_rejected_total{{reason=security_violation}} = {int(rejected)}")
    assert rejected and rejected >= 4

    # With allow_explain=True the same EXPLAIN query is accepted
    orch2 = build_orchestrator(
        {"postgres": executor},
        make_generator([("EXPLAIN SELECT 1", 15)]),
        security=SecurityConfig(allow_explain=True),
    )
    r = await orch2.execute_query(
        QueryRequest(question="Explain", database="postgres", return_type=ReturnType.RESULT)
    )
    print(f"  EXPLAIN allowed -> success={r.success} plan={r.data.rows}")
    assert r.success
    print("  OK: policy is enforced and configurable")


async def demo_rate_limiting() -> None:
    header("3. QUERY RATE LIMITING (concurrency guard)")
    limiter = MultiRateLimiter(query_limit=2, llm_limit=100)
    orch = build_orchestrator(
        {"postgres": make_executor([{"ok": 1}])},
        make_generator([("SELECT 1", 20)]),
        rate_limiter=limiter,
    )
    # Simulate two in-flight requests holding the only query slots
    await limiter.query_limiter.acquire()
    await limiter.query_limiter.acquire()
    r = await orch.execute_query(
        QueryRequest(question="Am I throttled?", database="postgres", return_type=ReturnType.RESULT)
    )
    print(f"  3rd concurrent request -> success={r.success} error.code={r.error.code!r}")
    assert not r.success
    limiter.query_limiter.release()
    limiter.query_limiter.release()
    r2 = await orch.execute_query(
        QueryRequest(question="Slot freed", database="postgres", return_type=ReturnType.RESULT)
    )
    print(f"  after releasing slots  -> success={r2.success}")
    assert r2.success
    print("  OK: overload is rejected; recovery works once capacity frees up")


async def demo_retry_backoff() -> None:
    header("4. SQL RETRY WITH EXPONENTIAL BACKOFF + FEEDBACK")
    import pg_mcp.services.orchestrator as orch_module

    sleeps: list[float] = []
    real_sleep = orch_module.asyncio.sleep

    async def recording_sleep(delay, *args, **kwargs):
        sleeps.append(delay)
        return None

    orch_module.asyncio.sleep = recording_sleep
    try:
        # First attempt violates the write policy; the retry loop feeds the
        # validation error back to the LLM and the second attempt succeeds.
        orch = build_orchestrator(
            {"postgres": make_executor([{"ok": 1}])},
            make_generator([("DELETE FROM posts", 12), ("SELECT 1", 9)]),
            resilience=ResilienceConfig(max_retries=1, retry_delay=0.1, backoff_factor=2.0),
        )
        r = await orch.execute_query(
            QueryRequest(question="retry me", database="postgres", return_type=ReturnType.RESULT)
        )
    finally:
        orch_module.asyncio.sleep = real_sleep

    print("  attempt 1: policy-violating SQL rejected, fed back to generator")
    print(f"  result: success={r.success} sql={r.generated_sql!r} tokens_used={r.tokens_used}")
    print(f"  backoff schedule observed: {sleeps}")
    assert r.success and r.generated_sql == "SELECT 1" and r.tokens_used == 9
    assert sleeps == [0.1], sleeps
    print("  OK: rejected SQL retried once after 0.1s backoff, then succeeded")


async def demo_observability() -> None:
    header("5. OBSERVABILITY (Prometheus metrics + request correlation)")

    orch = build_orchestrator(
        {"postgres": make_executor([{"n": 7}])},
        make_generator([("SELECT 7", 25)]),
    )
    r = await orch.execute_query(
        QueryRequest(question="count", database="postgres", return_type=ReturnType.RESULT)
    )
    print(f"  response.request_id = {r.request_id}")
    assert r.request_id

    def sample(name, labels=None):
        return REGISTRY.get_sample_value(name, labels or {})

    requests = sample("pg_mcp_query_requests_total", {"status": "success", "database": "postgres"})
    duration = sample("pg_mcp_query_duration_seconds_count")
    llm_calls = sample("pg_mcp_llm_calls_total", {"operation": "generate_sql"})
    tokens = sample("pg_mcp_llm_tokens_used_total", {"operation": "generate_sql"})
    print(f"  pg_mcp_query_requests_total{{status=success,database=postgres}} = {requests}")
    print(f"  pg_mcp_query_duration_seconds_count = {duration}")
    print(f"  pg_mcp_llm_calls_total{{operation=generate_sql}} = {llm_calls}")
    print(f"  pg_mcp_llm_tokens_used_total{{operation=generate_sql}} = {tokens}")
    assert requests and requests >= 1
    assert duration and duration >= 1
    assert llm_calls and llm_calls >= 1
    assert tokens and tokens >= 25
    print("  OK: counters, histograms and token accounting all recorded")


async def demo_response_model() -> None:
    header("6. RESPONSE MODEL INTEGRITY")

    source = inspect.getsource(QueryResponse)
    count = source.count("def to_dict")
    print(f"  QueryResponse defines to_dict {count} time(s)  (was 2 before the fix)")
    assert count == 1
    r = QueryResponse(
        success=True, generated_sql="SELECT 1", tokens_used=33, request_id="demo-req-1"
    )
    payload = r.to_dict()
    print(
        f"  to_dict() -> tokens_used={payload['tokens_used']}, request_id={payload['request_id']}"
    )
    assert payload["tokens_used"] == 33
    assert payload["request_id"] == "demo-req-1"
    print("  OK: single serializer; token usage and request_id preserved")


async def main() -> None:
    print(RULE)
    print("  PostgreSQL MCP Server - W5-S4 review-fix demonstration")
    print("  (real orchestrator pipeline, mocked database/LLM backends)")
    print(RULE)

    await demo_routing()
    await demo_security()
    await demo_rate_limiting()
    await demo_retry_backoff()
    await demo_observability()
    await demo_response_model()

    print()
    print(RULE)
    print("  ALL 6 DEMO SECTIONS PASSED")
    print(RULE)


if __name__ == "__main__":
    asyncio.run(main())
