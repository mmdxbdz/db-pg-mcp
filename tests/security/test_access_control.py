"""Security tests for config-driven access control (W5-S4 codex review #1).

Verifies that the table/column access restrictions and the EXPLAIN policy that
existed only as design promises are now actually enforced through settings, and
that the orchestrator wires the policy into the request pipeline.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import ResilienceConfig, SecurityConfig, ValidationConfig
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.sql_validator import SQLValidator


@pytest.fixture
def protected_config() -> SecurityConfig:
    """Security config protecting sensitive objects, EXPLAIN disabled."""
    return SecurityConfig(
        blocked_tables="salaries, audit_log",
        blocked_columns="password_hash, ssn",
        allow_explain=False,
    )


@pytest.fixture
def validator(protected_config: SecurityConfig) -> SQLValidator:
    """Validator built entirely from the security config (server wiring path)."""
    return SQLValidator.from_config(protected_config)


class TestBlockedTablePolicy:
    """Sensitive tables must be unreachable."""

    def test_direct_access_rejected(self, validator: SQLValidator) -> None:
        valid, error = validator.validate("SELECT * FROM salaries")
        assert not valid
        assert "salaries" in error

    def test_case_insensitive_rejection(self, validator: SQLValidator) -> None:
        valid, _ = validator.validate("select * from SALARIES")
        assert not valid

    def test_access_via_cte_rejected(self, validator: SQLValidator) -> None:
        valid, error = validator.validate(
            "WITH leak AS (SELECT * FROM audit_log) SELECT * FROM leak"
        )
        assert not valid
        assert "audit_log" in error

    def test_access_via_subquery_rejected(self, validator: SQLValidator) -> None:
        valid, _ = validator.validate("SELECT * FROM (SELECT * FROM salaries) sub")
        assert not valid

    def test_joined_blocked_table_rejected(self, validator: SQLValidator) -> None:
        valid, _ = validator.validate(
            "SELECT u.id FROM users u JOIN salaries s ON s.user_id = u.id"
        )
        assert not valid

    def test_normal_tables_still_allowed(self, validator: SQLValidator) -> None:
        assert validator.validate("SELECT id, name FROM users")[0]
        assert validator.validate("SELECT COUNT(*) FROM orders")[0]


class TestBlockedColumnPolicy:
    """Sensitive columns must be unreachable."""

    def test_direct_column_rejected(self, validator: SQLValidator) -> None:
        valid, error = validator.validate("SELECT password_hash FROM users")
        assert not valid
        assert "password_hash" in error

    def test_qualified_column_rejected(self, validator: SQLValidator) -> None:
        valid, _ = validator.validate("SELECT u.ssn FROM users u")
        assert not valid

    def test_column_in_where_rejected(self, validator: SQLValidator) -> None:
        valid, _ = validator.validate("SELECT id FROM users WHERE password_hash = 'x'")
        assert not valid

    def test_star_select_still_allowed(self, validator: SQLValidator) -> None:
        # SELECT * carries no column names in the AST, so it is not blocked
        assert validator.validate("SELECT * FROM users")[0]


class TestExplainPolicy:
    """The EXPLAIN policy is configurable and enforced."""

    def test_explain_blocked_by_default(self, validator: SQLValidator) -> None:
        valid, error = validator.validate("EXPLAIN SELECT 1")
        assert not valid
        assert "EXPLAIN" in error

    def test_explain_allowed_when_configured(self) -> None:
        validator = SQLValidator.from_config(SecurityConfig(allow_explain=True))
        assert validator.validate("EXPLAIN SELECT 1")[0]

    def test_other_commands_never_allowed(self) -> None:
        validator = SQLValidator.from_config(SecurityConfig(allow_explain=True))
        assert not validator.validate("VACUUM")[0]
        assert not validator.validate("COPY users TO '/tmp/x'")[0]


class TestWritePolicy:
    """Write operations stay blocked unless explicitly enabled."""

    def test_writes_blocked_by_default(self) -> None:
        validator = SQLValidator.from_config(SecurityConfig())
        for sql in (
            "INSERT INTO users VALUES (1)",
            "UPDATE users SET name = 'x'",
            "DELETE FROM users",
        ):
            assert not validator.validate(sql)[0], sql

    def test_writes_allowed_only_when_configured(self) -> None:
        validator = SQLValidator.from_config(SecurityConfig(allow_write_operations=True))
        assert validator.validate("UPDATE users SET name = 'x' WHERE id = 1")[0]

    def test_ddl_blocked_even_with_write_enabled(self) -> None:
        validator = SQLValidator.from_config(SecurityConfig(allow_write_operations=True))
        for sql in ("DROP TABLE users", "CREATE TABLE t (id INT)", "TRUNCATE users"):
            assert not validator.validate(sql)[0], sql

    def test_write_cannot_touch_blocked_table(self) -> None:
        validator = SQLValidator.from_config(
            SecurityConfig(allow_write_operations=True, blocked_tables=["salaries"])
        )
        assert not validator.validate("DELETE FROM salaries")[0]


class TestOrchestratorEnforcement:
    """The orchestrator pipeline must apply the policy (not just the validator)."""

    @pytest.fixture
    def orchestrator(self) -> QueryOrchestrator:
        generator = AsyncMock()
        generator.generate_with_usage.return_value = (
            "SELECT * FROM salaries",
            100,
        )
        return QueryOrchestrator(
            sql_generator=generator,
            sql_validator=SQLValidator.from_config(SecurityConfig(blocked_tables=["salaries"])),
            sql_executor=AsyncMock(),
            result_validator=AsyncMock(),
            schema_cache=MagicMock(),
            pools={"db1": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=0),
            validation_config=ValidationConfig(enabled=False),
            executors={"db1": AsyncMock()},
            metrics=MetricsCollector(),
        )

    @pytest.mark.asyncio
    async def test_protected_table_never_executes(self, orchestrator: QueryOrchestrator) -> None:
        from pg_mcp.models.query import QueryRequest, ReturnType

        response = await orchestrator.execute_query(
            QueryRequest(question="show salaries", database="db1", return_type=ReturnType.SQL)
        )
        assert not response.success
        assert response.error is not None
        assert response.error.code == "security_violation"
        assert "salaries" in response.error.message
        # The executor must never have been reached
        orchestrator.executors["db1"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejection_recorded_in_metrics(self, orchestrator: QueryOrchestrator) -> None:
        from prometheus_client import REGISTRY

        from pg_mcp.models.query import QueryRequest, ReturnType

        before = REGISTRY.get_sample_value(
            "pg_mcp_sql_rejected_total", {"reason": "security_violation"}
        )
        await orchestrator.execute_query(
            QueryRequest(question="show salaries", database="db1", return_type=ReturnType.SQL)
        )
        after = REGISTRY.get_sample_value(
            "pg_mcp_sql_rejected_total", {"reason": "security_violation"}
        )
        assert (after or 0) - (before or 0) == 1
