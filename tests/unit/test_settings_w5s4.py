"""Settings tests for the W5-S4 review fixes.

* Multi-database configuration: the server previously could only ever use one
  database; settings must now accept, normalize and validate a database map.
* Access-control policy fields: blocked tables, blocked columns, the EXPLAIN
  policy, the write policy and the rate limits are configurable.
* The unused duplicate field ValidationConfig.min_confidence_score is removed;
  confidence_threshold is the single source of truth.
"""

import json
import os

import pytest
from pydantic import ValidationError

from pg_mcp.config.settings import (
    ResilienceConfig,
    SecurityConfig,
    Settings,
    ValidationConfig,
)


@pytest.fixture(autouse=True)
def clean_db_env(monkeypatch: pytest.MonkeyPatch):
    """Keep DATABASES/DEFAULT_DATABASE env leakage out of these tests."""
    monkeypatch.delenv("DATABASES", raising=False)
    monkeypatch.delenv("DEFAULT_DATABASE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")


def _set_databases_env(payload: dict) -> None:
    os.environ["DATABASES"] = json.dumps(payload)


class TestMultiDatabaseSettings:
    def test_primary_database_becomes_single_entry(self) -> None:
        settings = Settings()
        assert list(settings.databases.keys()) == [settings.database.name]
        assert settings.default_database == settings.database.name

    def test_env_json_defines_multiple_databases(self) -> None:
        _set_databases_env(
            {
                "analytics": {"name": "analytics", "host": "a.example.com"},
                "billing": {"name": "billing", "host": "b.example.com"},
            }
        )
        try:
            settings = Settings()
            # All declared databases are present; the primary database is
            # always merged in as well (it must stay queryable).
            assert {"analytics", "billing"} <= set(settings.databases)
            assert settings.database.name in settings.databases
            assert settings.databases["billing"].host == "b.example.com"
        finally:
            os.environ.pop("DATABASES", None)

    def test_default_database_resolved_from_map(self) -> None:
        _set_databases_env(
            {
                "analytics": {"name": "analytics"},
                "billing": {"name": "billing"},
            }
        )
        os.environ["DEFAULT_DATABASE"] = "billing"
        try:
            assert Settings().default_database == "billing"
        finally:
            os.environ.pop("DATABASES", None)
            os.environ.pop("DEFAULT_DATABASE", None)

    def test_unknown_default_database_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings(default_database="ghost")

    def test_mismatched_key_and_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings(databases={"wrong": {"name": "right"}})

    def test_primary_database_always_queryable(self) -> None:
        settings = Settings(
            databases={"extra": {"name": "extra", "host": "x"}},
            default_database="extra",
        )
        # The primary config is merged in even if not listed explicitly
        assert settings.database.name in settings.databases


class TestSecurityPolicySettings:
    def test_blocked_objects_parse_from_csv_string(self) -> None:
        config = SecurityConfig(
            blocked_tables="Salaries, AuditLog , payments",
            blocked_columns="password_hash,ssn",
        )
        assert config.blocked_tables == ["salaries", "auditlog", "payments"]
        assert config.blocked_columns == ["password_hash", "ssn"]

    def test_explain_policy_default_off(self) -> None:
        assert SecurityConfig().allow_explain is False

    def test_write_policy_default_off(self) -> None:
        assert SecurityConfig().allow_write_operations is False

    def test_rate_limits_configurable(self) -> None:
        config = ResilienceConfig(query_rate_limit=3, llm_rate_limit=2, rate_limit_timeout=1.5)
        assert config.query_rate_limit == 3
        assert config.llm_rate_limit == 2
        assert config.rate_limit_timeout == 1.5


class TestValidationConfigCleanup:
    def test_duplicate_min_confidence_score_removed(self) -> None:
        assert not hasattr(ValidationConfig(), "min_confidence_score")

    def test_confidence_threshold_is_single_source(self) -> None:
        assert ValidationConfig(confidence_threshold=88).confidence_threshold == 88
