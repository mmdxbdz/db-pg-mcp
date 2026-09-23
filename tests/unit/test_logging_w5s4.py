"""Unit tests for structured logging and sensitive-data sanitization."""

import json
import logging

from pg_mcp.observability.logging import (
    JSONFormatter,
    SensitiveDataFilter,
    TextFormatter,
    configure_logging,
    get_logger,
)


def make_record(message: str, **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class TestSensitiveDataFilter:
    def test_always_allows_record(self) -> None:
        log_filter = SensitiveDataFilter()
        assert log_filter.filter(make_record("hello")) is True

    def test_redacts_sensitive_extra_fields(self) -> None:
        log_filter = SensitiveDataFilter()
        record = make_record("connect", password="s3cret", api_key="sk-123")
        log_filter.filter(record)
        assert record.password == "***REDACTED***"  # type: ignore[attr-defined]
        assert record.api_key == "***REDACTED***"  # type: ignore[attr-defined]

    def test_redacts_nested_dicts(self) -> None:
        log_filter = SensitiveDataFilter()
        record = make_record("db", config={"host": "h", "token": "abc"})
        log_filter.filter(record)
        sanitized = record.config  # type: ignore[attr-defined]
        assert sanitized["host"] == "h"
        assert sanitized["token"] == "***REDACTED***"

    def test_redacts_sensitive_values_in_args(self) -> None:
        log_filter = SensitiveDataFilter()
        record = make_record("value=%s")
        record.args = ({"password": "leak"},)
        log_filter.filter(record)
        assert record.args[0] == {"password": "***REDACTED***"}

    def test_case_insensitive_keys(self) -> None:
        log_filter = SensitiveDataFilter()
        record = make_record("x", PASSWORD="p", Auth="a")
        log_filter.filter(record)
        assert record.PASSWORD == "***REDACTED***"  # type: ignore[attr-defined]
        assert record.Auth == "***REDACTED***"  # type: ignore[attr-defined]


class TestJSONFormatter:
    def test_outputs_valid_json_with_core_fields(self) -> None:
        formatter = JSONFormatter()
        payload = json.loads(formatter.format(make_record("hello world")))
        assert payload["message"] == "hello world"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "test.logger"
        assert "timestamp" in payload

    def test_includes_request_id_when_present(self) -> None:
        formatter = JSONFormatter()
        payload = json.loads(formatter.format(make_record("m", request_id="req-9")))
        assert payload["request_id"] == "req-9"

    def test_extra_fields_grouped(self) -> None:
        formatter = JSONFormatter()
        payload = json.loads(formatter.format(make_record("m", database="db1", rows=3)))
        # taskName is a stdlib record attribute on Python 3.12+
        expected = {"database": "db1", "rows": 3, "taskName": None}
        assert payload["extra"] == expected

    def test_exception_included(self) -> None:
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = make_record("failed")
            record.exc_info = sys.exc_info()
        payload = json.loads(formatter.format(record))
        assert "ValueError: boom" in payload["exception"]


class TestTextFormatter:
    def test_readable_output(self) -> None:
        formatter = TextFormatter()
        text = formatter.format(make_record("plain message"))
        assert "[INFO]" in text
        assert "test.logger" in text
        assert "plain message" in text

    def test_request_id_suffix(self) -> None:
        formatter = TextFormatter()
        text = formatter.format(make_record("m", request_id="req-1"))
        assert "[request_id=req-1]" in text


class TestConfigureLogging:
    def teardown_method(self) -> None:
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    def test_json_mode_installs_json_formatter(self) -> None:
        configure_logging(level="INFO", log_format="json")
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)
        assert isinstance(root.handlers[0].filters[0], SensitiveDataFilter)
        assert root.level == logging.INFO

    def test_text_mode_installs_text_formatter(self) -> None:
        configure_logging(level="DEBUG", log_format="text", enable_sensitive_filter=False)
        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, TextFormatter)
        assert root.level == logging.DEBUG

    def test_no_sensitive_filter_when_disabled(self) -> None:
        configure_logging(log_format="text", enable_sensitive_filter=False)
        root = logging.getLogger()
        assert not any(isinstance(item, SensitiveDataFilter) for item in root.handlers[0].filters)

    def test_third_party_loggers_quieted(self) -> None:
        configure_logging()
        assert logging.getLogger("asyncpg").level == logging.WARNING
        assert logging.getLogger("openai").level == logging.WARNING

    def test_get_logger_returns_named_logger(self) -> None:
        assert get_logger("some.module").name == "some.module"
