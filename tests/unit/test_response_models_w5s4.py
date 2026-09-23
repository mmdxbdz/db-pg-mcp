"""Regression tests for response/model fixes (W5-S4 codex review #3).

QueryResponse previously defined to_dict twice: the later definition silently
overrode the earlier one and produced divergent response shapes. These tests
pin down the single canonical serializer. The response also now echoes the
request correlation id (request_id) so a query can be traced end to end.
"""

import inspect

from pg_mcp.models.query import ErrorDetail, QueryResponse


class TestSingleToDict:
    """The duplicate to_dict definitions must be gone."""

    def test_query_response_defines_to_dict_exactly_once(self) -> None:
        source = inspect.getsource(QueryResponse)
        assert source.count("def to_dict") == 1

    def test_to_dict_always_includes_tokens_used(self) -> None:
        response = QueryResponse(success=True, generated_sql="SELECT 1")
        payload = response.to_dict()
        assert payload["tokens_used"] == 0

    def test_to_dict_preserves_reported_token_usage(self) -> None:
        response = QueryResponse(success=True, generated_sql="SELECT 1", tokens_used=57)
        assert response.to_dict()["tokens_used"] == 57

    def test_to_dict_serializes_error_details(self) -> None:
        response = QueryResponse(
            success=False,
            error=ErrorDetail(
                code="security_violation",
                message="blocked",
                details={"table": "salaries"},
            ),
        )
        payload = response.to_dict()
        assert payload["error"]["code"] == "security_violation"
        assert payload["error"]["details"] == {"table": "salaries"}


class TestRequestIdPropagation:
    """Responses must carry the correlation id for full-chain tracing."""

    def test_request_id_field_defaults_to_none(self) -> None:
        assert QueryResponse(success=True).request_id is None

    def test_request_id_echoed(self) -> None:
        response = QueryResponse(success=True, request_id="req-abc-123")
        payload = response.to_dict()
        assert payload["request_id"] == "req-abc-123"
