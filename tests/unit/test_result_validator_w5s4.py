"""Unit tests for the LLM-based result validator."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import OpenAIConfig, ValidationConfig
from pg_mcp.models.errors import LLMError, LLMTimeoutError, LLMUnavailableError
from pg_mcp.services.result_validator import ResultValidator


def make_validator(**validation_overrides) -> ResultValidator:
    openai_config = OpenAIConfig(api_key="sk-test-key", model="gpt-4o-mini")
    validation_config = ValidationConfig(
        enabled=True, confidence_threshold=70, sample_rows=2, **validation_overrides
    )
    validator = ResultValidator(openai_config, validation_config)
    return validator


def make_llm_response(content: str | None) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    response.choices = [choice]
    return response


def patch_llm(validator: ResultValidator, response: MagicMock) -> None:
    validator.client.chat.completions.create = AsyncMock(return_value=response)


VALID_JSON = json.dumps({"confidence": 85, "explanation": "results match", "suggestion": None})


class TestValidate:
    @pytest.mark.asyncio
    async def test_disabled_config_returns_full_confidence(self) -> None:
        openai_config = OpenAIConfig(api_key="sk-test-key")
        validation_config = ValidationConfig(enabled=False)
        validator = ResultValidator(openai_config, validation_config)
        result = await validator.validate("q", "SELECT 1", [{"x": 1}], 1)
        assert result.confidence == 100
        assert result.is_acceptable

    @pytest.mark.asyncio
    async def test_acceptable_results(self) -> None:
        validator = make_validator()
        patch_llm(validator, make_llm_response(VALID_JSON))
        result = await validator.validate("count", "SELECT 1", [{"count": 42}], 1)
        assert result.confidence == 85
        assert result.is_acceptable
        assert result.explanation == "results match"

    @pytest.mark.asyncio
    async def test_below_threshold_not_acceptable(self) -> None:
        validator = make_validator()
        patch_llm(
            validator,
            make_llm_response(json.dumps({"confidence": 40, "explanation": "off"})),
        )
        result = await validator.validate("q", "SELECT 1", [], 0)
        assert result.confidence == 40
        assert not result.is_acceptable

    @pytest.mark.asyncio
    async def test_samples_rows_to_llm(self) -> None:
        validator = make_validator()
        create_mock = AsyncMock(return_value=make_llm_response(VALID_JSON))
        validator.client.chat.completions.create = create_mock
        rows = [{"i": i} for i in range(50)]
        await validator.validate("q", "SELECT 1", rows, 50)
        prompt = create_mock.call_args.kwargs["messages"][1]["content"]
        assert '"i": 49' not in prompt  # only sample_rows=2 included

    @pytest.mark.asyncio
    async def test_malformed_json_returns_moderate_confidence(self) -> None:
        validator = make_validator()
        patch_llm(validator, make_llm_response("not json at all {"))
        result = await validator.validate("q", "SELECT 1", [], 0)
        assert result.confidence == 60
        assert not result.is_acceptable

    @pytest.mark.asyncio
    async def test_out_of_bounds_confidence_clamped(self) -> None:
        validator = make_validator()
        patch_llm(
            validator,
            make_llm_response(json.dumps({"confidence": 250.0, "explanation": "x"})),
        )
        result = await validator.validate("q", "SELECT 1", [], 0)
        assert result.confidence == 100

    @pytest.mark.asyncio
    async def test_non_numeric_confidence_defaults_to_50(self) -> None:
        validator = make_validator()
        patch_llm(
            validator,
            make_llm_response(json.dumps({"confidence": "high", "explanation": "x"})),
        )
        result = await validator.validate("q", "SELECT 1", [], 0)
        assert result.confidence == 50

    @pytest.mark.asyncio
    async def test_empty_choices_raises_llm_error(self) -> None:
        validator = make_validator()
        empty = MagicMock()
        empty.choices = []
        patch_llm(validator, empty)
        with pytest.raises(LLMError, match="empty response"):
            await validator.validate("q", "SELECT 1", [], 0)

    @pytest.mark.asyncio
    async def test_empty_content_raises_llm_error(self) -> None:
        validator = make_validator()
        patch_llm(validator, make_llm_response(None))
        with pytest.raises(LLMError, match="empty message"):
            await validator.validate("q", "SELECT 1", [], 0)

    @pytest.mark.asyncio
    async def test_timeout_maps_to_llm_timeout_error(self) -> None:
        validator = make_validator()
        validator.client.chat.completions.create = AsyncMock(side_effect=TimeoutError())
        with pytest.raises(LLMTimeoutError):
            await validator.validate("q", "SELECT 1", [], 0)

    @pytest.mark.asyncio
    async def test_authentication_error_maps_to_unavailable(self) -> None:
        validator = make_validator()
        validator.client.chat.completions.create = AsyncMock(
            side_effect=Exception("invalid api_key provided")
        )
        with pytest.raises(LLMUnavailableError, match="authentication"):
            await validator.validate("q", "SELECT 1", [], 0)

    @pytest.mark.asyncio
    async def test_rate_limit_error_maps_to_unavailable(self) -> None:
        validator = make_validator()
        validator.client.chat.completions.create = AsyncMock(
            side_effect=Exception("rate_limit exceeded")
        )
        with pytest.raises(LLMUnavailableError, match="rate limit"):
            await validator.validate("q", "SELECT 1", [], 0)

    @pytest.mark.asyncio
    async def test_generic_error_maps_to_llm_error(self) -> None:
        validator = make_validator()
        validator.client.chat.completions.create = AsyncMock(
            side_effect=Exception("connection reset")
        )
        with pytest.raises(LLMError, match="validation failed"):
            await validator.validate("q", "SELECT 1", [], 0)
