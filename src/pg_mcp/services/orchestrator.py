"""Query orchestrator coordinating the complete request pipeline.

This module provides the QueryOrchestrator class that wires every component of
the query processing pipeline together: SQL generation, security validation,
per-database execution, and result validation. It integrates the resilience and
observability modules that previously existed only as standalone pieces:

* rate limiting (concurrency slots for queries and LLM calls),
* retry with exponential backoff for validation failures,
* Prometheus metrics for requests, rejections, LLM usage and DB latency,
* request-tracing context propagation (request_id) end to end.
"""

import asyncio
import logging
import time
from typing import Any

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    LLMError,
    PgMcpError,
    RateLimitExceededError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.observability.tracing import generate_request_id, request_context
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    The orchestrator owns the resilience and observability integration for
    every request: it acquires a rate-limit slot before doing any work,
    generates and validates SQL with exponential-backoff retries, routes
    execution to the executor bound to the resolved database (never another
    database's executor), records Prometheus metrics, and propagates a
    request_id through tracing context into every log record and response.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     sql_executor=executor,
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"mydb": pool},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ...     executors={"mydb": executor},
        ...     rate_limiter=limiter,
        ...     metrics=metrics,
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="mydb",
        ... ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        sql_executor: SQLExecutor,
        result_validator: ResultValidator,
        schema_cache: SchemaCache,
        pools: dict[str, Pool],
        resilience_config: ResilienceConfig,
        validation_config: ValidationConfig,
        executors: dict[str, SQLExecutor] | None = None,
        rate_limiter: MultiRateLimiter | None = None,
        metrics: MetricsCollector | None = None,
        default_database: str | None = None,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: SQL validation service.
            sql_executor: Default single SQL execution service. When the
                executors map is empty, every request uses this executor.
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries, circuit
                breaker and rate limits.
            validation_config: Validation configuration including thresholds.
            executors: Per-database executors keyed by database name. When
                provided, each request is routed strictly to its own database's
                executor; unknown databases are rejected instead of falling
                back to another database's executor.
            rate_limiter: Optional concurrency limiter for queries/LLM calls.
            metrics: Optional Prometheus metrics collector.
            default_database: Optional database used when a request does not
                specify one.
        """
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.sql_executor = sql_executor
        self.result_validator = result_validator
        self.schema_cache = schema_cache
        self.pools = pools
        self.resilience_config = resilience_config
        self.validation_config = validation_config
        self.executors = dict(executors) if executors else {}
        self.rate_limiter = rate_limiter
        self.metrics = metrics
        self.default_database = default_database

        # Create circuit breaker for LLM calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=resilience_config.circuit_breaker_threshold,
            recovery_timeout=resilience_config.circuit_breaker_timeout,
        )

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute the complete query flow from question to results.

        Pipeline:
        1. Open a tracing context with a fresh request_id
        2. Acquire a query rate-limit slot (reject with rate_limit_exceeded
           when the bounded wait expires)
        3. Resolve and validate the target database and its executor
        4. Load schema from cache
        5. Generate and validate SQL with retry and exponential backoff
        6. Execute SQL against the resolved database's executor
        7. Validate results (optional) and build the structured response
        8. Record Prometheus metrics for the whole request

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error info.
        """
        request_id = generate_request_id()
        start_ms = self._get_current_time_ms()
        database_name = request.database or self.default_database or "unknown"

        async with request_context(request_id):
            logger.info(
                "Starting query execution",
                extra={"request_id": request_id, "question": request.question[:100]},
            )

            # Step 0: query rate limiting (bounded wait, then reject)
            if self.rate_limiter is not None:
                acquired = await self.rate_limiter.query_limiter.acquire(
                    timeout=self.resilience_config.rate_limit_timeout
                )
                if not acquired:
                    logger.warning(
                        "Query rejected: rate limit exceeded",
                        extra={"request_id": request_id, "database": database_name},
                    )
                    if self.metrics is not None:
                        self.metrics.increment_query_request(
                            status="rate_limited", database=database_name
                        )
                    return QueryResponse(
                        success=False,
                        generated_sql=None,
                        validation=None,
                        data=None,
                        error=ErrorDetail(
                            code=ErrorCode.RATE_LIMIT_EXCEEDED.value,
                            message="Query rate limit exceeded; please retry later",
                            details={"database": database_name},
                        ),
                        confidence=0,
                        tokens_used=None,
                        request_id=request_id,
                    )

            try:
                response = await self._execute_pipeline(request, request_id, start_ms)
            finally:
                if self.rate_limiter is not None:
                    self.rate_limiter.query_limiter.release()

            # Final metrics for the request
            if self.metrics is not None:
                status = (
                    "success"
                    if response.success
                    else (response.error.code if response.error else "failed")
                )
                self.metrics.increment_query_request(status=status, database=database_name)
                self.metrics.observe_query_duration(
                    (self._get_current_time_ms() - start_ms) / 1000.0
                )
            return response

    async def _execute_pipeline(
        self,
        request: QueryRequest,
        request_id: str,
        start_ms: float,
    ) -> QueryResponse:
        """Run steps 1-7 of the pipeline (rate-limit slot already held)."""
        try:
            # Step 1: Resolve database name and executor
            database_name = self._resolve_database(request.database)
            executor = self._resolve_executor(database_name)
            logger.debug(
                "Resolved database",
                extra={"request_id": request_id, "database": database_name},
            )

            # Step 2: Get schema from cache
            schema = self.schema_cache.get(database_name)
            if schema is None:
                # Schema not in cache, load it
                pool = self.pools.get(database_name)
                if pool is None:
                    raise DatabaseError(
                        message=f"No connection pool available for database '{database_name}'",
                        details={"database": database_name},
                    )
                try:
                    schema = await self.schema_cache.load(database_name, pool)
                except Exception as e:
                    raise SchemaLoadError(
                        message=f"Failed to load schema for database '{database_name}': {e}",
                        details={"database": database_name, "error": str(e)},
                    ) from e

            if self.metrics is not None:
                cache_age = None
                if hasattr(self.schema_cache, "get_cache_age"):
                    cache_age = self.schema_cache.get_cache_age(database_name)
                if cache_age is not None:
                    self.metrics.set_schema_cache_age(database_name, cache_age)

            logger.debug(
                "Schema loaded",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "tables": len(schema.tables),
                },
            )

            # Step 3: Generate and validate SQL with retry + backoff
            generated_sql, validation_result, tokens_used = await self._generate_sql_with_retry(
                question=request.question,
                schema=schema,
                request_id=request_id,
            )

            # Step 4: If return_type is SQL, return early
            if request.return_type == ReturnType.SQL:
                logger.info(
                    "Returning SQL only",
                    extra={"request_id": request_id, "sql_length": len(generated_sql)},
                )
                return QueryResponse(
                    success=True,
                    generated_sql=generated_sql,
                    validation=validation_result,
                    data=None,
                    error=None,
                    confidence=100,
                    tokens_used=tokens_used,
                    request_id=request_id,
                )

            # Step 5: Execute SQL on the resolved database's own executor
            logger.debug("Executing SQL", extra={"request_id": request_id})
            db_start_ms = self._get_current_time_ms()

            results, total_count = await executor.execute(generated_sql)

            if self.metrics is not None:
                self.metrics.observe_db_query_duration(
                    (self._get_current_time_ms() - db_start_ms) / 1000.0
                )

            execution_time_ms = self._get_current_time_ms() - start_ms
            logger.info(
                "SQL executed successfully",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "row_count": total_count,
                    "execution_time_ms": execution_time_ms,
                },
            )

            # Step 6: Validate results (non-blocking, failures do not fail the request)
            result_confidence = await self._validate_results_safely(
                question=request.question,
                sql=generated_sql,
                results=results,
                row_count=total_count,
                request_id=request_id,
            )

            # Step 7: Build successful response
            query_result = QueryResult(
                columns=list(results[0].keys()) if results else [],
                rows=results,
                row_count=len(results),  # Limited row count (after max_rows applied)
                execution_time_ms=execution_time_ms,
            )

            return QueryResponse(
                success=True,
                generated_sql=generated_sql,
                validation=validation_result,
                data=query_result,
                error=None,
                confidence=result_confidence,
                tokens_used=tokens_used,
                request_id=request_id,
            )

        except PgMcpError as e:
            # Handle known application errors
            logger.warning(
                "Query execution failed with known error",
                extra={
                    "request_id": request_id,
                    "error_code": e.code,
                    "error_message": e.message,
                },
            )
            if self.metrics is not None and isinstance(e, (SecurityViolationError, SQLParseError)):
                reason = (
                    "security_violation"
                    if isinstance(e, SecurityViolationError)
                    else "sql_parse_error"
                )
                self.metrics.increment_sql_rejected(reason=reason)
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=e.code.value,
                    message=e.message,
                    details=e.details,
                ),
                confidence=0,
                tokens_used=None,
                request_id=request_id,
            )
        except Exception as e:
            # Handle unexpected errors
            logger.exception(
                "Query execution failed with unexpected error",
                extra={"request_id": request_id},
            )
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR.value,
                    message=f"Internal server error: {e}",
                    details={"error_type": type(e).__name__},
                ),
                confidence=0,
                tokens_used=None,
                request_id=request_id,
            )

    def _resolve_executor(self, database_name: str) -> SQLExecutor:
        """Resolve the executor bound to the requested database.

        With a per-database executor map, a request can never run against a
        different database's executor: an unknown database is rejected
        outright instead of silently falling back.

        Args:
            database_name: Resolved database name.

        Returns:
            SQLExecutor: The executor bound to the database.

        Raises:
            DatabaseError: If no executor is configured for the database.
        """
        if self.executors:
            executor = self.executors.get(database_name)
            if executor is None:
                raise DatabaseError(
                    message=f"No SQL executor configured for database '{database_name}'",
                    details={
                        "database": database_name,
                        "available_databases": sorted(self.executors),
                    },
                )
            return executor
        # Legacy single-executor mode: every request uses the shared executor.
        if self.sql_executor is not None:
            return self.sql_executor
        raise DatabaseError(
            message="No SQL executor configured",
            details={"database": database_name},
        )

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request, default, or auto-select.

        Resolution order:
        1. Explicit database from the request (validated against pools)
        2. Configured default_database
        3. The only available database (when exactly one exists)

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If the database is invalid or cannot be selected.
        """
        if database is not None:
            # Validate specified database exists
            if database not in self.pools:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": list(self.pools.keys()),
                    },
                )
            return database

        # Configured default database
        if self.default_database is not None:
            if self.default_database not in self.pools:
                raise DatabaseError(
                    message=(
                        f"Configured default database '{self.default_database}' is not available"
                    ),
                    details={
                        "default_database": self.default_database,
                        "available_databases": list(self.pools.keys()),
                    },
                )
            return self.default_database

        # Auto-select if only one database available
        available_dbs = list(self.pools.keys())
        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        # Multiple databases, must specify
        raise DatabaseError(
            message="Multiple databases available, please specify which database to query",
            details={"available_databases": available_dbs},
        )

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: Any,
        request_id: str,
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry, backoff and resilience hooks.

        Each attempt:
        1. Checks circuit breaker state
        2. Acquires an LLM rate-limit slot
        3. Generates SQL via the LLM (tokens captured from usage metadata)
        4. Validates the generated SQL
        5. On validation failure, waits retry_delay * backoff_factor**attempt
           seconds and retries with error feedback

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            request_id: Request ID for tracking.

        Returns:
            tuple: (generated_sql, validation_result, tokens_used)

        Raises:
            LLMError: If circuit breaker is open or generation fails.
            RateLimitExceededError: If the LLM rate-limit wait times out.
            SecurityViolationError: If SQL fails validation after all retries.
            SQLParseError: If SQL cannot be parsed.
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message=(
                    "SQL generation service is temporarily unavailable (circuit breaker open)"
                ),
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        tokens_used: int | None = None

        for attempt in range(max_retries + 1):
            # LLM rate limiting per attempt (bounded wait)
            if self.rate_limiter is not None:
                llm_acquired = await self.rate_limiter.llm_limiter.acquire(
                    timeout=self.resilience_config.rate_limit_timeout
                )
                if not llm_acquired:
                    raise RateLimitExceededError(
                        message="LLM rate limit exceeded; please retry later",
                        details={"request_id": request_id, "attempt": attempt + 1},
                    )
            try:
                logger.debug(
                    "Generating SQL",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                    },
                )

                if self.metrics is not None:
                    self.metrics.increment_llm_call(operation="generate_sql")
                llm_start_ms = self._get_current_time_ms()

                # Generate SQL (tokens captured from LLM usage metadata)
                generated_sql, tokens_used = await self.sql_generator.generate_with_usage(
                    question=question,
                    schema=schema,
                    previous_attempt=previous_sql,
                    error_feedback=error_feedback,
                )

                if self.metrics is not None:
                    self.metrics.observe_llm_latency(
                        operation="generate_sql",
                        duration=(self._get_current_time_ms() - llm_start_ms) / 1000.0,
                    )
                    if tokens_used:
                        self.metrics.increment_llm_tokens(
                            operation="generate_sql", tokens=tokens_used
                        )

                logger.debug(
                    "SQL generated",
                    extra={
                        "request_id": request_id,
                        "sql_length": len(generated_sql),
                    },
                )

                # Validate SQL
                try:
                    self.sql_validator.validate_or_raise(generated_sql)
                except (SecurityViolationError, SQLParseError) as validation_error:
                    if attempt < max_retries:
                        # Exponential backoff, then retry with feedback
                        delay = self.resilience_config.retry_delay * (
                            self.resilience_config.backoff_factor**attempt
                        )
                        logger.warning(
                            "SQL validation failed, retrying with feedback",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "backoff_seconds": delay,
                                "error": str(validation_error),
                            },
                        )
                        if delay > 0:
                            await asyncio.sleep(delay)
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)
                        continue
                    # Out of retries, record failure and raise
                    self.circuit_breaker.record_failure()
                    logger.error(
                        "SQL validation failed after all retries",
                        extra={
                            "request_id": request_id,
                            "attempts": attempt + 1,
                            "error": str(validation_error),
                        },
                    )
                    raise

                # Validation successful
                self.circuit_breaker.record_success()
                logger.info(
                    "SQL generated and validated successfully",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                    },
                )

                # Build validation result
                validation_result = ValidationResult(
                    is_valid=True,
                    is_select=True,
                    allows_data_modification=False,
                    uses_blocked_functions=[],
                    error_message=None,
                )

                return generated_sql, validation_result, tokens_used

            except (LLMError, SecurityViolationError, SQLParseError):
                # Re-raise known errors
                raise
            except Exception as e:
                # Unexpected error during generation
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e}",
                    details={"error_type": type(e).__name__},
                ) from e
            finally:
                if self.rate_limiter is not None:
                    self.rate_limiter.llm_limiter.release()

        # Should not reach here, but just in case
        self.circuit_breaker.record_failure()
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> int:
        """Validate query results with error handling (non-blocking).

        Attempts to validate results using the LLM, but failures never fail
        the overall query. Returns a confidence score instead.

        Args:
            question: User's original natural language question.
            sql: Generated SQL query.
            results: Query results.
            row_count: Total row count.
            request_id: Request ID for tracking.

        Returns:
            int: Confidence score (0-100). Returns 100 if validation
                is disabled or fails.
        """
        if not self.validation_config.enabled:
            return 100

        try:
            logger.debug(
                "Validating results",
                extra={"request_id": request_id},
            )

            if self.metrics is not None:
                self.metrics.increment_llm_call(operation="validate_result")
            llm_start_ms = self._get_current_time_ms()

            validation_result = await self.result_validator.validate(
                question=question,
                sql=sql,
                results=results,
                row_count=row_count,
            )

            if self.metrics is not None:
                self.metrics.observe_llm_latency(
                    operation="validate_result",
                    duration=(self._get_current_time_ms() - llm_start_ms) / 1000.0,
                )

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                },
            )

            return validation_result.confidence

        except Exception as e:
            # Log but do not fail the query
            logger.warning(
                "Result validation failed, continuing with default confidence",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
            )
            return 100  # Default to high confidence when validation fails

    @staticmethod
    def _get_current_time_ms() -> float:
        """Get current time in milliseconds.

        Returns:
            float: Current time in milliseconds since the epoch.
        """
        return time.time() * 1000
