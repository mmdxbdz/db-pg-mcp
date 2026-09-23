"""Unit tests for connection pool management (db/pool.py)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pg_mcp.config.settings import DatabaseConfig
from pg_mcp.db.pool import close_pools, create_pool, create_pools


def make_config(name: str = "postgres", **overrides) -> DatabaseConfig:
    return DatabaseConfig(name=name, **overrides)


class TestCreatePool:
    @pytest.mark.asyncio
    async def test_passes_full_configuration(self) -> None:
        config = make_config(
            name="analytics",
            host="db.example.com",
            port=5433,
            user="alice",
            password="secret",
            min_pool_size=2,
            max_pool_size=7,
            pool_timeout=12.5,
            command_timeout=25.0,
        )
        fake_pool = MagicMock()
        with patch(
            "pg_mcp.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=fake_pool)
        ) as factory:
            pool = await create_pool(config)

        assert pool is fake_pool
        kwargs = factory.call_args.kwargs
        assert kwargs["host"] == "db.example.com"
        assert kwargs["port"] == 5433
        assert kwargs["database"] == "analytics"
        assert kwargs["user"] == "alice"
        assert kwargs["password"] == "secret"
        assert kwargs["min_size"] == 2
        assert kwargs["max_size"] == 7
        assert kwargs["timeout"] == 12.5
        assert kwargs["command_timeout"] == 25.0

    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_pool_is_none(self) -> None:
        with (
            patch("pg_mcp.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=None)),
            pytest.raises(RuntimeError, match="analytics"),
        ):
            await create_pool(make_config(name="analytics"))


class TestCreatePools:
    @pytest.mark.asyncio
    async def test_creates_one_pool_per_database(self) -> None:
        configs = [make_config("db1"), make_config("db2")]
        pools_by_name = {"db1": MagicMock(), "db2": MagicMock()}

        async def fake_create(config):
            return pools_by_name[config.name]

        with patch("pg_mcp.db.pool.create_pool", new=fake_create):
            pools = await create_pools(configs)

        assert set(pools) == {"db1", "db2"}
        assert pools["db2"] is pools_by_name["db2"]


class TestClosePools:
    @pytest.mark.asyncio
    async def test_graceful_close(self) -> None:
        pool = MagicMock()
        pool.close = AsyncMock()
        await close_pools({"db1": pool}, grace_period=1.0)
        pool.close.assert_awaited_once()
        pool.terminate.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_terminates_on_timeout(self) -> None:
        pool = MagicMock()

        async def slow_close():
            await asyncio.sleep(10)

        pool.close = slow_close
        await close_pools({"db1": pool}, grace_period=0.01)
        pool.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_terminates_on_error_and_continues(self) -> None:
        good_pool, bad_pool = MagicMock(), MagicMock()
        good_pool.close = AsyncMock()
        bad_pool.close = AsyncMock(side_effect=RuntimeError("already dead"))

        await close_pools({"bad": bad_pool, "good": good_pool}, grace_period=1.0)

        bad_pool.terminate.assert_called_once()
        good_pool.close.assert_awaited_once()
