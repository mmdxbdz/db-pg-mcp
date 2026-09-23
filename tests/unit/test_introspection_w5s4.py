"""Unit tests for schema introspection against a mocked asyncpg connection."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.db.introspection import SchemaIntrospector


def row(**values):
    """Minimal dict-like row: introspection accesses keys via indexing."""
    return values


def make_connection() -> MagicMock:
    """Connection mock that routes fetch/fetchval by query fingerprint."""
    conn = MagicMock()

    async def fetch(query, *args):
        # Dispatch by distinctive fingerprints; several catalog queries share
        # tables (e.g. pg_attribute), so order matters.
        q = " ".join(query.split())
        if "pg_am am" in q:  # secondary index query (contains indisprimary too)
            return [
                row(
                    index_name="idx_users_email",
                    is_unique=True,
                    index_type="btree",
                    columns=["email"],
                )
            ]
        if "pg_constraint" in q:  # FKs also reference pg_attribute
            return [
                row(
                    constraint_name="fk_orders_user",
                    column_name="user_id",
                    referenced_table="users",
                    referenced_column="id",
                )
            ]
        if "indisprimary" in q:  # PK columns
            return [row(column_name="id")]
        if "pg_type t" in q:
            return [row(schema_name="public", type_name="order_status", values=["open", "paid"])]
        if "relkind = 'v'" in q:
            return [row(schema_name="public", table_name="active_users", comment=None)]
        if "relkind = 'r'" in q:
            return [row(schema_name="public", table_name="users", comment="users table")]
        if "pg_attribute" in q:  # columns (checked last: PK/FK queries hit it too)
            return [
                row(
                    column_name="id",
                    data_type="integer",
                    is_nullable=False,
                    default_value=None,
                    comment=None,
                ),
                row(
                    column_name="name",
                    data_type="varchar",
                    is_nullable=False,
                    default_value=None,
                    comment=None,
                ),
                row(
                    column_name="email",
                    data_type="varchar",
                    is_nullable=True,
                    default_value=None,
                    comment=None,
                ),
            ]
        return []

    async def fetchval(query, *args):
        q = " ".join(query.split())
        if "SELECT version()" in q:
            return "PostgreSQL 15.6, 64-bit"
        if "EXISTS(" in q:
            return True
        if "reltuples" in q:
            return 1234
        return None

    conn.fetch = AsyncMock(side_effect=fetch)
    conn.fetchval = AsyncMock(side_effect=fetchval)
    return conn


def make_pool() -> MagicMock:
    conn = make_connection()
    pool = MagicMock()
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire_cm
    pool._conn = conn  # keep a handle for assertions
    return pool


class TestIntrospect:
    @pytest.mark.asyncio
    async def test_builds_complete_schema(self) -> None:
        pool = make_pool()
        introspector = SchemaIntrospector(pool, "analytics")

        schema = await introspector.introspect()

        assert schema.database_name == "analytics"
        assert schema.version == "PostgreSQL 15.6"
        names = {t.table_name for t in schema.tables}
        assert names == {"users", "active_users"}

        users = next(t for t in schema.tables if t.table_name == "users")
        assert users.comment == "users table"
        # Primary key flag propagated onto the column
        id_column = next(c for c in users.columns if c.name == "id")
        assert id_column.is_primary_key
        # Unique secondary index captured
        assert users.indexes[0].name == "idx_users_email"
        assert users.indexes[0].is_unique
        assert users.indexes[0].index_type == "btree"
        # Foreign key captured for the enriched table
        assert users.foreign_keys[0].constraint_name == "fk_orders_user"
        assert users.foreign_keys[0].referenced_table == "users"
        # Row count estimate from pg_class statistics
        assert users.row_count_estimate == 1234

        assert schema.enum_types[0].type_name == "order_status"
        assert schema.enum_types[0].values == ["open", "paid"]

    @pytest.mark.asyncio
    async def test_column_uniqueness_checked(self) -> None:
        pool = make_pool()
        introspector = SchemaIntrospector(pool, "analytics")
        schema = await introspector.introspect()
        users = next(t for t in schema.tables if t.table_name == "users")
        email = next(c for c in users.columns if c.name == "email")
        assert email.is_unique

    @pytest.mark.asyncio
    async def test_version_fallback_when_unknown(self) -> None:
        pool = make_pool()
        pool._conn.fetchval = AsyncMock(return_value=None)
        introspector = SchemaIntrospector(pool, "db")
        schema = await introspector.introspect()
        assert schema.version is None

    @pytest.mark.asyncio
    async def test_empty_database_yields_empty_schema(self) -> None:
        pool = make_pool()
        pool._conn.fetch = AsyncMock(return_value=[])
        introspector = SchemaIntrospector(pool, "empty_db")
        schema = await introspector.introspect()
        assert schema.tables == []
        assert schema.enum_types == []
