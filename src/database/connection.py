"""Database connection pool management using asyncpg."""

import asyncio
from typing import Optional

import asyncpg
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)
from sqlalchemy.orm import sessionmaker

from src.core.config import DatabaseConfig
from src.core.exceptions import DatabaseError


class DatabaseConnection:
    """Async database connection pool manager."""

    def __init__(self, config: DatabaseConfig):
        """
        Initialize database connection manager.

        Args:
            config: Database configuration
        """
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[sessionmaker] = None

    async def create_pool(self) -> asyncpg.Pool:
        """
        Create and return asyncpg connection pool.

        Returns:
            asyncpg.Pool instance

        Raises:
            DatabaseError: If connection pool creation fails
        """
        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password,
                    min_size=self.config.min_pool_size,
                    max_size=self.config.max_pool_size,
                    command_timeout=60,
                )
            except Exception as e:
                raise DatabaseError(f"Failed to create connection pool: {e}") from e

        return self._pool

    async def get_connection(self) -> asyncpg.Connection:
        """
        Get a connection from the pool.

        Returns:
            asyncpg.Connection instance

        Raises:
            DatabaseError: If pool is not initialized or connection fails
        """
        if self._pool is None:
            await self.create_pool()

        try:
            return await self._pool.acquire()
        except Exception as e:
            raise DatabaseError(f"Failed to acquire connection: {e}") from e

    async def release_connection(self, connection: asyncpg.Connection):
        """
        Release a connection back to the pool.

        Args:
            connection: Connection to release
        """
        if self._pool is not None:
            await self._pool.release(connection)

    async def close_pool(self):
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def create_sqlalchemy_engine(self) -> AsyncEngine:
        """
        Create SQLAlchemy async engine.

        Returns:
            AsyncEngine instance
        """
        if self._engine is None:
            # Build async connection URL for SQLAlchemy
            url = (
                f"postgresql+asyncpg://{self.config.user}:{self.config.password}"
                f"@{self.config.host}:{self.config.port}/{self.config.database}"
            )

            self._engine = create_async_engine(
                url,
                pool_size=self.config.max_pool_size,
                max_overflow=0,
                echo=False,
            )

            # Create session factory
            self._session_factory = sessionmaker(
                self._engine, class_=AsyncSession, expire_on_commit=False
            )

        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get SQLAlchemy session factory."""
        if self._session_factory is None:
            self.create_sqlalchemy_engine()
        return self._session_factory

    async def execute_script(self, script: str):
        """
        Execute a SQL script (useful for migrations).

        Args:
            script: SQL script to execute

        Raises:
            DatabaseError: If script execution fails
        """
        conn = await self.get_connection()
        try:
            await conn.execute(script)
        except Exception as e:
            raise DatabaseError(f"Failed to execute script: {e}") from e
        finally:
            await self.release_connection(conn)

    async def run_migration(self, migration_file: str):
        """
        Run a database migration from a file.

        Args:
            migration_file: Path to migration SQL file

        Raises:
            DatabaseError: If migration fails
        """
        try:
            with open(migration_file, "r") as f:
                script = f.read()
            await self.execute_script(script)
        except FileNotFoundError as e:
            raise DatabaseError(f"Migration file not found: {migration_file}") from e
        except Exception as e:
            raise DatabaseError(f"Migration failed: {e}") from e

    async def close(self):
        """Close all database connections."""
        await self.close_pool()

        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None


# Global database connection instance
_db: Optional[DatabaseConnection] = None


def get_database() -> DatabaseConnection:
    """
    Get or create the global database connection instance.

    Returns:
        DatabaseConnection instance
    """
    global _db
    return _db


async def init_database(config: DatabaseConfig):
    """
    Initialize the global database connection.

    Args:
        config: Database configuration
    """
    global _db
    if _db is None:
        _db = DatabaseConnection(config)
        await _db.create_pool()
        _db.create_sqlalchemy_engine()


async def close_database():
    """Close the global database connection."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
