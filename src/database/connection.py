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
from src.core.logging_config import get_logger
from src.core.constants import DatabaseConstants
from src.core.validation import (
    validate_string, validate_id, validate_int_range,
    validate_file_path, validate_directory_path
)

logger = get_logger(__name__)


class DatabaseConnection:
    """Async database connection pool manager."""

    def __init__(self, config: DatabaseConfig):
        """
        Initialize database connection manager.

        Args:
            config: Database configuration
        """
        # Validate configuration parameters
        if not isinstance(config, DatabaseConfig):
            raise DatabaseError("initialization", original_error="DatabaseConfig instance required")

        # Validate host
        validated_host = validate_string(
            config.host,
            field_name="host",
            min_length=1,
            max_length=255,
            pattern=r"^[a-zA-Z0-9.-]+$",
            strip_whitespace=True
        )

        # Validate database name
        validated_database = validate_string(
            config.database,
            field_name="database",
            min_length=1,
            max_length=63,
            pattern=r"^[a-zA-Z0-9_]+$",
            strip_whitespace=True
        )

        # Validate user
        validated_user = validate_string(
            config.user,
            field_name="user",
            min_length=1,
            max_length=63,
            pattern=r"^[a-zA-Z0-9_]+$",
            strip_whitespace=True
        )

        # Validate port range
        validate_int_range(
            config.port,
            field_name="port",
            min_value=1,
            max_value=65535
        )

        # Validate pool sizes
        validate_int_range(
            config.min_pool_size,
            field_name="min_pool_size",
            min_value=1,
            max_value=config.max_pool_size
        )

        validate_int_range(
            config.max_pool_size,
            field_name="max_pool_size",
            min_value=config.min_pool_size,
            max_value=100
        )

        # Validate password length
        if len(config.password) < 8:
            raise DatabaseError("initialization", original_error="Password must be at least 8 characters")

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
                logger.info(
                    "Creating database connection pool: %s@%s:%s/%s",
                    self.config.user,
                    self.config.host,
                    self.config.port,
                    self.config.database,
                )
                self._pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password,
                    min_size=self.config.min_pool_size,
                    max_size=self.config.max_pool_size,
                    command_timeout=DatabaseConstants.COMMAND_TIMEOUT,
                )
                logger.info("Database connection pool created successfully")

            except asyncpg.PostgresError as e:
                logger.error(
                    "PostgreSQL error creating connection pool to %s: %s",
                    self.config.host,
                    e,
                    exc_info=True,
                )
                raise DatabaseError("creating connection pool", original_error=str(e)) from e
            except (OSError, ValueError) as e:
                logger.error(
                    "Invalid database configuration or connection error: %s", e, exc_info=True
                )
                raise DatabaseError("creating connection pool", original_error=str(e)) from e
            except Exception as e:
                logger.exception("Unexpected error creating database connection pool")
                raise DatabaseError("creating connection pool", original_error=str(e)) from e

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
        except asyncpg.PostgresError as e:
            logger.error(
                "PostgreSQL error acquiring database connection: %s", e, exc_info=True
            )
            raise DatabaseError("acquiring connection", original_error=str(e)) from e
        except Exception as e:
            logger.exception("Unexpected error acquiring database connection")
            raise DatabaseError("acquiring connection", original_error=str(e)) from e

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

        Sets proper isolation level to READ COMMITTED + enable read-only transactions
        where appropriate to prevent race conditions.

        Returns:
            AsyncEngine instance
        """
        if self._engine is None:
            # Build async connection URL for SQLAlchemy
            url = (
                f"postgresql+asyncpg://{self.config.user}:{self.config.password}"
                f"@{self.config.host}:{self.config.port}/{self.config.database}"
            )

            # Configure engine with proper isolation level
            # READ COMMITTED: Prevents dirty reads, ensures operations see committed data
            # Combined with SELECT FOR UPDATE for critical sections
            self._engine = create_async_engine(
                url,
                pool_size=self.config.max_pool_size,
                max_overflow=DatabaseConstants.MAX_OVERFLOW,
                echo=DatabaseConstants.ECHO_QUERIES,
                connect_args={
                    "server_settings": {
                        "default_transaction_isolation": "read committed",
                        "statement_timeout": DatabaseConstants.STATEMENT_TIMEOUT_MS,
                    }
                },
            )

            # Create session factory with explicit configuration
            self._session_factory = sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                # Join transactions conditionally with savepoint support
                join_transaction_mode="conditional_savepoint",
            )

            logger.info(
                "Created SQLAlchemy engine with READ COMMITTED isolation level"
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
        # Validate script
        validated_script = validate_string(
            script,
            field_name="script",
            min_length=1,
            max_length=1000000  # 1MB limit
        )

        conn = await self.get_connection()
        try:
            await conn.execute(validated_script)
        except Exception as e:
            raise DatabaseError("executing script", original_error=str(e)) from e
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
        # Validate migration file path
        validated_migration_file = validate_file_path(
            migration_file,
            allowed_extensions=[".sql"],
            field_name="migration_file"
        )

        try:
            with open(validated_migration_file, "r") as f:
                script = f.read()
            await self.execute_script(script)
        except FileNotFoundError as e:
            raise DatabaseError("finding migration file", original_error=f"Migration file not found: {validated_migration_file}") from e
        except Exception as e:
            raise DatabaseError("running migration", original_error=str(e)) from e

    async def close(self):
        """Close all database connections."""
        await self.close_pool()

        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    async def create_database_if_not_exists(self):
        """
        Create the database if it doesn't exist.

        Connects to the 'postgres' database first to check/create the target database.

        Raises:
            DatabaseError: If database creation fails
        """
        # Validate connection parameters for postgres database
        validated_host = validate_string(
            self.config.host,
            field_name="host",
            min_length=1,
            max_length=255,
            pattern=r"^[a-zA-Z0-9.-]+$",
            strip_whitespace=True
        )

        validated_user = validate_string(
            self.config.user,
            field_name="user",
            min_length=1,
            max_length=63,
            pattern=r"^[a-zA-Z0-9_]+$",
            strip_whitespace=True
        )

        validated_port = validate_int_range(
            self.config.port,
            field_name="port",
            min_value=1,
            max_value=65535
        )

        # Connect to postgres database to create our database
        conn = await asyncpg.connect(
            host=validated_host,
            port=validated_port,
            database="postgres",
            user=validated_user,
            password=self.config.password,
        )

        try:
            # Check if database exists
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = $1)",
                self.config.database
            )

            if not exists:
                # Create database with pgvector extension
                await conn.execute(f'CREATE DATABASE {self.config.database}')
                return True  # Database was created

            return False  # Database already existed

        except Exception as e:
            raise DatabaseError("creating database", original_error=str(e)) from e
        finally:
            await conn.close()


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

    Creates the database if it doesn't exist and runs migrations.

    Args:
        config: Database configuration
    """
    global _db
    if _db is None:
        _db = DatabaseConnection(config)

        # Create database if it doesn't exist
        created = await _db.create_database_if_not_exists()

        # Create connection pool
        await _db.create_pool()
        _db.create_sqlalchemy_engine()

        # Run migrations if database was created
        if created:
            await run_migrations()

        return created

    return False


async def run_migrations():
    """
    Run all database migrations.

    Raises:
        DatabaseError: If migration fails
    """
    from pathlib import Path

    db = get_database()
    if db is None:
        raise DatabaseError("initialization", original_error="Database not initialized")

    migrations_dir = Path(__file__).parent / "migrations"

    # Get all migration files and sort them
    migration_files = sorted(migrations_dir.glob("*.sql"))

    for migration_file in migration_files:
        try:
            await db.run_migration(str(migration_file))
        except Exception as e:
            raise DatabaseError("running migration", original_error=f"Migration {migration_file.name} failed: {e}") from e


async def close_database():
    """Close the global database connection."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
