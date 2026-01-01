"""Database migration runner."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from rich.console import Console

from src.core.config import AppConfig, get_config
from src.database.connection import init_database, get_database

logger = logging.getLogger(__name__)


async def run_migrations(config: AppConfig) -> None:
    """
    Run database migrations.

    Args:
        config: Application configuration
    """
    console = Console()

    console.print("[cyan]Initializing database...[/cyan]")
    logger.info("Initializing database and running migrations")

    try:
        # Initialize database connection
        await init_database(config.database)

        db = get_database()

        # Get migrations directory
        migrations_dir = Path(__file__).parent / "migrations"

        # Run migration files in order
        migration_files = sorted(migrations_dir.glob("*.sql"))

        if not migration_files:
            console.print("[yellow]No migration files found.[/yellow]")
            logger.warning("No migration files found")
            return

        for migration_file in migration_files:
            console.print(f"[dim]Running migration: {migration_file.name}[/dim]")
            logger.info("Running migration: %s", migration_file.name)

            try:
                await db.run_migration(str(migration_file))
                console.print(f"[green]✓[/green] {migration_file.name}")
                logger.info("Migration completed successfully: %s", migration_file.name)

            except Exception as e:
                console.print(f"[red]✗[/red] {migration_file.name}: {e}")
                logger.error("Migration failed: %s - %s", migration_file.name, e)
                raise

        console.print()
        console.print("[green]Database migrations completed successfully![/green]")
        logger.info("All database migrations completed successfully")

    except Exception as e:
        console.print(f"[red]Migration failed: {e}[/red]")
        logger.error("Database migration process failed: %s", e)
        raise
    finally:
        # Close database connection
        db = get_database()
        await db.close()


async def reset_database(config: AppConfig) -> None:
    """
    Reset database (drop and recreate all tables).

    WARNING: This will delete all data!

    Args:
        config: Application configuration
    """
    console = Console()

    console.print("[red]WARNING: This will delete all data![/red]")
    logger.warning("Attempting database reset - this will delete all data")

    from rich.prompt import Prompt

    confirm = Prompt.ask(
        "Are you sure you want to reset the database?",
        choices=["y", "n"],
        default="n",
    )

    if confirm != "y":
        console.print("[dim]Database reset cancelled.[/dim]")
        logger.info("Database reset cancelled by user")
        return

    console.print("[cyan]Resetting database...[/cyan]")
    logger.info("Resetting database - dropping all tables")

    try:
        # Initialize database connection
        await init_database(config.database)

        db = get_database()
        conn = await db.get_connection()

        try:
            # Drop all tables
            await conn.execute("""
                DROP TABLE IF EXISTS
                    scene_characters,
                    choices,
                    character_memories,
                    memories,
                    user_instructions,
                    scenes,
                    characters,
                    stories CASCADE;
            """)

            console.print("[green]Tables dropped successfully.[/green]")
            logger.info("All tables dropped successfully")

        finally:
            await db.release_connection(conn)

        # Run migrations
        await run_migrations(config)

        console.print("[green]Database reset completed![/green]")
        logger.info("Database reset completed successfully")

    except Exception as e:
        console.print(f"[red]Reset failed: {e}[/red]")
        logger.error("Database reset failed: %s", e)
        raise
    finally:
        # Close database connection
        db = get_database()
        await db.close()


def main() -> None:
    """Main entry point for migration runner."""
    import argparse

    parser = argparse.ArgumentParser(description="LivingWorld database management")
    parser.add_argument(
        "action",
        choices=["migrate", "reset"],
        help="Action to perform",
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Path to .env file",
    )

    args = parser.parse_args()

    # Load configuration
    env_file = Path(args.env) if args.env else None
    config = get_config(env_file)

    # Run action
    if args.action == "migrate":
        asyncio.run(run_migrations(config))
    elif args.action == "reset":
        asyncio.run(reset_database(config))


if __name__ == "__main__":
    main()
