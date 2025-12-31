"""Database migration runner."""

import asyncio
from pathlib import Path

from rich.console import Console

from src.core.config import AppConfig, get_config
from src.database.connection import init_database, get_database


async def run_migrations(config: AppConfig):
    """
    Run database migrations.

    Args:
        config: Application configuration
    """
    console = Console()

    console.print("[cyan]Initializing database...[/cyan]")

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
            return

        for migration_file in migration_files:
            console.print(f"[dim]Running migration: {migration_file.name}[/dim]")

            try:
                await db.run_migration(str(migration_file))
                console.print(f"[green]✓[/green] {migration_file.name}")

            except Exception as e:
                console.print(f"[red]✗[/red] {migration_file.name}: {e}")
                raise

        console.print()
        console.print("[green]Database migrations completed successfully![/green]")

    except Exception as e:
        console.print(f"[red]Migration failed: {e}[/red]")
        raise
    finally:
        # Close database connection
        db = get_database()
        await db.close()


async def reset_database(config: AppConfig):
    """
    Reset database (drop and recreate all tables).

    WARNING: This will delete all data!

    Args:
        config: Application configuration
    """
    console = Console()

    console.print("[red]WARNING: This will delete all data![/red]")

    from rich.prompt import Prompt

    confirm = Prompt.ask(
        "Are you sure you want to reset the database?",
        choices=["y", "n"],
        default="n",
    )

    if confirm != "y":
        console.print("[dim]Database reset cancelled.[/dim]")
        return

    console.print("[cyan]Resetting database...[/cyan]")

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

        finally:
            await db.release_connection(conn)

        # Run migrations
        await run_migrations(config)

        console.print("[green]Database reset completed![/green]")

    except Exception as e:
        console.print(f"[red]Reset failed: {e}[/red]")
        raise
    finally:
        # Close database connection
        db = get_database()
        await db.close()


def main():
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
