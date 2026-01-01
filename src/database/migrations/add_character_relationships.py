"""Migration: Add character_relationships table and update characters table.

This migration adds:
1. CharacterRelationship table for tracking relationships between characters
2. Emotional state fields to Character table (current_mood, emotional_state)
3. Updates to existing character agent infrastructure
"""

import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from src.core.config import get_config


async def upgrade():
    """Apply the migration."""
    config = get_config()

    # Build async database URL
    db_url = (
        f"postgresql+asyncpg://{config.database.user}:{config.database.password}"
        f"@{config.database.host}:{config.database.port}/{config.database.database}"
    )

    # Create async engine
    engine = create_async_engine(db_url, echo=False)

    # Create session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        try:
            # Add emotional state columns to characters table
            await session.execute(text("""
                ALTER TABLE characters
                ADD COLUMN IF NOT EXISTS current_mood VARCHAR(100),
                ADD COLUMN IF NOT EXISTS emotional_state JSON DEFAULT '{}'::jsonb;
            """))

            print("✓ Added emotional state columns to characters table")

            # Create character_relationships table
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS character_relationships (
                    id SERIAL PRIMARY KEY,
                    story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
                    character_a_id INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
                    character_b_id INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
                    sentiment_score FLOAT DEFAULT 0.0,
                    trust_level FLOAT DEFAULT 0.5,
                    familiarity FLOAT DEFAULT 0.0,
                    relationship_type VARCHAR(100),
                    interaction_count INTEGER DEFAULT 0,
                    last_interaction TIMESTAMP,
                    notes TEXT,
                    meta JSON DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(character_a_id, character_b_id, story_id)
                );
            """))

            print("✓ Created character_relationships table")

            # Create indexes for performance (one at a time)
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_character_relationships_story
                ON character_relationships(story_id);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_character_relationships_char_a
                ON character_relationships(character_a_id);
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_character_relationships_char_b
                ON character_relationships(character_b_id);
            """))

            print("✓ Created indexes for character_relationships table")

            await session.commit()
            print("\n✓ Migration completed successfully!")

        except Exception as e:
            await session.rollback()
            print(f"\n✗ Migration failed: {e}")
            raise
        finally:
            await engine.dispose()


async def downgrade():
    """Rollback the migration."""
    config = get_config()

    # Build async database URL
    db_url = (
        f"postgresql+asyncpg://{config.database.user}:{config.database.password}"
        f"@{config.database.host}:{config.database.port}/{config.database.database}"
    )

    # Create async engine
    engine = create_async_engine(db_url, echo=False)

    # Create session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        try:
            # Drop character_relationships table
            await session.execute(text("""
                DROP TABLE IF EXISTS character_relationships CASCADE;
            """))

            print("✓ Dropped character_relationships table")

            # Remove emotional state columns from characters table
            await session.execute(text("""
                ALTER TABLE characters
                DROP COLUMN IF EXISTS current_mood,
                DROP COLUMN IF EXISTS emotional_state;
            """))

            print("✓ Removed emotional state columns from characters table")

            await session.commit()
            print("\n✓ Rollback completed successfully!")

        except Exception as e:
            await session.rollback()
            print(f"\n✗ Rollback failed: {e}")
            raise
        finally:
            await engine.dispose()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "down":
        print("Rolling back migration...")
        asyncio.run(downgrade())
    else:
        print("Applying migration...")
        asyncio.run(upgrade())
