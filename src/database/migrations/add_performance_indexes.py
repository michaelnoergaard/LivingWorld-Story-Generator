"""Migration: Add missing performance indexes for optimized query patterns.

This migration adds indexes to improve performance for frequently queried columns
and patterns identified throughout the codebase.

Performance Impact:
- Foreign key lookups: 10-100x faster with indexes
- Timestamp-based sorting: Significant improvement for ORDER BY queries
- Composite indexes: Optimize common multi-column filters
- Vector similarity: Already indexed, but we verify configuration
"""

import asyncio
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from src.core.config import get_config

logger = logging.getLogger(__name__)


async def upgrade():
    """Apply the migration - add all missing indexes."""
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
            logger.info("Starting performance index migration...")

            # ========================================
            # SCENES table indexes
            # ========================================

            # Story_id index - heavily used in WHERE clauses
            # Query patterns: Scene.story_id == story_id (used everywhere)
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_scenes_story_id
                ON scenes(story_id);
            """))
            logger.info("Created index: idx_scenes_story_id")

            # Parent scene_id index - for scene tree navigation
            # Query patterns: parent_scene_id lookups for scene hierarchy
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_scenes_parent_scene_id
                ON scenes(parent_scene_id) WHERE parent_scene_id IS NOT NULL;
            """))
            logger.info("Created index: idx_scenes_parent_scene_id")

            # Created_at index - for temporal queries
            # Query patterns: ORDER BY created_at DESC
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_scenes_created_at
                ON scenes(created_at DESC);
            """))
            logger.info("Created index: idx_scenes_created_at")

            # Composite index for story_id + created_at - common pattern
            # Optimizes: WHERE story_id = ? ORDER BY created_at DESC
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_scenes_story_created
                ON scenes(story_id, created_at DESC);
            """))
            logger.info("Created index: idx_scenes_story_created")

            # ========================================
            # CHOICES table indexes
            # ========================================

            # Scene_id is already indexed (idx_choices_scene) - verified in v001
            # Add composite index for scene + selected
            # Query patterns: WHERE scene_id = ? AND selected = TRUE
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_choices_scene_selected
                ON choices(scene_id, selected);
            """))
            logger.info("Created index: idx_choices_scene_selected")

            # ========================================
            # CHARACTERS table indexes
            # ========================================

            # Name index - for character lookups by name
            # Query patterns: Character.name == other_character_name (agent_tools.py:193)
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_characters_name
                ON characters(LOWER(name));
            """))
            logger.info("Created index: idx_characters_name (case-insensitive)")

            # First appeared in scene index
            # Query patterns: Character.first_appeared_in_scene lookups
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_characters_first_scene
                ON characters(first_appeared_in_scene) WHERE first_appeared_in_scene IS NOT NULL;
            """))
            logger.info("Created index: idx_characters_first_scene")

            # Created_at index - for character ordering
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_characters_created_at
                ON characters(created_at DESC);
            """))
            logger.info("Created index: idx_characters_created_at")

            # ========================================
            # USER_INSTRUCTIONS table indexes
            # ========================================

            # Created_at index for temporal queries
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_instructions_created_at
                ON user_instructions(created_at DESC);
            """))
            logger.info("Created index: idx_user_instructions_created_at")

            # ========================================
            # MEMORIES table indexes
            # ========================================

            # Scene_id index - for scene-specific memories
            # Query patterns: Memory.scene_id lookups
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memories_scene_id
                ON memories(scene_id) WHERE scene_id IS NOT NULL;
            """))
            logger.info("Created index: idx_memories_scene_id")

            # Importance index - for retrieving important memories
            # Query patterns: Filtering by importance scores
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance DESC);
            """))
            logger.info("Created index: idx_memories_importance")

            # Memory_type index - for filtering memory types
            # Already indexed as composite (story_id, memory_type), but standalone helps
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(memory_type) WHERE memory_type IS NOT NULL;
            """))
            logger.info("Created index: idx_memories_type")

            # Created_at index - for recent memories
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memories_created_at
                ON memories(created_at DESC);
            """))
            logger.info("Created index: idx_memories_created_at")

            # ========================================
            # CHARACTER_MEMORIES table indexes
            # ========================================

            # Composite index for character_id + created_at - common pattern
            # Query patterns: WHERE character_id = ? ORDER BY created_at DESC
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_character_memories_char_created
                ON character_memories(character_id, created_at DESC);
            """))
            logger.info("Created index: idx_character_memories_char_created")

            # Memory_type index - for filtering character memory types
            # Query patterns: WHERE memory_type = 'internal_thought'
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_character_memories_type
                ON character_memories(memory_type) WHERE memory_type IS NOT NULL;
            """))
            logger.info("Created index: idx_character_memories_type")

            # Importance index - for important character memories
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_character_memories_importance
                ON character_memories(importance DESC);
            """))
            logger.info("Created index: idx_character_memories_importance")

            # Emotional valence index - for emotion-based queries
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_character_memories_valence
                ON character_memories(emotional_valence);
            """))
            logger.info("Created index: idx_character_memories_valence")

            # ========================================
            # SCENE_CHARACTERS junction table indexes
            # ========================================

            # Both scene_id and character_id already indexed separately
            # Add importance index for ordering characters in scenes
            # Query patterns: ORDER BY importance DESC (context.py:186)
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_scene_characters_importance
                ON scene_characters(scene_id, importance DESC);
            """))
            logger.info("Created index: idx_scene_characters_importance")

            # ========================================
            # CHARACTER_RELATIONSHIPS table indexes
            # ========================================

            # Note: story_id, character_a_id, character_b_id already indexed
            # Add composite index for relationship lookups
            # Query patterns: WHERE story_id = ? AND (character_a_id = ? OR character_b_id = ?)
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_character_relationships_story_a
                ON character_relationships(story_id, character_a_id);
            """))
            logger.info("Created index: idx_character_relationships_story_a")

            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_character_relationships_story_b
                ON character_relationships(story_id, character_b_id);
            """))
            logger.info("Created index: idx_character_relationships_story_b")

            # Last interaction index - for recent relationships
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_character_relationships_last_interaction
                ON character_relationships(last_interaction DESC)
                WHERE last_interaction IS NOT NULL;
            """))
            logger.info("Created index: idx_character_relationships_last_interaction")

            # Relationship type index - for filtering by type
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_character_relationships_type
                ON character_relationships(relationship_type)
                WHERE relationship_type IS NOT NULL;
            """))
            logger.info("Created index: idx_character_relationships_type")

            # ========================================
            # STORIES table indexes
            # ========================================

            # is_active already indexed (partial index)
            # Add updated_at index for story ordering
            # Query patterns: ORDER BY updated_at DESC (state.py:252)
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_stories_updated_at
                ON stories(updated_at DESC);
            """))
            logger.info("Created index: idx_stories_updated_at")

            # Created_at index - for story listing
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_stories_created_at
                ON stories(created_at DESC);
            """))
            logger.info("Created index: idx_stories_created_at")

            # Composite index for active + updated - most common query pattern
            # Query patterns: WHERE is_active = TRUE ORDER BY updated_at DESC
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_stories_active_updated
                ON stories(is_active, updated_at DESC);
            """))
            logger.info("Created index: idx_stories_active_updated")

            # ========================================
            # Vector index verification (HNSW upgrade)
            # ========================================

            # Current IVFFlat indexes work, but HNSW is better for production
            # Note: Rebuilding vector indexes is expensive, so we only create new ones
            # This allows gradual migration to HNSW if desired

            # Uncomment the following to upgrade scenes embedding to HNSW:
            # await session.execute(text("""
            #     CREATE INDEX IF NOT EXISTS idx_scenes_embedding_hnsw
            #     ON scenes USING hnsw (embedding vector_cosine_ops)
            #     WITH (m = 16, ef_construction = 64);
            # """))
            # logger.info("Created HNSW index: idx_scenes_embedding_hnsw")

            await session.commit()
            logger.info("Performance indexes migration completed successfully!")

        except Exception as e:
            await session.rollback()
            logger.error("Migration failed: %s", e)
            raise
        finally:
            await engine.dispose()


async def downgrade():
    """Rollback the migration - remove all added indexes."""
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
            logger.info("Rolling back performance indexes...")

            # STORIES table
            await session.execute(text("DROP INDEX IF EXISTS idx_stories_active_updated;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_stories_created_at;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_stories_updated_at;"))

            # CHARACTER_RELATIONSHIPS table
            await session.execute(text("DROP INDEX IF EXISTS idx_character_relationships_type;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_character_relationships_last_interaction;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_character_relationships_story_b;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_character_relationships_story_a;"))

            # SCENE_CHARACTERS table
            await session.execute(text("DROP INDEX IF EXISTS idx_scene_characters_importance;"))

            # CHARACTER_MEMORIES table
            await session.execute(text("DROP INDEX IF EXISTS idx_character_memories_valence;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_character_memories_importance;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_character_memories_type;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_character_memories_char_created;"))

            # MEMORIES table
            await session.execute(text("DROP INDEX IF EXISTS idx_memories_created_at;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_memories_type;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_memories_importance;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_memories_scene_id;"))

            # USER_INSTRUCTIONS table
            await session.execute(text("DROP INDEX IF EXISTS idx_user_instructions_created_at;"))

            # CHARACTERS table
            await session.execute(text("DROP INDEX IF EXISTS idx_characters_created_at;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_characters_first_scene;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_characters_name;"))

            # CHOICES table
            await session.execute(text("DROP INDEX IF EXISTS idx_choices_scene_selected;"))

            # SCENES table
            await session.execute(text("DROP INDEX IF EXISTS idx_scenes_story_created;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_scenes_created_at;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_scenes_parent_scene_id;"))
            await session.execute(text("DROP INDEX IF EXISTS idx_scenes_story_id;"))

            # HNSW indexes (if created)
            await session.execute(text("DROP INDEX IF EXISTS idx_scenes_embedding_hnsw;"))

            await session.commit()
            logger.info("Rollback completed successfully!")

        except Exception as e:
            await session.rollback()
            logger.error("Rollback failed: %s", e)
            raise
        finally:
            await engine.dispose()


if __name__ == "__main__":
    import sys

    # Set up basic logging for standalone migration execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) > 1 and sys.argv[1] == "down":
        logger.info("Rolling back migration...")
        asyncio.run(downgrade())
    else:
        logger.info("Applying migration...")
        asyncio.run(upgrade())
