"""Semantic search using pgvector for story content and memories."""

import asyncio
from typing import List, Optional, Tuple

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.exceptions import SemanticSearchError
from src.database.models import Scene, Character, Memory, CharacterMemory
from src.embeddings.encoder import EmbeddingEncoder


class SemanticSearch:
    """
    Vector similarity search using pgvector.

    Provides semantic search capabilities for scenes, characters, and memories.
    """

    def __init__(self, encoder: EmbeddingEncoder):
        """
        Initialize semantic search.

        Args:
            encoder: Embedding encoder instance
        """
        self.encoder = encoder

    async def find_similar_scenes(
        self,
        session: AsyncSession,
        query_text: str,
        story_id: int,
        limit: int = 5,
        threshold: float = 0.7,
        exclude_scene_id: Optional[int] = None,
    ) -> List[Tuple[Scene, float]]:
        """
        Find semantically similar scenes using cosine similarity.

        Args:
            session: Database session
            query_text: Query text to search for
            story_id: Story ID to search within
            limit: Maximum number of results
            threshold: Minimum similarity score (0-1)
            exclude_scene_id: Optional scene ID to exclude from results

        Returns:
            List of (Scene, similarity_score) tuples

        Raises:
            SemanticSearchError: If search fails
        """
        try:
            # Generate embedding for query
            query_embedding = await self.encoder.encode_async(query_text)

            # Build SQL query with pgvector cosine similarity
            # Note: Using raw SQL for pgvector operators
            exclude_clause = f"AND id != {exclude_scene_id}" if exclude_scene_id else ""

            sql = text("""
                SELECT
                    id,
                    story_id,
                    parent_scene_id,
                    scene_number,
                    content,
                    raw_response,
                    choices_generated,
                    created_at,
                    meta,
                    1 - (embedding <=> :embedding) as similarity_score
                FROM scenes
                WHERE story_id = :story_id
                    AND embedding IS NOT NULL
                    {exclude_clause}
                ORDER BY embedding <=> :embedding
                LIMIT :limit
            """.format(exclude_clause=exclude_clause))

            result = await session.execute(
                sql,
                {
                    "embedding": str(query_embedding),
                    "story_id": story_id,
                    "limit": limit * 2,  # Get more results, filter by threshold
                },
            )

            rows = result.fetchall()

            # Convert to Scene objects and filter by threshold
            scenes = []
            for row in rows:
                similarity_score = float(row[-1])  # Last column is similarity_score

                if similarity_score >= threshold:
                    scene = Scene(
                        id=row[0],
                        story_id=row[1],
                        parent_scene_id=row[2],
                        scene_number=row[3],
                        content=row[4],
                        raw_response=row[5],
                        choices_generated=row[6],
                        created_at=row[7],
                        meta=row[8],
                    )
                    scenes.append((scene, similarity_score))

                    if len(scenes) >= limit:
                        break

            return scenes

        except Exception as e:
            raise SemanticSearchError(f"Failed to find similar scenes: {e}") from e

    async def find_relevant_characters(
        self,
        session: AsyncSession,
        query: str,
        story_id: Optional[int] = None,
        limit: int = 5,
    ) -> List[Tuple[Character, float]]:
        """
        Find characters relevant to a query.

        Args:
            session: Database session
            query: Query text
            story_id: Optional story ID to filter by
            limit: Maximum number of results

        Returns:
            List of (Character, relevance_score) tuples
        """
        try:
            # Generate embedding for query
            query_embedding = await self.encoder.encode_async(query)

            # For now, use simple character name/description matching
            # TODO: Implement proper vector search on characters table

            result = await session.execute(
                select(Character)
                .options(selectinload(Character.first_scene))
                .limit(limit * 2)
            )

            characters = result.scalars().all()

            # Simple relevance based on text matching
            # In production, use proper vector similarity search
            relevant_chars = []
            for char in characters:
                # Calculate simple text relevance
                text = f"{char.name} {char.description or ''} {char.personality or ''}"
                relevance = self._calculate_text_relevance(query, text)

                if relevance > 0:
                    relevant_chars.append((char, relevance))

            # Sort by relevance and limit
            relevant_chars.sort(key=lambda x: x[1], reverse=True)
            return relevant_chars[:limit]

        except Exception as e:
            raise SemanticSearchError(f"Failed to find relevant characters: {e}") from e

    async def retrieve_memories(
        self,
        session: AsyncSession,
        story_id: int,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Tuple[Memory, float]]:
        """
        Retrieve story memories relevant to a query.

        Args:
            session: Database session
            story_id: Story ID
            query: Query text
            memory_types: Optional list of memory types to filter by
            limit: Maximum number of results

        Returns:
            List of (Memory, relevance_score) tuples
        """
        try:
            # Generate embedding for query
            query_embedding = await self.encoder.encode_async(query)

            # Build SQL query
            type_filter = ""
            params = {
                "embedding": str(query_embedding),
                "story_id": story_id,
                "limit": limit * 2,
            }

            if memory_types:
                type_filter = "AND memory_type = ANY(:memory_types)"
                params["memory_types"] = memory_types

            sql = text(f"""
                SELECT
                    id,
                    story_id,
                    scene_id,
                    content,
                    memory_type,
                    importance,
                    created_at,
                    1 - (embedding <=> :embedding) as relevance_score
                FROM memories
                WHERE story_id = :story_id
                    AND embedding IS NOT NULL
                    {type_filter}
                ORDER BY embedding <=> :embedding
                LIMIT :limit
            """)

            result = await session.execute(sql, params)
            rows = result.fetchall()

            # Convert to Memory objects
            memories = []
            for row in rows:
                memory = Memory(
                    id=row[0],
                    story_id=row[1],
                    scene_id=row[2],
                    content=row[3],
                    memory_type=row[4],
                    importance=row[5],
                    created_at=row[6],
                )
                relevance = float(row[7])
                memories.append((memory, relevance))

            return memories

        except Exception as e:
            raise SemanticSearchError(f"Failed to retrieve memories: {e}") from e

    async def retrieve_character_memories(
        self,
        session: AsyncSession,
        character_id: int,
        story_id: int,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Tuple[CharacterMemory, float]]:
        """
        Retrieve character memories relevant to a query.

        Args:
            session: Database session
            character_id: Character ID
            story_id: Story ID
            query: Query text
            memory_types: Optional list of memory types to filter by
            limit: Maximum number of results

        Returns:
            List of (CharacterMemory, relevance_score) tuples
        """
        try:
            # Generate embedding for query
            query_embedding = await self.encoder.encode_async(query)

            # Build SQL query
            type_filter = ""
            params = {
                "embedding": str(query_embedding),
                "character_id": character_id,
                "story_id": story_id,
                "limit": limit * 2,
            }

            if memory_types:
                type_filter = "AND memory_type = ANY(:memory_types)"
                params["memory_types"] = memory_types

            sql = text(f"""
                SELECT
                    id,
                    character_id,
                    story_id,
                    memory_type,
                    content,
                    emotional_valence,
                    importance,
                    created_at,
                    1 - (embedding <=> :embedding) as relevance_score
                FROM character_memories
                WHERE character_id = :character_id
                    AND story_id = :story_id
                    AND embedding IS NOT NULL
                    {type_filter}
                ORDER BY embedding <=> :embedding
                LIMIT :limit
            """)

            result = await session.execute(sql, params)
            rows = result.fetchall()

            # Convert to CharacterMemory objects
            memories = []
            for row in rows:
                memory = CharacterMemory(
                    id=row[0],
                    character_id=row[1],
                    story_id=row[2],
                    memory_type=row[3],
                    content=row[4],
                    emotional_valence=row[5],
                    importance=row[6],
                    created_at=row[7],
                )
                relevance = float(row[8])
                memories.append((memory, relevance))

            return memories

        except Exception as e:
            raise SemanticSearchError(f"Failed to retrieve character memories: {e}") from e

    def _calculate_text_relevance(self, query: str, text: str) -> float:
        """
        Calculate simple text relevance (fallback when vectors not available).

        Args:
            query: Query text
            text: Text to compare against

        Returns:
            Relevance score between 0 and 1
        """
        if not query or not text:
            return 0.0

        query_lower = query.lower()
        text_lower = text.lower()

        # Count matching words
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())

        if not query_words:
            return 0.0

        matches = len(query_words & text_words)
        return matches / len(query_words)


# Global semantic search instance
_search: Optional[SemanticSearch] = None


def get_semantic_search(encoder: Optional[EmbeddingEncoder] = None) -> SemanticSearch:
    """
    Get or create the global semantic search instance.

    Args:
        encoder: Optional embedding encoder. If not provided, uses global encoder.

    Returns:
        SemanticSearch instance
    """
    global _search
    if _search is None:
        if encoder is None:
            from src.embeddings.encoder import get_encoder

            encoder = get_encoder()
        _search = SemanticSearch(encoder)
    return _search


def reset_semantic_search():
    """Reset the global semantic search instance (mainly for testing)."""
    global _search
    _search = None
