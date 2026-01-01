"""LangChain tools for character agents."""

from typing import Optional, List
from langchain_core.tools import Tool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import asyncio

from src.core.exceptions import SemanticSearchError
from src.database.models import CharacterMemory, Character, CharacterRelationship
from src.embeddings.search import SemanticSearch


class CharacterAgentTools:
    """
    Tools for LangChain character agents.

    Provides agents with capabilities to query memories,
    observe scenes, recall conversations, and track relationships.
    """

    def __init__(
        self,
        session: AsyncSession,
        character_id: int,
        story_id: int,
        semantic_search: SemanticSearch,
    ):
        """
        Initialize agent tools.

        Args:
            session: Database session
            character_id: Character ID
            story_id: Story ID
            semantic_search: Semantic search instance
        """
        self.session = session
        self.character_id = character_id
        self.story_id = story_id
        self.semantic_search = semantic_search

    async def query_memories(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 5,
    ) -> str:
        """
        Search character's memories by semantic similarity.

        Args:
            query: Query text
            memory_types: Optional filter by memory types
            limit: Maximum number of results

        Returns:
            Formatted string with relevant memories
        """
        try:
            memories = await self.semantic_search.retrieve_character_memories(
                session=self.session,
                character_id=self.character_id,
                story_id=self.story_id,
                query=query,
                memory_types=memory_types,
                limit=limit,
            )

            if not memories:
                return "No relevant memories found."

            result_parts = []
            for memory, relevance in memories:
                # Include emotional valence in memory display
                valence_str = ""
                if memory.emotional_valence != 0:
                    if memory.emotional_valence > 0:
                        valence_str = f" [positive: {memory.emotional_valence:.2f}]"
                    else:
                        valence_str = f" [negative: {memory.emotional_valence:.2f}]"

                result_parts.append(f"- {memory.content}{valence_str} (relevance: {relevance:.2f})")

            return "Relevant memories:\n" + "\n".join(result_parts)

        except Exception as e:
            return f"Error querying memories: {e}"

    async def store_memory(
        self,
        content: str,
        memory_type: str,
        emotional_valence: float = 0.0,
        importance: float = 0.5,
    ) -> str:
        """
        Store a new memory with emotional valence.

        Args:
            content: Memory content
            memory_type: Type of memory (conversation, observation, action_taken)
            emotional_valence: Emotional score (-1.0 to 1.0)
            importance: Importance score (0.0 to 1.0)

        Returns:
            Confirmation message
        """
        try:
            from src.embeddings.encoder import get_encoder

            encoder = get_encoder()
            embedding = await encoder.encode_async(content)

            memory = CharacterMemory(
                character_id=self.character_id,
                story_id=self.story_id,
                memory_type=memory_type,
                content=content,
                emotional_valence=emotional_valence,
                importance=importance,
                embedding=embedding,
            )

            self.session.add(memory)
            await self.session.flush()

            return f"Memory stored: {content[:50]}..."

        except Exception as e:
            return f"Error storing memory: {e}"

    async def observe_scene(self, scene_content: str) -> str:
        """
        Get current scene context.

        Args:
            scene_content: Current scene content

        Returns:
            Scene observation
        """
        return f"Current scene: {scene_content[:200]}..."

    async def get_relationships(self, other_character_name: Optional[str] = None) -> str:
        """
        Query relationships with other characters.

        Args:
            other_character_name: Optional specific character name

        Returns:
            Relationship information
        """
        try:
            # Get relationships from database
            result = await self.session.execute(
                select(CharacterRelationship).where(
                    CharacterRelationship.story_id == self.story_id,
                    (CharacterRelationship.character_a_id == self.character_id) |
                    (CharacterRelationship.character_b_id == self.character_id)
                )
            )
            relationships = result.scalars().all()

            if not relationships:
                return "No relationships established yet."

            # Filter by specific character if requested
            if other_character_name:
                # Get the character ID for the name
                char_result = await self.session.execute(
                    select(Character).where(Character.name == other_character_name)
                )
                other_char = char_result.scalar_one_or_none()

                if not other_char:
                    return f"Character '{other_character_name}' not found."

                # Filter relationships
                filtered = [
                    r for r in relationships
                    if r.character_a_id == other_char.id or r.character_b_id == other_char.id
                ]

                if not filtered:
                    return f"No relationship established with {other_character_name}."

                relationships = filtered

            # Format relationships
            lines = []
            for rel in relationships:
                # Determine which character is the other one
                other_id = rel.character_b_id if rel.character_a_id == self.character_id else rel.character_a_id

                # Get other character name
                other_char_result = await self.session.execute(
                    select(Character).where(Character.id == other_id)
                )
                other_char = other_char_result.scalar_one_or_none()
                other_name = other_char.name if other_char else "Unknown"

                # Format relationship
                rel_type = rel.relationship_type or "acquaintance"
                sentiment = "neutral" if rel.sentiment_score == 0 else ("positive" if rel.sentiment_score > 0 else "negative")
                trust = f"{rel.trust_level*100:.0f}%"

                lines.append(
                    f"- {other_name}: {rel_type} ({sentiment}, trust: {trust}, "
                    f"familiarity: {rel.familiarity*100:.0f}%)"
                )

            return "Your relationships:\n" + "\n".join(lines)

        except Exception as e:
            return f"Error querying relationships: {e}"

    async def update_relationship(
        self,
        other_character_id: int,
        sentiment_delta: float = 0.0,
        trust_delta: float = 0.0,
        interaction_type: Optional[str] = None,
    ) -> str:
        """
        Update relationship with another character.

        Args:
            other_character_id: ID of other character
            sentiment_delta: Change in sentiment (-1.0 to 1.0)
            trust_delta: Change in trust (0.0 to 1.0)
            interaction_type: Type of interaction

        Returns:
            Confirmation message
        """
        try:
            from datetime import datetime

            # Get or create relationship
            result = await self.session.execute(
                select(CharacterRelationship).where(
                    CharacterRelationship.story_id == self.story_id,
                    CharacterRelationship.character_a_id == self.character_id,
                    CharacterRelationship.character_b_id == other_character_id
                )
            )
            relationship = result.scalar_one_or_none()

            if relationship:
                # Update existing relationship
                relationship.sentiment_score = max(-1.0, min(1.0, relationship.sentiment_score + sentiment_delta))
                relationship.trust_level = max(0.0, min(1.0, relationship.trust_level + trust_delta))
                relationship.familiarity = min(1.0, relationship.familiarity + 0.1)  # Increase with each interaction
                relationship.interaction_count += 1
                relationship.last_interaction = datetime.utcnow()

                if interaction_type:
                    relationship.relationship_type = interaction_type
            else:
                # Create new relationship
                relationship = CharacterRelationship(
                    story_id=self.story_id,
                    character_a_id=self.character_id,
                    character_b_id=other_character_id,
                    sentiment_score=max(-1.0, min(1.0, sentiment_delta)),
                    trust_level=max(0.0, min(1.0, 0.5 + trust_delta)),
                    familiarity=0.1,
                    relationship_type=interaction_type or "acquaintance",
                    interaction_count=1,
                    last_interaction=datetime.utcnow(),
                )
                self.session.add(relationship)

            await self.session.flush()

            # Get other character name
            char_result = await self.session.execute(
                select(Character).where(Character.id == other_character_id)
            )
            other_char = char_result.scalar_one_or_none()
            other_name = other_char.name if other_char else "Unknown"

            return f"Relationship updated with {other_name}"

        except Exception as e:
            return f"Error updating relationship: {e}"

    async def recall_conversation(self, topic: str, limit: int = 3) -> str:
        """
        Recall past conversations about a topic.

        Args:
            topic: Topic to recall
            limit: Maximum number of conversations

        Returns:
            Relevant conversation memories
        """
        return await self.query_memories(
            query=topic,
            memory_types=["conversation"],
            limit=limit,
        )

    def get_langchain_tools(self) -> List[Tool]:
        """
        Convert agent methods to LangChain tools.

        Returns:
            List of LangChain Tool objects
        """
        # Create synchronous wrappers for async methods
        def sync_query_memories(query: str) -> str:
            """Synchronous wrapper for query_memories."""
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.query_memories(query))

        def sync_store_memory(content: str) -> str:
            """Synchronous wrapper for store_memory."""
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.store_memory(content, "observation", 0.0, 0.5)
            )

        def sync_observe_scene(scene_content: str) -> str:
            """Synchronous wrapper for observe_scene."""
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.observe_scene(scene_content))

        def sync_get_relationships(other_name: Optional[str] = None) -> str:
            """Synchronous wrapper for get_relationships."""
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_relationships(other_name))

        def sync_recall_conversation(topic: str) -> str:
            """Synchronous wrapper for recall_conversation."""
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.recall_conversation(topic))

        return [
            Tool(
                name="query_memories",
                func=sync_query_memories,
                description="Search your memories by semantic similarity. Input: query string.",
            ),
            Tool(
                name="store_memory",
                func=sync_store_memory,
                description="Store a new memory. Input: memory content string.",
            ),
            Tool(
                name="observe_scene",
                func=sync_observe_scene,
                description="Observe the current scene. Input: scene content string.",
            ),
            Tool(
                name="get_relationships",
                func=sync_get_relationships,
                description="Get relationship information. Input: optional character name.",
            ),
            Tool(
                name="recall_conversation",
                func=sync_recall_conversation,
                description="Recall past conversations. Input: topic string.",
            ),
        ]
