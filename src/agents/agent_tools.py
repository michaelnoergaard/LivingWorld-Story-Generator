"""LangChain tools for character agents."""

from typing import Optional, List
from langchain_core.tools import Tool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.exceptions import SemanticSearchError
from src.database.models import CharacterMemory, Character
from src.embeddings.search import SemanticSearch


class CharacterAgentTools:
    """
    Tools for LangChain character agents.

    Provides agents with capabilities to query memories,
    observe scenes, and recall conversations.
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
                result_parts.append(f"- {memory.content} (relevance: {relevance:.2f})")

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
        # For now, return basic info
        # In a full implementation, this would query relationship data
        if other_character_name:
            return f"Relationship with {other_character_name}: Not yet tracked"
        return "No specific relationships tracked yet"

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
        return [
            Tool(
                name="query_memories",
                func=lambda q: self.query_memories(q),
                description="Search your memories by semantic similarity. Input: query string.",
            ),
            Tool(
                name="store_memory",
                func=lambda c: self.store_memory(c, "observation", 0.0, 0.5),
                description="Store a new memory. Input: memory content string.",
            ),
            Tool(
                name="observe_scene",
                func=lambda s: self.observe_scene(s),
                description="Observe the current scene. Input: scene content string.",
            ),
            Tool(
                name="get_relationships",
                func=lambda n: self.get_relationships(n),
                description="Get relationship information. Input: optional character name.",
            ),
            Tool(
                name="recall_conversation",
                func=lambda t: self.recall_conversation(t),
                description="Recall past conversations. Input: topic string.",
            ),
        ]
