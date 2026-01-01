"""Factory for creating character agents."""

from typing import Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.exceptions import AgentError
from src.database.models import Character
from src.agents.character_agent import CharacterAgent
from src.llm.ollama_client import OllamaClient
from src.llm.prompt_builder import PromptBuilder
from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.search import SemanticSearch


class AgentFactory:
    """
    Factory for creating and managing character agents.

    Creates CharacterAgent instances from database records
    and manages their lifecycle.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        prompt_builder: PromptBuilder,
        encoder: EmbeddingEncoder,
        semantic_search: SemanticSearch,
        show_internal_thoughts: bool = False,
    ):
        """
        Initialize agent factory.

        Args:
            ollama_client: Ollama API client
            prompt_builder: Prompt builder
            encoder: Embedding encoder
            semantic_search: Semantic search instance
            show_internal_thoughts: Whether to show internal thoughts by default
        """
        self.ollama_client = ollama_client
        self.prompt_builder = prompt_builder
        self.encoder = encoder
        self.semantic_search = semantic_search
        self.show_internal_thoughts = show_internal_thoughts

        # Cache of active agents
        self._agent_cache: Dict[int, CharacterAgent] = {}

    def set_show_internal_thoughts(self, show: bool):
        """
        Set whether to show internal thoughts for all agents.

        Args:
            show: Whether to show internal thoughts
        """
        self.show_internal_thoughts = show
        # Update all cached agents
        for agent in self._agent_cache.values():
            agent.set_show_internal_thoughts(show)

    async def create_agent(
        self,
        character: Character,
        session: AsyncSession,
        story_id: int,
        show_internal_thoughts: Optional[bool] = None,
    ) -> CharacterAgent:
        """
        Create or retrieve a character agent.

        Args:
            character: Character database model
            session: Database session
            story_id: Story ID
            show_internal_thoughts: Override default setting for this agent

        Returns:
            Initialized CharacterAgent instance

        Raises:
            AgentError: If agent creation fails
        """
        try:
            # Determine if we should show internal thoughts
            show_thoughts = show_internal_thoughts if show_internal_thoughts is not None else self.show_internal_thoughts

            # Check cache first
            if character.id in self._agent_cache:
                agent = self._agent_cache[character.id]
                agent.set_show_internal_thoughts(show_thoughts)
            else:
                # Create new agent
                agent = CharacterAgent(
                    character=character,
                    ollama_client=self.ollama_client,
                    encoder=self.encoder,
                    semantic_search=self.semantic_search,
                    prompt_builder=self.prompt_builder,
                    show_internal_thoughts=show_thoughts,
                )
                self._agent_cache[character.id] = agent

            # Initialize session
            await agent.initialize_session(session, story_id)

            return agent

        except Exception as e:
            raise AgentError(f"Failed to create agent for character {character.id}: {e}") from e

    async def create_agents_for_scene(
        self,
        character_ids: list[int],
        session: AsyncSession,
        story_id: int,
        show_internal_thoughts: Optional[bool] = None,
    ) -> Dict[int, CharacterAgent]:
        """
        Create multiple agents for characters in a scene.

        Args:
            character_ids: List of character IDs
            session: Database session
            story_id: Story ID
            show_internal_thoughts: Override default setting for these agents

        Returns:
            Dictionary mapping character IDs to agents

        Raises:
            AgentError: If any agent creation fails
        """
        agents = {}

        for char_id in character_ids:
            # Get character from database
            result = await session.execute(
                select(Character).where(Character.id == char_id)
            )
            character = result.scalar_one_or_none()

            if not character:
                raise AgentError(f"Character {char_id} not found in database")

            # Create agent
            agent = await self.create_agent(
                character, 
                session, 
                story_id,
                show_internal_thoughts=show_internal_thoughts,
            )
            agents[char_id] = agent

        return agents

    async def get_or_create_character(
        self,
        session: AsyncSession,
        name: str,
        description: Optional[str] = None,
        personality: Optional[str] = None,
        goals: Optional[str] = None,
        background: Optional[str] = None,
        first_scene_id: Optional[int] = None,
    ) -> Character:
        """
        Get existing character or create a new one.

        Args:
            session: Database session
            name: Character name
            description: Optional description
            personality: Optional personality traits
            goals: Optional goals/motivations
            background: Optional backstory
            first_scene_id: Optional first scene ID

        Returns:
            Character database model
        """
        # Try to find existing character
        result = await session.execute(
            select(Character).where(Character.name == name)
        )
        character = result.scalar_one_or_none()

        if character:
            return character

        # Create new character
        character = Character(
            name=name,
            description=description,
            personality=personality,
            goals=goals,
            background=background,
            first_appeared_in_scene=first_scene_id,
        )

        session.add(character)
        await session.flush()

        # Generate embedding for character
        character_text = f"{name} {description or ''} {personality or ''} {goals or ''}"
        embedding = await self.encoder.encode_async(character_text)
        character.embedding = embedding

        await session.commit()
        await session.refresh(character)

        return character

    def clear_cache(self):
        """Clear the agent cache."""
        self._agent_cache.clear()

    def get_cached_agent(self, character_id: int) -> Optional[CharacterAgent]:
        """
        Get cached agent if exists.

        Args:
            character_id: Character ID

        Returns:
            Cached agent or None
        """
        return self._agent_cache.get(character_id)


# Global agent factory instance
_factory: Optional[AgentFactory] = None


def get_agent_factory(
    ollama_client: Optional[OllamaClient] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    encoder: Optional[EmbeddingEncoder] = None,
    semantic_search: Optional[SemanticSearch] = None,
    show_internal_thoughts: bool = False,
) -> AgentFactory:
    """
    Get or create the global agent factory instance.

    Args:
        ollama_client: Optional Ollama client
        prompt_builder: Optional prompt builder
        encoder: Optional embedding encoder
        semantic_search: Optional semantic search
        show_internal_thoughts: Whether to show internal thoughts by default

    Returns:
        AgentFactory instance
    """
    global _factory

    if _factory is None:
        # Import and get instances if not provided
        if ollama_client is None:
            from src.llm.ollama_client import get_ollama_client

            ollama_client = get_ollama_client()

        if prompt_builder is None:
            from src.core.config import get_config

            config = get_config()
            prompt_builder = PromptBuilder(config.story.default_system_prompt_path)

        if encoder is None:
            from src.embeddings.encoder import get_encoder

            encoder = get_encoder()

        if semantic_search is None:
            from src.embeddings.search import get_semantic_search

            semantic_search = get_semantic_search(encoder)

        _factory = AgentFactory(
            ollama_client=ollama_client,
            prompt_builder=prompt_builder,
            encoder=encoder,
            semantic_search=semantic_search,
            show_internal_thoughts=show_internal_thoughts,
        )

    return _factory


def reset_agent_factory():
    """Reset the global agent factory (mainly for testing)."""
    global _factory
    _factory = None
