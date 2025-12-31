"""Character agent implementation using LangChain."""

from typing import Optional, List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import AgentError
from src.database.models import Character
from src.llm.prompt_builder import PromptBuilder
from src.llm.ollama_client import OllamaClient
from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.search import SemanticSearch
from src.agents.agent_tools import CharacterAgentTools


class CharacterAgent:
    """
    LangChain-based agent for autonomous NPC characters.

    Each character has their own agent with unique personality,
    goals, memories, and ability to generate in-character responses.
    """

    def __init__(
        self,
        character: Character,
        ollama_client: OllamaClient,
        encoder: EmbeddingEncoder,
        semantic_search: SemanticSearch,
        prompt_builder: PromptBuilder,
    ):
        """
        Initialize character agent.

        Args:
            character: Character database model
            ollama_client: Ollama API client
            encoder: Embedding encoder
            semantic_search: Semantic search instance
            prompt_builder: Prompt builder
        """
        self.character = character
        self.ollama_client = ollama_client
        self.encoder = encoder
        self.semantic_search = semantic_search
        self.prompt_builder = prompt_builder

        # Build system prompt for this character
        self.system_prompt = self._build_character_prompt()

        # Tools will be initialized per-session
        self.tools: Optional[CharacterAgentTools] = None

    def _build_character_prompt(self) -> str:
        """
        Build system prompt for this character.

        Returns:
            Character-specific system prompt
        """
        personality = self.character.personality or "A friendly person"
        goals = self.character.goals or "To interact with others"
        background = self.character.background or "Living in this world"

        return self.prompt_builder.build_character_system_prompt(
            character_name=self.character.name,
            personality=personality,
            goals=goals,
            background=background,
        )

    async def initialize_session(
        self,
        session: AsyncSession,
        story_id: int,
    ):
        """
        Initialize agent session with tools.

        Args:
            session: Database session
            story_id: Story ID
        """
        self.tools = CharacterAgentTools(
            session=session,
            character_id=self.character.id,
            story_id=story_id,
            semantic_search=self.semantic_search,
        )

    async def respond_to(
        self,
        context: str,
        scene_content: str,
        other_characters: Optional[List[str]] = None,
    ) -> str:
        """
        Generate character's response based on context.

        Args:
            context: Current situation/context
            scene_content: Current scene content
            other_characters: Optional list of other character names present

        Returns:
            Character's response/dialogue

        Raises:
            AgentError: If response generation fails
        """
        if self.tools is None:
            raise AgentError("Agent session not initialized. Call initialize_session() first.")

        try:
            # Retrieve relevant memories
            memory_query = f"{context} {scene_content}"
            memories_result = await self.tools.query_memories(
                query=memory_query,
                limit=3,
            )

            # Build prompt
            prompt_parts = [
                f"## Current Situation\n{context}\n",
                f"## Scene\n{scene_content[:500]}\n",
            ]

            if other_characters:
                prompt_parts.append(f"## Others Present\n{', '.join(other_characters)}\n")

            if memories_result and "No relevant memories" not in memories_result:
                prompt_parts.append(f"## Your Memories\n{memories_result}\n")

            prompt_parts.append(
                "\n## Your Response\n"
                "Respond in character. Show your personality through your words. "
                "Consider your goals and how you would react to this situation."
            )

            prompt = "".join(prompt_parts)

            # Generate response
            response = await self.ollama_client.generate_with_retry(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=self._get_temperature(),
            )

            # Store this interaction as a memory
            await self.tools.store_memory(
                content=f"Responded to: {context[:100]}... | Said: {response[:100]}...",
                memory_type="conversation",
                emotional_valence=0.0,  # Could be calculated from response
                importance=0.5,
            )

            return response

        except Exception as e:
            raise AgentError(f"Failed to generate response: {e}") from e

    def _get_temperature(self) -> float:
        """Get temperature from agent config."""
        config = self.character.agent_config or {}
        return config.get("temperature", 0.8)

    async def observe_and_react(
        self,
        event: str,
        emotional_valence: float = 0.0,
    ) -> str:
        """
        Observe an event and react to it.

        Args:
            event: Event description
            emotional_valence: Emotional impact (-1.0 to 1.0)

        Returns:
            Character's reaction
        """
        if self.tools is None:
            raise AgentError("Agent session not initialized. Call initialize_session() first.")

        # Store observation as memory
        await self.tools.store_memory(
            content=f"Observed: {event}",
            memory_type="observation",
            emotional_valence=emotional_valence,
            importance=0.5,
        )

        # Generate reaction
        prompt = f"Something just happened: {event}\n\nHow do you react?"

        return await self.ollama_client.generate_with_retry(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self._get_temperature(),
        )

    async def decide_action(
        self,
        situation: str,
        available_actions: List[str],
    ) -> str:
        """
        Decide on an action based on personality and goals.

        Args:
            situation: Current situation
            available_actions: List of possible actions

        Returns:
            Chosen action and reasoning
        """
        if self.tools is None:
            raise AgentError("Agent session not initialized. Call initialize_session() first.")

        actions_text = "\n".join(f"{i+1}. {action}" for i, action in enumerate(available_actions))

        prompt = (
            f"## Situation\n{situation}\n\n"
            f"## Available Actions\n{actions_text}\n\n"
            "Which action would you take? Consider your personality and goals. "
            "Explain your reasoning and state your choice."
        )

        return await self.ollama_client.generate_with_retry(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self._get_temperature(),
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get character summary.

        Returns:
            Dictionary with character info
        """
        return {
            "id": self.character.id,
            "name": self.character.name,
            "description": self.character.description,
            "personality": self.character.personality,
            "goals": self.character.goals,
            "background": self.character.background,
            "agent_config": self.character.agent_config,
        }
