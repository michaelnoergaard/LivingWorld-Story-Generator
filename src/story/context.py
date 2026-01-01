"""Story context builder with semantic search and agent coordination."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database.models import Story, Scene, Character
from src.embeddings.search import SemanticSearch
from src.agents.agent_factory import AgentFactory
from src.agents.character_agent import CharacterAgent
from src.core.exceptions import AgentError
from src.core.logging_config import get_logger
from src.core.constants import StoryConstants, PromptConstants

logger = get_logger(__name__)


@dataclass
class StoryContext:
    """Complete context for scene generation."""

    story_id: int
    current_scene_content: str
    recent_scenes: List[str]
    relevant_memories: List[str]
    character_contexts: Dict[str, str]
    user_instruction: Optional[str] = None
    location: Optional[str] = None
    active_characters: List[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "story_id": self.story_id,
            "current_scene_content": self.current_scene_content,
            "recent_scenes": self.recent_scenes,
            "relevant_memories": self.relevant_memories,
            "character_contexts": self.character_contexts,
            "user_instruction": self.user_instruction,
            "location": self.location,
            "active_characters": self.active_characters or [],
        }


class StoryContextBuilder:
    """
    Build story context using semantic search and character agents.

    Coordinates between:
    - Semantic search for relevant past scenes
    - Memory retrieval
    - Character agent responses
    - Location and setting information
    """

    def __init__(
        self,
        semantic_search: SemanticSearch,
        agent_factory: AgentFactory,
    ):
        """
        Initialize context builder.

        Args:
            semantic_search: Semantic search instance
            agent_factory: Agent factory for character agents
        """
        self.semantic_search = semantic_search
        self.agent_factory = agent_factory

    async def build_context(
        self,
        session: AsyncSession,
        story_id: int,
        current_scene_id: Optional[int] = None,
        user_instruction: Optional[str] = None,
        use_agents: bool = True,
    ) -> StoryContext:
        """
        Build full context for scene generation.

        Args:
            session: Database session
            story_id: Story ID
            current_scene_id: Current scene ID
            user_instruction: Optional user instruction
            use_agents: Whether to use character agents

        Returns:
            Complete StoryContext
        """
        # Get current scene
        if current_scene_id:
            result = await session.execute(
                select(Scene).where(Scene.id == current_scene_id)
            )
            current_scene = result.scalar_one_or_none()
            current_scene_content = current_scene.content if current_scene else ""
        else:
            current_scene_content = "Beginning of story"

        # Get recent scenes
        recent_scenes = await self._get_recent_scenes(session, story_id, limit=StoryConstants.DEFAULT_RECENT_SCENES_LIMIT)

        # Get relevant memories
        relevant_memories = await self._get_relevant_memories(
            session, story_id, current_scene_content
        )

        # Get character contexts
        character_contexts = {}
        active_characters = []

        if use_agents:
            character_contexts, active_characters = await self._build_character_contexts(
                session, story_id, current_scene_id, current_scene_content
            )

        # Get location if available
        location = await self._get_location(session, story_id, current_scene_id)

        return StoryContext(
            story_id=story_id,
            current_scene_content=current_scene_content,
            recent_scenes=recent_scenes,
            relevant_memories=relevant_memories,
            character_contexts=character_contexts,
            user_instruction=user_instruction,
            location=location,
            active_characters=active_characters,
        )

    async def _get_recent_scenes(
        self,
        session: AsyncSession,
        story_id: int,
        limit: int = None,
    ) -> List[str]:
        """Get recent scene contents."""
        if limit is None:
            limit = StoryConstants.DEFAULT_RECENT_SCENES_LIMIT

        result = await session.execute(
            select(Scene)
            .where(Scene.story_id == story_id)
            .order_by(Scene.scene_number.desc())
            .limit(limit)
        )

        scenes = list(reversed(result.scalars().all()))
        return [scene.content for scene in scenes]

    async def _get_relevant_memories(
        self,
        session: AsyncSession,
        story_id: int,
        query: str,
        limit: int = None,
    ) -> List[str]:
        """Get relevant story memories."""
        if limit is None:
            limit = StoryConstants.DEFAULT_RECENT_SCENES_LIMIT

        memories = await self.semantic_search.retrieve_memories(
            session=session,
            story_id=story_id,
            query=query,
            limit=limit,
        )

        return [memory.content for memory, _ in memories]

    async def _build_character_contexts(
        self,
        session: AsyncSession,
        story_id: int,
        scene_id: Optional[int],
        current_scene_content: str,
    ) -> tuple[Dict[str, str], List[int]]:
        """
        Build character contexts using agents.

        Returns:
            Tuple of (character_contexts dict, active_character_ids list)
        """
        # Get characters in current scene
        if scene_id:
            from src.database.models import SceneCharacter

            result = await session.execute(
                select(SceneCharacter)
                .where(SceneCharacter.scene_id == scene_id)
                .order_by(SceneCharacter.importance.desc())
            )

            scene_characters = result.scalars().all()
            character_ids = [sc.character_id for sc in scene_characters]
        else:
            # No characters yet
            return {}, []

        if not character_ids:
            return {}, []

        # Create agents for characters
        agents = await self.agent_factory.create_agents_for_scene(
            character_ids=character_ids,
            session=session,
            story_id=story_id,
        )

        # Get character perspectives
        character_contexts = {}
        for char_id, agent in agents.items():
            try:
                # Ask agent what they think about current situation
                response = await agent.respond_to(
                    context="The story continues...",
                    scene_content=current_scene_content[:PromptConstants.SCENE_CONTENT_TRUNCATION],
                )
                character_contexts[agent.character.name] = response
            except Exception as e:
                # If agent fails, use basic character info and log the error
                logger.warning(
                    "Agent for character %s (ID: %s) failed to generate response, using fallback description: %s",
                    agent.character.name,
                    char_id,
                    e,
                    exc_info=True,
                )
                character_contexts[agent.character.name] = agent.character.description or ""

        return character_contexts, character_ids

    async def _get_location(
        self,
        session: AsyncSession,
        story_id: int,
        scene_id: Optional[int],
    ) -> Optional[str]:
        """Get current location from scene metadata."""
        if not scene_id:
            return None

        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()

        if scene and scene.metadata:
            return scene.metadata.get("location")

        return None

    async def build_character_dialogue(
        self,
        session: AsyncSession,
        story_id: int,
        character_id: int,
        situation: str,
        other_characters: Optional[List[str]] = None,
    ) -> str:
        """
        Generate character dialogue using agent.

        Args:
            session: Database session
            story_id: Story ID
            character_id: Character ID
            situation: Current situation
            other_characters: Other characters present

        Returns:
            Character's dialogue/response
        """
        # Get character
        result = await session.execute(
            select(Character).where(Character.id == character_id)
        )
        character = result.scalar_one_or_none()

        if not character:
            raise AgentError("finding character", character_id=character_id, error_details=f"Character {character_id} not found")

        # Create agent
        agent = await self.agent_factory.create_agent(character, session, story_id)

        # Generate response
        return await agent.respond_to(
            context=situation,
            scene_content="",
            other_characters=other_characters,
        )
