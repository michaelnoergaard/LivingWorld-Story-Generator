"""Story generation orchestrator with autonomous character integration."""

import re
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.exceptions import StoryGenerationError
from src.core.validation import (
    validate_id,
    validate_string,
    validate_choice,
    validate_content,
    validate_story_setting,
    ValidationError,
)
from src.database.models import Story, Scene, Choice, Character
from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.search import SemanticSearch
from src.llm.ollama_client import OllamaClient
from src.llm.prompt_builder import PromptBuilder
from src.story.state import StoryStateManager
from src.story.context import StoryContextBuilder
from src.agents.agent_factory import AgentFactory
from src.core.constants import (
    StoryConstants,
    LLMConstants,
    AgentConstants,
    SceneParsingConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class GeneratedScene:
    """Result of scene generation."""

    id: int  # Database ID of the saved scene
    content: str
    choices: List[str]
    raw_response: str
    character_actions: Optional[List[dict]] = None


@dataclass
class ParsedScene:
    """Parsed scene from AI response."""

    scene_content: str
    choices: List[str]


class StoryGenerator:
    """Main story generation orchestration with autonomous character actions."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        prompt_builder: PromptBuilder,
        encoder: EmbeddingEncoder,
        session_factory: type,
        use_agents: bool = True,
        show_internal_thoughts: bool = False,
    ) -> None:
        """
        Initialize story generator.

        Args:
            ollama_client: Ollama API client
            prompt_builder: Prompt builder
            encoder: Embedding encoder
            session_factory: Database session factory
            use_agents: Whether to use character agents
            show_internal_thoughts: Whether to show internal thoughts by default
        """
        self.ollama = ollama_client
        self.prompt_builder = prompt_builder
        self.encoder = encoder
        self.session_factory = session_factory
        self.state_manager = StoryStateManager(session_factory)
        self.use_agents = use_agents
        self.show_internal_thoughts = show_internal_thoughts

        # Initialize semantic search and agent components
        self.semantic_search = SemanticSearch(encoder)

        if use_agents:
            self.agent_factory = AgentFactory(
                ollama_client=ollama_client,
                prompt_builder=prompt_builder,
                encoder=encoder,
                semantic_search=self.semantic_search,
                show_internal_thoughts=show_internal_thoughts,
            )
            self.context_builder = StoryContextBuilder(
                semantic_search=self.semantic_search,
                agent_factory=self.agent_factory,
            )
        else:
            self.agent_factory = None
            self.context_builder = None

    def set_show_internal_thoughts(self, show: bool) -> None:
        """
        Set whether to show internal thoughts.

        Args:
            show: Whether to show internal thoughts
        """
        self.show_internal_thoughts = show
        if self.agent_factory:
            self.agent_factory.set_show_internal_thoughts(show)

    def parse_scene_response(self, response: str) -> ParsedScene:
        """
        Parse AI response to extract scene and choices.

        Args:
            response: Raw AI response

        Returns:
            ParsedScene with scene content and choices

        Raises:
            StoryGenerationError: If parsing fails
        """
        # Try to extract numbered choices
        choice_pattern = SceneParsingConfig.choice_pattern
        lines = response.split("\n")

        scene_parts = []
        choices = []
        found_choices = False

        for line in lines:
            match = re.match(choice_pattern, line.strip())
            if match:
                found_choices = True
                choices.append(match.group(1).strip())
            elif not found_choices:
                scene_parts.append(line)

        scene_content = "\n".join(scene_parts).strip()

        if not choices:
            raise StoryGenerationError(
                "parsing response",
                error_details="No choices found in response"
            )

        if len(choices) != SceneParsingConfig.expected_choices:
            raise StoryGenerationError(
                "parsing response",
                error_details=f"Expected {SceneParsingConfig.expected_choices} choices, found {len(choices)}"
            )

        return ParsedScene(scene_content=scene_content, choices=choices)

    async def generate_initial_scene(
        self,
        story_id: int,
        story_setting: str,
        user_instructions: Optional[str] = None,
    ) -> GeneratedScene:
        """
        Generate the initial scene for a new story.

        Args:
            story_id: Story ID
            story_setting: Story setting/premise
            user_instructions: Optional user instructions

        Returns:
            GeneratedScene with content and choices

        Raises:
            StoryGenerationError: If generation fails
        """
        try:
            # Validate parameters
            validated_story_id = validate_id(story_id, field_name="story_id")
            validated_setting = validate_story_setting(story_setting)

            if user_instructions:
                user_instructions = validate_string(
                    user_instructions,
                    field_name="user_instructions",
                    min_length=1,
                    max_length=500,
                    strip_whitespace=True,
                )

            # Build prompt
            prompt = self.prompt_builder.build_initial_scene_prompt(
                story_setting=validated_setting,
                user_instructions=user_instructions,
            )

            system_prompt = self.prompt_builder.build_system_prompt()

            # Generate response
            response = await self.ollama.generate_with_retry(
                prompt=prompt,
                system_prompt=system_prompt,
            )

            # Parse response
            parsed = self.parse_scene_response(response)

            # Save to database
            scene_id = await self._save_scene(
                story_id=validated_story_id,
                parent_scene_id=None,
                scene_number=1,
                content=parsed.scene_content,
                choices=parsed.choices,
                raw_response=response,
            )

            # Extract and create characters if using agents
            if self.use_agents:
                async with self.session_factory() as session:
                    await self.extract_and_create_characters(
                        scene_id=scene_id,
                        scene_content=parsed.scene_content,
                        session=session,
                    )

            return GeneratedScene(
                id=scene_id,
                content=parsed.scene_content,
                choices=parsed.choices,
                raw_response=response,
            )

        except Exception as e:
            raise StoryGenerationError("generating initial scene", story_id=story_id, error_details=str(e)) from e

    async def generate_next_scene(
        self,
        story_id: int,
        choice: int,
        user_instruction: Optional[str] = None,
    ) -> GeneratedScene:
        """
        Generate the next scene based on player's choice.

        Args:
            story_id: Story ID
            choice: Selected choice number (1-3)
            user_instruction: Optional user instruction

        Returns:
            GeneratedScene with content and choices

        Raises:
            StoryGenerationError: If generation fails
        """
        try:
            # Validate parameters
            validated_story_id = validate_id(story_id, field_name="story_id")
            validated_choice = validate_choice(choice, field_name="choice")

            if user_instruction:
                user_instruction = validate_string(
                    user_instruction,
                    field_name="user_instruction",
                    min_length=1,
                    max_length=500,
                    strip_whitespace=True,
                )

            # Load current state
            state = await self.state_manager.load_story(validated_story_id)

            if not state.current_scene_id:
                raise StoryGenerationError(
                    "loading story state",
                    story_id=story_id,
                    error_details="No current scene found"
                )

            # Get current scene and recent scenes
            async with self.session_factory() as session:
                # Get current scene
                result = await session.execute(
                    select(Scene).where(Scene.id == state.current_scene_id)
                )
                current_scene = result.scalar_one_or_none()

                if not current_scene:
                    raise StoryGenerationError(
                        "loading scene",
                        story_id=story_id,
                        scene_id=state.current_scene_id,
                        error_details="Current scene not found"
                    )

                # Get selected choice
                result = await session.execute(
                    select(Choice)
                    .where(
                        Choice.scene_id == state.current_scene_id,
                        Choice.choice_number == validated_choice,
                    )
                )
                selected_choice = result.scalar_one_or_none()

                choice_text = selected_choice.content if selected_choice else f"Choice {validated_choice}"

                # Get recent scenes for context
                result = await session.execute(
                    select(Scene)
                    .where(Scene.story_id == story_id)
                    .order_by(Scene.scene_number.desc())
                    .limit(StoryConstants.DEFAULT_RECENT_SCENES_LIMIT)
                )
                recent_scenes = list(reversed(result.scalars().all()))

                # Get characters in current scene if using agents
                character_actions = []
                if self.use_agents:
                    character_actions = await self._generate_character_autonomous_actions(
                        session=session,
                        story_id=story_id,
                        scene_id=state.current_scene_id,
                        situation=f"Player chose: {choice_text}",
                    )

            # Build context
            current_scene_content = f"You chose: {choice_text}\n\n{current_scene.content}"

            # Add character actions if any
            if character_actions:
                current_scene_content += "\n\nCharacter Actions:\n"
                for action in character_actions:
                    # Include internal thoughts if enabled
                    action_text = action['action']
                    if 'internal_thought' in action and self.show_internal_thoughts:
                        action_text += f"\n  *Internal: {action['internal_thought']}*"
                    current_scene_content += f"- {action['character_name']}: {action_text[:StoryConstants.ACTION_PREVIEW_LENGTH]}...\n"

            recent_scene_summaries = [scene.content for scene in recent_scenes]

            # Build prompt
            prompt = self.prompt_builder.build_scene_prompt(
                current_scene_content=current_scene_content,
                recent_scenes=recent_scene_summaries,
                user_instruction=user_instruction,
            )

            system_prompt = self.prompt_builder.build_system_prompt()

            # Generate response
            response = await self.ollama.generate_with_retry(
                prompt=prompt,
                system_prompt=system_prompt,
            )

            # Parse response
            parsed = self.parse_scene_response(response)

            # Save to database
            scene_id = await self._save_scene(
                story_id=validated_story_id,
                parent_scene_id=state.current_scene_id,
                scene_number=state.scene_number + 1,
                content=parsed.scene_content,
                choices=parsed.choices,
                raw_response=response,
            )

            # Extract and create characters if using agents
            if self.use_agents:
                async with self.session_factory() as session:
                    await self.extract_and_create_characters(
                        scene_id=scene_id,
                        scene_content=parsed.scene_content,
                        session=session,
                    )

            return GeneratedScene(
                id=scene_id,
                content=parsed.scene_content,
                choices=parsed.choices,
                raw_response=response,
                character_actions=character_actions if character_actions else None,
            )

        except Exception as e:
            raise StoryGenerationError("generating next scene", story_id=story_id, error_details=str(e)) from e

    async def _generate_character_autonomous_actions(
        self,
        session: AsyncSession,
        story_id: int,
        scene_id: int,
        situation: str,
    ) -> List[dict]:
        """
        Generate autonomous actions for characters in the scene.

        Args:
            session: Database session
            story_id: Story ID
            scene_id: Current scene ID
            situation: Current situation

        Returns:
            List of character actions with optional internal thoughts
        """
        if not self.use_agents or not self.agent_factory:
            return []

        try:
            # Get characters in scene
            from src.database.models import SceneCharacter

            result = await session.execute(
                select(SceneCharacter)
                .where(SceneCharacter.scene_id == scene_id)
                .order_by(SceneCharacter.importance.desc())
            )

            scene_characters = result.scalars().all()

            if not scene_characters:
                return []

            # Create agents for characters
            character_ids = [sc.character_id for sc in scene_characters]
            agents = await self.agent_factory.create_agents_for_scene(
                character_ids=character_ids,
                session=session,
                story_id=story_id,
                show_internal_thoughts=self.show_internal_thoughts,
            )

            # Get autonomous actions from each character
            actions = []
            for char_id, agent in agents.items():
                try:
                    # Randomly decide if character acts (not everyone acts every time)
                    import random
                    if random.random() < StoryConstants.AUTONOMOUS_ACTION_PROBABILITY:
                        action_result = await agent.autonomous_action(
                            situation=situation,
                            other_characters_present=character_ids,
                        )
                        actions.append(action_result)
                except Exception as e:
                    # If one character fails, continue with others
                    logger.warning("Character %s autonomous action failed: %s", char_id, e)
                    continue

            return actions

        except Exception as e:
            logger.warning("Autonomous action generation failed: %s", e)
            return []

    async def _save_scene(
        self,
        story_id: int,
        parent_scene_id: Optional[int],
        scene_number: int,
        content: str,
        choices: List[str],
        raw_response: str,
    ) -> int:
        """
        Save scene and choices to database.

        Uses SELECT FOR UPDATE on the parent scene to prevent race conditions
        when multiple processes try to add scenes to the same story simultaneously.

        Args:
            story_id: Story ID
            parent_scene_id: Parent scene ID
            scene_number: Scene number
            content: Scene content
            choices: List of 3 choices
            raw_response: Raw AI response

        Returns:
            Scene ID
        """
        # Validate parameters
        validated_story_id = validate_id(story_id, field_name="story_id")
        validated_scene_number = validate_id(scene_number, field_name="scene_number", min_value=1)
        validated_content = validate_content(content, field_name="content")
        validated_choices = validate_list(
            choices,
            field_name="choices",
            min_length=1,
            max_length=5,
            item_validator=lambda x: validate_string(
                x, field_name="choice", min_length=1, max_length=200
            )
        )

        async with self.session_factory() as session:
            try:
                # Lock the story row to prevent concurrent scene additions
                # This ensures scene_number uniqueness and proper sequential ordering
                from sqlalchemy import select as sqlalchemy_select
                from src.database.models import Story as StoryModel

                result = await session.execute(
                    sqlalchemy_select(StoryModel)
                    .where(StoryModel.id == validated_story_id)
                    .with_for_update()
                )
                story = result.scalar_one_or_none()

                if not story:
                    raise StoryGenerationError(
                        "loading story",
                        story_id=validated_story_id,
                        error_details=f"Story {validated_story_id} not found"
                    )

                # Double-check scene_number doesn't already exist (defensive programming)
                existing_check = await session.execute(
                    sqlalchemy_select(Scene)
                    .where(
                        Scene.story_id == validated_story_id,
                        Scene.scene_number == validated_scene_number
                    )
                )
                if existing_check.scalar_one_or_none():
                    logger.warning(
                        "Scene %d already exists for story %d, may indicate race condition",
                        validated_scene_number,
                        validated_story_id
                    )
                    raise StoryGenerationError(
                        f"Scene {validated_scene_number} already exists for story {validated_story_id}"
                    )

                # Generate embedding for scene content
                embedding = await self.encoder.encode_async(validated_content)

                # Create scene
                scene = Scene(
                    story_id=validated_story_id,
                    parent_scene_id=parent_scene_id,
                    scene_number=validated_scene_number,
                    content=validated_content,
                    raw_response=raw_response,
                    choices_generated=validated_choices,
                    embedding=embedding,
                )

                session.add(scene)
                await session.flush()

                logger.debug(
                    "Created scene %d for story %d with scene_number=%d",
                    scene.id,
                    validated_story_id,
                    validated_scene_number
                )

                # Create choices
                for i, choice_text in enumerate(validated_choices, start=1):
                    choice = Choice(
                        scene_id=scene.id,
                        choice_number=i,
                        content=choice_text,
                    )
                    session.add(choice)

                # Update story timestamp atomically
                story.updated_at = datetime.now(timezone.utc)

                await session.commit()

                logger.info(
                    "Successfully saved scene %d (number %d) for story %d",
                    scene.id,
                    validated_scene_number,
                    validated_story_id
                )

                return scene.id

            except StoryGenerationError:
                await session.rollback()
                raise
            except Exception as e:
                await session.rollback()
                logger.error(
                    "Failed to save scene for story %d: %s",
                    validated_story_id,
                    e,
                    exc_info=True
                )
                raise StoryGenerationError("saving scene", story_id=validated_story_id, scene_id=scene_id, error_details=str(e)) from e

    async def extract_and_create_characters(
        self,
        scene_id: int,
        scene_content: str,
        session: AsyncSession,
    ) -> List[Character]:
        """
        Extract character mentions from scene and create character records.

        Args:
            scene_id: Scene ID
            scene_content: Scene content
            session: Database session

        Returns:
            List of created characters
        """
        if not self.use_agents or not self.agent_factory:
            return []

        # Use AI to extract characters
        prompt = f"""Analyze this scene and extract all named characters.

Scene:
{scene_content}

For each character found, provide:
1. Name
2. Brief description (appearance, role)
3. Personality traits (if discernible)
4. Goals/motivations (if evident)

Format as JSON:
{{
    "characters": [
        {{
            "name": "Character Name",
            "description": "Description",
            "personality": "Personality traits",
            "goals": "Goals"
        }}
    ]
}}

Only include clearly named characters, not generic references."""

        try:
            response = await self.ollama.generate_with_retry(
                prompt=prompt,
                system_prompt="You are a character extraction assistant. Extract character information accurately.",
                temperature=LLMConstants.EXTRACTION_TEMPERATURE,
            )

            # Parse JSON response
            import json

            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                return []

            characters_created = []

            for char_data in data.get("characters", []):
                try:
                    character = await self.agent_factory.get_or_create_character(
                        session=session,
                        name=char_data["name"],
                        description=char_data.get("description"),
                        personality=char_data.get("personality"),
                        goals=char_data.get("goals"),
                        first_scene_id=scene_id,
                    )

                    # Link character to scene
                    from src.database.models import SceneCharacter

                    scene_char = SceneCharacter(
                        scene_id=scene_id,
                        character_id=character.id,
                        role=CharacterExtractionConfig.default_role,
                        importance=CharacterExtractionConfig.default_importance,
                    )
                    session.add(scene_char)

                    characters_created.append(character)

                except Exception as e:
                    # Continue with other characters if one fails
                    logger.warning("Failed to create character: %s", e)
                    continue

            await session.commit()
            return characters_created

        except Exception as e:
            logger.warning("Character extraction failed: %s", e)
            return []

    async def generate_scene_with_agents(
        self,
        story_id: int,
        situation: str,
        session: AsyncSession,
        include_characters: bool = True,
    ) -> GeneratedScene:
        """
        Generate a scene with character agent integration.

        Args:
            story_id: Story ID
            situation: Current situation
            session: Database session
            include_characters: Whether to include character responses

        Returns:
            GeneratedScene with enhanced content
        """
        if not self.use_agents or not self.context_builder:
            # Fall back to basic generation
            return await self.generate_next_scene(story_id, 1, situation)

        # Build full context
        context = await self.context_builder.build_context(
            session=session,
            story_id=story_id,
            user_instruction=situation,
            use_agents=include_characters,
        )

        # Build prompt with character contexts
        prompt_parts = []

        if context.recent_scenes:
            prompt_parts.append("## Story So Far\n")
            for i, scene in enumerate(context.recent_scenes[-3:], 1):
                prompt_parts.append(f"{i}. {scene[:200]}...\n")

        if context.character_contexts:
            prompt_parts.append("\n## Character Perspectives\n")
            for name, perspective in context.character_contexts.items():
                prompt_parts.append(f"**{name}:** {perspective[:200]}...\n")

        prompt_parts.append(f"\n## Current Situation\n{situation}\n")
        prompt_parts.append(
            "\nContinue the story with vivid details. "
            "If characters are present, incorporate their perspectives and responses."
        )

        prompt = "".join(prompt_parts)

        # Generate response
        response = await self.ollama.generate_with_retry(
            prompt=prompt,
            system_prompt=self.prompt_builder.build_system_prompt(),
        )

        # Parse and save
        parsed = self.parse_scene_response(response)

        state = await self.state_manager.load_story(story_id)
        scene_id = await self._save_scene(
            story_id=story_id,
            parent_scene_id=state.current_scene_id,
            scene_number=state.scene_number + 1,
            content=parsed.scene_content,
            choices=parsed.choices,
            raw_response=response,
        )

        # Extract characters
        await self.extract_and_create_characters(
            scene_id=scene_id,
            scene_content=parsed.scene_content,
            session=session,
        )

        return GeneratedScene(
            id=scene_id,
            content=parsed.scene_content,
            choices=parsed.choices,
            raw_response=response,
        )
