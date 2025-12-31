"""Tests for story generator module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from src.llm.story_generator import StoryGenerator, GeneratedScene, ParsedScene
from src.core.exceptions import StoryGenerationError


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client."""
    client = MagicMock()
    client.generate_with_retry = AsyncMock(return_value="Test response")
    return client


@pytest.fixture
def mock_prompt_builder():
    """Mock prompt builder."""
    builder = MagicMock()
    builder.build_system_prompt.return_value = "System prompt"
    builder.build_scene_prompt.return_value = "Scene prompt"
    builder.build_initial_scene_prompt.return_value = "Initial scene prompt"
    builder.build_character_system_prompt.return_value = "Character prompt"
    return builder


@pytest.fixture
def mock_encoder():
    """Mock embedding encoder."""
    encoder = MagicMock()
    encoder.encode_async = AsyncMock(return_value=[0.1] * 384)
    return encoder


@pytest.fixture
def mock_session_factory():
    """Mock session factory."""
    session = MagicMock(spec=AsyncSession)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    factory = MagicMock()
    factory.return_value.__aenter__.return_value = session
    return factory


@pytest.fixture
def story_generator(mock_ollama_client, mock_prompt_builder, mock_encoder, mock_session_factory):
    """Create story generator instance."""
    generator = StoryGenerator(
        ollama_client=mock_ollama_client,
        prompt_builder=mock_prompt_builder,
        encoder=mock_encoder,
        session_factory=mock_session_factory,
        use_agents=False,  # Disable agents for basic tests
    )
    return generator


class TestParseSceneResponse:
    """Tests for parse_scene_response method."""

    def test_parse_valid_response(self, story_generator):
        """Test parsing valid response with 3 choices."""
        response = """The sun sets over the horizon.

1. Go inside
2. Stay outside
3. Walk away"""

        result = story_generator.parse_scene_response(response)

        assert result.scene_content == "The sun sets over the horizon."
        assert result.choices == ["Go inside", "Stay outside", "Walk away"]

    def test_parse_response_with_context(self, story_generator):
        """Test parsing response with scene context before choices."""
        response = """You stand at the crossroads. The wind blows cold.
People pass by, ignoring you.

1. Follow the path
2. Wait here
3. Turn back"""

        result = story_generator.parse_scene_response(response)

        assert "crossroads" in result.scene_content
        assert len(result.choices) == 3

    def test_parse_response_no_choices_raises_error(self, story_generator):
        """Test that response without choices raises error."""
        response = "Just a simple response with no choices."

        with pytest.raises(StoryGenerationError, match="No choices found"):
            story_generator.parse_scene_response(response)

    def test_parse_response_wrong_number_of_choices_raises_error(self, story_generator):
        """Test that response with wrong number of choices raises error."""
        response = """Scene content here.

1. Choice one
2. Choice two"""

        with pytest.raises(StoryGenerationError, match="Expected 3 choices, found 2"):
            story_generator.parse_scene_response(response)


@pytest.mark.asyncio
class TestGenerateInitialScene:
    """Tests for generate_initial_scene method."""

    async def test_generate_initial_scene_success(
        self, story_generator, mock_ollama_client, mock_session_factory
    ):
        """Test successful initial scene generation."""
        mock_response = """The adventure begins in a small village.

1. Explore the village
2. Visit the tavern
3. Head to the forest"""

        mock_ollama_client.generate_with_retry.return_value = mock_response

        result = await story_generator.generate_initial_scene(
            story_id=1,
            story_setting="A fantasy village adventure",
        )

        assert isinstance(result, GeneratedScene)
        assert "village" in result.content.lower()
        assert len(result.choices) == 3
        assert result.raw_response == mock_response


@pytest.mark.asyncio
class TestGenerateNextScene:
    """Tests for generate_next_scene method."""

    async def test_generate_next_scene_with_choice(
        self, story_generator, mock_ollama_client
    ):
        """Test generating next scene with a choice."""
        # Mock state manager
        story_generator.state_manager.load_story = AsyncMock(
            return_value=MagicMock(
                story_id=1,
                current_scene_id=1,
                scene_number=1,
            )
        )

        # Mock scene and choice queries
        mock_scene = MagicMock()
        mock_scene.content = "Previous scene content"
        mock_choice = MagicMock()
        mock_choice.content = "Go left"

        # Setup session mocks
        async with story_generator.session_factory() as session:
            session.execute = AsyncMock()

            # First call gets scene
            scene_result = MagicMock()
            scene_result.scalar_one_or_none.return_value = mock_scene

            # Second call gets choice
            choice_result = MagicMock()
            choice_result.scalar_one_or_none.return_value = mock_choice

            # Third call gets recent scenes
            scenes_result = MagicMock()
            scenes_result.scalars.return_value.all.return_value = []

            session.execute.side_effect = [
                scene_result,
                choice_result,
                scenes_result,
            ]

        mock_response = """You continue forward.

1. Keep going
2. Stop and rest
3. Look around"""

        mock_ollama_client.generate_with_retry.return_value = mock_response

        result = await story_generator.generate_next_scene(
            story_id=1,
            choice=1,
        )

        assert isinstance(result, GeneratedScene)
        assert len(result.choices) == 3


@pytest.mark.asyncio
class TestCharacterExtraction:
    """Tests for extract_and_create_characters method."""

    async def test_extract_characters_from_scene(
        self, story_generator, mock_ollama_client
    ):
        """Test extracting characters from scene content."""
        scene_content = "Sreykeo walks down the path. Her brother Kai waves from the house."

        # Mock Ollama response with character data
        mock_response = '''```json
{
    "characters": [
        {
            "name": "Sreykeo",
            "description": "A young Cambodian village girl",
            "personality": "Curious and friendly",
            "goals": "Learn about the world"
        },
        {
            "name": "Kai",
            "description": "Sreykeo's older brother",
            "personality": "Protective and hardworking",
            "goals": "Take care of his family"
        }
    ]
}
```'''

        mock_ollama_client.generate_with_retry.return_value = mock_response

        # Mock agent factory
        story_generator.use_agents = True
        story_generator.agent_factory = MagicMock()
        story_generator.agent_factory.get_or_create_character = AsyncMock(
            return_value=MagicMock(id=1, name="Test Character")
        )

        # Mock session
        async with story_generator.session_factory() as session:
            session.add = MagicMock()
            session.commit = AsyncMock()

            result = await story_generator.extract_and_create_characters(
                scene_id=1,
                scene_content=scene_content,
                session=session,
            )

            # Should have called get_or_create_character twice
            assert story_generator.agent_factory.get_or_create_character.call_count == 2


@pytest.mark.asyncio
class TestGenerateSceneWithAgents:
    """Tests for agent-integrated scene generation."""

    async def test_generate_scene_without_agents_fallback(
        self, story_generator, mock_ollama_client
    ):
        """Test that agent generation falls back to basic when agents disabled."""
        story_generator.use_agents = False
        story_generator.context_builder = None

        # Mock the basic generation flow
        story_generator.state_manager.load_story = AsyncMock(
            return_value=MagicMock(
                story_id=1,
                current_scene_id=1,
                scene_number=1,
            )
        )

        async with story_generator.session_factory() as session:
            session.execute = AsyncMock()
            result_mock = MagicMock()
            result_mock.scalar_one_or_none.return_value = MagicMock(content="Test")
            result_mock.scalars.return_value.all.return_value = []
            session.execute.return_value = result_mock

        mock_response = "Test scene\n1. Choice 1\n2. Choice 2\n3. Choice 3"
        mock_ollama_client.generate_with_retry.return_value = mock_response

        result = await story_generator.generate_scene_with_agents(
            story_id=1,
            situation="Test situation",
            session=MagicMock(),
        )

        assert isinstance(result, GeneratedScene)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_parse_empty_response(self, story_generator):
        """Test parsing empty response."""
        with pytest.raises(StoryGenerationError):
            story_generator.parse_scene_response("")

    def test_parse_response_with_no_content_before_choices(self, story_generator):
        """Test parsing response with choices immediately."""
        response = "1. First\n2. Second\n3. Third"

        result = story_generator.parse_scene_response(response)

        assert result.scene_content == ""
        assert len(result.choices) == 3

    @pytest.mark.asyncio
    async def test_save_scene_database_error(
        self, story_generator, mock_session_factory
    ):
        """Test handling database error during scene save."""
        # Mock session that raises exception
        async with story_generator.session_factory() as session:
            session.commit = AsyncMock(side_effect=Exception("Database error"))

        with pytest.raises(StoryGenerationError, match="Failed to save scene"):
            await story_generator._save_scene(
                story_id=1,
                parent_scene_id=None,
                scene_number=1,
                content="Test content",
                choices=["A", "B", "C"],
                raw_response="Raw response",
            )
