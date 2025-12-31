"""Tests for story export and import functionality."""

import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from src.story.io import StoryExporter, StoryImporter
from src.core.exceptions import StoryGenerationError
from src.database.models import (
    Story,
    Scene,
    Choice,
    Character,
    SceneCharacter,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_session():
    """Mock database session."""
    session = MagicMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def mock_story():
    """Mock story object."""
    story = MagicMock()
    story.id = 1
    story.title = "The Adventure Begins"
    story.created_at = datetime(2024, 1, 1, 12, 0, 0)
    story.updated_at = datetime(2024, 1, 1, 13, 0, 0)
    story.system_prompt = "You are a storyteller."
    return story


@pytest.fixture
def mock_scenes():
    """Mock scene objects."""
    scene1 = MagicMock()
    scene1.id = 1
    scene1.story_id = 1
    scene1.parent_scene_id = None
    scene1.scene_number = 1
    scene1.content = "You stand at a crossroads."
    scene1.raw_response = "You stand at a crossroads.\n1. Go left\n2. Go right\n3. Stay"
    scene1.choices_generated = None
    scene1.created_at = datetime(2024, 1, 1, 12, 30, 0)
    scene1.meta = {"genre": "fantasy"}

    scene2 = MagicMock()
    scene2.id = 2
    scene2.story_id = 1
    scene2.parent_scene_id = 1
    scene2.scene_number = 2
    scene2.content = "You walk down the left path."
    scene2.raw_response = "You walk left.\n1. Continue\n2. Turn back\n3. Rest"
    scene2.choices_generated = None
    scene2.created_at = datetime(2024, 1, 1, 12, 45, 0)
    scene2.meta = {}

    return [scene1, scene2]


@pytest.fixture
def mock_choices():
    """Mock choice objects."""
    choice1 = MagicMock()
    choice1.id = 1
    choice1.scene_id = 1
    choice1.choice_number = 1
    choice1.content = "Go left"

    choice2 = MagicMock()
    choice2.id = 2
    choice2.scene_id = 1
    choice2.choice_number = 2
    choice2.content = "Go right"

    choice3 = MagicMock()
    choice3.id = 3
    choice3.scene_id = 1
    choice3.choice_number = 3
    choice3.content = "Stay put"

    return [choice1, choice2, choice3]


@pytest.fixture
def mock_characters():
    """Mock character objects."""
    char1 = MagicMock()
    char1.id = 1
    char1.name = "Sreykeo"
    char1.description = "A young village girl"
    char1.personality = "Curious and brave"
    char1.goals = "Explore the world"
    char1.background = "Born in a small village"
    char1.meta = {}

    char2 = MagicMock()
    char2.id = 2
    char2.name = "Kai"
    char2.description = "Sreykeo's brother"
    char2.personality = "Protective and strong"
    char2.goals = "Keep his sister safe"
    char2.background = "A village farmer"
    char2.meta = {}

    return [char1, char2]


@pytest.fixture
def mock_scene_characters():
    """Mock scene character associations."""
    sc1_scene = MagicMock()
    sc1_scene.id = 1
    sc1_scene.name = "Sreykeo"

    sc1 = MagicMock()
    sc1.role = "protagonist"
    sc1.importance = 5

    sc2_scene = MagicMock()
    sc2_scene.id = 2
    sc2_scene.name = "Kai"

    sc2 = MagicMock()
    sc2.role = "companion"
    sc2.importance = 3

    return [(sc1, sc1_scene), (sc2, sc2_scene)]


@pytest.fixture
def sample_json_data():
    """Sample JSON export data."""
    return {
        "story": {
            "id": 1,
            "title": "The Adventure Begins",
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T13:00:00",
            "system_prompt": "You are a storyteller.",
        },
        "scenes": [
            {
                "id": 1,
                "scene_number": 1,
                "content": "You stand at a crossroads.",
                "raw_response": "You stand at a crossroads.\n1. Go left",
                "choices": [
                    {"number": 1, "content": "Go left"},
                    {"number": 2, "content": "Go right"},
                    {"number": 3, "content": "Stay"},
                ],
                "characters": [
                    {"id": 1, "name": "Sreykeo", "role": "protagonist", "importance": 5},
                ],
                "metadata": {"genre": "fantasy"},
                "created_at": "2024-01-01T12:30:00",
            }
        ],
        "characters": [
            {
                "id": 1,
                "name": "Sreykeo",
                "description": "A young village girl",
                "personality": "Curious and brave",
                "goals": "Explore the world",
                "background": "Born in a small village",
                "metadata": {},
            }
        ],
        "exported_at": "2024-01-01T14:00:00",
        "format_version": "1.0",
    }


@pytest.fixture
def temp_json_file(tmp_path, sample_json_data):
    """Create temporary JSON file for import tests."""
    json_file = tmp_path / "story_export.json"
    json_file.write_text(json.dumps(sample_json_data, indent=2), encoding="utf-8")
    return json_file


@pytest.fixture
def temp_markdown_file(tmp_path):
    """Create temporary markdown file."""
    md_file = tmp_path / "story.md"
    return md_file


# =============================================================================
# StoryExporter Tests - JSON Export
# =============================================================================

@pytest.mark.asyncio
class TestStoryExporterExportToJson:
    """Tests for export_to_json method."""

    async def test_export_to_json_success(
        self, mock_session, mock_story, mock_scenes, mock_choices, mock_scene_characters
    ):
        """Test successful JSON export."""
        # Setup mock responses
        # Story query
        story_result = MagicMock()
        story_result.scalar_one_or_none.return_value = mock_story

        # Scenes query
        scenes_result = MagicMock()
        scenes_result.scalars.return_value.all.return_value = mock_scenes

        # Choices query (for scene 1)
        choices_result = MagicMock()
        choices_result.scalars.return_value.all.return_value = mock_choices

        # Scene characters query
        scene_chars_result = MagicMock()
        scene_chars_result.all.return_value = mock_scene_characters

        # Character query
        char_result = MagicMock()
        char_result.scalar_one_or_none.side_effect = [
            mock_scene_characters[0][1],
            mock_scene_characters[1][1],
        ]

        mock_session.execute.side_effect = [
            story_result,
            scenes_result,
            choices_result,
            scene_chars_result,
            char_result,
            char_result,
        ]

        exporter = StoryExporter()
        json_str = await exporter.export_to_json(mock_session, story_id=1)

        # Parse result
        data = json.loads(json_str)

        assert data["story"]["id"] == 1
        assert data["story"]["title"] == "The Adventure Begins"
        assert len(data["scenes"]) == 2
        assert data["scenes"][0]["content"] == "You stand at a crossroads."
        assert len(data["scenes"][0]["choices"]) == 3
        assert data["format_version"] == "1.0"

    async def test_export_to_json_with_file_output(
        self, mock_session, mock_story, mock_scenes, mock_choices, mock_scene_characters, tmp_path
    ):
        """Test JSON export to file."""
        output_path = tmp_path / "output.json"

        # Setup mocks
        story_result = MagicMock()
        story_result.scalar_one_or_none.return_value = mock_story

        scenes_result = MagicMock()
        scenes_result.scalars.return_value.all.return_value = mock_scenes

        choices_result = MagicMock()
        choices_result.scalars.return_value.all.return_value = mock_choices

        scene_chars_result = MagicMock()
        scene_chars_result.all.return_value = mock_scene_characters

        char_result = MagicMock()
        char_result.scalar_one_or_none.side_effect = [
            mock_scene_characters[0][1],
            mock_scene_characters[1][1],
        ]

        mock_session.execute.side_effect = [
            story_result,
            scenes_result,
            choices_result,
            scene_chars_result,
            char_result,
            char_result,
        ]

        exporter = StoryExporter()
        result_path = await exporter.export_to_json(
            mock_session, story_id=1, output_path=output_path
        )

        assert result_path == str(output_path)
        assert output_path.exists()

        # Verify content
        data = json.loads(output_path.read_text())
        assert data["story"]["title"] == "The Adventure Begins"

    async def test_export_to_json_story_not_found(self, mock_session):
        """Test export with non-existent story."""
        story_result = MagicMock()
        story_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = story_result

        exporter = StoryExporter()

        with pytest.raises(StoryGenerationError, match="Story 1 not found"):
            await exporter.export_to_json(mock_session, story_id=1)

    async def test_export_to_json_with_empty_scenes(
        self, mock_session, mock_story
    ):
        """Test export with story that has no scenes."""
        story_result = MagicMock()
        story_result.scalar_one_or_none.return_value = mock_story

        scenes_result = MagicMock()
        scenes_result.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [story_result, scenes_result]

        exporter = StoryExporter()
        json_str = await exporter.export_to_json(mock_session, story_id=1)

        data = json.loads(json_str)
        assert len(data["scenes"]) == 0

    async def test_export_to_json_database_error(self, mock_session):
        """Test handling of database error."""
        mock_session.execute.side_effect = Exception("Database connection lost")

        exporter = StoryExporter()

        with pytest.raises(StoryGenerationError, match="Failed to export story"):
            await exporter.export_to_json(mock_session, story_id=1)


# =============================================================================
# StoryExporter Tests - Markdown Export
# =============================================================================

@pytest.mark.asyncio
class TestStoryExporterExportToMarkdown:
    """Tests for export_to_markdown method."""

    async def test_export_to_markdown_success(
        self, mock_session, mock_story, mock_scenes, mock_choices, temp_markdown_file
    ):
        """Test successful markdown export."""
        # Setup mocks
        story_result = MagicMock()
        story_result.scalar_one_or_none.return_value = mock_story

        scenes_result = MagicMock()
        scenes_result.scalars.return_value.all.return_value = mock_scenes

        choices_result = MagicMock()
        choices_result.scalars.return_value.all.return_value = mock_choices

        mock_session.execute.side_effect = [
            story_result,
            scenes_result,
            choices_result,
            choices_result,  # For second scene
        ]

        exporter = StoryExporter()
        result_path = await exporter.export_to_markdown(
            mock_session, story_id=1, output_path=temp_markdown_file
        )

        assert result_path == str(temp_markdown_file)
        assert temp_markdown_file.exists()

        # Verify content
        content = temp_markdown_file.read_text()
        assert "# The Adventure Begins" in content
        assert "## Scene 1" in content
        assert "You stand at a crossroads." in content
        assert "**Choices:**" in content
        assert "1. Go left" in content

    async def test_export_to_markdown_story_not_found(self, mock_session, temp_markdown_file):
        """Test markdown export with non-existent story."""
        story_result = MagicMock()
        story_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = story_result

        exporter = StoryExporter()

        with pytest.raises(StoryGenerationError, match="Story 1 not found"):
            await exporter.export_to_markdown(
                mock_session, story_id=1, output_path=temp_markdown_file
            )

    async def test_export_to_markdown_format(self, mock_session, mock_story, mock_scenes, mock_choices, temp_markdown_file):
        """Test markdown export format."""
        story_result = MagicMock()
        story_result.scalar_one_or_none.return_value = mock_story

        scenes_result = MagicMock()
        scenes_result.scalars.return_value.all.return_value = mock_scenes

        choices_result = MagicMock()
        choices_result.scalars.return_value.all.return_value = mock_choices

        mock_session.execute.side_effect = [
            story_result,
            scenes_result,
            choices_result,
            choices_result,
        ]

        exporter = StoryExporter()
        await exporter.export_to_markdown(
            mock_session, story_id=1, output_path=temp_markdown_file
        )

        content = temp_markdown_file.read_text()

        # Check markdown structure
        assert content.startswith("#")
        assert "---" in content
        assert "## Scene" in content
        assert "###" not in content  # Should not have deeper nesting

    async def test_export_to_markdown_error(self, mock_session, temp_markdown_file):
        """Test handling of error in markdown export."""
        mock_session.execute.side_effect = Exception("Database error")

        exporter = StoryExporter()

        with pytest.raises(StoryGenerationError, match="Failed to export to markdown"):
            await exporter.export_to_markdown(
                mock_session, story_id=1, output_path=temp_markdown_file
            )


# =============================================================================
# StoryImporter Tests - JSON Import
# =============================================================================

@pytest.mark.asyncio
class TestStoryImporterImportFromJson:
    """Tests for import_from_json method."""

    async def test_import_from_json_create_new(self, mock_session, temp_json_file):
        """Test importing JSON to create new story."""
        importer = StoryImporter()
        story_id = await importer.import_from_json(
            mock_session, temp_json_file, create_new_story=True
        )

        # Verify story was created
        assert mock_session.add.call_count > 0
        mock_session.commit.assert_called_once()

    async def test_import_from_json_update_existing(self, mock_session, temp_json_file):
        """Test importing JSON to update existing story."""
        # Setup mock for existing story
        existing_story = MagicMock()
        existing_story.id = 1

        story_result = MagicMock()
        story_result.scalar_one_or_none.return_value = existing_story

        mock_session.execute.return_value = story_result

        importer = StoryImporter()
        story_id = await importer.import_from_json(
            mock_session, temp_json_file, create_new_story=False
        )

        assert story_id == 1

    async def test_import_from_json_file_not_found(self, mock_session):
        """Test import from non-existent file."""
        importer = StoryImporter()

        with pytest.raises(StoryGenerationError, match="File not found"):
            await importer.import_from_json(
                mock_session, Path("/nonexistent/file.json")
            )

    async def test_import_from_json_unsupported_version(self, mock_session, tmp_path):
        """Test import with unsupported format version."""
        # Create JSON with unsupported version
        data = {
            "story": {"id": 1, "title": "Test"},
            "scenes": [],
            "characters": [],
            "format_version": "2.0",  # Unsupported
        }

        json_file = tmp_path / "wrong_version.json"
        json_file.write_text(json.dumps(data))

        importer = StoryImporter()

        with pytest.raises(StoryGenerationError, match="Unsupported format version"):
            await importer.import_from_json(mock_session, json_file)

    async def test_import_from_json_update_nonexistent(self, mock_session, temp_json_file):
        """Test updating story that doesn't exist."""
        story_result = MagicMock()
        story_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = story_result

        importer = StoryImporter()

        with pytest.raises(StoryGenerationError, match="Story 1 not found for update"):
            await importer.import_from_json(
                mock_session, temp_json_file, create_new_story=False
            )

    async def test_import_from_json_with_characters(self, mock_session, tmp_path):
        """Test import with characters."""
        data = {
            "story": {"id": 1, "title": "Test"},
            "scenes": [
                {
                    "id": 1,
                    "scene_number": 1,
                    "content": "Scene content",
                    "raw_response": None,
                    "choices": [],
                    "characters": [{"id": 1, "name": "Hero", "role": "protagonist", "importance": 5}],
                    "metadata": {},
                }
            ],
            "characters": [
                {
                    "id": 1,
                    "name": "Hero",
                    "description": "A brave hero",
                    "personality": "Brave",
                    "goals": "Save the world",
                    "background": "From a small village",
                    "metadata": {},
                }
            ],
            "format_version": "1.0",
        }

        json_file = tmp_path / "with_characters.json"
        json_file.write_text(json.dumps(data))

        importer = StoryImporter()
        await importer.import_from_json(mock_session, json_file, create_new_story=True)

        # Verify characters were added
        assert mock_session.add.call_count > 0

    async def test_import_from_json_rollback_on_error(self, mock_session, temp_json_file):
        """Test that import rolls back on error."""
        mock_session.commit = AsyncMock(side_effect=Exception("Database error"))

        importer = StoryImporter()

        with pytest.raises(StoryGenerationError, match="Failed to import story"):
            await importer.import_from_json(mock_session, temp_json_file)

        mock_session.rollback.assert_called_once()


# =============================================================================
# StoryImporter Tests - Text Import
# =============================================================================

@pytest.mark.asyncio
class TestStoryImporterImportFromText:
    """Tests for import_from_text method."""

    async def test_import_from_text_success(self, mock_session):
        """Test successful text import."""
        importer = StoryImporter()
        story_id = await importer.import_from_text(
            mock_session,
            title="My Story",
            content="This is the beginning of a great adventure.",
        )

        assert isinstance(story_id, int)
        assert mock_session.add.call_count > 0  # Story + Scene + 3 Choices

        # Verify choices were created
        add_calls = mock_session.add.call_args_list
        choice_contents = [call[0][0].content for call in add_calls[2:]]  # Skip story and scene
        assert all("Continue the story" in content for content in choice_contents)

    async def test_import_from_text_creates_default_choices(self, mock_session):
        """Test that text import creates 3 default choices."""
        importer = StoryImporter()
        await importer.import_from_text(
            mock_session,
            title="Test",
            content="Content",
        )

        # Should have: 1 story + 1 scene + 3 choices = 5 add calls
        assert mock_session.add.call_count == 5

    async def test_import_from_text_long_content(self, mock_session):
        """Test import with long text content."""
        long_content = "This is a very long story. " * 100

        importer = StoryImporter()
        story_id = await importer.import_from_text(
            mock_session,
            title="Epic Story",
            content=long_content,
        )

        assert isinstance(story_id, int)

    async def test_import_from_text_unicode(self, mock_session):
        """Test import with unicode content."""
        unicode_content = 'The dragon said "Hello, world!" in Cambodian: "'

        importer = StoryImporter()
        story_id = await importer.import_from_text(
            mock_session,
            title="Unicode Story",
            content=unicode_content,
        )

        assert isinstance(story_id, int)

    async def test_import_from_text_rollback_on_error(self, mock_session):
        """Test that text import rolls back on error."""
        mock_session.commit = AsyncMock(side_effect=Exception("Database error"))

        importer = StoryImporter()

        with pytest.raises(StoryGenerationError, match="Failed to import text"):
            await importer.import_from_text(mock_session, "Test", "Content")

        mock_session.rollback.assert_called_once()

    async def test_import_from_text_empty_content(self, mock_session):
        """Test import with empty content."""
        importer = StoryImporter()
        story_id = await importer.import_from_text(
            mock_session,
            title="Empty Story",
            content="",
        )

        assert isinstance(story_id, int)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

@pytest.mark.asyncio
class TestStoryIOEdgeCases:
    """Tests for edge cases in story I/O."""

    async def test_export_import_roundtrip(self, mock_session, tmp_path):
        """Test exporting and re-importing a story."""
        # This would require a more complex setup with actual mock data
        # For now, just verify the workflow doesn't crash
        exporter = StoryExporter()
        importer = StoryImporter()

        # Minimal test
        json_file = tmp_path / "test.json"

        # Create minimal valid export data
        data = {
            "story": {"id": 1, "title": "Test", "created_at": "2024-01-01T00:00:00", "updated_at": "2024-01-01T00:00:00"},
            "scenes": [],
            "characters": [],
            "format_version": "1.0",
        }
        json_file.write_text(json.dumps(data))

        story_id = await importer.import_from_json(mock_session, json_file)
        assert story_id is not None

    async def test_export_with_none_fields(self, mock_session, mock_story):
        """Test export with None/optional fields."""
        mock_story.system_prompt = None

        story_result = MagicMock()
        story_result.scalar_one_or_none.return_value = mock_story

        scenes_result = MagicMock()
        scenes_result.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [story_result, scenes_result]

        exporter = StoryExporter()
        json_str = await exporter.export_to_json(mock_session, story_id=1)

        data = json.loads(json_str)
        assert data["story"]["system_prompt"] is None
