"""Story export and import functionality."""

import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from dataclasses import asdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.exceptions import StoryGenerationError
from src.database.models import Story, Scene, Choice, Character, SceneCharacter
from src.core.validation import validate_id, validate_string, validate_file_path, validate_content


class StoryExporter:
    """Export stories to various formats."""

    async def export_to_json(
        self,
        session: AsyncSession,
        validated_story_id: int,
        validated_output_path: Optional[Path] = None,
    ) -> str:
        """
        Export story to JSON format.

        Args:
            session: Database session
            validated_story_id: Story ID to export
            validated_output_path: Optional output file path

        Returns:
            JSON string or file path if validated_output_path provided

        Raises:
            StoryGenerationError: If export fails
        """
        # Validate parameters
        validated_validated_story_id = validate_id(validated_story_id, field_name="validated_story_id")

        if validated_output_path is not None:
            validated_validated_output_path = validate_file_path(
                validated_output_path,
                allowed_extensions=[".json"],
                field_name="validated_output_path"
            )

        try:
            # Get story
            result = await session.execute(
                select(Story).where(Story.id == validated_validated_story_id)
            )
            story = result.scalar_one_or_none()

            if not story:
                raise StoryGenerationError("loading story", validated_story_id=validated_story_id, error_details=f"Story {validated_story_id} not found")

            # Get all scenes
            result = await session.execute(
                select(Scene)
                .where(Scene.validated_story_id == validated_story_id)
                .order_by(Scene.scene_number)
            )
            scenes = result.scalars().all()

            # Build export data
            export_data = {
                "story": {
                    "id": story.id,
                    "title": story.title,
                    "created_at": story.created_at.isoformat(),
                    "updated_at": story.updated_at.isoformat(),
                    "system_prompt": story.system_prompt,
                },
                "scenes": [],
                "characters": [],
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "format_version": "1.0",
            }

            # Export scenes with choices
            for scene in scenes:
                # Get choices for this scene
                result = await session.execute(
                    select(Choice)
                    .where(Choice.scene_id == scene.id)
                    .order_by(Choice.choice_number)
                )
                choices = result.scalars().all()

                # Get characters in this scene
                result = await session.execute(
                    select(SceneCharacter, Character)
                    .join(Character, SceneCharacter.character_id == Character.id)
                    .where(SceneCharacter.scene_id == scene.id)
                )
                scene_characters = result.all()

                scene_data = {
                    "id": scene.id,
                    "scene_number": scene.scene_number,
                    "content": scene.content,
                    "raw_response": scene.raw_response,
                    "choices": [
                        {
                            "number": choice.choice_number,
                            "content": choice.content,
                        }
                        for choice in choices
                    ],
                    "characters": [
                        {
                            "id": char.id,
                            "name": char.name,
                            "role": scene_char.role,
                            "importance": scene_char.importance,
                        }
                        for scene_char, char in scene_characters
                    ],
                    "metadata": scene.meta,
                    "created_at": scene.created_at.isoformat(),
                }

                export_data["scenes"].append(scene_data)

            # Export unique characters
            character_ids = set()
            for scene_data in export_data["scenes"]:
                for char in scene_data["characters"]:
                    character_ids.add(char["id"])

            for char_id in character_ids:
                result = await session.execute(
                    select(Character).where(Character.id == char_id)
                )
                character = result.scalar_one_or_none()

                if character:
                    export_data["characters"].append({
                        "id": character.id,
                        "name": character.name,
                        "description": character.description,
                        "personality": character.personality,
                        "goals": character.goals,
                        "background": character.background,
                        "metadata": character.meta,
                    })

            # Convert to JSON
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

            # Write to file if path provided
            if validated_output_path:
                validated_output_path = Path(validated_output_path)
                validated_output_path.write_text(json_str, encoding="utf-8")
                return str(validated_output_path)

            return json_str

        except Exception as e:
            raise StoryGenerationError("exporting story", error_details=str(e)) from e

    async def export_to_markdown(
        self,
        session: AsyncSession,
        story_id: int,
        validated_output_path: Path,
    ) -> str:
        """
        Export story to Markdown format for reading.

        Args:
            session: Database session
            story_id: Story ID to export
            validated_output_path: Output file path

        Returns:
            File path

        Raises:
            StoryGenerationError: If export fails
        """
        # Validate parameters
        validated_story_id = validate_id(story_id, field_name="story_id")
        validated_validated_output_path = validate_file_path(
            validated_output_path,
            allowed_extensions=[".md"],
            field_name="validated_output_path"
        )

        try:
            # Get story
            result = await session.execute(
                select(Story).where(Story.id == validated_story_id)
            )
            story = result.scalar_one_or_none()

            if not story:
                raise StoryGenerationError("loading story", validated_story_id=validated_story_id, error_details=f"Story {validated_story_id} not found")

            # Get all scenes
            result = await session.execute(
                select(Scene)
                .where(Scene.validated_story_id == validated_story_id)
                .order_by(Scene.scene_number)
            )
            scenes = result.scalars().all()

            # Build markdown content
            lines = [
                f"# {story.title}\n",
                f"*Exported on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}*\n",
                "---\n\n",
            ]

            for scene in scenes:
                # Get choices
                result = await session.execute(
                    select(Choice)
                    .where(Choice.scene_id == scene.id)
                    .order_by(Choice.choice_number)
                )
                choices = result.scalars().all()

                lines.append(f"## Scene {scene.scene_number}\n\n")
                lines.append(f"{scene.content}\n\n")
                lines.append("**Choices:**\n\n")

                for choice in choices:
                    lines.append(f"{choice.choice_number}. {choice.content}\n")

                lines.append("\n---\n\n")

            # Write to file
            validated_output_path = Path(validated_output_path)
            validated_output_path.write_text("".join(lines), encoding="utf-8")

            return str(validated_output_path)

        except Exception as e:
            raise StoryGenerationError("exporting to markdown", error_details=str(e)) from e


class StoryImporter:
    """Import stories from various formats."""

    async def import_from_json(
        self,
        session: AsyncSession,
        json_path: Path,
        create_new_story: bool = True,
    ) -> int:
        """
        Import story from JSON file.

        Args:
            session: Database session
            json_path: Path to JSON file
            create_new_story: Whether to create new story or update existing

        Returns:
            Imported story ID

        Raises:
            StoryGenerationError: If import fails
        """
        # Validate parameters
        validated_json_path = validate_file_path(
            json_path,
            allowed_extensions=[".json"],
            field_name="json_path"
        )

        # Validate create_new_story
        if not isinstance(create_new_story, bool):
            raise StoryGenerationError("validating import parameters", error_details="create_new_story must be a boolean")

        try:
            # Read JSON file
            validated_json_path = Path(validated_json_path)
            if not validated_json_path.exists():
                raise StoryGenerationError("loading story file", error_details=f"File not found: {validated_json_path}")

            json_str = validated_json_path.read_text(encoding="utf-8")
            data = json.loads(json_str)

            # Validate JSON structure
            if not isinstance(data, dict):
                raise StoryGenerationError("validating JSON structure", error_details="Root JSON object must be a dictionary")

            # Validate format version
            format_version = data.get("format_version", "1.0")
            if format_version != "1.0":
                raise StoryGenerationError("validating story format", error_details=f"Unsupported format version: {format_version}")

            # Validate story data
            if "story" not in data:
                raise StoryGenerationError("validating JSON structure", error_details="Missing 'story' field in JSON")

            story_data = data["story"]
            if not isinstance(story_data, dict):
                raise StoryGenerationError("validating story data", error_details="Story data must be a dictionary")

            # Validate required story fields
            if "title" not in story_data:
                raise StoryGenerationError("validating story data", error_details="Missing 'title' field in story")

            # Validate title
            validated_title = validate_string(
                story_data["title"],
                field_name="title",
                min_length=1,
                max_length=255,
                strip_whitespace=True
            )

            # Create or update story
            if create_new_story:
                story = Story(
                    title=validated_title,
                    system_prompt=story_data.get("system_prompt"),
                    meta={},
                )
                session.add(story)
                await session.flush()

                validated_story_id = story.id
            else:
                validated_story_id = story_data["id"]
                result = await session.execute(
                    select(Story).where(Story.id == validated_story_id)
                )
                story = result.scalar_one_or_none()

                if not story:
                    raise StoryGenerationError("updating story", validated_story_id=validated_story_id, error_details=f"Story {validated_story_id} not found for update")

                story.title = story_data["title"]
                story.system_prompt = story_data.get("system_prompt")

            # Import characters
            character_map = {}  # Maps old IDs to new IDs

            for char_data in data.get("characters", []):
                if create_new_story:
                    character = Character(
                        name=char_data["name"],
                        description=char_data.get("description"),
                        personality=char_data.get("personality"),
                        goals=char_data.get("goals"),
                        background=char_data.get("background"),
                        meta=char_data.get("metadata", {}),
                    )
                    session.add(character)
                    await session.flush()

                    character_map[char_data["id"]] = character.id
                else:
                    character_map[char_data["id"]] = char_data["id"]

            # Import scenes
            for scene_data in data["scenes"]:
                if create_new_story:
                    scene = Scene(
                        validated_story_id=validated_story_id,
                        parent_scene_id=None,  # Simplified for import
                        scene_number=scene_data["scene_number"],
                        content=scene_data["content"],
                        raw_response=scene_data.get("raw_response"),
                        meta=scene_data.get("metadata", {}),
                    )
                    session.add(scene)
                    await session.flush()

                    # Import choices
                    for choice_data in scene_data["choices"]:
                        choice = Choice(
                            scene_id=scene.id,
                            choice_number=choice_data["number"],
                            content=choice_data["content"],
                        )
                        session.add(choice)

                    # Import character associations
                    for char_data in scene_data.get("characters", []):
                        old_char_id = char_data["id"]
                        new_char_id = character_map.get(old_char_id, old_char_id)

                        scene_char = SceneCharacter(
                            scene_id=scene.id,
                            character_id=new_char_id,
                            role=char_data.get("role"),
                            importance=char_data.get("importance", 1),
                        )
                        session.add(scene_char)

            await session.commit()

            return validated_story_id

        except Exception as e:
            await session.rollback()
            raise StoryGenerationError("importing story", error_details=str(e)) from e

    async def import_from_text(
        self,
        session: AsyncSession,
        title: str,
        content: str,
    ) -> int:
        """
        Import story from plain text (creates initial scene).

        Args:
            session: Database session
            title: Story title
            content: Story content

        Returns:
            Created story ID
        """
        # Validate parameters
        validated_title = validate_string(
            title,
            field_name="title",
            min_length=1,
            max_length=255,
            strip_whitespace=True
        )

        validated_content = validate_content(
            content,
            field_name="content",
            max_length=10000
        )

        try:
            # Create story
            story = Story(title=validated_title, meta={})
            session.add(story)
            await session.flush()

            # Create initial scene
            scene = Scene(
                story_id=story.id,
                scene_number=1,
                content=validated_content,
                meta={},
            )
            session.add(scene)
            await session.flush()

            # Create default choices
            for i in range(1, 4):
                choice = Choice(
                    scene_id=scene.id,
                    choice_number=i,
                    content=f"[Continue the story - option {i}]",
                )
                session.add(choice)

            await session.commit()

            return story.id

        except Exception as e:
            await session.rollback()
            raise StoryGenerationError("importing text", error_details=str(e)) from e
