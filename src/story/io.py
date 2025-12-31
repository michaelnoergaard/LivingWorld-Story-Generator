"""Story export and import functionality."""

import json
import asyncio
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from dataclasses import asdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.exceptions import StoryGenerationError
from src.database.models import Story, Scene, Choice, Character, SceneCharacter


class StoryExporter:
    """Export stories to various formats."""

    async def export_to_json(
        self,
        session: AsyncSession,
        story_id: int,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Export story to JSON format.

        Args:
            session: Database session
            story_id: Story ID to export
            output_path: Optional output file path

        Returns:
            JSON string or file path if output_path provided

        Raises:
            StoryGenerationError: If export fails
        """
        try:
            # Get story
            result = await session.execute(
                select(Story).where(Story.id == story_id)
            )
            story = result.scalar_one_or_none()

            if not story:
                raise StoryGenerationError(f"Story {story_id} not found")

            # Get all scenes
            result = await session.execute(
                select(Scene)
                .where(Scene.story_id == story_id)
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
                "exported_at": datetime.utcnow().isoformat(),
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
            if output_path:
                output_path = Path(output_path)
                output_path.write_text(json_str, encoding="utf-8")
                return str(output_path)

            return json_str

        except Exception as e:
            raise StoryGenerationError(f"Failed to export story: {e}") from e

    async def export_to_markdown(
        self,
        session: AsyncSession,
        story_id: int,
        output_path: Path,
    ) -> str:
        """
        Export story to Markdown format for reading.

        Args:
            session: Database session
            story_id: Story ID to export
            output_path: Output file path

        Returns:
            File path

        Raises:
            StoryGenerationError: If export fails
        """
        try:
            # Get story
            result = await session.execute(
                select(Story).where(Story.id == story_id)
            )
            story = result.scalar_one_or_none()

            if not story:
                raise StoryGenerationError(f"Story {story_id} not found")

            # Get all scenes
            result = await session.execute(
                select(Scene)
                .where(Scene.story_id == story_id)
                .order_by(Scene.scene_number)
            )
            scenes = result.scalars().all()

            # Build markdown content
            lines = [
                f"# {story.title}\n",
                f"*Exported on {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}*\n",
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
            output_path = Path(output_path)
            output_path.write_text("".join(lines), encoding="utf-8")

            return str(output_path)

        except Exception as e:
            raise StoryGenerationError(f"Failed to export to markdown: {e}") from e


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
        try:
            # Read JSON file
            json_path = Path(json_path)
            if not json_path.exists():
                raise StoryGenerationError(f"File not found: {json_path}")

            json_str = json_path.read_text(encoding="utf-8")
            data = json.loads(json_str)

            # Validate format version
            format_version = data.get("format_version", "1.0")
            if format_version != "1.0":
                raise StoryGenerationError(f"Unsupported format version: {format_version}")

            story_data = data["story"]

            # Create or update story
            if create_new_story:
                story = Story(
                    title=story_data["title"],
                    system_prompt=story_data.get("system_prompt"),
                    meta={},
                )
                session.add(story)
                await session.flush()

                story_id = story.id
            else:
                story_id = story_data["id"]
                result = await session.execute(
                    select(Story).where(Story.id == story_id)
                )
                story = result.scalar_one_or_none()

                if not story:
                    raise StoryGenerationError(f"Story {story_id} not found for update")

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
                        story_id=story_id,
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

            return story_id

        except Exception as e:
            await session.rollback()
            raise StoryGenerationError(f"Failed to import story: {e}") from e

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
        try:
            # Create story
            story = Story(title=title, meta={})
            session.add(story)
            await session.flush()

            # Create initial scene
            scene = Scene(
                story_id=story.id,
                scene_number=1,
                content=content,
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
            raise StoryGenerationError(f"Failed to import text: {e}") from e
