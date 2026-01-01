"""Story state management."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Set, Dict, Any
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.core.exceptions import DatabaseError
from src.database.models import Story, Scene, Character
from src.database.connection import get_database


class StoryStatus(Enum):
    """Story status enumeration."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class StoryState:
    """Current story state."""

    story_id: Optional[int] = None
    title: str = ""
    current_scene_id: Optional[int] = None
    scene_number: int = 0
    location: str = ""
    active_characters: Set[int] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: StoryStatus = StoryStatus.ACTIVE
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "story_id": self.story_id,
            "title": self.title,
            "current_scene_id": self.current_scene_id,
            "scene_number": self.scene_number,
            "location": self.location,
            "active_characters": list(self.active_characters),
            "metadata": self.metadata,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoryState":
        """Create state from dictionary."""
        return cls(
            story_id=data.get("story_id"),
            title=data.get("title", ""),
            current_scene_id=data.get("current_scene_id"),
            scene_number=data.get("scene_number", 0),
            location=data.get("location", ""),
            active_characters=set(data.get("active_characters", [])),
            metadata=data.get("metadata", {}),
            status=StoryStatus(data.get("status", StoryStatus.ACTIVE.value)),
        )


class StoryStateManager:
    """Manage story state persistence."""

    def __init__(self, session_factory):
        """
        Initialize story state manager.

        Args:
            session_factory: SQLAlchemy session factory
        """
        self.session_factory = session_factory

    async def create_story(
        self,
        title: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoryState:
        """
        Create a new story session.

        Args:
            title: Story title
            system_prompt: Optional custom system prompt
            metadata: Optional metadata dictionary

        Returns:
            StoryState instance

        Raises:
            DatabaseError: If creation fails
        """
        async with self.session_factory() as session:
            try:
                story = Story(
                    title=title,
                    system_prompt=system_prompt,
                    meta=metadata or {},
                )

                session.add(story)
                await session.commit()
                await session.refresh(story)

                return StoryState(
                    story_id=story.id,
                    title=story.title,
                    status=StoryStatus.ACTIVE,
                    created_at=story.created_at,
                    updated_at=story.updated_at,
                )

            except Exception as e:
                await session.rollback()
                raise DatabaseError(f"Failed to create story: {e}") from e

    async def load_story(self, story_id: int) -> StoryState:
        """
        Load an existing story state.

        Args:
            story_id: Story ID to load

        Returns:
            StoryState instance

        Raises:
            DatabaseError: If story not found or loading fails
        """
        async with self.session_factory() as session:
            try:
                result = await session.execute(
                    select(Story)
                    .options(selectinload(Story.scenes))
                    .where(Story.id == story_id)
                )
                story = result.scalar_one_or_none()

                if not story:
                    raise DatabaseError(f"Story {story_id} not found")

                # Get current scene
                current_scene_id = None
                scene_number = 0

                if story.scenes:
                    latest_scene = max(story.scenes, key=lambda s: s.scene_number, default=None)
                    if latest_scene:
                        current_scene_id = latest_scene.id
                        scene_number = latest_scene.scene_number

                # Get active characters from latest scene
                active_characters = set()
                if current_scene_id:
                    result = await session.execute(
                        select(Character)
                        .join(Character.scene_characters)
                        .where(Scene.id == current_scene_id)
                    )
                    active_characters = {c.id for c in result.scalars()}

                return StoryState(
                    story_id=story.id,
                    title=story.title,
                    current_scene_id=current_scene_id,
                    scene_number=scene_number,
                    active_characters=active_characters,
                    metadata=story.meta or {},
                    status=StoryStatus.ACTIVE if story.is_active else StoryStatus.PAUSED,
                    created_at=story.created_at,
                    updated_at=story.updated_at,
                )

            except Exception as e:
                raise DatabaseError(f"Failed to load story: {e}") from e

    async def update_scene(
        self,
        story_id: int,
        scene_id: int,
        scene_number: int,
        location: Optional[str] = None,
    ) -> StoryState:
        """
        Update story state after adding a new scene.

        Args:
            story_id: Story ID
            scene_id: New scene ID
            scene_number: Scene number
            location: Optional location string

        Returns:
            Updated StoryState

        Raises:
            DatabaseError: If update fails
        """
        async with self.session_factory() as session:
            try:
                result = await session.execute(
                    select(Story).where(Story.id == story_id)
                )
                story = result.scalar_one_or_none()

                if not story:
                    raise DatabaseError(f"Story {story_id} not found")

                # Update story timestamp
                story.updated_at = datetime.utcnow()

                await session.commit()

                # Build updated state
                state = await self.load_story(story_id)
                state.current_scene_id = scene_id
                state.scene_number = scene_number

                if location:
                    state.location = location

                return state

            except Exception as e:
                await session.rollback()
                raise DatabaseError(f"Failed to update scene: {e}") from e

    async def list_stories(self, active_only: bool = True) -> list[Story]:
        """
        List available stories.

        Args:
            active_only: Whether to only return active stories

        Returns:
            List of Story objects
        """
        async with self.session_factory() as session:
            try:
                query = select(Story)
                if active_only:
                    query = query.where(Story.is_active == True)

                result = await session.execute(query.order_by(Story.updated_at.desc()))
                return list(result.scalars().all())

            except Exception as e:
                raise DatabaseError(f"Failed to list stories: {e}") from e
