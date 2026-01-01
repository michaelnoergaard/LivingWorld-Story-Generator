"""SQLAlchemy ORM models for LivingWorld."""

from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
    and_,
)
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Story(Base):
    """Story model representing an interactive story session."""

    __tablename__ = "stories"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    # Relationships
    scenes: Mapped[List["Scene"]] = relationship(
        "Scene", back_populates="story", cascade="all, delete-orphan"
    )
    memories: Mapped[List["Memory"]] = relationship(
        "Memory", back_populates="story", cascade="all, delete-orphan"
    )


class Scene(Base):
    """Scene model representing a single story scene."""

    __tablename__ = "scenes"

    id: Mapped[int] = mapped_column(primary_key=True)
    story_id: Mapped[int] = mapped_column(ForeignKey("stories.id", ondelete="CASCADE"))
    parent_scene_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("scenes.id", ondelete="SET NULL"), nullable=True
    )
    scene_number: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    raw_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    choices_generated: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(384), nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    # Relationships
    story: Mapped["Story"] = relationship("Story", back_populates="scenes")
    parent_scene: Mapped[Optional["Scene"]] = relationship(
        "Scene", remote_side=[id], backref="child_scenes"
    )
    choices: Mapped[List["Choice"]] = relationship(
        "Choice", back_populates="scene", cascade="all, delete-orphan"
    )
    scene_characters: Mapped[List["SceneCharacter"]] = relationship(
        "SceneCharacter", back_populates="scene", cascade="all, delete-orphan"
    )
    user_instructions: Mapped[List["UserInstruction"]] = relationship(
        "UserInstruction", back_populates="scene", cascade="all, delete-orphan"
    )
    memories: Mapped[List["Memory"]] = relationship(
        "Memory", back_populates="scene", cascade="all, delete-orphan"
    )

    # Characters appearing in this scene
    characters: Mapped[List["Character"]] = relationship(
        "Character",
        secondary="scene_characters",
        primaryjoin="Scene.id == SceneCharacter.scene_id",
        secondaryjoin="Character.id == SceneCharacter.character_id",
        viewonly=True
    )


class Choice(Base):
    """Choice model representing player choices."""

    __tablename__ = "choices"

    id: Mapped[int] = mapped_column(primary_key=True)
    scene_id: Mapped[int] = mapped_column(ForeignKey("scenes.id", ondelete="CASCADE"))
    choice_number: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    selected: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # Relationships
    scene: Mapped["Scene"] = relationship("Scene", back_populates="choices")


class Character(Base):
    """Character model representing NPCs in the story."""

    __tablename__ = "characters"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    first_appeared_in_scene: Mapped[Optional[int]] = mapped_column(
        ForeignKey("scenes.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(384), nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    # Agent-specific fields
    personality: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    goals: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    background: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    agent_config: Mapped[dict] = mapped_column(JSON, default=dict)

    # Relationships
    first_scene: Mapped[Optional["Scene"]] = relationship("Scene", foreign_keys=[first_appeared_in_scene])
    scene_characters: Mapped[List["SceneCharacter"]] = relationship(
        "SceneCharacter", back_populates="character", cascade="all, delete-orphan"
    )
    character_memories: Mapped[List["CharacterMemory"]] = relationship(
        "CharacterMemory", back_populates="character", cascade="all, delete-orphan"
    )


class SceneCharacter(Base):
    """Junction table linking scenes and characters."""

    __tablename__ = "scene_characters"

    scene_id: Mapped[int] = mapped_column(
        ForeignKey("scenes.id", ondelete="CASCADE"), primary_key=True
    )
    character_id: Mapped[int] = mapped_column(
        ForeignKey("characters.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    importance: Mapped[int] = mapped_column(Integer, default=1)

    # Relationships
    scene: Mapped["Scene"] = relationship("Scene", back_populates="scene_characters")
    character: Mapped["Character"] = relationship("Character", back_populates="scene_characters")


class UserInstruction(Base):
    """User instruction model for parenthetical instructions."""

    __tablename__ = "user_instructions"

    id: Mapped[int] = mapped_column(primary_key=True)
    scene_id: Mapped[int] = mapped_column(ForeignKey("scenes.id", ondelete="CASCADE"))
    instruction: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(384), nullable=True)

    # Relationships
    scene: Mapped["Scene"] = relationship("Scene", back_populates="user_instructions")


class Memory(Base):
    """Memory model for story events and context."""

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(primary_key=True)
    story_id: Mapped[int] = mapped_column(ForeignKey("stories.id", ondelete="CASCADE"))
    scene_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("scenes.id", ondelete="SET NULL"), nullable=True
    )
    content: Mapped[str] = mapped_column(Text)
    memory_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(384), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # Relationships
    story: Mapped["Story"] = relationship("Story", back_populates="memories")
    scene: Mapped[Optional["Scene"]] = relationship("Scene", back_populates="memories")


class CharacterMemory(Base):
    """Character memory model for NPC memories and experiences."""

    __tablename__ = "character_memories"

    id: Mapped[int] = mapped_column(primary_key=True)
    character_id: Mapped[int] = mapped_column(ForeignKey("characters.id", ondelete="CASCADE"))
    story_id: Mapped[int] = mapped_column(ForeignKey("stories.id", ondelete="CASCADE"))
    memory_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    emotional_valence: Mapped[float] = mapped_column(Float, default=0.0)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(384), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # Relationships
    character: Mapped["Character"] = relationship("Character", back_populates="character_memories")
