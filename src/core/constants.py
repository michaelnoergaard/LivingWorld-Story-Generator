"""
Constants module for LivingWorld application.

This module contains centralized constants organized by category.
Values that were previously hard-coded throughout the codebase are now
defined here with clear documentation about their purpose and usage.
"""

from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# LLM Configuration Constants
# =============================================================================

class LLMConstants:
    """
    Constants for LLM (Large Language Model) operations.

    These values control generation behavior, retry logic, and temperature settings.
    """

    # Default temperature for LLM generation
    DEFAULT_TEMPERATURE: float = 0.8

    # Temperature for character agent responses (higher = more creative/variable)
    CHARACTER_TEMPERATURE: float = 0.8

    # Temperature for internal thought generation (lower = more focused)
    THOUGHT_TEMPERATURE: float = 0.7

    # Temperature for character extraction (lower = more consistent)
    EXTRACTION_TEMPERATURE: float = 0.3

    # Maximum number of retry attempts for failed LLM calls
    MAX_RETRIES: int = 3

    # Exponential backoff base for retries (sleep = 2^attempt seconds)
    RETRY_BACKOFF_BASE: int = 2

    # Maximum iterations for LangChain agent execution
    AGENT_MAX_ITERATIONS: int = 5


# =============================================================================
# Story Generation Constants
# =============================================================================

class StoryConstants:
    """
    Constants for story generation operations.

    These values control scene structure, choices, and generation behavior.
    """

    # Number of choices to generate per scene
    NUM_CHOICES_PER_SCENE: int = 3

    # Number of recent scenes to include in context
    DEFAULT_RECENT_SCENES_LIMIT: int = 5

    # Number of scenes to include in prompt context
    CONTEXT_SCENE_COUNT: int = 3

    # Maximum length of scene content to display in character prompts
    SCENE_CONTENT_PREVIEW_LENGTH: int = 500

    # Maximum length of action text to display
    ACTION_PREVIEW_LENGTH: int = 100

    # Probability that a character acts autonomously (0.0 to 1.0)
    AUTONOMOUS_ACTION_PROBABILITY: float = 0.4

    # Number of characters to retrieve when searching
    DEFAULT_CHARACTER_LIMIT: int = 5

    # Truncation length for memory/context content
    MEMORY_TRUNCATION_LENGTH: int = 200

    # Truncation length for scene previews in context
    SCENE_PREVIEW_LENGTH: int = 200


# =============================================================================
# Embedding and Semantic Search Constants
# =============================================================================

class EmbeddingConstants:
    """
    Constants for embedding operations and semantic search.

    These values control caching, search limits, and similarity thresholds.
    """

    # Default maximum cache size for embedding encoder
    DEFAULT_MAX_CACHE_SIZE: int = 1000

    # Default similarity threshold for semantic search (0.0 to 1.0)
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.7

    # Limit for semantic search results (before threshold filtering)
    SEARCH_RESULT_MULTIPLIER: int = 2

    # Default number of similar scenes to retrieve
    DEFAULT_SIMILAR_SCENES_LIMIT: int = 5

    # Default number of memories to retrieve
    DEFAULT_MEMORY_RETRIEVAL_LIMIT: int = 10

    # Default number of character memories to retrieve
    DEFAULT_CHARACTER_MEMORY_LIMIT: int = 10

    # Limit for retrieving relevant characters
    RELEVANT_CHARACTERS_LIMIT: int = 5


# =============================================================================
# Agent and Character Constants
# =============================================================================

class AgentConstants:
    """
    Constants for character agent operations.

    These values control memory, emotion, and relationship management.
    """

    # Number of recent memories to retrieve for context
    DEFAULT_MEMORY_QUERY_LIMIT: int = 5

    # Number of memories to retrieve for autonomous action decisions
    AUTONOMOUS_MEMORY_LIMIT: int = 3

    # Number of conversation memories to retrieve
    CONVERSATION_MEMORY_LIMIT: int = 3

    # Default importance score for stored memories (0.0 to 1.0)
    DEFAULT_MEMORY_IMPORTANCE: float = 0.5

    # Importance score for internal thought memories
    INTERNAL_THOUGHT_IMPORTANCE: float = 0.6

    # Importance score for action memories
    ACTION_MEMORY_IMPORTANCE: float = 0.6

    # Default emotional valence (neutral)
    DEFAULT_EMOTIONAL_VALENCE: float = 0.0

    # Valence threshold for "positive" mood classification
    POSITIVE_VALENCE_THRESHOLD: float = 0.3

    # Valence threshold for "negative" mood classification
    NEGATIVE_VALENCE_THRESHOLD: float = -0.3

    # Emotional state exponential moving average coefficient
    # (new_valence = EMA_COEFFICIENT * current + (1 - EMA_COEFFICIENT) * valence)
    EMOTIONAL_STATE_EMA_COEFFICIENT: float = 0.7

    # Default initial emotional state values
    INITIAL_EMOTIONAL_AROUSAL: float = 0.5
    INITIAL_EMOTIONAL_VALENCE: float = 0.0
    INITIAL_EMOTIONAL_DOMINANCE: float = 0.5

    # Relationship sentiment delta multiplier for interactions
    RELATIONMENT_SENTIMENT_MULTIPLIER: float = 0.2

    # Trust delta for positive interactions
    TRUST_DELTA_POSITIVE: float = 0.05

    # Trust delta for negative interactions
    TRUST_DELTA_NEGATIVE: float = -0.05

    # Familiarity increment per interaction
    FAMILIARITY_INCREMENT: float = 0.1

    # Initial familiarity for new relationships
    INITIAL_FAMILIARITY: float = 0.1

    # Default trust level for new relationships
    INITIAL_TRUST_LEVEL: float = 0.5


# =============================================================================
# Database Constants
# =============================================================================

class DatabaseConstants:
    """
    Constants for database operations.

    These values control timeouts, pool sizes, and connection behavior.
    """

    # Command timeout for database queries (seconds)
    COMMAND_TIMEOUT: int = 60

    # Statement timeout for SQLAlchemy connections (milliseconds)
    STATEMENT_TIMEOUT_MS: str = "30000"

    # Maximum overflow for SQLAlchemy connection pool
    MAX_OVERFLOW: int = 0

    # Echo setting for SQLAlchemy (False = no query logging)
    ECHO_QUERIES: bool = False


# =============================================================================
# Prompt and Text Processing Constants
# =============================================================================

class PromptConstants:
    """
    Constants for prompt building and text processing.

    These values control text lengths and formatting.
    """

    # Number of recent scenes to show in story context
    RECENT_SCENES_IN_CONTEXT: int = 3

    # Truncation length for scene content in prompts
    SCENE_CONTENT_TRUNCATION: int = 500

    # Truncation length for character perspective display
    CHARACTER_PERSPECTIVE_TRUNCATION: int = 200


# =============================================================================
# Positive and Negative Emotion Words (for sentiment analysis)
# =============================================================================

class SentimentWords:
    """
    Word lists for basic sentiment analysis.

    Used in character agent to detect emotional valence of text.
    """

    # Positive emotion words
    POSITIVE_WORDS: Tuple[str, ...] = (
        "happy", "joy", "love", "good", "great", "wonderful", "excited",
        "pleased", "delighted", "thank", "grateful", "hope", "friend"
    )

    # Negative emotion words
    NEGATIVE_WORDS: Tuple[str, ...] = (
        "sad", "angry", "hate", "bad", "terrible", "awful", "furious",
        "upset", "disappointed", "worried", "fear", "enemy", "pain"
    )


# =============================================================================
# Dataclass Containers for Complex Constants
# =============================================================================

@dataclass(frozen=True)
class SceneParsingConfig:
    """Configuration for parsing AI-generated scenes."""

    expected_choices: int = 3
    choice_pattern: str = r"^\d+\.\s*(.+)$"


@dataclass(frozen=True)
class CharacterExtractionConfig:
    """Configuration for character extraction from scenes."""

    min_confidence: float = 0.5
    default_importance: int = 5
    default_role: str = "mentioned"


# =============================================================================
# Module Metadata
# =============================================================================

__all__ = [
    "LLMConstants",
    "StoryConstants",
    "EmbeddingConstants",
    "AgentConstants",
    "DatabaseConstants",
    "PromptConstants",
    "SentimentWords",
    "SceneParsingConfig",
    "CharacterExtractionConfig",
]
