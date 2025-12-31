"""Custom exceptions for LivingWorld application."""


class LivingWorldError(Exception):
    """Base exception for all LivingWorld errors."""

    pass


class ConfigurationError(LivingWorldError):
    """Raised when there's a configuration problem."""

    pass


class DatabaseError(LivingWorldError):
    """Raised when there's a database-related error."""

    pass


class OllamaError(LivingWorldError):
    """Raised when there's an error communicating with Ollama."""

    pass


class EmbeddingError(LivingWorldError):
    """Raised when there's an error generating embeddings."""

    pass


class StoryGenerationError(LivingWorldError):
    """Raised when story generation fails."""

    pass


class AgentError(LivingWorldError):
    """Raised when there's an error with a character agent."""

    pass


class SemanticSearchError(LivingWorldError):
    """Raised when semantic search fails."""

    pass
