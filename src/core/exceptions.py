"""Custom exceptions for LivingWorld application."""

from typing import Any, Optional


class LivingWorldError(Exception):
    """Base exception for all LivingWorld errors."""

    def __init__(self, message: str, context: Optional[dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with consistent capitalization and punctuation."""
        return self.message


class ConfigurationError(LivingWorldError):
    """Raised when there's a configuration problem."""

    def __init__(self, message: str, config_key: Optional[str] = None, config_value: Any = None):
        context = {}
        if config_key:
            context['config_key'] = config_key
        if config_value is not None:
            context['config_value'] = config_value
        super().__init__(f"Configuration error: {message}", context)


class DatabaseError(LivingWorldError):
    """Raised when there's a database-related error."""

    def __init__(self, operation: str, table: Optional[str] = None,
                 entity_id: Optional[Any] = None, original_error: Optional[str] = None):
        message_parts = [f"Database error during {operation}"]
        if table:
            message_parts.append(f"table {table}")
        if entity_id is not None:
            message_parts.append(f"ID {entity_id}")
        if original_error:
            message_parts.append(f"Details: {original_error}")

        context = {}
        if table:
            context['table'] = table
        if entity_id is not None:
            context['entity_id'] = entity_id
        if original_error:
            context['original_error'] = original_error

        super().__init__(" ".join(message_parts), context)


class OllamaError(LivingWorldError):
    """Raised when there's an error communicating with Ollama."""

    def __init__(self, operation: str, model: str, error_details: Optional[str] = None):
        message = f"Ollama API error during {operation} for model {model}"
        if error_details:
            message += f": {error_details}"

        context = {
            'operation': operation,
            'model': model,
            'error_details': error_details
        }
        super().__init__(message, context)


class EmbeddingError(LivingWorldError):
    """Raised when there's an error generating embeddings."""

    def __init__(self, operation: str, model: Optional[str] = None,
                 text_length: Optional[int] = None, original_error: Optional[str] = None):
        message_parts = [f"Embedding error during {operation}"]
        if model:
            message_parts.append(f"model {model}")
        if text_length is not None:
            message_parts.append(f"text length {text_length}")
        if original_error:
            message_parts.append(f"Details: {original_error}")

        context = {}
        if model:
            context['model'] = model
        if text_length is not None:
            context['text_length'] = text_length
        if original_error:
            context['original_error'] = original_error

        super().__init__(" ".join(message_parts), context)


class StoryGenerationError(LivingWorldError):
    """Raised when story generation fails."""

    def __init__(self, operation: str, story_id: Optional[int] = None,
                 scene_id: Optional[int] = None, error_details: Optional[str] = None):
        message_parts = [f"Story generation error during {operation}"]
        if story_id is not None:
            message_parts.append(f"story ID {story_id}")
        if scene_id is not None:
            message_parts.append(f"scene ID {scene_id}")
        if error_details:
            message_parts.append(f"Details: {error_details}")

        context = {}
        if story_id is not None:
            context['story_id'] = story_id
        if scene_id is not None:
            context['scene_id'] = scene_id
        if error_details:
            context['error_details'] = error_details

        super().__init__(" ".join(message_parts), context)


class AgentError(LivingWorldError):
    """Raised when there's an error with a character agent."""

    def __init__(self, operation: str, character_id: Optional[int] = None,
                 character_name: Optional[str] = None, session_id: Optional[int] = None,
                 error_details: Optional[str] = None):
        message_parts = [f"Agent error during {operation}"]
        if character_id is not None:
            message_parts.append(f"character ID {character_id}")
        if character_name:
            message_parts.append(f"character {character_name}")
        if session_id is not None:
            message_parts.append(f"session ID {session_id}")
        if error_details:
            message_parts.append(f"Details: {error_details}")

        context = {}
        if character_id is not None:
            context['character_id'] = character_id
        if character_name:
            context['character_name'] = character_name
        if session_id is not None:
            context['session_id'] = session_id
        if error_details:
            context['error_details'] = error_details

        super().__init__(" ".join(message_parts), context)


class SemanticSearchError(LivingWorldError):
    """Raised when semantic search fails."""

    def __init__(self, operation: str, search_type: Optional[str] = None,
                 query: Optional[str] = None, limit: Optional[int] = None,
                 original_error: Optional[str] = None):
        message_parts = [f"Semantic search error during {operation}"]
        if search_type:
            message_parts.append(f"search type {search_type}")
        if query:
            message_parts.append(f"query {query[:50]}{'...' if len(query) > 50 else ''}")
        if limit is not None:
            message_parts.append(f"limit {limit}")
        if original_error:
            message_parts.append(f"Details: {original_error}")

        context = {}
        if search_type:
            context['search_type'] = search_type
        if query:
            context['query'] = query
        if limit is not None:
            context['limit'] = limit
        if original_error:
            context['original_error'] = original_error

        super().__init__(" ".join(message_parts), context)
