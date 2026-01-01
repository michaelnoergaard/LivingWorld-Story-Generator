# Error Message Consistency Guidelines

## Overview

This document establishes consistent patterns for error messages throughout the LivingWorld codebase. All error messages should follow these guidelines to ensure clarity, consistency, and ease of debugging.

## General Principles

### 1. Message Format
- **Start with capital letter**: All error messages should begin with a capital letter
- **No ending period**: Single-line error messages should not end with a period
- **Multi-sentence messages**: If multiple sentences are needed, use periods appropriately
- **Consistent terminology**: Use the same terms for similar operations (e.g., "generating", "loading", "saving")

### 2. Structure
Error messages should follow this pattern:
```
[Error Type] during [Operation]: [Specific Details]
```

Example:
```
Story generation error during creating initial scene: Invalid response format
```

### 3. Context Information
Include relevant context when available:
- IDs (story_id, character_id, scene_id)
- Model names (for LLM/embedding operations)
- Operation names (generating, loading, saving, parsing)
- Specific error details when helpful

### 4. Exception Classes Usage
Use appropriate exception classes:
- `ConfigurationError`: For configuration-related issues
- `DatabaseError`: For database operations
- `OllamaError`: For Ollama API interactions
- `EmbeddingError`: For embedding generation
- `StoryGenerationError`: For story-related operations
- `AgentError`: For character agent issues
- `SemanticSearchError`: For semantic search failures

## Exception Class Templates

### LivingWorldError (Base)
```python
class LivingWorldError(Exception):
    """Base exception for all LivingWorld errors."""
    def __init__(self, message: str, context: Optional[dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self._format_message())
```

### DatabaseError
```python
# Parameters: operation, table=None, entity_id=None, original_error=None
raise DatabaseError("creating story", original_error=str(e))
raise DatabaseError("loading story", entity_id=story_id, original_error="Story not found")
```

### OllamaError
```python
# Parameters: operation, model, error_details=None
raise OllamaError("generate response", "llama2", "Connection timeout")
```

### StoryGenerationError
```python
# Parameters: operation, story_id=None, scene_id=None, error_details=None
raise StoryGenerationError("generating initial scene", story_id=1, error_details=str(e))
raise StoryGenerationError("parsing response", error_details="No choices found")
```

### AgentError
```python
# Parameters: operation, character_id=None, character_name=None, session_id=None, error_details=None
raise AgentError("generating response", character_id=42, error_details=str(e))
raise AgentError("session initialization", error_details="Call initialize_session() first")
```

### EmbeddingError
```python
# Parameters: operation, model=None, text_length=None, original_error=None
raise EmbeddingError("encoding text", model="all-MiniLM-L6-v2", original_error=str(e))
```

### SemanticSearchError
```python
# Parameters: operation, search_type=None, query=None, limit=None, original_error=None
raise SemanticSearchError("finding similar scenes", original_error=str(e))
```

## Specific Guidelines by Module

### LLM Module (src/llm/)
- Use `OllamaError` for all Ollama API interactions
- Include model name in context
- Specify operation type (generate, parse, pull, etc.)
- Use consistent retry error messages

### Database Module (src/database/)
- Use `DatabaseError` for all database operations
- Include operation type (create, read, update, delete, migrate)
- Include table name when relevant
- Include entity ID when applicable

### Agents Module (src/agents/)
- Use `AgentError` for character agent issues
- Include character information (ID or name)
- Specify agent operation type (create, respond, execute)
- Include session context when relevant

### Story Module (src/story/)
- Use `StoryGenerationError` for story-related operations
- Include story_id and scene_id when available
- Specify operation type (export, import, generate, load)
- Include file operations context for I/O operations

### Embeddings Module (src/embeddings/)
- Use `EmbeddingError` for embedding operations
- Include model name
- Specify operation type (encode, load, batch encode)
- Use `SemanticSearchError` for search operations
- Include search type and query context

## Error Message Examples

### Before (Inconsistent)
```python
raise StoryGenerationError("Failed to generate initial scene: Connection error")
raise OllamaError("Failed to generate response: Invalid format")
raise DatabaseError("Database not initialized")
```

### After (Consistent)
```python
raise StoryGenerationError("generating initial scene", story_id=1, error_details="Connection error")
raise OllamaError("generate response", "llama2", "Invalid format")
raise DatabaseError("initialization", original_error="Database not initialized")
```

## Best Practices

1. **Be specific**: Include all relevant context (IDs, names, parameters)
2. **Be concise**: Avoid unnecessary verbosity
3. **Be consistent**: Follow the established patterns
4. **Use proper exceptions**: Choose the right exception class for the situation
5. **Include original errors**: Chain original exceptions when appropriate
6. **Test error messages**: Ensure error messages are clear and helpful

## Checklist for New Error Messages

- [ ] Started with capital letter?
- [ ] No ending period (unless multi-sentence)?
- [ ] Used appropriate exception class?
- [ ] Included relevant context (IDs, names, parameters)?
- [ ] Specified operation type?
- [ ] Consistent terminology?
- [ ] Properly chained original exception if applicable?