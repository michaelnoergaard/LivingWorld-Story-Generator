"""Pytest configuration and fixtures."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from src.core.config import AppConfig, DatabaseConfig, OllamaConfig, EmbeddingConfig, StoryConfig


# Test configuration
@pytest.fixture
def test_config():
    """Create test configuration."""
    return AppConfig(
        database=DatabaseConfig(
            host="localhost",
            port=5432,
            database="livingworld_test",
            user="test",
            password="test",
        ),
        ollama=OllamaConfig(
            base_url="http://localhost:11434",
            model="test-model",
            timeout=30,
        ),
        embeddings=EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        ),
        story=StoryConfig(
            default_system_prompt_path="prompts/system_prompt.txt",
            context_window_size=5,
            max_retries=2,
        ),
    )


@pytest.fixture
def mock_session():
    """Create mock database session."""
    session = MagicMock(spec=AsyncSession)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()

    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))

    session.execute.return_value = result

    return session


@pytest.fixture
def mock_ollama_client():
    """Create mock Ollama client."""
    client = MagicMock()
    client.generate_with_retry = AsyncMock(return_value="Test response")
    client.check_model_available = MagicMock(return_value=True)
    client.pull_model = AsyncMock()
    return client


@pytest.fixture
def mock_encoder():
    """Create mock embedding encoder."""
    encoder = MagicMock()
    encoder.encode_async = AsyncMock(return_value=[0.1] * 384)
    encoder.encode_batch_async = AsyncMock(return_value=[[0.1] * 384, [0.2] * 384])
    return encoder


@pytest.fixture
def sample_scene_response():
    """Sample scene response for testing."""
    return """The sun sets over the distant mountains, painting the sky in shades of orange and purple. You stand at the crossroads, uncertain of your path ahead.

1. Head toward the mountains
2. Make camp for the night
3. Continue along the road"""


@pytest.fixture
def sample_story_state():
    """Sample story state for testing."""
    state = MagicMock()
    state.story_id = 1
    state.title = "Test Adventure"
    state.current_scene_id = 1
    state.scene_number = 1
    state.location = "A crossroads"
    state.active_characters = set()
    state.metadata = {}
    state.status = "active"
    return state


# Async test event loop
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
