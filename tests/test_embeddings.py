"""Tests for embedding encoder and semantic search modules."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from unittest import mock
import numpy as np

from src.embeddings.encoder import EmbeddingEncoder, get_encoder, reset_encoder
from src.embeddings.search import SemanticSearch, get_semantic_search, reset_semantic_search
from src.core.config import EmbeddingConfig
from src.core.exceptions import EmbeddingError, SemanticSearchError
from src.database.models import Scene, Character, Memory, CharacterMemory


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def embedding_config():
    """Create embedding configuration for testing."""
    return EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        batch_size=32,
    )


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model."""
    model = MagicMock()
    # Create mock embeddings (384-dimensional)
    model.encode.return_value = np.random.rand(384)
    return model


@pytest.fixture
def embedding_encoder(embedding_config, mock_sentence_transformer):
    """Create embedding encoder with mocked model."""
    encoder = EmbeddingEncoder(embedding_config)
    encoder._model = mock_sentence_transformer
    return encoder


@pytest.fixture
def mock_encoder():
    """Mock embedding encoder for semantic search tests."""
    encoder = MagicMock()
    encoder.encode_async = AsyncMock(return_value=[0.1] * 384)
    return encoder


@pytest.fixture
def semantic_search(mock_encoder):
    """Create semantic search instance."""
    return SemanticSearch(mock_encoder)


@pytest.fixture
def mock_session():
    """Mock database session."""
    session = MagicMock()
    session.execute = AsyncMock()
    return session


# =============================================================================
# EmbeddingEncoder Tests
# =============================================================================

class TestEmbeddingEncoderInit:
    """Tests for EmbeddingEncoder initialization."""

    def test_init_with_config(self, embedding_config):
        """Test encoder initialization with config."""
        encoder = EmbeddingEncoder(embedding_config)
        assert encoder.config == embedding_config
        assert encoder._model is None
        assert encoder._cache == {}

    def test_model_lazy_loading(self, embedding_config):
        """Test that model is loaded lazily."""
        with patch('src.embeddings.encoder.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            encoder = EmbeddingEncoder(embedding_config)
            assert encoder._model is None

            model = encoder._load_model()
            assert model is mock_model
            assert encoder._model is mock_model

            # Subsequent calls should return cached model
            model2 = encoder._load_model()
            assert model2 is model


class TestEmbeddingEncoderEncode:
    """Tests for encode method."""

    def test_encode_single_text(self, embedding_encoder):
        """Test encoding a single text."""
        result = embedding_encoder.encode("Hello world")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)

    def test_encode_uses_cache(self, embedding_encoder):
        """Test that cache is used when enabled."""
        text = "Test text"
        embedding = np.array([0.1] * 384)
        embedding_encoder._model.encode.return_value = embedding

        # First call should use model
        result1 = embedding_encoder.encode(text, use_cache=True)
        assert embedding_encoder._model.encode.call_count == 1

        # Second call should use cache
        result2 = embedding_encoder.encode(text, use_cache=True)
        assert embedding_encoder._model.encode.call_count == 1  # No additional call
        assert result1 == result2

    def test_encode_bypasses_cache(self, embedding_encoder):
        """Test bypassing cache."""
        text = "Test text"
        embedding_encoder.encode(text, use_cache=False)

        result = embedding_encoder.encode(text, use_cache=True)
        assert text in embedding_encoder._cache

        # Clear cache and verify
        embedding_encoder.clear_cache()
        assert text not in embedding_encoder._cache

    def test_encode_empty_text(self, embedding_encoder):
        """Test encoding empty text."""
        result = embedding_encoder.encode("")
        assert len(result) == 384

    def test_encode_model_error(self, embedding_config):
        """Test handling of model encoding error."""
        encoder = EmbeddingEncoder(embedding_config)
        encoder._model = MagicMock()
        encoder._model.encode.side_effect = Exception("Model error")

        with pytest.raises(EmbeddingError, match="Failed to encode text"):
            encoder.encode("Test text")


class TestEmbeddingEncoderEncodeBatch:
    """Tests for encode_batch method."""

    def test_encode_batch_empty_list(self, embedding_encoder):
        """Test encoding empty list."""
        result = embedding_encoder.encode_batch([])
        assert result == []

    def test_encode_batch_multiple_texts(self, embedding_encoder):
        """Test encoding multiple texts."""
        texts = ["Text one", "Text two", "Text three"]

        # Mock batch encoding
        embeddings = np.random.rand(3, 384)
        embedding_encoder._model.encode.return_value = embeddings

        results = embedding_encoder.encode_batch(texts, use_cache=False)

        assert len(results) == 3
        assert all(len(r) == 384 for r in results)

    def test_encode_batch_with_cache(self, embedding_encoder):
        """Test batch encoding with partial cache hits."""
        texts = ["New text 1", "New text 2", "New text 3"]

        # Pre-populate cache for one text
        cached_embedding = np.array([0.5] * 384)
        embedding_encoder._cache[texts[1]] = cached_embedding

        # Mock for uncached texts
        embeddings = np.random.rand(2, 384)
        embedding_encoder._model.encode.return_value = embeddings

        results = embedding_encoder.encode_batch(texts, use_cache=True)

        assert len(results) == 3
        # Should only encode 2 texts (one was cached)
        assert embedding_encoder._model.encode.call_count == 1

    def test_encode_batch_all_cached(self, embedding_encoder):
        """Test batch encoding when all texts are cached."""
        texts = ["Text 1", "Text 2"]
        cached_embedding = np.array([0.5] * 384)
        embedding_encoder._cache[texts[0]] = cached_embedding
        embedding_encoder._cache[texts[1]] = cached_embedding

        results = embedding_encoder.encode_batch(texts, use_cache=True)

        assert len(results) == 2
        # Should not call model at all
        assert embedding_encoder._model.encode.call_count == 0

    def test_encode_batch_error(self, embedding_encoder):
        """Test handling of batch encoding error."""
        embedding_encoder._model.encode.side_effect = Exception("Batch error")

        with pytest.raises(EmbeddingError, match="Failed to encode batch"):
            embedding_encoder.encode_batch(["Text 1", "Text 2"], use_cache=False)


@pytest.mark.asyncio
class TestEmbeddingEncoderAsync:
    """Tests for async encoding methods."""

    async def test_encode_async(self, embedding_encoder):
        """Test async encoding of single text."""
        result = await embedding_encoder.encode_async("Test text")

        assert isinstance(result, list)
        assert len(result) == 384

    async def test_encode_batch_async(self, embedding_encoder):
        """Test async batch encoding."""
        texts = ["Text 1", "Text 2", "Text 3"]

        # Mock batch encoding
        embeddings = np.random.rand(3, 384)
        embedding_encoder._model.encode.return_value = embeddings

        results = await embedding_encoder.encode_batch_async(texts, use_cache=False)

        assert len(results) == 3
        assert all(len(r) == 384 for r in results)


class TestEmbeddingEncoderCache:
    """Tests for cache management."""

    def test_clear_cache(self, embedding_encoder):
        """Test clearing the cache."""
        embedding_encoder._cache["key1"] = np.array([0.1] * 384)
        embedding_encoder._cache["key2"] = np.array([0.2] * 384)

        assert embedding_encoder.get_cache_size() == 2

        embedding_encoder.clear_cache()

        assert embedding_encoder.get_cache_size() == 0

    def test_get_cache_size(self, embedding_encoder):
        """Test getting cache size."""
        assert embedding_encoder.get_cache_size() == 0

        embedding_encoder._cache["key1"] = np.array([0.1] * 384)
        assert embedding_encoder.get_cache_size() == 1

        embedding_encoder._cache["key2"] = np.array([0.2] * 384)
        assert embedding_encoder.get_cache_size() == 2


class TestGlobalEncoder:
    """Tests for global encoder instance."""

    def test_get_encoder_creates_instance(self, embedding_config):
        """Test that get_encoder creates new instance."""
        reset_encoder()

        with patch('src.core.config.get_config') as mock_get_config:
            mock_get_config.return_value = MagicMock(embeddings=embedding_config)

            encoder = get_encoder()
            assert isinstance(encoder, EmbeddingEncoder)

    def test_get_encoder_returns_cached(self, embedding_config):
        """Test that get_encoder returns cached instance."""
        reset_encoder()

        with patch('src.core.config.get_config') as mock_get_config:
            mock_get_config.return_value = MagicMock(embeddings=embedding_config)

            encoder1 = get_encoder()
            encoder2 = get_encoder()
            assert encoder1 is encoder2

    def test_get_encoder_with_custom_config(self, embedding_config):
        """Test get_encoder with custom config."""
        reset_encoder()

        encoder = get_encoder(embedding_config)
        assert isinstance(encoder, EmbeddingEncoder)
        assert encoder.config == embedding_config


# =============================================================================
# SemanticSearch Tests
# =============================================================================

@pytest.mark.asyncio
class TestSemanticSearchFindSimilarScenes:
    """Tests for find_similar_scenes method."""

    async def test_find_similar_scenes_success(self, semantic_search, mock_session):
        """Test successful scene search."""
        # Mock database response
        mock_rows = [
            # Mock row: (id, story_id, parent_scene_id, scene_number, content,
            #            raw_response, choices_generated, created_at, metadata, similarity)
            (1, 1, None, 1, "Scene 1 content", "response", {}, "2024-01-01", {}, 0.95),
            (2, 1, None, 2, "Scene 2 content", "response", {}, "2024-01-02", {}, 0.85),
            (3, 1, None, 3, "Scene 3 content", "response", {}, "2024-01-03", {}, 0.65),  # Below threshold
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result

        results = await semantic_search.find_similar_scenes(
            session=mock_session,
            query_text="Search query",
            story_id=1,
            limit=5,
            threshold=0.7,
        )

        # Should return 2 scenes (above threshold)
        assert len(results) == 2
        assert all(isinstance(scene, Scene) for scene, _ in results)
        assert all(isinstance(score, float) for _, score in results)

        # Verify ordering by similarity
        assert results[0][1] > results[1][1]

    async def test_find_similar_scenes_with_exclude(self, semantic_search, mock_session):
        """Test scene search with scene exclusion."""
        mock_rows = [
            (2, 1, None, 2, "Scene 2", "response", {}, "2024-01-02", {}, 0.85),
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result

        results = await semantic_search.find_similar_scenes(
            session=mock_session,
            query_text="Query",
            story_id=1,
            exclude_scene_id=1,
        )

        assert len(results) == 1

    async def test_find_similar_scenes_empty_results(self, semantic_search, mock_session):
        """Test scene search with no results."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        results = await semantic_search.find_similar_scenes(
            session=mock_session,
            query_text="Query",
            story_id=1,
        )

        assert results == []

    async def test_find_similar_scenes_database_error(self, semantic_search, mock_session):
        """Test handling of database error."""
        mock_session.execute.side_effect = Exception("Database error")

        with pytest.raises(SemanticSearchError, match="Failed to find similar scenes"):
            await semantic_search.find_similar_scenes(
                session=mock_session,
                query_text="Query",
                story_id=1,
            )


@pytest.mark.asyncio
class TestSemanticSearchFindRelevantCharacters:
    """Tests for find_relevant_characters method."""

    async def test_find_relevant_characters_success(self, semantic_search, mock_session):
        """Test successful character search."""
        # Mock characters
        mock_char1 = MagicMock()
        mock_char1.id = 1
        mock_char1.name = "Alice"
        mock_char1.description = "A brave warrior"
        mock_char1.personality = "Courageous"

        mock_char2 = MagicMock()
        mock_char2.id = 2
        mock_char2.name = "Bob"
        mock_char2.description = "A wise mage"
        mock_char2.personality = "Thoughtful"

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_char1, mock_char2]
        mock_session.execute.return_value = mock_result

        results = await semantic_search.find_relevant_characters(
            session=mock_session,
            query="brave warrior",
            limit=5,
        )

        assert len(results) > 0
        assert all(isinstance(char, MagicMock) for char, _ in results)
        assert all(isinstance(score, float) for _, score in results)

    async def test_find_relevant_characters_no_matches(self, semantic_search, mock_session):
        """Test character search with no matches."""
        mock_char = MagicMock()
        mock_char.id = 1
        mock_char.name = "Alice"
        mock_char.description = "A character"
        mock_char.personality = None

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_char]
        mock_session.execute.return_value = mock_result

        # Query with no matching words
        results = await semantic_search.find_relevant_characters(
            session=mock_session,
            query="xyzabc123",  # No matching words
        )

        # Should return empty since no text relevance
        assert results == []

    async def test_find_relevant_characters_error(self, semantic_search, mock_session):
        """Test handling of error in character search."""
        mock_session.execute.side_effect = Exception("Database error")

        with pytest.raises(SemanticSearchError, match="Failed to find relevant characters"):
            await semantic_search.find_relevant_characters(
                session=mock_session,
                query="Query",
            )


@pytest.mark.asyncio
class TestSemanticSearchRetrieveMemories:
    """Tests for retrieve_memories method."""

    async def test_retrieve_memories_success(self, semantic_search, mock_session):
        """Test successful memory retrieval."""
        mock_rows = [
            (1, 1, 1, "Memory 1", "observation", 0.8, "2024-01-01", 0.95),
            (2, 1, 2, "Memory 2", "conversation", 0.6, "2024-01-02", 0.75),
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result

        results = await semantic_search.retrieve_memories(
            session=mock_session,
            story_id=1,
            query="What happened",
            limit=10,
        )

        assert len(results) == 2
        assert all(isinstance(memory, Memory) for memory, _ in results)
        assert all(isinstance(score, float) for _, score in results)

    async def test_retrieve_memories_with_type_filter(self, semantic_search, mock_session):
        """Test memory retrieval with type filter."""
        mock_rows = [
            (1, 1, 1, "Memory 1", "conversation", 0.8, "2024-01-01", 0.95),
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result

        results = await semantic_search.retrieve_memories(
            session=mock_session,
            story_id=1,
            query="Conversations",
            memory_types=["conversation"],
            limit=10,
        )

        assert len(results) == 1

    async def test_retrieve_memories_empty(self, semantic_search, mock_session):
        """Test memory retrieval with no results."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        results = await semantic_search.retrieve_memories(
            session=mock_session,
            story_id=1,
            query="Query",
        )

        assert results == []

    async def test_retrieve_memories_error(self, semantic_search, mock_session):
        """Test handling of error in memory retrieval."""
        mock_session.execute.side_effect = Exception("Database error")

        with pytest.raises(SemanticSearchError, match="Failed to retrieve memories"):
            await semantic_search.retrieve_memories(
                session=mock_session,
                story_id=1,
                query="Query",
            )


@pytest.mark.asyncio
class TestSemanticSearchRetrieveCharacterMemories:
    """Tests for retrieve_character_memories method."""

    async def test_retrieve_character_memories_success(self, semantic_search, mock_session):
        """Test successful character memory retrieval."""
        mock_rows = [
            (1, 1, 1, "conversation", "Spoke to hero", 0.5, 0.8, "2024-01-01", 0.95),
            (2, 1, 1, "observation", "Saw something", 0.0, 0.6, "2024-01-02", 0.75),
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result

        results = await semantic_search.retrieve_character_memories(
            session=mock_session,
            character_id=1,
            story_id=1,
            query="What do I remember",
            limit=10,
        )

        assert len(results) == 2
        assert all(isinstance(memory, CharacterMemory) for memory, _ in results)
        assert all(isinstance(score, float) for _, score in results)

    async def test_retrieve_character_memories_with_type_filter(self, semantic_search, mock_session):
        """Test character memory retrieval with type filter."""
        mock_rows = [
            (1, 1, 1, "conversation", "Spoke to hero", 0.5, 0.8, "2024-01-01", 0.95),
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result

        results = await semantic_search.retrieve_character_memories(
            session=mock_session,
            character_id=1,
            story_id=1,
            query="Conversations",
            memory_types=["conversation"],
            limit=10,
        )

        assert len(results) == 1

    async def test_retrieve_character_memories_empty(self, semantic_search, mock_session):
        """Test character memory retrieval with no results."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        results = await semantic_search.retrieve_character_memories(
            session=mock_session,
            character_id=1,
            story_id=1,
            query="Query",
        )

        assert results == []

    async def test_retrieve_character_memories_error(self, semantic_search, mock_session):
        """Test handling of error in character memory retrieval."""
        mock_session.execute.side_effect = Exception("Database error")

        with pytest.raises(SemanticSearchError, match="Failed to retrieve character memories"):
            await semantic_search.retrieve_character_memories(
                session=mock_session,
                character_id=1,
                story_id=1,
                query="Query",
            )


class TestSemanticSearchTextRelevance:
    """Tests for _calculate_text_relevance method."""

    def test_calculate_text_relevance_full_match(self, semantic_search):
        """Test text relevance with full word match."""
        query = "brave warrior"
        text = "brave warrior fights"

        relevance = semantic_search._calculate_text_relevance(query, text)

        assert relevance == 1.0

    def test_calculate_text_relevance_partial_match(self, semantic_search):
        """Test text relevance with partial match."""
        query = "brave warrior"
        text = "brave hero fights"

        relevance = semantic_search._calculate_text_relevance(query, text)

        assert relevance == 0.5

    def test_calculate_text_relevance_no_match(self, semantic_search):
        """Test text relevance with no match."""
        query = "brave warrior"
        text = "timid healer sleeps"

        relevance = semantic_search._calculate_text_relevance(query, text)

        assert relevance == 0.0

    def test_calculate_text_relevance_case_insensitive(self, semantic_search):
        """Test that relevance is case insensitive."""
        query = "Brave Warrior"
        text = "BRAVE WARRIOR fights"

        relevance = semantic_search._calculate_text_relevance(query, text)

        assert relevance == 1.0

    def test_calculate_text_relevance_empty_inputs(self, semantic_search):
        """Test text relevance with empty inputs."""
        assert semantic_search._calculate_text_relevance("", "text") == 0.0
        assert semantic_search._calculate_text_relevance("query", "") == 0.0
        assert semantic_search._calculate_text_relevance("", "") == 0.0


class TestGlobalSemanticSearch:
    """Tests for global semantic search instance."""

    def test_get_semantic_search_creates_instance(self, mock_encoder):
        """Test that get_semantic_search creates new instance."""
        reset_semantic_search()

        search = get_semantic_search(mock_encoder)
        assert isinstance(search, SemanticSearch)
        assert search.encoder is mock_encoder

    def test_get_semantic_search_returns_cached(self, mock_encoder):
        """Test that get_semantic_search returns cached instance."""
        reset_semantic_search()

        search1 = get_semantic_search(mock_encoder)
        search2 = get_semantic_search()
        assert search1 is search2

    def test_get_semantic_search_without_encoder(self):
        """Test get_semantic_search without explicit encoder."""
        reset_semantic_search()
        reset_encoder()

        with patch('src.embeddings.encoder.get_encoder') as mock_get_encoder:
            mock_get_encoder.return_value = mock_encoder

            search = get_semantic_search()
            assert isinstance(search, SemanticSearch)
