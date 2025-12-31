"""Embedding encoder using SentenceTransformer."""

import asyncio
from functools import lru_cache
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.config import EmbeddingConfig
from src.core.exceptions import EmbeddingError


class EmbeddingEncoder:
    """
    Wrapper for SentenceTransformer with caching and batch processing.

    Generates 384-dimensional embeddings using all-MiniLM-L6-v2 model.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding encoder.

        Args:
            config: Embedding configuration
        """
        self.config = config
        self._model: Optional[SentenceTransformer] = None
        self._cache: dict[str, np.ndarray] = {}

    def _load_model(self) -> SentenceTransformer:
        """
        Load SentenceTransformer model lazily.

        Returns:
            SentenceTransformer instance

        Raises:
            EmbeddingError: If model loading fails
        """
        if self._model is None:
            try:
                self._model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                )
            except Exception as e:
                raise EmbeddingError(f"Failed to load embedding model: {e}") from e

        return self._model

    def encode(
        self, text: str, use_cache: bool = True
    ) -> List[float]:
        """
        Encode a single text to embedding vector.

        Args:
            text: Text to encode
            use_cache: Whether to use in-memory cache

        Returns:
            List of floats representing the embedding vector

        Raises:
            EmbeddingError: If encoding fails
        """
        if use_cache and text in self._cache:
            return self._cache[text].tolist()

        try:
            model = self._load_model()
            embedding = model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            result = embedding.tolist()

            if use_cache:
                self._cache[text] = embedding

            return result

        except Exception as e:
            raise EmbeddingError(f"Failed to encode text: {e}") from e

    def encode_batch(
        self, texts: List[str], use_cache: bool = True
    ) -> List[List[float]]:
        """
        Encode multiple texts to embedding vectors.

        Args:
            texts: List of texts to encode
            use_cache: Whether to use in-memory cache for individual texts

        Returns:
            List of embedding vectors (each is a list of floats)

        Raises:
            EmbeddingError: If encoding fails
        """
        if not texts:
            return []

        # Check cache for each text
        results = []
        uncached_texts = []
        uncached_indices = []

        if use_cache:
            for i, text in enumerate(texts):
                if text in self._cache:
                    results.append((i, self._cache[text].tolist()))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Encode uncached texts in batch
        if uncached_texts:
            try:
                model = self._load_model()
                embeddings = model.encode(
                    uncached_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=self.config.batch_size,
                )

                for idx, text, embedding in zip(uncached_indices, uncached_texts, embeddings):
                    result = embedding.tolist()
                    results.append((idx, result))
                    if use_cache:
                        self._cache[text] = embedding

            except Exception as e:
                raise EmbeddingError(f"Failed to encode batch: {e}") from e

        # Sort results by original index and return just the embeddings
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    async def encode_async(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Async wrapper for encoding a single text.

        Args:
            text: Text to encode
            use_cache: Whether to use in-memory cache

        Returns:
            List of floats representing the embedding vector
        """
        # Run encoding in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode, text, use_cache)

    async def encode_batch_async(
        self, texts: List[str], use_cache: bool = True
    ) -> List[List[float]]:
        """
        Async wrapper for encoding multiple texts.

        Args:
            texts: List of texts to encode
            use_cache: Whether to use in-memory cache

        Returns:
            List of embedding vectors
        """
        # Run encoding in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode_batch, texts, use_cache)

    def clear_cache(self):
        """Clear the in-memory embedding cache."""
        self._cache.clear()

    def get_cache_size(self) -> int:
        """Get the number of items in the cache."""
        return len(self._cache)


# Global embedding encoder instance
_encoder: Optional[EmbeddingEncoder] = None


def get_encoder(config: Optional[EmbeddingConfig] = None) -> EmbeddingEncoder:
    """
    Get or create the global embedding encoder instance.

    Args:
        config: Optional embedding configuration. If not provided, uses config from environment.

    Returns:
        EmbeddingEncoder instance
    """
    global _encoder
    if _encoder is None:
        if config is None:
            from src.core.config import get_config

            app_config = get_config()
            config = app_config.embeddings
        _encoder = EmbeddingEncoder(config)
    return _encoder


def reset_encoder():
    """Reset the global encoder instance (mainly for testing)."""
    global _encoder
    _encoder = None
