"""Ollama API client for story generation."""

import asyncio
from typing import AsyncIterator, List, Optional, Dict, Any

import ollama

from src.core.config import OllamaConfig
from src.core.exceptions import OllamaError


class OllamaClient:
    """
    Wrapper for Ollama API with retry logic and async support.

    Handles communication with Ollama for LLM-based story generation.
    """

    def __init__(self, config: OllamaConfig):
        """
        Initialize Ollama client.

        Args:
            config: Ollama configuration
        """
        self.config = config
        self._client: Optional[ollama.Client] = None

    def _get_client(self) -> ollama.Client:
        """
        Get or create Ollama client.

        Returns:
            ollama.Client instance
        """
        if self._client is None:
            self._client = ollama.Client(host=self.config.base_url)
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[int]] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text using Ollama model.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            context: Optional context from previous conversation (ignored for now)
            temperature: Optional temperature override

        Returns:
            Generated text

        Raises:
            OllamaError: If generation fails
        """
        client = self._get_client()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build options
        options = {
            "temperature": temperature or self.config.temperature,
        }

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.chat(
                    model=self.config.model,
                    messages=messages,
                    options=options,
                ),
            )

            return response["message"]["content"]

        except Exception as e:
            raise OllamaError(f"Failed to generate response: {e}") from e

    async def generate_with_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[int]] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Generate text with streaming output.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            context: Optional context from previous conversation (ignored for now)
            temperature: Optional temperature override

        Yields:
            Chunks of generated text

        Raises:
            OllamaError: If generation fails
        """
        client = self._get_client()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build options
        options = {
            "temperature": temperature or self.config.temperature,
        }

        loop = asyncio.get_event_loop()

        try:
            # Start streaming in thread pool
            stream = await loop.run_in_executor(
                None,
                lambda: client.chat(
                    model=self.config.model,
                    messages=messages,
                    options=options,
                    stream=True,
                ),
            )

            # Yield chunks as they arrive
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

        except Exception as e:
            raise OllamaError(f"Failed to generate streaming response: {e}") from e

    async def generate_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[int]] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Generate text with automatic retry on failure.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            context: Optional context from previous conversation
            temperature: Optional temperature override
            max_retries: Maximum number of retry attempts

        Returns:
            Generated text

        Raises:
            OllamaError: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return await self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    context=context,
                    temperature=temperature,
                )
            except OllamaError as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2**attempt)
                    continue
                else:
                    raise OllamaError(
                        f"Failed after {max_retries} attempts: {e}"
                    ) from e

        # Should never reach here, but just in case
        raise OllamaError(f"Failed after {max_retries} attempts") from last_error

    def check_model_available(self) -> bool:
        """
        Check if the configured model is available in Ollama.

        Returns:
            True if model is available, False otherwise
        """
        try:
            client = self._get_client()
            models = client.list()
            model_names = [model["name"] for model in models.get("models", [])]

            # Check if exact model name or partial match
            return any(
                self.config.model in name or name in self.config.model
                for name in model_names
            )
        except Exception:
            return False

    async def pull_model(self) -> None:
        """
        Pull the configured model from Ollama library.

        Raises:
            OllamaError: If model pull fails
        """
        client = self._get_client()
        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(
                None,
                lambda: client.pull(self.config.model),
            )
        except Exception as e:
            raise OllamaError(f"Failed to pull model {self.config.model}: {e}") from e


# Global Ollama client instance
_client: Optional[OllamaClient] = None


def get_ollama_client(config: Optional[OllamaConfig] = None) -> OllamaClient:
    """
    Get or create the global Ollama client instance.

    Args:
        config: Optional Ollama configuration. If not provided, uses config from environment.

    Returns:
        OllamaClient instance
    """
    global _client
    if _client is None:
        if config is None:
            from src.core.config import get_config

            app_config = get_config()
            config = app_config.ollama
        _client = OllamaClient(config)
    return _client


def reset_ollama_client():
    """Reset the global Ollama client instance (mainly for testing)."""
    global _client
    _client = None
