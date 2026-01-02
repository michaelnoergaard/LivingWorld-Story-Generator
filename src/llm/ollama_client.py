"""Ollama API client for story generation."""

import asyncio
from typing import AsyncIterator, List, Optional, Dict, Any

import ollama

from src.core.config import OllamaConfig
from src.core.exceptions import OllamaError
from src.core.logging_config import get_logger
from src.core.constants import LLMConstants

logger = get_logger(__name__)


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
        logger.debug("Starting Ollama generation for model %s", self.config.model)
        logger.debug("Prompt length: %d characters", len(prompt))
        if system_prompt:
            logger.debug("System prompt length: %d characters", len(system_prompt))

        client = self._get_client()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build options
        temperature_value = temperature or self.config.temperature
        options = {
            "temperature": temperature_value,
        }
        logger.debug("Generation temperature: %.2f", temperature_value)

        # Run in thread pool to avoid blocking
        loop = asyncio.get_running_loop()

        try:
            logger.info("Calling Ollama API for model %s", self.config.model)
            response = await loop.run_in_executor(
                None,
                lambda: client.chat(
                    model=self.config.model,
                    messages=messages,
                    options=options,
                ),
            )

            content = response["message"]["content"]
            logger.info("Ollama API call completed, response length: %d characters", len(content))
            return content

        except (ollama.ResponseError, ollama.RequestError) as e:
            logger.error(
                "Ollama API error during generate for model %s: %s",
                self.config.model,
                e,
                exc_info=True,
            )
            raise OllamaError("generate response", self.config.model, str(e)) from e
        except (KeyError, TypeError) as e:
            logger.error(
                "Invalid response format from Ollama API for model %s: %s",
                self.config.model,
                e,
                exc_info=True,
            )
            raise OllamaError("parse response", self.config.model, str(e)) from e
        except Exception as e:
            logger.exception(
                "Unexpected error generating response with model %s",
                self.config.model,
            )
            raise OllamaError("generate response", self.config.model, str(e)) from e

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

        loop = asyncio.get_running_loop()

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

        except (ollama.ResponseError, ollama.RequestError) as e:
            logger.error(
                "Ollama API error during streaming generation for model %s: %s",
                self.config.model,
                e,
                exc_info=True,
            )
            raise OllamaError("generate streaming response", self.config.model, str(e)) from e
        except (KeyError, TypeError) as e:
            logger.error(
                "Invalid chunk format from Ollama API streaming for model %s: %s",
                self.config.model,
                e,
                exc_info=True,
            )
            raise OllamaError("parse streaming chunk", self.config.model, str(e)) from e
        except Exception as e:
            logger.exception(
                "Unexpected error during streaming generation with model %s",
                self.config.model,
            )
            raise OllamaError("generate streaming response", self.config.model, str(e)) from e

    async def generate_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[int]] = None,
        temperature: Optional[float] = None,
        max_retries: int = None,
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
        if max_retries is None:
            max_retries = LLMConstants.MAX_RETRIES

        logger.debug("Starting generation with retry (max_retries=%d)", max_retries)
        last_error = None

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info("Retry attempt %d/%d for model %s", attempt + 1, max_retries, self.config.model)

                result = await self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    context=context,
                    temperature=temperature,
                )

                if attempt > 0:
                    logger.info("Generation succeeded on attempt %d", attempt + 1)

                return result
            except OllamaError as e:
                last_error = e
                logger.warning("Attempt %d/%d failed: %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    # Exponential backoff
                    backoff_time = LLMConstants.RETRY_BACKOFF_BASE ** attempt
                    logger.debug("Backing off for %.2f seconds before retry", backoff_time)
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    logger.error("All %d retry attempts failed for model %s", max_retries, self.config.model)
                    raise OllamaError("retry attempts", self.config.model, f"Failed after {max_retries} attempts: {e}") from e

        # Should never reach here, but just in case
        raise OllamaError("retry limit exceeded", self.config.model, f"Failed after {max_retries} attempts") from last_error

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
            available = any(
                self.config.model in name or name in self.config.model
                for name in model_names
            )

            if not available:
                logger.warning(
                    "Model %s not found in available models: %s",
                    self.config.model,
                    model_names,
                )

            return available

        except (ollama.ResponseError, ollama.RequestError) as e:
            logger.error(
                "Failed to check model availability for %s: %s",
                self.config.model,
                e,
                exc_info=True,
            )
            return False
        except (KeyError, TypeError) as e:
            logger.error(
                "Invalid response format when checking model %s: %s",
                self.config.model,
                e,
                exc_info=True,
            )
            return False
        except Exception as e:
            logger.exception(
                "Unexpected error checking model availability for %s",
                self.config.model,
            )
            return False

    async def pull_model(self) -> None:
        """
        Pull the configured model from Ollama library.

        Raises:
            OllamaError: If model pull fails
        """
        client = self._get_client()
        loop = asyncio.get_running_loop()

        try:
            logger.info("Pulling model %s from Ollama library", self.config.model)
            await loop.run_in_executor(
                None,
                lambda: client.pull(self.config.model),
            )
            logger.info("Successfully pulled model %s", self.config.model)

        except (ollama.ResponseError, ollama.RequestError) as e:
            logger.error(
                "Ollama API error pulling model %s: %s",
                self.config.model,
                e,
                exc_info=True,
            )
            raise OllamaError("pull model", self.config.model, str(e)) from e
        except Exception as e:
            logger.exception(
                "Unexpected error pulling model %s",
                self.config.model,
            )
            raise OllamaError("pull model", self.config.model, str(e)) from e


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
