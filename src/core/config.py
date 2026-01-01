"""Configuration management for LivingWorld application."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from src.core.validation import (
    validate_string, validate_id, validate_int_range, validate_file_path,
    validate_float_range
)


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection configuration."""

    host: str
    port: int
    database: str
    user: str
    password: str
    min_pool_size: int = 2
    max_pool_size: int = 10

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate host
        validated_host = validate_string(
            self.host,
            field_name="host",
            min_length=1,
            max_length=255,
            pattern=r"^[a-zA-Z0-9.-]+$",
            strip_whitespace=True
        )

        # Validate database name
        validated_database = validate_string(
            self.database,
            field_name="database",
            min_length=1,
            max_length=63,
            pattern=r"^[a-zA-Z0-9_]+$",
            strip_whitespace=True
        )

        # Validate user
        validated_user = validate_string(
            self.user,
            field_name="user",
            min_length=1,
            max_length=63,
            pattern=r"^[a-zA-Z0-9_]+$",
            strip_whitespace=True
        )

        # Validate port range
        validate_int_range(
            self.port,
            field_name="port",
            min_value=1,
            max_value=65535
        )

        # Validate pool sizes
        validate_int_range(
            self.min_pool_size,
            field_name="min_pool_size",
            min_value=1,
            max_value=self.max_pool_size
        )

        validate_int_range(
            self.max_pool_size,
            field_name="max_pool_size",
            min_value=self.min_pool_size,
            max_value=100
        )

        # Validate password length
        if len(self.password) < 8:
            raise ValueError("Password must be at least 8 characters")

        # Ensure all validated values are assigned to make dataclass work with frozen=True
        object.__setattr__(self, "_validated", True)


@dataclass(frozen=True)
class OllamaConfig:
    """Ollama API configuration."""

    base_url: str
    model: str
    timeout: int = 120
    temperature: float = 0.8

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate base URL
        validated_base_url = validate_string(
            self.base_url,
            field_name="base_url",
            min_length=1,
            max_length=2048,
            pattern=r"^(https?|ftp)://[a-zA-Z0-9.-]+(?::[0-9]+)?(?:/.*)?$",
            strip_whitespace=True
        )

        # Validate model name
        validated_model = validate_string(
            self.model,
            field_name="model",
            min_length=1,
            max_length=255,
            pattern=r"^[a-zA-Z0-9._/-]+$",
            strip_whitespace=True
        )

        # Validate timeout
        validate_int_range(
            self.timeout,
            field_name="timeout",
            min_value=1,
            max_value=3600  # 1 hour max
        )

        # Validate temperature
        validate_float_range(
            self.temperature,
            field_name="temperature",
            min_value=0.0,
            max_value=2.0
        )

        # Ensure all validated values are assigned to make dataclass work with frozen=True
        object.__setattr__(self, "_validated", True)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model configuration."""

    model_name: str
    device: str = "cpu"
    batch_size: int = 32

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate model name
        validated_model_name = validate_string(
            self.model_name,
            field_name="model_name",
            min_length=1,
            max_length=255,
            pattern=r"^[a-zA-Z0-9._/-]+$",
            strip_whitespace=True
        )

        # Validate device
        validated_device = validate_string(
            self.device,
            field_name="device",
            min_length=1,
            max_length=50,
            pattern=r"^[a-zA-Z0-9-]+$",
            strip_whitespace=True
        )

        # Validate batch size
        validate_int_range(
            self.batch_size,
            field_name="batch_size",
            min_value=1,
            max_value=1024
        )

        # Ensure all validated values are assigned to make dataclass work with frozen=True
        object.__setattr__(self, "_validated", True)


@dataclass(frozen=True)
class StoryConfig:
    """Story generation configuration."""

    default_system_prompt_path: str
    context_window_size: int = 10
    max_retries: int = 3

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate system prompt path
        validated_prompt_path = validate_file_path(
            self.default_system_prompt_path,
            allowed_extensions=[".txt", ".md", ".yaml", ".yml"],
            field_name="default_system_prompt_path"
        )

        # Validate context window size
        validate_int_range(
            self.context_window_size,
            field_name="context_window_size",
            min_value=1,
            max_value=100
        )

        # Validate max retries
        validate_int_range(
            self.max_retries,
            field_name="max_retries",
            min_value=0,
            max_value=10
        )

        # Ensure all validated values are assigned to make dataclass work with frozen=True
        object.__setattr__(self, "_validated", True)


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""

    database: DatabaseConfig
    ollama: OllamaConfig
    embeddings: EmbeddingConfig
    story: StoryConfig

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "AppConfig":
        """
        Load configuration from environment variables.

        Args:
            env_file: Optional path to .env file. If not provided, searches in default locations.

        Returns:
            AppConfig instance with values from environment
        """
        # Validate env_file parameter
        if env_file is not None:
            validated_env_file = validate_file_path(
                env_file,
                allowed_extensions=[".env"],
                field_name="env_file"
            )
        # Load environment variables from .env file if it exists
        if env_file is None:
            # Try to find .env in current directory or parent directories
            current_path = Path.cwd()
            while current_path != current_path.parent:
                env_file = current_path / ".env"
                if env_file.exists():
                    break
                current_path = current_path.parent
            else:
                env_file = None

        if env_file and env_file.exists():
            load_dotenv(env_file)

        # Database configuration
        try:
            database = DatabaseConfig(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                database=os.getenv("DB_NAME", "livingworld"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", ""),
                min_pool_size=int(os.getenv("DB_MIN_POOL_SIZE", "2")),
                max_pool_size=int(os.getenv("DB_MAX_POOL_SIZE", "10")),
            )
        except ValueError as e:
            raise ValueError(f"Database configuration error: {e}") from e

        # Ollama configuration
        try:
            ollama = OllamaConfig(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "hf.co/TheDrummer/Cydonia-24B-v4.3-GGUF:Q4_K_M"),
                timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
                temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.8")),
            )
        except ValueError as e:
            raise ValueError(f"Ollama configuration error: {e}") from e

        # Embedding configuration
        try:
            embeddings = EmbeddingConfig(
                model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                device=os.getenv("EMBEDDING_DEVICE", "cpu"),
                batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            )
        except ValueError as e:
            raise ValueError(f"Embedding configuration error: {e}") from e

        # Story configuration
        try:
            story = StoryConfig(
                default_system_prompt_path=os.getenv(
                    "DEFAULT_SYSTEM_PROMPT_PATH", "prompts/system_prompt.txt"
                ),
                context_window_size=int(os.getenv("CONTEXT_WINDOW_SIZE", "10")),
                max_retries=int(os.getenv("MAX_RETRIES", "3")),
            )
        except ValueError as e:
            raise ValueError(f"Story configuration error: {e}") from e

        return cls(database=database, ollama=ollama, embeddings=embeddings, story=story)


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config(env_file: Optional[Path] = None) -> AppConfig:
    """
    Get or create the global configuration instance.

    Args:
        env_file: Optional path to .env file

    Returns:
        AppConfig instance
    """
    global _config
    if _config is None:
        _config = AppConfig.from_env(env_file)
    return _config


def reset_config():
    """Reset the global configuration instance (mainly for testing)."""
    global _config
    _config = None
