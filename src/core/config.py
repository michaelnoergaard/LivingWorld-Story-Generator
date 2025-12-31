"""Configuration management for LivingWorld application."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


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


@dataclass(frozen=True)
class OllamaConfig:
    """Ollama API configuration."""

    base_url: str
    model: str
    timeout: int = 120
    temperature: float = 0.8


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model configuration."""

    model_name: str
    device: str = "cpu"
    batch_size: int = 32


@dataclass(frozen=True)
class StoryConfig:
    """Story generation configuration."""

    default_system_prompt_path: str
    context_window_size: int = 10
    max_retries: int = 3


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
        database = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "livingworld"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            min_pool_size=int(os.getenv("DB_MIN_POOL_SIZE", "2")),
            max_pool_size=int(os.getenv("DB_MAX_POOL_SIZE", "10")),
        )

        # Ollama configuration
        ollama = OllamaConfig(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "hf.co/TheDrummer/Cydonia-24B-v4.3-GGUF:Q4_K_M"),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.8")),
        )

        # Embedding configuration
        embeddings = EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        )

        # Story configuration
        story = StoryConfig(
            default_system_prompt_path=os.getenv(
                "DEFAULT_SYSTEM_PROMPT_PATH", "prompts/system_prompt.txt"
            ),
            context_window_size=int(os.getenv("CONTEXT_WINDOW_SIZE", "10")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
        )

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
