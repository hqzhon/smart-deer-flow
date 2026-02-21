"""Configuration for memory mechanism."""

from pydantic import BaseModel, Field

from src.config.base_config import ConfigManager


class MemoryConfig(BaseModel):
    """Configuration for global memory mechanism."""

    enabled: bool = Field(
        default=True,
        description="Whether to enable memory mechanism",
    )
    storage_path: str = Field(
        default=".deer-flow/memory.json",
        description="Path to store memory data (relative to backend directory)",
    )
    debounce_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Seconds to wait before processing queued updates (debounce)",
    )
    model_name: str | None = Field(
        default=None,
        description="Model name to use for memory updates (None = use default model)",
    )
    max_facts: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of facts to store",
    )
    fact_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for storing facts",
    )
    injection_enabled: bool = Field(
        default=True,
        description="Whether to inject memory into system prompt",
    )
    max_injection_tokens: int = Field(
        default=2000,
        ge=100,
        le=8000,
        description="Maximum tokens to use for memory injection",
    )


_memory_config_manager = ConfigManager(MemoryConfig)


def get_memory_config() -> MemoryConfig:
    """Get the current memory configuration."""
    return _memory_config_manager.get()


def set_memory_config(config: MemoryConfig) -> None:
    """Set the memory configuration."""
    _memory_config_manager.set(config)


def load_memory_config_from_dict(config_dict: dict) -> None:
    """Load memory configuration from a dictionary."""
    _memory_config_manager.load_from_dict(config_dict)


def reset_memory_config() -> None:
    """Reset the memory configuration to defaults."""
    _memory_config_manager.reset()
