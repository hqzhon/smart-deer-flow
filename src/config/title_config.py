"""Configuration for automatic thread title generation."""

from pydantic import BaseModel, Field

from src.config.base_config import ConfigManager


class TitleConfig(BaseModel):
    """Configuration for automatic thread title generation."""

    enabled: bool = Field(
        default=True,
        description="Whether to enable automatic title generation",
    )
    max_words: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum number of words in the generated title",
    )
    max_chars: int = Field(
        default=60,
        ge=10,
        le=200,
        description="Maximum number of characters in the generated title",
    )
    model_name: str | None = Field(
        default=None,
        description="Model name to use for title generation (None = use default model)",
    )
    prompt_template: str = Field(
        default=(
            "Generate a concise title (max {max_words} words) for this conversation.\nUser: {user_msg}\nAssistant: {assistant_msg}\n\nReturn ONLY the title, no quotes, no explanation."
        ),
        description="Prompt template for title generation",
    )


_title_config_manager = ConfigManager(TitleConfig)


def get_title_config() -> TitleConfig:
    """Get the current title configuration."""
    return _title_config_manager.get()


def set_title_config(config: TitleConfig) -> None:
    """Set the title configuration."""
    _title_config_manager.set(config)


def load_title_config_from_dict(config_dict: dict) -> None:
    """Load title configuration from a dictionary."""
    _title_config_manager.load_from_dict(config_dict)


def reset_title_config() -> None:
    """Reset the title configuration to defaults."""
    _title_config_manager.reset()
