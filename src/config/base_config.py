# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Base configuration manager for singleton pattern."""

from typing import Generic, TypeVar, Type, Optional, Callable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ConfigManager(Generic[T]):
    """Generic configuration manager with singleton pattern.

    Provides a reusable pattern for configuration management with:
    - Lazy initialization
    - Thread-safe singleton access
    - Configuration reset capability
    - Dictionary-based loading

    Example:
        ```python
        class MyConfig(BaseModel):
            enabled: bool = True
            max_items: int = 10

        my_config_manager = ConfigManager(MyConfig)

        def get_my_config() -> MyConfig:
            return my_config_manager.get()

        def set_my_config(config: MyConfig) -> None:
            my_config_manager.set(config)

        def load_my_config_from_dict(config_dict: dict) -> None:
            my_config_manager.load_from_dict(config_dict)
        ```
    """

    def __init__(
        self,
        config_class: Type[T],
        default_factory: Optional[Callable[[], T]] = None,
    ):
        """Initialize the configuration manager.

        Args:
            config_class: The Pydantic model class for the configuration.
            default_factory: Optional factory function to create default instance.
                            If None, uses config_class() with default values.
        """
        self._config_class = config_class
        self._instance: Optional[T] = None
        self._default_factory = default_factory

    def get(self) -> T:
        """Get the configuration instance.

        Returns the cached singleton instance. Creates a new instance
        with default values if not yet initialized.

        Returns:
            The configuration instance.
        """
        if self._instance is None:
            self._instance = self._create_default()
        return self._instance

    def set(self, config: T) -> None:
        """Set the configuration instance.

        Args:
            config: The configuration instance to use.
        """
        self._instance = config

    def load_from_dict(self, config_dict: dict) -> None:
        """Load configuration from a dictionary.

        Args:
            config_dict: Dictionary with configuration values.
        """
        self._instance = self._config_class(**config_dict)

    def reset(self) -> None:
        """Reset the cached configuration instance.

        Clears the singleton cache, causing the next call to get()
        to create a new instance with default values.
        """
        self._instance = None

    def _create_default(self) -> T:
        """Create a default configuration instance.

        Returns:
            A new configuration instance with default values.
        """
        if self._default_factory:
            return self._default_factory()
        return self._config_class()
