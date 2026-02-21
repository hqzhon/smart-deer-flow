# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Singleton pattern utilities for managing global instances.

This module provides a standardized way to implement singleton pattern
across the codebase, reducing code duplication.
"""

import threading
from typing import Any, Callable, Dict, Optional, TypeVar, Generic

T = TypeVar("T")


class SingletonMeta(type):
    """
    Thread-safe singleton metaclass.

    Usage:
        class MyClass(metaclass=SingletonMeta):
            def __init__(self):
                self.value = 42
    """

    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def reset_instance(mcs, cls: type) -> None:
        """Reset the singleton instance for a class."""
        with mcs._lock:
            if cls in mcs._instances:
                del mcs._instances[cls]

    @classmethod
    def has_instance(mcs, cls: type) -> bool:
        """Check if a singleton instance exists."""
        return cls in mcs._instances


class SingletonManager(Generic[T]):
    """
    Generic singleton manager for managing global instances.

    This class provides a standardized way to create and manage singleton instances
    with thread safety and reset capabilities.

    Usage:
        # Create a manager
        manager = SingletonManager(lambda: MyExpensiveClass())

        # Get the singleton instance
        instance = manager.get_instance()

        # Reset the instance
        manager.reset_instance()
    """

    def __init__(self, factory: Callable[[], T]):
        """
        Initialize the singleton manager.

        Args:
            factory: A callable that creates the singleton instance.
        """
        self._factory = factory
        self._instance: Optional[T] = None
        self._lock = threading.Lock()

    def get_instance(self) -> T:
        """
        Get the singleton instance, creating it if necessary.

        Returns:
            The singleton instance.
        """
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance

    def reset_instance(self) -> None:
        """Reset the singleton instance."""
        with self._lock:
            self._instance = None

    def has_instance(self) -> bool:
        """Check if an instance exists."""
        return self._instance is not None


def create_singleton_functions(
    factory: Callable[[], T],
) -> tuple[Callable[[], T], Callable[[], None]]:
    """
    Create get_instance and reset_instance functions for a singleton.

    This is a convenience function for creating module-level singleton accessors.

    Args:
        factory: A callable that creates the singleton instance.

    Returns:
        A tuple of (get_instance, reset_instance) functions.

    Usage:
        class MyManager:
            def __init__(self):
                self.data = {}

        get_my_manager, reset_my_manager = create_singleton_functions(MyManager)

        # Use the singleton
        manager = get_my_manager()
        reset_my_manager()  # Reset when needed
    """
    manager = SingletonManager(factory)

    def get_instance() -> T:
        return manager.get_instance()

    def reset_instance() -> None:
        manager.reset_instance()

    return get_instance, reset_instance


class GlobalInstanceRegistry:
    """
    Central registry for all global instances.

    This class provides a single point of management for all singleton instances
    across the application, useful for testing and debugging.
    """

    _registry: Dict[str, SingletonManager] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, name: str, factory: Callable[[], Any]) -> SingletonManager:
        """
        Register a singleton factory.

        Args:
            name: Unique name for the singleton.
            factory: A callable that creates the singleton instance.

        Returns:
            The SingletonManager for this registration.
        """
        with cls._lock:
            if name not in cls._registry:
                cls._registry[name] = SingletonManager(factory)
            return cls._registry[name]

    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        """
        Get a singleton instance by name.

        Args:
            name: The registered name of the singleton.

        Returns:
            The singleton instance or None if not registered.
        """
        if name in cls._registry:
            return cls._registry[name].get_instance()
        return None

    @classmethod
    def reset(cls, name: str) -> None:
        """
        Reset a specific singleton instance.

        Args:
            name: The registered name of the singleton.
        """
        if name in cls._registry:
            cls._registry[name].reset_instance()

    @classmethod
    def reset_all(cls) -> None:
        """Reset all singleton instances."""
        with cls._lock:
            for manager in cls._registry.values():
                manager.reset_instance()

    @classmethod
    def list_instances(cls) -> list[str]:
        """List all registered singleton names."""
        return list(cls._registry.keys())
