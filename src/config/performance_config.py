# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""
Performance configuration management.
Provides functions to get, update, and reset performance configuration.
"""

import threading
from typing import Dict, Any

from .models import PerformanceSettings
from .config_loader import get_settings

# Thread-safe configuration management
_config_lock = threading.RLock()
_cached_config: PerformanceSettings = None
_config_overrides: Dict[str, Any] = {}


def get_performance_config() -> PerformanceSettings:
    """Get the current performance configuration.

    Returns:
        The current performance configuration with any applied overrides.
    """
    global _cached_config

    with _config_lock:
        if _cached_config is None:
            # Load base configuration from settings
            try:
                settings = get_settings()
                _cached_config = settings.performance
            except Exception:
                # Fallback to default configuration
                _cached_config = PerformanceSettings()

        # Apply any runtime overrides
        if _config_overrides:
            config_dict = _cached_config.model_dump()
            _apply_overrides(config_dict, _config_overrides)
            return PerformanceSettings(**config_dict)

        return _cached_config


def update_performance_config(updates: Dict[str, Any]) -> None:
    """Update performance configuration with new values.

    Args:
        updates: Dictionary of configuration updates to apply.
    """
    global _config_overrides

    with _config_lock:
        # Validate updates by creating a test configuration
        current_config = get_performance_config()
        test_config_dict = current_config.model_dump()
        _apply_overrides(test_config_dict, updates)

        # This will raise validation errors if the config is invalid
        PerformanceSettings(**test_config_dict)

        # If validation passes, apply the overrides
        _apply_overrides(_config_overrides, updates)


def reset_performance_config() -> None:
    """Reset performance configuration to defaults."""
    global _cached_config, _config_overrides

    with _config_lock:
        _cached_config = None
        _config_overrides.clear()


def _apply_overrides(target: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Apply configuration overrides to target dictionary.

    Args:
        target: Target dictionary to update.
        overrides: Override values to apply.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _apply_overrides(target[key], value)
        else:
            target[key] = value


# Backward compatibility exports
__all__ = [
    "get_performance_config",
    "update_performance_config",
    "reset_performance_config",
    "PerformanceSettings",
]
