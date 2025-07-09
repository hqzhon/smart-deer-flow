# -*- coding: utf-8 -*-
"""
Configuration Management System
Provides unified configuration loading, validation, hot reload and environment management functionality
"""

import os
import json
import yaml
import toml
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from enum import Enum
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pydantic import BaseModel
from functools import wraps

from .structured_logging import get_logger, EventType

logger = get_logger(__name__)


class ConfigFormat(Enum):
    """Configuration file format"""

    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"


class Environment(Enum):
    """Environment type"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ConfigSource:
    """Configuration source"""

    path: str
    format: ConfigFormat
    required: bool = True
    watch: bool = False
    priority: int = 0  # Priority, higher number means higher priority


class ConfigLoader(ABC):
    """Base class for configuration loaders"""

    @abstractmethod
    def load(self, path: str) -> Dict[str, Any]:
        """Load configuration"""
        pass

    @abstractmethod
    def can_handle(self, format: ConfigFormat) -> bool:
        """Check if can handle specified format"""
        pass


class JsonConfigLoader(ConfigLoader):
    """JSON configuration loader"""

    def load(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def can_handle(self, format: ConfigFormat) -> bool:
        return format == ConfigFormat.JSON


class YamlConfigLoader(ConfigLoader):
    """YAML configuration loader"""

    def load(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def can_handle(self, format: ConfigFormat) -> bool:
        return format == ConfigFormat.YAML


class TomlConfigLoader(ConfigLoader):
    """TOML configuration loader"""

    def load(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return toml.load(f)

    def can_handle(self, format: ConfigFormat) -> bool:
        return format == ConfigFormat.TOML


class EnvConfigLoader(ConfigLoader):
    """Environment variable configuration loader"""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def load(self, path: str = None) -> Dict[str, Any]:
        config = {}
        for key, value in os.environ.items():
            if self.prefix and not key.startswith(self.prefix):
                continue

            # Remove prefix
            config_key = key[len(self.prefix) :] if self.prefix else key

            # Try to convert type
            config[config_key] = self._convert_value(value)

        return config

    def _convert_value(self, value: str) -> Any:
        """Convert environment variable value type"""
        # Boolean values
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Numbers
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # String
        return value

    def can_handle(self, format: ConfigFormat) -> bool:
        return format == ConfigFormat.ENV


class ConfigWatcher(FileSystemEventHandler):
    """Configuration file watcher"""

    def __init__(
        self, config_manager: "ConfigManager", watched_files: Dict[str, ConfigSource]
    ):
        self.config_manager = config_manager
        self.watched_files = watched_files
        self._last_modified = {}

    def on_modified(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        if file_path not in self.watched_files:
            return

        # Prevent duplicate triggers
        current_time = time.time()
        if file_path in self._last_modified:
            if (
                current_time - self._last_modified[file_path] < 1.0
            ):  # Ignore duplicate events within 1 second
                return

        self._last_modified[file_path] = current_time

        logger.info(
            f"Configuration file changed: {file_path}", event_type=EventType.SYSTEM
        )

        try:
            self.config_manager.reload_config()
        except Exception as e:
            logger.error(
                f"Failed to reload configuration: {e}",
                error=e,
                event_type=EventType.ERROR,
            )


class ConfigManager:
    """Configuration manager"""

    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self.sources: List[ConfigSource] = []
        self.loaders: List[ConfigLoader] = []
        self.config: Dict[str, Any] = {}
        self.validators: List[Callable[[Dict[str, Any]], bool]] = []
        self.change_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._lock = threading.RLock()
        self._observer: Optional[Observer] = None
        self._watched_files: Dict[str, ConfigSource] = {}

        # Register default loaders
        self._register_default_loaders()

    def _register_default_loaders(self):
        """Register default configuration loaders"""
        self.loaders.extend(
            [
                JsonConfigLoader(),
                YamlConfigLoader(),
                TomlConfigLoader(),
                EnvConfigLoader(),
            ]
        )

    def add_source(self, source: ConfigSource):
        """Add configuration source"""
        with self._lock:
            self.sources.append(source)
            # Sort by priority
            self.sources.sort(key=lambda s: s.priority, reverse=True)

        logger.info(
            f"Added configuration source: {source.path}",
            event_type=EventType.SYSTEM,
            data={"format": source.format.value, "priority": source.priority},
        )

    def add_loader(self, loader: ConfigLoader):
        """Add configuration loader"""
        self.loaders.append(loader)

    def add_validator(self, validator: Callable[[Dict[str, Any]], bool]):
        """Add configuration validator"""
        self.validators.append(validator)

    def add_change_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add configuration change callback"""
        self.change_callbacks.append(callback)

    def load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        with self._lock:
            merged_config = {}

            # Load configuration in priority order
            for source in reversed(self.sources):  # Load low priority first
                try:
                    config_data = self._load_source(source)
                    if config_data:
                        merged_config = self._deep_merge(merged_config, config_data)
                        logger.debug(
                            f"Loaded configuration from {source.path}",
                            event_type=EventType.SYSTEM,
                            data={"keys_count": len(config_data)},
                        )

                except Exception as e:
                    if source.required:
                        logger.error(
                            f"Failed to load required configuration from {source.path}: {e}",
                            error=e,
                            event_type=EventType.ERROR,
                        )
                        raise
                    else:
                        logger.warning(
                            f"Failed to load optional configuration from {source.path}: {e}",
                            event_type=EventType.SYSTEM,
                        )

            # Environment variable substitution
            merged_config = self._substitute_env_vars(merged_config)

            # Validate configuration
            self._validate_config(merged_config)

            # Update configuration
            old_config = self.config.copy()
            self.config = merged_config

            # Trigger change callbacks
            if old_config != merged_config:
                self._trigger_change_callbacks(merged_config)

            # Setup file watching
            self._setup_file_watching()

            logger.info(
                "Configuration loaded successfully",
                event_type=EventType.SYSTEM,
                data={"total_keys": len(merged_config)},
            )

            return merged_config

    def _load_source(self, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """Load single configuration source"""
        # Find suitable loader
        loader = None
        for l in self.loaders:
            if l.can_handle(source.format):
                loader = l
                break

        if not loader:
            raise ValueError(f"No loader found for format: {source.format}")

        # Check if file exists (except for environment variables)
        if source.format != ConfigFormat.ENV:
            if not os.path.exists(source.path):
                if source.required:
                    raise FileNotFoundError(
                        f"Configuration file not found: {source.path}"
                    )
                else:
                    return None

        # Load configuration
        return loader.load(source.path)

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables"""

        def substitute_value(value):
            if isinstance(value, str):
                # Support ${VAR_NAME} and ${VAR_NAME:default_value} format
                import re

                pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

                def replace_match(match):
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else ""
                    return os.getenv(var_name, default_value)

                return re.sub(pattern, replace_match, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value

        return substitute_value(config)

    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration"""
        for validator in self.validators:
            try:
                if not validator(config):
                    raise ValueError("Configuration validation failed")
            except Exception as e:
                logger.error(
                    f"Configuration validation error: {e}",
                    error=e,
                    event_type=EventType.ERROR,
                )
                raise

    def _trigger_change_callbacks(self, config: Dict[str, Any]):
        """Trigger configuration change callbacks"""
        for callback in self.change_callbacks:
            try:
                callback(config)
            except Exception as e:
                logger.error(
                    f"Configuration change callback error: {e}",
                    error=e,
                    event_type=EventType.ERROR,
                )

    def _setup_file_watching(self):
        """Setup file watching"""
        # Collect files that need to be watched
        watch_files = {}
        for source in self.sources:
            if (
                source.watch
                and source.format != ConfigFormat.ENV
                and os.path.exists(source.path)
            ):
                watch_files[os.path.abspath(source.path)] = source

        if not watch_files:
            return

        # Stop existing watcher
        if self._observer:
            self._observer.stop()
            self._observer.join()

        # Create new watcher
        self._watched_files = watch_files
        self._observer = Observer()

        # Group files by directory for watching
        watched_dirs = set()
        for file_path in watch_files.keys():
            dir_path = os.path.dirname(file_path)
            if dir_path not in watched_dirs:
                event_handler = ConfigWatcher(self, watch_files)
                self._observer.schedule(event_handler, dir_path, recursive=False)
                watched_dirs.add(dir_path)

        self._observer.start()
        logger.info(
            f"Started watching {len(watch_files)} configuration files",
            event_type=EventType.SYSTEM,
        )

    def reload_config(self):
        """Reload configuration"""
        logger.info("Reloading configuration", event_type=EventType.SYSTEM)
        self.load_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value (runtime)"""
        with self._lock:
            keys = key.split(".")
            config = self.config

            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            config[keys[-1]] = value

            logger.debug(
                f"Configuration value updated: {key}",
                event_type=EventType.SYSTEM,
                data={"key": key, "value": str(value)[:100]},
            )

    def has(self, key: str) -> bool:
        """Check if configuration key exists"""
        return self.get(key) is not None

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self.config.copy()

    def stop_watching(self):
        """Stop file watching"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info(
                "Stopped configuration file watching", event_type=EventType.SYSTEM
            )

    def __del__(self):
        """Destructor"""
        self.stop_watching()


class ConfigSchema(BaseModel):
    """Base class for configuration schema"""

    class Config:
        extra = "allow"  # Allow extra fields
        validate_assignment = True  # Validate on assignment


def config_property(
    key: str, default: Any = None, config_manager: ConfigManager = None
):
    """Configuration property decorator"""

    def decorator(cls):
        def getter(self):
            manager = config_manager or getattr(self, "_config_manager", None)
            if manager:
                return manager.get(key, default)
            return default

        def setter(self, value):
            manager = config_manager or getattr(self, "_config_manager", None)
            if manager:
                manager.set(key, value)

        setattr(cls, key.replace(".", "_"), property(getter, setter))
        return cls

    return decorator


def with_config(config_manager: ConfigManager):
    """Configuration injection decorator"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Inject configuration manager
            if "config" not in kwargs:
                kwargs["config"] = config_manager
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Inject configuration manager
            if "config" not in kwargs:
                kwargs["config"] = config_manager
            return await func(*args, **kwargs)

        if hasattr(func, "__call__") and hasattr(func, "__await__"):
            return async_wrapper
        else:
            return wrapper

    return decorator


# Global configuration manager
global_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager"""
    global global_config_manager
    if global_config_manager is None:
        # Get environment type from environment variable
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        try:
            environment = Environment(env_name)
        except ValueError:
            environment = Environment.DEVELOPMENT
            logger.warning(
                f"Unknown environment '{env_name}', using development",
                event_type=EventType.SYSTEM,
            )

        global_config_manager = ConfigManager(environment)

    return global_config_manager


def configure_from_files(*file_paths: str, watch: bool = True):
    """Configure from files"""
    manager = get_config_manager()

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            logger.warning(
                f"Configuration file not found: {file_path}",
                event_type=EventType.SYSTEM,
            )
            continue

        # Determine format based on file extension
        format_map = {
            ".json": ConfigFormat.JSON,
            ".yaml": ConfigFormat.YAML,
            ".yml": ConfigFormat.YAML,
            ".toml": ConfigFormat.TOML,
        }

        file_format = format_map.get(path.suffix.lower())
        if not file_format:
            logger.warning(
                f"Unknown configuration file format: {file_path}",
                event_type=EventType.SYSTEM,
            )
            continue

        source = ConfigSource(
            path=str(path.absolute()),
            format=file_format,
            watch=watch,
            priority=len(manager.sources),  # Later added has higher priority
        )

        manager.add_source(source)

    # Load configuration
    manager.load_config()

    return manager


def configure_from_env(prefix: str = ""):
    """Configure from environment variables"""
    manager = get_config_manager()

    # Add environment variable loader
    env_loader = EnvConfigLoader(prefix)
    manager.add_loader(env_loader)

    # Add environment variable source
    source = ConfigSource(
        path="",  # Environment variables don't need path
        format=ConfigFormat.ENV,
        priority=1000,  # Environment variables have highest priority
    )

    manager.add_source(source)
    manager.load_config()

    return manager


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any):
    """Set configuration value"""
    get_config_manager().set(key, value)
