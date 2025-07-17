# SPDX-License-Identifier: MIT
"""
Unified Configuration Loader - New Pydantic-based system.
Replaces both old and intermediate configuration systems.
"""

import os
import yaml
import logging
from typing import Any, Dict, Optional
from pathlib import Path

from .models import AppSettings
from .validators import validate_configuration

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Unified configuration loader using Pydantic models."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the configuration loader.

        Args:
            config_dir: Directory containing configuration files. Defaults to project root.
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()
        self._settings: Optional[AppSettings] = None

    def load_from_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            Dictionary containing configuration data.

        Raises:
            FileNotFoundError: If YAML file doesn't exist.
            yaml.YamLError: If YAML is invalid.
        """
        yaml_file = self.config_dir / yaml_path
        if not yaml_file.exists():
            logger.warning(f"Configuration file not found: {yaml_file}")
            return {}

        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Successfully loaded configuration file: {yaml_file}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {yaml_file}: {e}")
            return {}

    def load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Dictionary containing environment-based configuration.
        """
        env_config = {}

        # Environment variable mappings
        env_mappings = {
            # Core settings
            "DEER_REPORT_STYLE": "report_style",
            "DEER_RESOURCES": "resources",
            # LLM settings
            "DEER_LLM_TEMPERATURE": ["llm", "temperature"],
            "DEER_LLM_TIMEOUT": ["llm", "timeout"],
            "DEER_LLM_MAX_TOKENS": ["llm", "max_tokens"],
            # Agent settings
            "DEER_MAX_PLAN_ITERATIONS": ["agents", "max_plan_iterations"],
            "DEER_MAX_STEP_NUM": ["agents", "max_step_num"],
            "DEER_MAX_SEARCH_RESULTS": ["agents", "max_search_results"],
            "DEER_ENABLE_DEEP_THINKING": ["agents", "enable_deep_thinking"],
            "DEER_ENABLE_PARALLEL_EXECUTION": ["agents", "enable_parallel_execution"],
            "DEER_MAX_PARALLEL_TASKS": ["agents", "max_parallel_tasks"],
            "DEER_MAX_CONTEXT_STEPS_PARALLEL": ["agents", "max_context_steps_parallel"],
            "DEER_DISABLE_CONTEXT_PARALLEL": ["agents", "disable_context_parallel"],
            # Research settings
            "DEER_ENABLE_RESEARCHER_ISOLATION": [
                "research",
                "enable_researcher_isolation",
            ],
            "DEER_RESEARCHER_ISOLATION_LEVEL": [
                "research",
                "researcher_isolation_level",
            ],
            "DEER_RESEARCHER_MAX_LOCAL_CONTEXT": [
                "research",
                "researcher_max_local_context",
            ],
            "DEER_RESEARCHER_ISOLATION_THRESHOLD": [
                "research",
                "researcher_isolation_threshold",
            ],
            "DEER_RESEARCHER_AUTO_ISOLATION": ["research", "researcher_auto_isolation"],
            "DEER_RESEARCHER_ISOLATION_METRICS": [
                "research",
                "researcher_isolation_metrics",
            ],
            "DEER_MAX_CONTEXT_STEPS_RESEARCHER": [
                "research",
                "max_context_steps_researcher",
            ],
            # Reflection settings
            "DEER_ENABLE_ENHANCED_REFLECTION": [
                "reflection",
                "enable_enhanced_reflection",
            ],
            "DEER_MAX_REFLECTION_LOOPS": ["reflection", "max_reflection_loops"],
            "DEER_REFLECTION_TEMPERATURE": ["reflection", "reflection_temperature"],
            "DEER_REFLECTION_TRIGGER_THRESHOLD": [
                "reflection",
                "reflection_trigger_threshold",
            ],
            "DEER_REFLECTION_CONFIDENCE_THRESHOLD": [
                "reflection",
                "reflection_confidence_threshold",
            ],
            "DEER_ENABLE_REFLECTION_INTEGRATION": [
                "reflection",
                "enable_reflection_integration",
            ],
            "DEER_ENABLE_PROGRESSIVE_REFLECTION": [
                "reflection",
                "enable_progressive_reflection",
            ],
            "DEER_ENABLE_REFLECTION_METRICS": [
                "reflection",
                "enable_reflection_metrics",
            ],
            # Iterative research settings
            "DEER_MAX_FOLLOW_UP_ITERATIONS": [
                "iterative_research",
                "max_follow_up_iterations",
            ],
            "DEER_SUFFICIENCY_THRESHOLD": [
                "iterative_research",
                "sufficiency_threshold",
            ],
            "DEER_ENABLE_ITERATIVE_RESEARCH": [
                "iterative_research",
                "enable_iterative_research",
            ],
            "DEER_MAX_QUERIES_PER_ITERATION": [
                "iterative_research",
                "max_queries_per_iteration",
            ],
            "DEER_FOLLOW_UP_DELAY_SECONDS": [
                "iterative_research",
                "follow_up_delay_seconds",
            ],
            # Content settings
            "DEER_ENABLE_CONTENT_SUMMARIZATION": [
                "content",
                "enable_content_summarization",
            ],
            "DEER_ENABLE_SMART_FILTERING": ["content", "enable_smart_filtering"],
            "DEER_SUMMARY_TYPE": ["content", "summary_type"],
            # Advanced context settings
            "DEER_MAX_CONTEXT_RATIO": ["advanced_context", "max_context_ratio"],
            "DEER_SLIDING_WINDOW_SIZE": ["advanced_context", "sliding_window_size"],
            "DEER_OVERLAP_RATIO": ["advanced_context", "overlap_ratio"],
            "DEER_COMPRESSION_THRESHOLD": ["advanced_context", "compression_threshold"],
            "DEER_DEFAULT_STRATEGY": ["advanced_context", "default_strategy"],
            "DEER_ENABLE_CACHING": ["advanced_context", "enable_caching"],
            "DEER_ENABLE_ANALYTICS": ["advanced_context", "enable_analytics"],
            "DEER_DEBUG_MODE": ["advanced_context", "debug_mode"],
            # MCP settings
            "DEER_MCP_ENABLED": ["mcp", "enabled"],
            "DEER_MCP_TIMEOUT": ["mcp", "timeout"],
            # Tool settings
            "SELECTED_SEARCH_ENGINE": ["tools", "search_engine"],
            "SELECTED_RAG_PROVIDER": ["tools", "rag_provider"],
            # Performance settings
            "DEER_ENABLE_ADVANCED_OPTIMIZATION": [
                "performance",
                "enable_advanced_optimization",
            ],
            "DEER_ENABLE_COLLABORATION": ["performance", "enable_collaboration"],
            "DEER_PERFORMANCE_DEBUG_MODE": ["performance", "debug_mode"],
            # Connection pool settings
            "DEER_MAX_CONNECTIONS": [
                "performance",
                "connection_pool",
                "max_connections",
            ],
            "DEER_INITIAL_CONNECTIONS": [
                "performance",
                "connection_pool",
                "initial_connections",
            ],
            "DEER_CONNECTION_TIMEOUT": [
                "performance",
                "connection_pool",
                "connection_timeout",
            ],
            "DEER_IDLE_TIMEOUT": ["performance", "connection_pool", "idle_timeout"],
            # Batch processing settings
            "DEER_BATCH_SIZE": ["performance", "batch_processing", "batch_size"],
            "DEER_BATCH_TIMEOUT": ["performance", "batch_processing", "batch_timeout"],
            "DEER_MAX_QUEUE_SIZE": [
                "performance",
                "batch_processing",
                "max_queue_size",
            ],
            # Cache settings
            "DEER_L1_CACHE_SIZE": ["performance", "cache", "l1_size"],
            "DEER_L2_CACHE_SIZE": ["performance", "cache", "l2_size"],
            "DEER_L3_CACHE_SIZE": ["performance", "cache", "l3_size"],
            "DEER_CACHE_TTL": ["performance", "cache", "default_ttl"],
            # Rate limit settings
            "DEER_INITIAL_RATE": ["performance", "rate_limit", "initial_rate"],
            "DEER_MAX_RATE": ["performance", "rate_limit", "max_rate"],
            "DEER_MIN_RATE": ["performance", "rate_limit", "min_rate"],
            # Agent LLM mapping settings
            "DEER_COORDINATOR_LLM": ["agent_llm_map", "coordinator"],
            "DEER_PLANNER_LLM": ["agent_llm_map", "planner"],
            "DEER_RESEARCHER_LLM": ["agent_llm_map", "researcher"],
            "DEER_CODER_LLM": ["agent_llm_map", "coder"],
            "DEER_REPORTER_LLM": ["agent_llm_map", "reporter"],
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if value.lower() in ("true", "1", "yes", "on"):
                    value = True
                elif value.lower() in ("false", "0", "no", "off"):
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit():
                    value = float(value)

                # Handle nested keys (supports multi-level nesting)
                if isinstance(config_key, list):
                    current = env_config
                    for key in config_key[:-1]:
                        if key not in current:
                            current[key] = {}
                        elif not isinstance(current[key], dict):
                            # If existing value is not a dict, convert it to dict
                            current[key] = {}
                        current = current[key]
                    current[config_key[-1]] = value
                else:
                    env_config[config_key] = value

        return env_config

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries with deep merging.

        Args:
            *configs: Configuration dictionaries to merge.

        Returns:
            Merged configuration dictionary.
        """

        def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
            """Deep merge two dictionaries."""
            result = dict1.copy()

            for key, value in dict2.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value

            return result

        merged = {}
        for config in configs:
            if config:
                merged = deep_merge(merged, config)

        return merged

    def load_configuration(
        self, yaml_path: str = "conf.yaml", validate: bool = True
    ) -> AppSettings:
        """Load complete configuration from all sources.

        Args:
            yaml_path: Path to YAML configuration file.
            validate: Whether to validate configuration before creating AppSettings.

        Returns:
            Complete application settings.

        Raises:
            ValidationError: If configuration is invalid.
        """
        # Load from YAML file
        yaml_config = self.load_from_yaml(yaml_path)

        # Load from environment variables
        env_config = self.load_from_env()

        # Merge configurations (env vars override YAML)
        merged_config = self.merge_configs(yaml_config, env_config)

        # Validate configuration if requested
        if validate:
            if not validate_configuration(merged_config):
                logger.warning(
                    "Configuration validation failed, but continuing with current config"
                )

        # Create and validate AppSettings
        try:
            self._settings = AppSettings(**merged_config)
            logger.info("Successfully loaded and validated configuration")
            return self._settings
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}")

    def get_settings(self) -> AppSettings:
        """Get the loaded settings.

        Returns:
            Application settings.

        Raises:
            RuntimeError: If settings haven't been loaded yet.
        """
        if self._settings is None:
            return self.load_configuration()
        return self._settings

    def reload_configuration(self, yaml_path: str = "conf.yaml") -> AppSettings:
        """Reload configuration from all sources.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            Updated application settings.
        """
        self._settings = None
        return self.load_configuration(yaml_path)


# Global configuration loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get the global configuration loader instance.

    Returns:
        Configuration loader instance.
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_configuration(yaml_path: str = "conf.yaml") -> AppSettings:
    """Convenience function to load configuration.

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        Application settings.
    """
    loader = get_config_loader()
    return loader.load_configuration(yaml_path)


def get_settings() -> AppSettings:
    """Convenience function to get loaded settings.

    Returns:
        Application settings.
    """
    loader = get_config_loader()
    return loader.get_settings()
