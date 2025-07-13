# SPDX-License-Identifier: MIT
"""
Configuration Loader
Load configuration from YAML files, including model token limits
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

from src.utils.tokens.content_processor import ModelTokenLimits
from src.config.configuration import Configuration, AdvancedContextConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration Loader"""

    def __init__(self, config_path: Optional[str] = None):
        # Use relative path, relative to project root directory
        self.config_path = config_path or "conf.yaml"
        # Simplified default model limits, only provide a general default value
        self.default_model_limits = ModelTokenLimits(
            input_limit=32000,
            output_limit=4096,
            context_window=32000,
            safety_margin=0.8,
        )

    def load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        if not os.path.exists(self.config_path):
            logger.warning(
                f"Configuration file {self.config_path} does not exist, using default configuration"
            )
            return {}

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Successfully loaded configuration file: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            return {}

    def parse_model_limits(self, config: Dict[str, Any]) -> Dict[str, ModelTokenLimits]:
        """Parse model token limit configuration"""
        model_limits = {}

        # Read token limits from BASIC_MODEL configuration
        basic_model = config.get("BASIC_MODEL", {})
        basic_model_name = basic_model.get("model", "")
        basic_token_limits = basic_model.get("token_limits", {})

        # Read token limits from REASONING_MODEL configuration (if exists)
        reasoning_model = config.get("REASONING_MODEL", {})
        reasoning_model_name = reasoning_model.get("model", "")
        reasoning_token_limits = reasoning_model.get("token_limits", {})

        # If BASIC_MODEL has token_limits configured, use user configuration
        if basic_model_name and basic_token_limits:
            try:
                model_limits[basic_model_name] = ModelTokenLimits(
                    input_limit=basic_token_limits.get("input_limit", 32000),
                    output_limit=basic_token_limits.get("output_limit", 4096),
                    context_window=basic_token_limits.get("context_window", 32000),
                    safety_margin=basic_token_limits.get("safety_margin", 0.8),
                )
                logger.info(f"Loading BASIC_MODEL token limits: {basic_model_name}")
            except Exception as e:
                logger.error(
                    f"Failed to parse BASIC_MODEL {basic_model_name} token limit configuration: {e}"
                )

        # If REASONING_MODEL has token_limits configured, use user configuration
        if reasoning_model_name and reasoning_token_limits:
            try:
                model_limits[reasoning_model_name] = ModelTokenLimits(
                    input_limit=reasoning_token_limits.get("input_limit", 32000),
                    output_limit=reasoning_token_limits.get("output_limit", 4096),
                    context_window=reasoning_token_limits.get("context_window", 32000),
                    safety_margin=reasoning_token_limits.get("safety_margin", 0.8),
                )
                logger.info(
                    f"Loading REASONING_MODEL token limits: {reasoning_model_name}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to parse REASONING_MODEL {reasoning_model_name} token limit configuration: {e}"
                )

        # Compatible with legacy MODEL_TOKEN_LIMITS configuration (if exists)
        custom_limits = config.get("MODEL_TOKEN_LIMITS", {})

        for model_name, limits_config in custom_limits.items():
            try:
                if isinstance(limits_config, dict):
                    model_limits[model_name] = ModelTokenLimits(
                        input_limit=limits_config.get("input_limit", 4000),
                        output_limit=limits_config.get("output_limit", 1000),
                        context_window=limits_config.get("context_window", 8000),
                        safety_margin=limits_config.get("safety_margin", 0.8),
                    )
                    logger.info(f"Loading custom model limits: {model_name}")
            except Exception as e:
                logger.error(
                    f"Failed to parse model {model_name} token limit configuration: {e}"
                )

        # If no model limits found, add default deepseek configurations
        if not model_limits:
            logger.warning(
                "No model token limits found in configuration, using defaults"
            )
            default_models = {
                "deepseek-chat": ModelTokenLimits(
                    input_limit=65536,
                    output_limit=8192,
                    context_window=131072,
                    safety_margin=0.2,
                ),
                "deepseek-reasoner": ModelTokenLimits(
                    input_limit=65536,
                    output_limit=8192,
                    context_window=131072,
                    safety_margin=0.2,
                ),
            }
            model_limits.update(default_models)
            logger.info(
                f"Added default configurations for models: {list(default_models.keys())}"
            )

        return model_limits

    def parse_advanced_context_config(
        self, config: Dict[str, Any]
    ) -> AdvancedContextConfig:
        """Parse advanced context management configuration"""
        advanced_config = config.get("ADVANCED_CONTEXT_MANAGEMENT", {})

        try:
            return AdvancedContextConfig(
                max_context_ratio=advanced_config.get("max_context_ratio", 0.6),
                sliding_window_size=advanced_config.get("sliding_window_size", 5),
                overlap_ratio=advanced_config.get("overlap_ratio", 0.2),
                compression_threshold=advanced_config.get("compression_threshold", 0.8),
                default_strategy=advanced_config.get("default_strategy", "adaptive"),
                priority_weights=advanced_config.get(
                    "priority_weights",
                    {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.1},
                ),
                enable_caching=advanced_config.get("enable_caching", True),
                enable_analytics=advanced_config.get("enable_analytics", True),
                debug_mode=advanced_config.get("debug_mode", False),
            )
        except Exception as e:
            logger.error(
                f"Failed to parse advanced context management configuration: {e}"
            )
            return AdvancedContextConfig()  # Return default configuration

    def create_configuration(self) -> Configuration:
        """Create configuration object"""
        config_data = self.load_config()
        model_limits = self.parse_model_limits(config_data)

        # If no model limits are configured, use default values
        if not model_limits:
            model_limits = {"default": self.default_model_limits}

        # Parse intelligent content processing configuration
        content_config = config_data.get("CONTENT_PROCESSING", {})

        # Parse parallel execution configuration
        parallel_config = config_data.get("PARALLEL_EXECUTION", {})

        # Parse advanced context management configuration
        advanced_context_config = self.parse_advanced_context_config(config_data)

        # Create model instances
        basic_model = None
        reasoning_model = None

        try:
            from src.llms.llm import get_llm_by_type

            # Try to create basic model instance
            if config_data.get("BASIC_MODEL"):
                basic_model = get_llm_by_type("basic")
            # Try to create reasoning model instance
            if config_data.get("REASONING_MODEL"):
                reasoning_model = get_llm_by_type("reasoning")
        except Exception as e:
            logger.warning(f"Failed to create model instances: {e}")

        return Configuration(
            # enable_smart_chunking removed - smart chunking is now always enabled
            enable_content_summarization=content_config.get(
                "enable_content_summarization", True
            ),
            enable_smart_filtering=content_config.get("enable_smart_filtering", True),
            # chunk_strategy removed - smart chunking now uses 'auto' strategy by default
            summary_type=content_config.get("summary_type", "comprehensive"),
            model_token_limits=model_limits,
            max_search_results=config_data.get("max_search_results", 3),
            enable_parallel_execution=parallel_config.get(
                "enable_parallel_execution", True
            ),
            max_parallel_tasks=parallel_config.get("max_parallel_tasks", 3),
            max_context_steps_parallel=parallel_config.get(
                "max_context_steps_parallel", 1
            ),  # Reduced for token optimization
            disable_context_parallel=parallel_config.get(
                "disable_context_parallel", False
            ),
            basic_model=basic_model,
            reasoning_model=reasoning_model,
            advanced_context_config=advanced_context_config,
        )

    def save_example_config(self, output_path: Optional[str] = None):
        """Save example configuration file"""
        output_path = output_path or "conf.yaml.example"

        example_config = {
            "BASIC_MODEL": {
                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                "model": "doubao-1-5-pro-32k-250115",
                "api_key": "xxxx",
                "token_limits": {
                    "input_limit": 32000,
                    "output_limit": 4096,
                    "context_window": 32000,
                    "safety_margin": 0.8,
                },
            },
            "REASONING_MODEL": {
                "base_url": "https://ark-cn-beijing.bytedance.net/api/v3",
                "model": "doubao-1-5-thinking-pro-m-250428",
                "api_key": "xxxx",
                "token_limits": {
                    "input_limit": 32000,
                    "output_limit": 8192,
                    "context_window": 32000,
                    "safety_margin": 0.8,
                },
            },
            "CONTENT_PROCESSING": {
                # 'enable_smart_chunking': True,  # Removed - smart chunking is now always enabled
                "enable_content_summarization": True,
                "enable_smart_filtering": True,  # Enable LLM-based search result filtering
                # 'chunk_strategy': 'auto',  # Removed - smart chunking now uses 'auto' strategy by default
                "summary_type": "key_points",  # comprehensive, key_points, abstract
            },
            "PARALLEL_EXECUTION": {
                "enable_parallel_execution": True,
                "max_parallel_tasks": 2,  # Maximum number of parallel tasks
                "max_context_steps_parallel": 3,  # Maximum context steps in parallel execution
            },
            "max_search_results": 3,
        }

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    example_config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                )
            logger.info(f"Example configuration file saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save example configuration file: {e}")


# Global configuration loader instance
config_loader = ConfigLoader()
