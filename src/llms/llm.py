# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any, Dict
import os
import httpx

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from typing import get_args

from src.config import get_settings
from src.config.agents import LLMType
from src.utils.tokens.content_processor import ModelTokenLimits

# Cache for LLM instances
_llm_cache: dict[LLMType, ChatOpenAI] = {}

# Global registry for model token limits
_model_token_limits_registry: dict[str, ModelTokenLimits] = {}


def _get_config_file_path() -> str:
    """Get the path to the configuration file."""
    return str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())


def _store_model_token_limits(model_name: str, token_limits: Dict[str, Any]) -> None:
    """Store model token limits in global registry."""
    try:
        limits = ModelTokenLimits(
            input_limit=token_limits.get("input_limit", 32000),
            output_limit=token_limits.get("output_limit", 4096),
            context_window=token_limits.get("context_window", 32000),
            safety_margin=token_limits.get("safety_margin", 0.8),
        )
        _model_token_limits_registry[model_name] = limits
    except Exception as e:
        print(f"Warning: Failed to store token limits for model {model_name}: {e}")


def get_model_token_limits_registry() -> dict[str, ModelTokenLimits]:
    """Get the global model token limits registry."""
    return _model_token_limits_registry.copy()


# Legacy function removed - no longer needed with new configuration system


def _get_env_llm_conf(llm_type: str) -> Dict[str, Any]:
    """
    Get LLM configuration from environment variables.
    Environment variables should follow the format: {LLM_TYPE}__{KEY}
    e.g., BASIC_MODEL__api_key, BASIC_MODEL__base_url
    """
    prefix = f"{llm_type.upper()}_MODEL__"
    conf = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            conf_key = key[len(prefix) :].lower()
            conf[conf_key] = value
    return conf


def _create_llm_use_conf(llm_type: LLMType, conf: Any) -> ChatOpenAI | ChatDeepSeek:
    """Create LLM instance using new configuration system."""
    from src.config.config_loader import load_configuration

    # Load configuration using new unified system
    settings = load_configuration()

    # Get LLM configuration based on type
    llm_config = {}
    if llm_type == "basic":
        if settings.llm.basic_model:
            llm_config = settings.llm.basic_model.model_dump()
    elif llm_type == "reasoning":
        if settings.llm.reasoning_model:
            llm_config = settings.llm.reasoning_model.model_dump()
    elif llm_type == "vision":
        # Vision model configuration can be added here in the future
        pass

    if not llm_config:
        raise ValueError(f"No configuration found for LLM type: {llm_type}")

    # Get configuration from environment variables (still supported for flexibility)
    env_conf = _get_env_llm_conf(llm_type)

    # Merge configurations, with environment variables taking precedence
    merged_conf = {**llm_config, **env_conf}

    if llm_type == "reasoning":
        merged_conf["api_base"] = merged_conf.pop("base_url", None)

    # Handle SSL verification settings
    verify_ssl = merged_conf.pop("verify_ssl", True)

    # Store model token limits from new configuration system
    model_name = merged_conf.get("model")
    if model_name and model_name in settings.model_token_limits:
        model_limits = settings.model_token_limits[model_name]
        _store_model_token_limits(
            model_name,
            {
                "input_limit": model_limits.get("input_limit", 32000),
                "output_limit": model_limits.get("output_limit", 4096),
                "context_window": model_limits.get("context_window", 32000),
                "safety_margin": model_limits.get("safety_margin", 0.8),
            },
        )

    # Create custom HTTP client if SSL verification is disabled
    if not verify_ssl:
        http_client = httpx.Client(verify=False)
        http_async_client = httpx.AsyncClient(verify=False)
        merged_conf["http_client"] = http_client
        merged_conf["http_async_client"] = http_async_client

    # Rename 'model' to 'model_name' for LangChain compatibility
    if "model" in merged_conf:
        merged_conf["model_name"] = merged_conf.pop("model")

    return (
        ChatOpenAI(**merged_conf)
        if llm_type != "reasoning"
        else ChatDeepSeek(**merged_conf)
    )


def get_llm_by_type(
    llm_type: LLMType,
) -> ChatOpenAI:
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    conf = get_settings().llm
    llm = _create_llm_use_conf(llm_type, conf)
    _llm_cache[llm_type] = llm
    return llm


def get_configured_llm_models() -> dict[str, list[str]]:
    """
    Get all configured LLM models grouped by type.

    Returns:
        Dictionary mapping LLM type to list of configured model names.
    """
    try:
        from src.config.config_loader import load_configuration

        # Load configuration using new unified system
        settings = load_configuration()

        configured_models: dict[str, list[str]] = {}

        for llm_type in get_args(LLMType):
            # Get configuration from new system
            llm_config = {}
            if llm_type == "basic" and settings.llm.basic_model:
                # Convert LLMModelConfig to dict
                llm_config = settings.llm.basic_model.model_dump()
            elif llm_type == "reasoning" and settings.llm.reasoning_model:
                # Convert LLMModelConfig to dict
                llm_config = settings.llm.reasoning_model.model_dump()
            elif llm_type == "vision":
                # Vision model configuration can be added here in the future
                pass

            # Get configuration from environment variables
            env_conf = _get_env_llm_conf(llm_type)

            # Merge configurations, with environment variables taking precedence
            merged_conf = {**llm_config, **env_conf}

            # Check if model is configured
            model_name = merged_conf.get("model")
            if model_name:
                configured_models.setdefault(llm_type, []).append(model_name)

        return configured_models

    except Exception as e:
        # Log error and return empty dict to avoid breaking the application
        print(f"Warning: Failed to load LLM configuration: {e}")
        return {}


# In the future, we will use reasoning_llm and vl_llm for different purposes
# reasoning_llm = get_llm_by_type("reasoning")
# vl_llm = get_llm_by_type("vision")
