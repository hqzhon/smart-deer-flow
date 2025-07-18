# -*- coding: utf-8 -*-
"""
Configuration Validators
Provides validation functions for configuration data
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.config.models import AppSettings

logger = logging.getLogger(__name__)


def validate_llm_model_config(model_config: Dict[str, Any], model_name: str) -> bool:
    """Validate individual LLM model configuration"""
    required_fields = ["model", "api_key", "base_url"]

    for field in required_fields:
        if field not in model_config:
            logger.error(f"Missing required field '{field}' in {model_name}")
            return False
        if not model_config[field]:
            logger.error(f"Field '{field}' cannot be empty in {model_name}")
            return False

    # Validate temperature if present
    temperature = model_config.get("temperature")
    if temperature is not None and not (0 <= temperature <= 2):
        logger.error(
            f"Invalid temperature in {model_name}: {temperature}. Must be between 0 and 2"
        )
        return False

    # Validate max_tokens if present
    max_tokens = model_config.get("max_tokens")
    if max_tokens is not None and max_tokens <= 0:
        logger.error(
            f"Invalid max_tokens in {model_name}: {max_tokens}. Must be positive"
        )
        return False

    # Validate timeout if present
    timeout = model_config.get("timeout")
    if timeout is not None and timeout <= 0:
        logger.error(f"Invalid timeout in {model_name}: {timeout}. Must be positive")
        return False

    return True


def validate_app_settings(config: Dict[str, Any]) -> bool:
    """Validate application settings configuration"""
    try:
        # Try to create AppSettings instance to validate structure
        AppSettings(**config)
        return True
    except Exception as e:
        logger.error(f"AppSettings validation failed: {e}")
        return False


def validate_llm_config(config: Dict[str, Any]) -> bool:
    """Validate LLM configuration"""
    llm_config = config.get("llm", {})

    # Check that at least basic_model is configured
    basic_model = llm_config.get("basic_model")
    if not basic_model:
        logger.error("Missing required LLM field: basic_model")
        return False

    # Validate basic_model configuration
    if not validate_llm_model_config(basic_model, "basic_model"):
        return False

    # Validate reasoning_model if present
    reasoning_model = llm_config.get("reasoning_model")
    if reasoning_model and not validate_llm_model_config(
        reasoning_model, "reasoning_model"
    ):
        return False

    # Validate temperature range
    temperature = llm_config.get("temperature", 0.7)
    if not 0 <= temperature <= 2:
        logger.error(f"Invalid temperature: {temperature}. Must be between 0 and 2")
        return False

    # Validate max_tokens
    max_tokens = llm_config.get("max_tokens")
    if max_tokens is not None and max_tokens <= 0:
        logger.error(f"Invalid max_tokens: {max_tokens}. Must be positive")
        return False

    # Validate timeout
    timeout = llm_config.get("timeout")
    if timeout is not None and timeout <= 0:
        logger.error(f"Invalid timeout: {timeout}. Must be positive")
        return False

    return True


def validate_database_config(config: Dict[str, Any]) -> bool:
    """Validate database configuration"""
    db_config = config.get("database", {})

    # Check database type
    db_type = db_config.get("type", "sqlite")
    valid_types = ["sqlite", "postgresql", "mysql"]
    if db_type not in valid_types:
        logger.error(f"Invalid database type: {db_type}. Valid types: {valid_types}")
        return False

    # Validate SQLite path
    if db_type == "sqlite":
        db_path = db_config.get("path", "data/deer_flow.db")
        db_dir = Path(db_path).parent
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Cannot create database directory {db_dir}: {e}")
                return False

    # Validate connection parameters for other databases
    elif db_type in ["postgresql", "mysql"]:
        required_fields = ["host", "port", "database", "username"]
        for field in required_fields:
            if field not in db_config:
                logger.error(f"Missing required database field: {field}")
                return False

    return True


def validate_agent_config(config: Dict[str, Any]) -> bool:
    """Validate agent configuration"""
    agents_config = config.get("agents", {})

    # Validate max_iterations
    max_iterations = agents_config.get("max_iterations", 10)
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        logger.error(
            f"Invalid max_iterations: {max_iterations}. Must be positive integer"
        )
        return False

    # Validate timeout
    timeout = agents_config.get("timeout", 300)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        logger.error(f"Invalid timeout: {timeout}. Must be positive number")
        return False

    return True


def validate_research_config(config: Dict[str, Any]) -> bool:
    """Validate research configuration"""
    research_config = config.get("research", {})

    # Validate max_sources
    max_sources = research_config.get("max_sources", 10)
    if not isinstance(max_sources, int) or max_sources <= 0:
        logger.error(f"Invalid max_sources: {max_sources}. Must be positive integer")
        return False

    # Validate search_depth
    search_depth = research_config.get("search_depth", 3)
    if not isinstance(search_depth, int) or search_depth <= 0:
        logger.error(f"Invalid search_depth: {search_depth}. Must be positive integer")
        return False

    return True


def validate_file_paths(config: Dict[str, Any]) -> bool:
    """Validate file paths in configuration"""
    paths_to_check = []

    # Collect paths from various config sections
    database_config = config.get("database", {})
    if database_config.get("type") == "sqlite":
        db_path = database_config.get("path")
        if db_path:
            paths_to_check.append(
                ("database.path", db_path, True)
            )  # True means create if not exists

    content_config = config.get("content", {})
    output_dir = content_config.get("output_dir")
    if output_dir:
        paths_to_check.append(("content.output_dir", output_dir, True))

    # Validate paths
    for path_name, path_value, create_if_missing in paths_to_check:
        path = Path(path_value)

        if create_if_missing:
            # Create directory if it doesn't exist
            if path.suffix:  # It's a file path
                parent_dir = path.parent
            else:  # It's a directory path
                parent_dir = path

            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Cannot create directory for {path_name}: {e}")
                return False
        else:
            # Check if path exists
            if not path.exists():
                logger.error(f"Path does not exist for {path_name}: {path_value}")
                return False

    return True


def validate_environment_variables(config: Dict[str, Any]) -> bool:
    """Validate required environment variables"""
    # For now, just validate that basic environment is set up
    # This can be extended based on actual requirements
    return True


def get_all_validators() -> List[callable]:
    """Get all configuration validators"""
    return [
        validate_app_settings,
        validate_llm_config,
        validate_database_config,
        validate_agent_config,
        validate_research_config,
        validate_file_paths,
        validate_environment_variables,
    ]


def validate_configuration(
    config: Dict[str, Any], validators: Optional[List[callable]] = None
) -> bool:
    """Validate configuration using all or specified validators"""
    if validators is None:
        validators = get_all_validators()

    for validator in validators:
        try:
            if not validator(config):
                logger.error(f"Configuration validation failed in {validator.__name__}")
                return False
        except Exception as e:
            logger.error(f"Error in validator {validator.__name__}: {e}")
            return False

    logger.info("Configuration validation passed")
    return True
