# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import yaml
from typing import Dict, Any


def replace_env_vars(value: str) -> str:
    """Replace environment variables in string values."""
    if not isinstance(value, str):
        return value
    if value.startswith("$"):
        env_var = value[1:]
        return os.getenv(env_var, env_var)
    return value


def process_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively process dictionary to replace environment variables."""
    if not config:
        return {}
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = process_dict(value)
        elif isinstance(value, str):
            result[key] = replace_env_vars(value)
        else:
            result[key] = value
    return result


_config_cache: Dict[str, Dict[str, Any]] = {}


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load and process YAML configuration file."""
    # If file doesn't exist, return {}
    if not os.path.exists(file_path):
        return {}

    # Check if configuration already exists in cache
    if file_path in _config_cache:
        return _config_cache[file_path]

    # If not in cache, load and process configuration
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    processed_config = process_dict(config)

    # Store processed configuration in cache
    _config_cache[file_path] = processed_config
    return processed_config
