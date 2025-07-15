# SPDX-License-Identifier: MIT
"""
Simplified Configuration Loader using pydantic-settings.
This replaces the manual env_mappings with automatic environment variable loading.
"""

import yaml
import logging
from typing import Any, Dict, Optional
from pathlib import Path

from .settings import AppSettings
from .validators import validate_configuration

logger = logging.getLogger(__name__)


class ConfigLoaderV2:
    """Simplified configuration loader using pydantic-settings for automatic env loading."""
    
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
            yaml.YAMLError: If YAML is invalid.
        """
        yaml_file = self.config_dir / yaml_path
        if not yaml_file.exists():
            logger.warning(f"Configuration file not found: {yaml_file}")
            return {}
        
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Successfully loaded configuration file: {yaml_file}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {yaml_file}: {e}")
            return {}
    
    def merge_configs(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge YAML configuration with environment variables.
        
        Args:
            yaml_config: Configuration from YAML file.
            
        Returns:
            Merged configuration dictionary.
            
        Note:
            Environment variables are automatically loaded by pydantic-settings,
            so we only need to provide the YAML config as override values.
        """
        # With pydantic-settings, we can pass the YAML config directly
        # Environment variables will be automatically loaded and take precedence
        return yaml_config
    
    def load_configuration(self, yaml_path: str = "conf.yaml", validate: bool = True) -> AppSettings:
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
        
        # Validate configuration if requested
        if validate and yaml_config:
            if not validate_configuration(yaml_config):
                logger.warning("Configuration validation failed, but continuing with current config")
        
        # Create AppSettings - pydantic-settings will automatically:
        # 1. Load environment variables based on env_prefix
        # 2. Apply YAML config as overrides
        # 3. Use nested delimiter for complex structures
        try:
            # Pass YAML config as keyword arguments to override defaults
            # Environment variables will be automatically loaded
            self._settings = AppSettings(**yaml_config)
            logger.info("Successfully loaded and validated configuration using pydantic-settings")
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
_config_loader_v2: Optional[ConfigLoaderV2] = None


def get_config_loader_v2() -> ConfigLoaderV2:
    """Get the global configuration loader instance.
    
    Returns:
        Configuration loader instance.
    """
    global _config_loader_v2
    if _config_loader_v2 is None:
        _config_loader_v2 = ConfigLoaderV2()
    return _config_loader_v2


def load_configuration_v2(yaml_path: str = "conf.yaml") -> AppSettings:
    """Convenience function to load configuration using pydantic-settings.
    
    Args:
        yaml_path: Path to YAML configuration file.
        
    Returns:
        Application settings.
    """
    loader = get_config_loader_v2()
    return loader.load_configuration(yaml_path)


def get_settings_v2() -> AppSettings:
    """Convenience function to get loaded settings using pydantic-settings.
    
    Returns:
        Application settings.
    """
    loader = get_config_loader_v2()
    return loader.get_settings()