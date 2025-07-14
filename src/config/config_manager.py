# SPDX-License-Identifier: MIT
"""
Configuration Manager
Provides unified configuration access interface with backward compatibility
"""

import logging
from typing import Dict, Any, Optional, Union
from functools import lru_cache

from src.config.configuration import Configuration
from src.config.config_loader import ConfigLoader
from src.config.config_integration import ConfigurationIntegrator
from src.config.researcher_config_loader import ResearcherConfig, EnhancedReflectionConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Unified configuration manager with backward compatibility."""

    def __init__(self, 
                 main_config_path: Optional[str] = None,
                 researcher_config_path: Optional[str] = None):
        self.main_loader = ConfigLoader(main_config_path)
        self.integrator = ConfigurationIntegrator(researcher_config_path)
        self._main_config: Optional[Configuration] = None
        self._researcher_config: Optional[ResearcherConfig] = None
        self._integrated: bool = False

    @lru_cache(maxsize=1)
    def get_main_config(self) -> Configuration:
        """Get main configuration with caching."""
        if self._main_config is None:
            self._main_config = self.main_loader.create_configuration()
            logger.info("Main configuration loaded")
        return self._main_config

    @lru_cache(maxsize=1)
    def get_researcher_config(self) -> ResearcherConfig:
        """Get researcher configuration with caching."""
        if self._researcher_config is None:
            self._researcher_config = self.integrator.load_researcher_config()
            logger.info("Researcher configuration loaded")
        return self._researcher_config

    def get_integrated_config(self) -> Configuration:
        """Get integrated configuration."""
        if not self._integrated:
            main_config = self.get_main_config()
            self.integrator.integrate_with_main_config(main_config)
            self._integrated = True
            logger.info("Configuration integration completed")
        return self._main_config

    def is_reflection_enabled(self) -> bool:
        """Check if enhanced reflection is enabled."""
        researcher_config = self.get_researcher_config()
        return researcher_config.enhanced_reflection.enable_enhanced_reflection

    def is_isolation_enabled(self) -> bool:
        """Check if researcher isolation is enabled."""
        researcher_config = self.get_researcher_config()
        return researcher_config.isolation.enable_isolation

    def get_reflection_config(self) -> Dict[str, Any]:
        """Get reflection configuration."""
        return self.integrator.get_integrated_reflection_config()

    def get_isolation_config(self) -> Dict[str, Any]:
        """Get isolation configuration."""
        return self.integrator.get_integrated_isolation_config()

    def get_config_value(self, key: str, default: Any = None, section: Optional[str] = None) -> Any:
        """Get configuration value with backward compatibility."""
        try:
            if section == "reflection" or key.startswith("reflection_"):
                reflection_config = self.get_reflection_config()
                clean_key = key.replace("reflection_", "").replace("enable_enhanced_", "")
                return reflection_config.get(clean_key, default)
            
            elif section == "isolation" or key.startswith("isolation_"):
                isolation_config = self.get_isolation_config()
                clean_key = key.replace("isolation_", "")
                return isolation_config.get(clean_key, default)
            
            else:
                # Try main config first
                main_config = self.get_integrated_config()
                if hasattr(main_config, key):
                    return getattr(main_config, key)
                
                # Fallback to researcher config
                researcher_config = self.get_researcher_config()
                return self._get_nested_value(researcher_config.__dict__, key, default)
                
        except Exception as e:
            logger.warning(f"Failed to get config value for key '{key}': {e}")
            return default

    def _get_nested_value(self, config_dict: Dict[str, Any], key: str, default: Any) -> Any:
        """Get nested configuration value."""
        keys = key.split('.')
        current = config_dict
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            elif hasattr(current, k):
                current = getattr(current, k)
            else:
                return default
        
        return current

    def set_config_value(self, key: str, value: Any, section: Optional[str] = None) -> bool:
        """Set configuration value (runtime only)."""
        try:
            if section == "reflection" or key.startswith("reflection_"):
                # Update reflection config in main config
                main_config = self.get_integrated_config()
                clean_key = key.replace("reflection_", "")
                
                if clean_key == "enabled" or clean_key == "enable_enhanced_reflection":
                    main_config.enable_enhanced_reflection = bool(value)
                elif clean_key == "max_loops":
                    main_config.max_reflection_loops = int(value)
                elif clean_key == "temperature":
                    main_config.reflection_temperature = float(value)
                elif clean_key == "trigger_threshold":
                    main_config.reflection_trigger_threshold = int(value)
                elif clean_key == "confidence_threshold":
                    main_config.reflection_confidence_threshold = float(value)
                
                logger.info(f"Updated reflection config: {key} = {value}")
                return True
            
            elif section == "isolation" or key.startswith("isolation_"):
                # Update isolation config in main config
                main_config = self.get_integrated_config()
                clean_key = key.replace("isolation_", "")
                
                if clean_key == "enabled" or clean_key == "enable_researcher_isolation":
                    main_config.enable_researcher_isolation = bool(value)
                elif clean_key == "max_local_context":
                    main_config.researcher_max_local_context = int(value)
                elif clean_key == "level":
                    main_config.researcher_isolation_level = str(value)
                
                logger.info(f"Updated isolation config: {key} = {value}")
                return True
            
            else:
                # Update main config
                main_config = self.get_integrated_config()
                if hasattr(main_config, key):
                    setattr(main_config, key, value)
                    logger.info(f"Updated main config: {key} = {value}")
                    return True
            
            logger.warning(f"Unknown configuration key: {key}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to set config value for key '{key}': {e}")
            return False

    def validate_configuration(self) -> bool:
        """Validate all configurations."""
        try:
            # Validate main configuration
            main_config = self.get_integrated_config()
            if main_config is None:
                logger.error("Main configuration is None")
                return False
            
            # Validate researcher configuration
            if not self.integrator.validate_configuration():
                logger.error("Researcher configuration validation failed")
                return False
            
            # Validate integration consistency
            if self.is_reflection_enabled():
                reflection_config = self.get_reflection_config()
                if not reflection_config.get("enabled", False):
                    logger.error("Reflection configuration inconsistency detected")
                    return False
            
            logger.info("All configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get comprehensive configuration summary."""
        try:
            main_config = self.get_integrated_config()
            researcher_summary = self.integrator.get_configuration_summary()
            
            return {
                "main_config": {
                    "parallel_execution": main_config.enable_parallel_execution,
                    "max_parallel_tasks": main_config.max_parallel_tasks,
                    "max_search_results": main_config.max_search_results,
                    "content_summarization": main_config.enable_content_summarization,
                    "smart_filtering": main_config.enable_smart_filtering
                },
                "researcher_config": researcher_summary,
                "integration_status": {
                    "integrated": self._integrated,
                    "reflection_enabled": self.is_reflection_enabled(),
                    "isolation_enabled": self.is_isolation_enabled()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate configuration summary: {e}")
            return {"error": str(e)}

    def reload_configuration(self) -> bool:
        """Reload all configurations."""
        try:
            # Clear caches
            self.get_main_config.cache_clear()
            self.get_researcher_config.cache_clear()
            
            # Reset state
            self._main_config = None
            self._researcher_config = None
            self._integrated = False
            
            # Reload configurations
            self.get_integrated_config()
            
            logger.info("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False

    def export_configuration(self, output_path: str, format: str = "json") -> bool:
        """Export current configuration to file."""
        try:
            import json
            import yaml
            
            config_summary = self.get_configuration_summary()
            
            if format.lower() == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(config_summary, f, indent=2, ensure_ascii=False)
            elif format.lower() == "yaml":
                with open(output_path, "w", encoding="utf-8") as f:
                    yaml.dump(config_summary, f, default_flow_style=False, allow_unicode=True)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Configuration exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False


# Global configuration manager instance
config_manager = ConfigManager()