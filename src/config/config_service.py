# -*- coding: utf-8 -*-
"""
Configuration Service Implementation
Provides unified configuration service that integrates with dependency injection system
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar
from abc import ABC, abstractmethod

from src.config.models import AppSettings
from src.config.config_loader import ConfigLoader, get_config_loader
from src.config.validators import validate_configuration
from src.config.cache import get_config_cache, invalidate_config_cache
from src.utils.system.dependency_injection import IConfigurationService

logger = logging.getLogger(__name__)

T = TypeVar('T')


class IAppConfigurationService(ABC):
    """Application configuration service interface"""
    
    @abstractmethod
    def get_app_settings(self) -> AppSettings:
        """Get application settings"""
        pass
    
    @abstractmethod
    def get_section(self, section_name: str, section_type: Type[T]) -> T:
        """Get configuration section with type validation"""
        pass
    
    @abstractmethod
    def reload_configuration(self) -> None:
        """Reload configuration from sources"""
        pass


class ConfigurationService(IConfigurationService, IAppConfigurationService):
    """Unified configuration service implementation"""
    
    def __init__(self, 
                 config_loader: Optional[ConfigLoader] = None):
        self._config_loader = config_loader or get_config_loader()
        self._app_settings: Optional[AppSettings] = None
        self._cache = get_config_cache()
        
    def get_app_settings(self) -> AppSettings:
        """Get application settings"""
        if self._app_settings is None:
            try:
                # Load configuration using the config loader
                self._app_settings = self._config_loader.load_configuration()
                logger.info("Application settings loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load application settings: {e}")
                # Return default settings on error
                self._app_settings = AppSettings()
                
        return self._app_settings
    
    def get_section(self, section_name: str, section_type: Type[T]) -> T:
        """Get configuration section with type validation"""
        app_settings = self.get_app_settings()
        
        if not hasattr(app_settings, section_name):
            raise ValueError(f"Configuration section '{section_name}' not found")
            
        section = getattr(app_settings, section_name)
        
        if not isinstance(section, section_type):
            raise TypeError(f"Configuration section '{section_name}' is not of type {section_type.__name__}")
            
        return section
    
    def reload_configuration(self, validate: bool = True) -> AppSettings:
        """Reload configuration from all sources"""
        logger.info("Reloading configuration")
        self._app_settings = None  # Clear cache
        
        # Validate configuration if requested
        if validate:
            app_settings = self.get_app_settings()
            if not validate_configuration(app_settings.model_dump()):
                logger.warning("Configuration validation failed during reload")
        
        return self.get_app_settings()
    
    def validate_current_configuration(self) -> bool:
        """Validate current configuration"""
        app_settings = self.get_app_settings()
        return validate_configuration(app_settings.model_dump())
    
    def clear_cache(self) -> None:
        """Clear all cached configuration data"""
        self._cache.clear()
        self._app_settings = None
        logger.info("Configuration cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get configuration cache statistics"""
        return self._cache.get_stats()
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries"""
        return self._cache.cleanup_expired()
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path"""
        try:
            # Get from app settings (for structured config)
            app_settings = self.get_app_settings()
            return self._get_nested_value(app_settings.model_dump(), key, default)
            
        except Exception as e:
            logger.warning(f"Failed to get configuration value for key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value (not supported in this implementation)"""
        logger.warning("Setting configuration values at runtime is not supported in this implementation")
        
    def _get_nested_value(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get nested value from dictionary using dot notation"""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
                
        return current
    
    def has(self, key: str) -> bool:
        """Check if configuration key exists"""
        return self.get(key) is not None
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        app_settings = self.get_app_settings()
        return app_settings.model_dump()


# Global configuration service instance
_config_service: Optional[ConfigurationService] = None


def get_configuration_service() -> ConfigurationService:
    """Get global configuration service instance"""
    global _config_service
    if _config_service is None:
        _config_service = ConfigurationService()
    return _config_service


def configure_configuration_service(config_loader: Optional[ConfigLoader] = None) -> ConfigurationService:
    """Configure global configuration service"""
    global _config_service
    _config_service = ConfigurationService(config_loader)
    return _config_service