# -*- coding: utf-8 -*-
"""
Dependency Injection Configuration for Configuration Services
Configures and registers configuration-related services in the DI container
"""

import logging
from typing import Optional

from src.config.config_service import (
    ConfigurationService, 
    IAppConfigurationService,
    get_configuration_service
)
from src.config.config_loader import ConfigLoader, get_config_loader
from src.config.models import AppSettings
from src.utils.system.dependency_injection import (
    DependencyInjectionContainer,
    IConfigurationService,
    global_container,
    configure_services
)

logger = logging.getLogger(__name__)


def configure_configuration_services(container: DependencyInjectionContainer) -> None:
    """Configure configuration services in DI container"""
    
    # Register ConfigLoader as singleton
    container.register_singleton(
        ConfigLoader,
        factory=lambda: get_config_loader()
    )
    
    # Register ConfigurationService as singleton
    container.register_singleton(
        ConfigurationService,
        factory=lambda config_loader: ConfigurationService(
            config_loader=config_loader
        )
    )
    
    # Register interface implementations
    container.register_singleton(
        IConfigurationService,
        factory=lambda: container.resolve(ConfigurationService)
    )
    
    container.register_singleton(
        IAppConfigurationService,
        factory=lambda: container.resolve(ConfigurationService)
    )
    
    # Register AppSettings as singleton (lazy loaded)
    container.register_singleton(
        AppSettings,
        factory=lambda config_service: config_service.get_app_settings()
    )
    
    logger.info("Configuration services registered in DI container")


def setup_configuration_di() -> DependencyInjectionContainer:
    """Setup configuration dependency injection"""
    configure_services(configure_configuration_services)
    return global_container


def get_app_settings_from_di() -> AppSettings:
    """Get AppSettings from DI container"""
    return global_container.resolve(AppSettings)


def get_config_service_from_di() -> ConfigurationService:
    """Get ConfigurationService from DI container"""
    return global_container.resolve(ConfigurationService)