# SPDX-License-Identifier: MIT
"""
Configuration Integration
Integrate researcher configuration with main configuration system
"""

import logging
from typing import Dict, Any, Optional

from src.config.configuration import Configuration
from src.config.researcher_config_loader import (
    ResearcherConfigLoader,
    ResearcherConfig,
    EnhancedReflectionConfig
)

logger = logging.getLogger(__name__)


class ConfigurationIntegrator:
    """Integrates researcher configuration with main configuration system."""

    def __init__(self, researcher_config_path: Optional[str] = None):
        self.researcher_loader = ResearcherConfigLoader(researcher_config_path)
        self._researcher_config: Optional[ResearcherConfig] = None
        self._integrated_config: Optional[Configuration] = None

    def load_researcher_config(self) -> ResearcherConfig:
        """Load researcher configuration."""
        if self._researcher_config is None:
            self._researcher_config = self.researcher_loader.create_researcher_config()
            logger.info("Researcher configuration loaded successfully")
        return self._researcher_config

    def integrate_with_main_config(self, main_config: Configuration) -> Configuration:
        """Integrate researcher configuration with main configuration."""
        researcher_config = self.load_researcher_config()
        
        # Update main configuration with researcher settings
        self._update_reflection_settings(main_config, researcher_config.enhanced_reflection)
        self._update_isolation_settings(main_config, researcher_config.isolation)
        self._update_performance_settings(main_config, researcher_config.performance)
        
        self._integrated_config = main_config
        logger.info("Configuration integration completed")
        return main_config

    def _update_reflection_settings(self, main_config: Configuration, reflection_config: EnhancedReflectionConfig):
        """Update main configuration with reflection settings."""
        # Only update if reflection is enabled in researcher config
        if reflection_config.enable_enhanced_reflection:
            main_config.enable_enhanced_reflection = True
            main_config.max_reflection_loops = reflection_config.max_reflection_loops
            main_config.reflection_temperature = reflection_config.reflection_temperature
            main_config.reflection_trigger_threshold = reflection_config.reflection_trigger_threshold
            main_config.reflection_confidence_threshold = reflection_config.reflection_confidence_threshold
            main_config.enable_reflection_integration = reflection_config.enable_reflection_integration
            main_config.enable_progressive_reflection = reflection_config.enable_progressive_reflection
            main_config.enable_reflection_metrics = reflection_config.enable_reflection_metrics
            
            logger.info("Enhanced reflection settings integrated")
        else:
            logger.info("Enhanced reflection is disabled in researcher configuration")

    def _update_isolation_settings(self, main_config: Configuration, isolation_config):
        """Update main configuration with isolation settings."""
        if isolation_config.enable_isolation:
            main_config.enable_researcher_isolation = True
            main_config.researcher_max_local_context = isolation_config.max_local_context_size
            
            # Map isolation settings to main config
            if isolation_config.compression_threshold > 0.8:
                main_config.researcher_isolation_level = "aggressive"
            elif isolation_config.compression_threshold > 0.6:
                main_config.researcher_isolation_level = "moderate"
            else:
                main_config.researcher_isolation_level = "minimal"
                
            logger.info(f"Isolation settings integrated with level: {main_config.researcher_isolation_level}")

    def _update_performance_settings(self, main_config: Configuration, performance_config):
        """Update main configuration with performance settings."""
        # Map performance thresholds to main config
        if hasattr(main_config, 'advanced_context_config'):
            main_config.advanced_context_config.compression_threshold = performance_config.performance_threshold
            logger.info("Performance settings integrated")

    def get_integrated_reflection_config(self) -> Dict[str, Any]:
        """Get integrated reflection configuration."""
        if self._researcher_config is None:
            self.load_researcher_config()
            
        reflection_config = self._researcher_config.enhanced_reflection
        
        return {
            "enabled": reflection_config.enable_enhanced_reflection,
            "max_loops": reflection_config.max_reflection_loops,
            "model": reflection_config.reflection_model,
            "temperature": reflection_config.reflection_temperature,
            "knowledge_gap_threshold": reflection_config.knowledge_gap_threshold,
            "sufficiency_threshold": reflection_config.sufficiency_threshold,
            "enable_caching": reflection_config.enable_reflection_caching,
            "trigger_threshold": reflection_config.reflection_trigger_threshold,
            "confidence_threshold": reflection_config.reflection_confidence_threshold,
            "integration_enabled": reflection_config.enable_reflection_integration,
            "progressive_enabled": reflection_config.enable_progressive_reflection,
            "metrics_enabled": reflection_config.enable_reflection_metrics,
            "timeout_seconds": reflection_config.reflection_timeout_seconds,
            "max_knowledge_gaps": reflection_config.max_knowledge_gaps,
            "max_follow_up_queries": reflection_config.max_follow_up_queries,
            "debugging_enabled": reflection_config.enable_reflection_debugging
        }

    def get_integrated_isolation_config(self) -> Dict[str, Any]:
        """Get integrated isolation configuration."""
        if self._researcher_config is None:
            self.load_researcher_config()
            
        isolation_config = self._researcher_config.isolation
        
        return {
            "enabled": isolation_config.enable_isolation,
            "max_local_context_size": isolation_config.max_local_context_size,
            "token_budget": isolation_config.token_budget,
            "compression_threshold": isolation_config.compression_threshold,
            "max_concurrent_sessions": isolation_config.max_concurrent_sessions,
            "session_timeout": isolation_config.session_timeout
        }

    def validate_configuration(self) -> bool:
        """Validate the integrated configuration."""
        try:
            if self._researcher_config is None:
                self.load_researcher_config()
                
            reflection_config = self._researcher_config.enhanced_reflection
            
            # Validate reflection configuration
            if reflection_config.enable_enhanced_reflection:
                if reflection_config.max_reflection_loops <= 0:
                    logger.error("Invalid max_reflection_loops: must be positive")
                    return False
                    
                if not (0.0 <= reflection_config.reflection_temperature <= 2.0):
                    logger.error("Invalid reflection_temperature: must be between 0.0 and 2.0")
                    return False
                    
                if not (0.0 <= reflection_config.knowledge_gap_threshold <= 1.0):
                    logger.error("Invalid knowledge_gap_threshold: must be between 0.0 and 1.0")
                    return False
                    
                if not (0.0 <= reflection_config.sufficiency_threshold <= 1.0):
                    logger.error("Invalid sufficiency_threshold: must be between 0.0 and 1.0")
                    return False
                    
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the integrated configuration."""
        if self._researcher_config is None:
            self.load_researcher_config()
            
        return {
            "reflection": {
                "enabled": self._researcher_config.enhanced_reflection.enable_enhanced_reflection,
                "integration_enabled": self._researcher_config.enhanced_reflection.enable_reflection_integration,
                "progressive_enabled": self._researcher_config.enhanced_reflection.enable_progressive_reflection,
                "metrics_enabled": self._researcher_config.enhanced_reflection.enable_reflection_metrics
            },
            "isolation": {
                "enabled": self._researcher_config.isolation.enable_isolation,
                "max_context_size": self._researcher_config.isolation.max_local_context_size,
                "compression_threshold": self._researcher_config.isolation.compression_threshold
            },
            "performance": {
                "threshold": self._researcher_config.performance.performance_threshold,
                "success_rate_threshold": self._researcher_config.performance.success_rate_threshold,
                "complexity_threshold": self._researcher_config.performance.complexity_threshold
            },
            "monitoring": {
                "enabled": self._researcher_config.monitoring.enable_predictive_analytics,
                "interval": self._researcher_config.monitoring.monitoring_interval,
                "retention_days": self._researcher_config.monitoring.metrics_retention_days
            }
        }


# Global configuration integrator instance
config_integrator = ConfigurationIntegrator()