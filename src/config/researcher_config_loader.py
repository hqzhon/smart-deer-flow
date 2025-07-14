# SPDX-License-Identifier: MIT
"""
Researcher Configuration Loader
Load researcher-specific configuration from JSON files, including enhanced reflection settings
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EnhancedReflectionConfig:
    """Configuration for enhanced reflection capabilities."""
    
    enable_enhanced_reflection: bool = True
    max_reflection_loops: int = 3
    reflection_model: str = "gpt-4"
    reflection_temperature: float = 0.1
    knowledge_gap_threshold: float = 0.7
    sufficiency_threshold: float = 0.8
    enable_reflection_caching: bool = True
    reflection_trigger_threshold: int = 2
    reflection_confidence_threshold: float = 0.7
    enable_reflection_integration: bool = True
    enable_progressive_reflection: bool = True
    enable_reflection_metrics: bool = True
    reflection_timeout_seconds: int = 30
    max_knowledge_gaps: int = 5
    max_follow_up_queries: int = 3
    enable_reflection_debugging: bool = False


@dataclass
class IsolationConfig:
    """Configuration for researcher isolation."""
    
    enable_isolation: bool = True
    max_local_context_size: int = 8000
    token_budget: int = 16000
    compression_threshold: float = 0.7
    max_concurrent_sessions: int = 8
    session_timeout: int = 1800


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    
    memory_cleanup_interval: int = 300
    performance_threshold: float = 0.3
    success_rate_threshold: float = 0.8
    complexity_threshold: float = 0.6


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and metrics."""
    
    monitoring_interval: int = 60
    alert_sensitivity: str = "medium"
    metrics_retention_days: int = 7
    enable_predictive_analytics: bool = True


@dataclass
class OptimizationConfig:
    """Configuration for optimization settings."""
    
    auto_tuning_enabled: bool = False
    optimization_level: str = "balanced"
    min_data_points_for_tuning: int = 50


@dataclass
class ResearcherConfig:
    """Complete researcher configuration."""
    
    isolation: IsolationConfig = field(default_factory=IsolationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    enhanced_reflection: EnhancedReflectionConfig = field(default_factory=EnhancedReflectionConfig)


class ResearcherConfigLoader:
    """Researcher Configuration Loader for JSON files."""

    def __init__(self, config_path: Optional[str] = None):
        # Use relative path, relative to project root directory
        self.config_path = config_path or "config/researcher_config.json"

    def load_config(self) -> Dict[str, Any]:
        """Load configuration file."""
        if not os.path.exists(self.config_path):
            logger.warning(
                f"Researcher configuration file {self.config_path} does not exist, using default configuration"
            )
            return {}

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f) or {}
            logger.info(f"Successfully loaded researcher configuration file: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load researcher configuration file: {e}")
            return {}

    def parse_enhanced_reflection_config(self, config: Dict[str, Any]) -> EnhancedReflectionConfig:
        """Parse enhanced reflection configuration."""
        reflection_config = config.get("enhanced_reflection", {})
        
        try:
            return EnhancedReflectionConfig(
                enable_enhanced_reflection=reflection_config.get("enable_enhanced_reflection", True),
                max_reflection_loops=reflection_config.get("max_reflection_loops", 3),
                reflection_model=reflection_config.get("reflection_model", "gpt-4"),
                reflection_temperature=reflection_config.get("reflection_temperature", 0.1),
                knowledge_gap_threshold=reflection_config.get("knowledge_gap_threshold", 0.7),
                sufficiency_threshold=reflection_config.get("sufficiency_threshold", 0.8),
                enable_reflection_caching=reflection_config.get("enable_reflection_caching", True),
                reflection_trigger_threshold=reflection_config.get("reflection_trigger_threshold", 2),
                reflection_confidence_threshold=reflection_config.get("reflection_confidence_threshold", 0.7),
                enable_reflection_integration=reflection_config.get("enable_reflection_integration", False),
                enable_progressive_reflection=reflection_config.get("enable_progressive_reflection", True),
                enable_reflection_metrics=reflection_config.get("enable_reflection_metrics", True),
                reflection_timeout_seconds=reflection_config.get("reflection_timeout_seconds", 30),
                max_knowledge_gaps=reflection_config.get("max_knowledge_gaps", 5),
                max_follow_up_queries=reflection_config.get("max_follow_up_queries", 3),
                enable_reflection_debugging=reflection_config.get("enable_reflection_debugging", False),
            )
        except Exception as e:
            logger.error(f"Failed to parse enhanced reflection configuration: {e}")
            return EnhancedReflectionConfig()  # Return default configuration

    def parse_isolation_config(self, config: Dict[str, Any]) -> IsolationConfig:
        """Parse isolation configuration."""
        isolation_config = config.get("isolation", {})
        
        try:
            return IsolationConfig(
                enable_isolation=isolation_config.get("enable_isolation", True),
                max_local_context_size=isolation_config.get("max_local_context_size", 8000),
                token_budget=isolation_config.get("token_budget", 16000),
                compression_threshold=isolation_config.get("compression_threshold", 0.7),
                max_concurrent_sessions=isolation_config.get("max_concurrent_sessions", 8),
                session_timeout=isolation_config.get("session_timeout", 1800),
            )
        except Exception as e:
            logger.error(f"Failed to parse isolation configuration: {e}")
            return IsolationConfig()  # Return default configuration

    def parse_performance_config(self, config: Dict[str, Any]) -> PerformanceConfig:
        """Parse performance configuration."""
        performance_config = config.get("performance", {})
        
        try:
            return PerformanceConfig(
                memory_cleanup_interval=performance_config.get("memory_cleanup_interval", 300),
                performance_threshold=performance_config.get("performance_threshold", 0.3),
                success_rate_threshold=performance_config.get("success_rate_threshold", 0.8),
                complexity_threshold=performance_config.get("complexity_threshold", 0.6),
            )
        except Exception as e:
            logger.error(f"Failed to parse performance configuration: {e}")
            return PerformanceConfig()  # Return default configuration

    def parse_monitoring_config(self, config: Dict[str, Any]) -> MonitoringConfig:
        """Parse monitoring configuration."""
        monitoring_config = config.get("monitoring", {})
        
        try:
            return MonitoringConfig(
                monitoring_interval=monitoring_config.get("monitoring_interval", 60),
                alert_sensitivity=monitoring_config.get("alert_sensitivity", "medium"),
                metrics_retention_days=monitoring_config.get("metrics_retention_days", 7),
                enable_predictive_analytics=monitoring_config.get("enable_predictive_analytics", True),
            )
        except Exception as e:
            logger.error(f"Failed to parse monitoring configuration: {e}")
            return MonitoringConfig()  # Return default configuration

    def parse_optimization_config(self, config: Dict[str, Any]) -> OptimizationConfig:
        """Parse optimization configuration."""
        optimization_config = config.get("optimization", {})
        
        try:
            return OptimizationConfig(
                auto_tuning_enabled=optimization_config.get("auto_tuning_enabled", False),
                optimization_level=optimization_config.get("optimization_level", "balanced"),
                min_data_points_for_tuning=optimization_config.get("min_data_points_for_tuning", 50),
            )
        except Exception as e:
            logger.error(f"Failed to parse optimization configuration: {e}")
            return OptimizationConfig()  # Return default configuration

    def create_researcher_config(self) -> ResearcherConfig:
        """Create researcher configuration object."""
        config_data = self.load_config()
        
        return ResearcherConfig(
            isolation=self.parse_isolation_config(config_data),
            performance=self.parse_performance_config(config_data),
            monitoring=self.parse_monitoring_config(config_data),
            optimization=self.parse_optimization_config(config_data),
            enhanced_reflection=self.parse_enhanced_reflection_config(config_data),
        )

    def save_example_config(self, output_path: Optional[str] = None):
        """Save example configuration file."""
        output_path = output_path or "config/researcher_config.json.example"

        example_config = {
            "isolation": {
                "enable_isolation": True,
                "max_local_context_size": 8000,
                "token_budget": 16000,
                "compression_threshold": 0.7,
                "max_concurrent_sessions": 8,
                "session_timeout": 1800
            },
            "performance": {
                "memory_cleanup_interval": 300,
                "performance_threshold": 0.3,
                "success_rate_threshold": 0.8,
                "complexity_threshold": 0.6
            },
            "monitoring": {
                "monitoring_interval": 60,
                "alert_sensitivity": "medium",
                "metrics_retention_days": 7,
                "enable_predictive_analytics": True
            },
            "optimization": {
                "auto_tuning_enabled": False,
                "optimization_level": "balanced",
                "min_data_points_for_tuning": 50
            },
            "enhanced_reflection": {
                "enable_enhanced_reflection": True,
                "max_reflection_loops": 3,
                "reflection_model": "gpt-4",
                "reflection_temperature": 0.1,
                "knowledge_gap_threshold": 0.7,
                "sufficiency_threshold": 0.8,
                "enable_reflection_caching": True,
                "reflection_trigger_threshold": 2,
                "reflection_confidence_threshold": 0.7,
                "enable_reflection_integration": True,
                "enable_progressive_reflection": True,
                "enable_reflection_metrics": True,
                "reflection_timeout_seconds": 30,
                "max_knowledge_gaps": 5,
                "max_follow_up_queries": 3,
                "enable_reflection_debugging": False
            }
        }

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(example_config, f, indent=2, ensure_ascii=False)
            logger.info(f"Example researcher configuration file saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save example researcher configuration file: {e}")


# Global researcher configuration loader instance
researcher_config_loader = ResearcherConfigLoader()