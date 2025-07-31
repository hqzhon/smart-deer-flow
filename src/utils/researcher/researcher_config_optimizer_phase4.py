# -*- coding: utf-8 -*-
"""
Researcher Configuration Optimizer - Phase 4

Intelligent configuration management with auto-tuning,
smart defaults, and performance-based recommendations.
"""

import json
import time
import logging
import statistics
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class ConfigOptimizationLevel(Enum):
    """Configuration optimization levels"""

    CONSERVATIVE = "conservative"  # Safe, minimal changes
    BALANCED = "balanced"  # Moderate optimization
    AGGRESSIVE = "aggressive"  # Maximum performance
    CUSTOM = "custom"  # User-defined rules


class ConfigParameter(Enum):
    """Configuration parameters that can be optimized"""

    # Isolation settings
    ENABLE_ISOLATION = "enable_isolation"
    MAX_LOCAL_CONTEXT_SIZE = "max_local_context_size"
    TOKEN_BUDGET = "token_budget"
    COMPRESSION_THRESHOLD = "compression_threshold"

    # Performance settings
    MAX_CONCURRENT_SESSIONS = "max_concurrent_sessions"
    SESSION_TIMEOUT = "session_timeout"
    MEMORY_CLEANUP_INTERVAL = "memory_cleanup_interval"

    # Progressive enablement
    COMPLEXITY_THRESHOLD = "complexity_threshold"
    SUCCESS_RATE_THRESHOLD = "success_rate_threshold"
    PERFORMANCE_THRESHOLD = "performance_threshold"

    # Monitoring
    MONITORING_INTERVAL = "monitoring_interval"
    ALERT_SENSITIVITY = "alert_sensitivity"
    METRICS_RETENTION_DAYS = "metrics_retention_days"


@dataclass
class ConfigRecommendation:
    """Configuration change recommendation"""

    parameter: ConfigParameter
    current_value: Any
    recommended_value: Any
    reason: str
    confidence: float  # 0-1
    impact_estimate: str  # "low", "medium", "high"
    priority: int  # 1-10, higher = more important
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter.value,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "reason": self.reason,
            "confidence": self.confidence,
            "impact_estimate": self.impact_estimate,
            "priority": self.priority,
            "timestamp": self.timestamp,
        }


@dataclass
class ConfigProfile:
    """Configuration profile for different scenarios"""

    name: str
    description: str
    target_scenario: str  # "high_load", "low_latency", "memory_constrained", etc.
    config_values: Dict[str, Any]
    performance_characteristics: Dict[str, float]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfigurationOptimizer:
    """Intelligent configuration optimizer"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/researcher_config.json"
        self.optimization_history: deque = deque(maxlen=100)
        self.performance_history: deque = deque(maxlen=500)

        # Load or create default configuration
        self.current_config = self._load_or_create_config()

        # Built-in profiles
        self.built_in_profiles = self._create_built_in_profiles()

        # Auto-tuning settings
        self.auto_tuning_enabled = False
        self.optimization_level = ConfigOptimizationLevel.BALANCED

        logger.info("Initialized ConfigurationOptimizer")

    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load existing config or create intelligent defaults"""
        config_path = Path(self.config_file)

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")

        # Create intelligent defaults
        default_config = self._generate_intelligent_defaults()
        self._save_config(default_config)
        return default_config

    def _generate_intelligent_defaults(self) -> Dict[str, Any]:
        """Generate intelligent default configuration based on system characteristics"""
        # Detect system characteristics
        import psutil
        import os

        cpu_count = os.cpu_count() or 4
        memory_gb = psutil.virtual_memory().total / (1024**3)

        # Base configuration
        config = {
            "isolation": {
                "enable_isolation": True,
                "max_local_context_size": min(
                    8000, int(memory_gb * 1000)
                ),  # Scale with memory
                "token_budget": min(16000, int(memory_gb * 2000)),
                "compression_threshold": 0.7,
                "max_concurrent_sessions": min(cpu_count * 2, 8),
                "session_timeout": 1800,  # 30 minutes
            },
            "performance": {
                "memory_cleanup_interval": 300,  # 5 minutes
                "performance_threshold": 0.3,
                "success_rate_threshold": 0.8,
                "complexity_threshold": 0.6,
            },
            "monitoring": {
                "monitoring_interval": 60,  # 1 minute
                "alert_sensitivity": "medium",
                "metrics_retention_days": 7,
                "enable_predictive_analytics": True,
            },
            "optimization": {
                "auto_tuning_enabled": False,
                "optimization_level": "balanced",
                "min_data_points_for_tuning": 50,
            },
        }

        # Adjust for system characteristics
        if memory_gb < 4:
            # Memory-constrained system
            config["isolation"]["max_local_context_size"] = 4000
            config["isolation"]["token_budget"] = 8000
            config["isolation"]["max_concurrent_sessions"] = 2
            config["monitoring"]["monitoring_interval"] = 120
        elif memory_gb > 16:
            # High-memory system
            config["isolation"]["max_local_context_size"] = 16000
            config["isolation"]["token_budget"] = 32000
            config["isolation"]["max_concurrent_sessions"] = min(cpu_count * 3, 12)

        if cpu_count <= 2:
            # Low-CPU system
            config["isolation"]["max_concurrent_sessions"] = 1
            config["performance"]["performance_threshold"] = 0.5

        logger.info(
            f"Generated intelligent defaults for system: {cpu_count} CPUs, {memory_gb:.1f}GB RAM"
        )
        return config

    def _create_built_in_profiles(self) -> Dict[str, ConfigProfile]:
        """Create built-in configuration profiles"""
        profiles = {}

        # High-performance profile
        profiles["high_performance"] = ConfigProfile(
            name="High Performance",
            description="Optimize performance for high-load scenarios",
            target_scenario="high_load",
            config_values={
                "isolation.max_concurrent_sessions": 12,
                "isolation.token_budget": 32000,
                "isolation.compression_threshold": 0.8,
                "performance.performance_threshold": 0.2,
                "monitoring.monitoring_interval": 30,
            },
            performance_characteristics={
                "throughput": 0.9,
                "latency": 0.7,
                "memory_usage": 0.8,
                "cpu_usage": 0.8,
            },
        )

        # Low-latency profile
        profiles["low_latency"] = ConfigProfile(
            name="Low Latency",
            description="Optimize response time for interactive scenarios",
            target_scenario="low_latency",
            config_values={
                "isolation.max_local_context_size": 4000,
                "isolation.token_budget": 8000,
                "isolation.session_timeout": 600,
                "performance.memory_cleanup_interval": 60,
                "monitoring.monitoring_interval": 15,
            },
            performance_characteristics={
                "throughput": 0.6,
                "latency": 0.9,
                "memory_usage": 0.5,
                "cpu_usage": 0.6,
            },
        )

        # Memory-efficient profile
        profiles["memory_efficient"] = ConfigProfile(
            name="Memory Optimized",
            description="Minimize memory usage for resource-constrained environments",
            target_scenario="memory_constrained",
            config_values={
                "isolation.max_local_context_size": 2000,
                "isolation.token_budget": 4000,
                "isolation.max_concurrent_sessions": 2,
                "performance.memory_cleanup_interval": 120,
                "monitoring.metrics_retention_days": 3,
            },
            performance_characteristics={
                "throughput": 0.4,
                "latency": 0.6,
                "memory_usage": 0.3,
                "cpu_usage": 0.4,
            },
        )

        # Balanced profile (default)
        profiles["balanced"] = ConfigProfile(
            name="Balanced",
            description="Balance performance and resource usage for most scenarios",
            target_scenario="general",
            config_values={
                "isolation.max_local_context_size": 8000,
                "isolation.token_budget": 16000,
                "isolation.max_concurrent_sessions": 4,
                "isolation.compression_threshold": 0.7,
                "performance.performance_threshold": 0.3,
                "monitoring.monitoring_interval": 60,
            },
            performance_characteristics={
                "throughput": 0.7,
                "latency": 0.7,
                "memory_usage": 0.6,
                "cpu_usage": 0.6,
            },
        )

        return profiles

    def analyze_performance_data(
        self, metrics_data: Dict[str, Any]
    ) -> List[ConfigRecommendation]:
        """Analyze performance data and generate configuration recommendations"""
        recommendations = []

        # Store performance data
        self.performance_history.append(
            {"timestamp": time.time(), "metrics": metrics_data}
        )

        if len(self.performance_history) < 10:
            return recommendations  # Need more data

        # Analyze recent performance
        recent_data = list(self.performance_history)[-20:]

        # Check success rate
        success_rates = [d["metrics"].get("success_rate_1h", 1.0) for d in recent_data]
        avg_success_rate = statistics.mean(success_rates)

        if avg_success_rate < 0.8:
            recommendations.append(
                ConfigRecommendation(
                    parameter=ConfigParameter.SUCCESS_RATE_THRESHOLD,
                    current_value=self.current_config.get("performance", {}).get(
                        "success_rate_threshold", 0.8
                    ),
                    recommended_value=max(0.6, avg_success_rate - 0.1),
                    reason=f"Current success rate {avg_success_rate:.1%} is below threshold, suggest lowering requirements",
                    confidence=0.8,
                    impact_estimate="medium",
                    priority=7,
                )
            )

        # Check performance overhead
        overheads = [
            d["metrics"].get("performance_overhead_1h", 0.0) for d in recent_data
        ]
        avg_overhead = statistics.mean(overheads)

        if avg_overhead > 0.3:
            current_sessions = self.current_config.get("isolation", {}).get(
                "max_concurrent_sessions", 4
            )
            recommended_sessions = max(1, int(current_sessions * 0.8))

            recommendations.append(
                ConfigRecommendation(
                    parameter=ConfigParameter.MAX_CONCURRENT_SESSIONS,
                    current_value=current_sessions,
                    recommended_value=recommended_sessions,
                    reason=f"Performance overhead {avg_overhead:.1%} is too high, suggest reducing concurrent sessions",
                    confidence=0.9,
                    impact_estimate="high",
                    priority=8,
                )
            )

        # Check token efficiency
        compression_ratios = [
            d["metrics"].get("avg_compression_ratio_1h", 1.0) for d in recent_data
        ]
        avg_compression = statistics.mean(compression_ratios)

        if avg_compression > 0.8:  # Poor compression
            current_budget = self.current_config.get("isolation", {}).get(
                "token_budget", 16000
            )
            recommended_budget = int(current_budget * 1.2)

            recommendations.append(
                ConfigRecommendation(
                    parameter=ConfigParameter.TOKEN_BUDGET,
                    current_value=current_budget,
                    recommended_value=recommended_budget,
                    reason=f"Compression ratio {avg_compression:.1%} is poor, suggest increasing token budget",
                    confidence=0.7,
                    impact_estimate="medium",
                    priority=5,
                )
            )

        # Check resource utilization
        utilizations = [
            d["metrics"].get("resource_utilization", 0.0) for d in recent_data
        ]
        avg_utilization = statistics.mean(utilizations)

        if avg_utilization > 0.9:
            current_cleanup = self.current_config.get("performance", {}).get(
                "memory_cleanup_interval", 300
            )
            recommended_cleanup = max(60, int(current_cleanup * 0.7))

            recommendations.append(
                ConfigRecommendation(
                    parameter=ConfigParameter.MEMORY_CLEANUP_INTERVAL,
                    current_value=current_cleanup,
                    recommended_value=recommended_cleanup,
                    reason=f"Resource utilization {avg_utilization:.1%} is too high, suggest increasing cleanup frequency",
                    confidence=0.8,
                    impact_estimate="medium",
                    priority=6,
                )
            )

        # Sort recommendations by priority
        recommendations.sort(key=lambda r: r.priority, reverse=True)

        return recommendations

    def apply_recommendations(
        self, recommendations: List[ConfigRecommendation], auto_apply: bool = False
    ) -> Dict[str, Any]:
        """Apply configuration recommendations"""
        applied_changes = {}

        for rec in recommendations:
            if auto_apply or rec.confidence >= 0.8:
                # Apply the recommendation
                param_path = rec.parameter.value.split(".")

                # Navigate to the correct config section
                config_section = self.current_config
                for part in param_path[:-1]:
                    if part not in config_section:
                        config_section[part] = {}
                    config_section = config_section[part]

                # Apply the change
                old_value = config_section.get(param_path[-1], None)
                config_section[param_path[-1]] = rec.recommended_value

                applied_changes[rec.parameter.value] = {
                    "old_value": old_value,
                    "new_value": rec.recommended_value,
                    "reason": rec.reason,
                    "confidence": rec.confidence,
                }

                logger.info(
                    f"Applied config change: {rec.parameter.value} = {rec.recommended_value}"
                )

        if applied_changes:
            self._save_config(self.current_config)

            # Record optimization
            self.optimization_history.append(
                {
                    "timestamp": time.time(),
                    "changes": applied_changes,
                    "auto_applied": auto_apply,
                }
            )

        return applied_changes

    def apply_profile(self, profile_name: str) -> bool:
        """Apply a configuration profile"""
        if profile_name not in self.built_in_profiles:
            logger.error(f"Profile '{profile_name}' not found")
            return False

        profile = self.built_in_profiles[profile_name]
        applied_changes = {}

        for param_path, value in profile.config_values.items():
            parts = param_path.split(".")

            # Navigate to config section
            config_section = self.current_config
            for part in parts[:-1]:
                if part not in config_section:
                    config_section[part] = {}
                config_section = config_section[part]

            # Apply change
            old_value = config_section.get(parts[-1], None)
            config_section[parts[-1]] = value
            applied_changes[param_path] = {"old": old_value, "new": value}

        self._save_config(self.current_config)

        logger.info(
            f"Applied profile '{profile_name}' with {len(applied_changes)} changes"
        )
        return True

    def auto_tune_configuration(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically tune configuration based on performance data"""
        if not self.auto_tuning_enabled:
            return {"message": "Auto-tuning is disabled"}

        # Generate recommendations
        recommendations = self.analyze_performance_data(metrics_data)

        if not recommendations:
            return {"message": "No recommendations generated"}

        # Filter recommendations based on optimization level
        filtered_recs = self._filter_recommendations_by_level(recommendations)

        # Apply recommendations automatically
        applied_changes = self.apply_recommendations(filtered_recs, auto_apply=True)

        return {
            "auto_tuning_applied": True,
            "recommendations_count": len(recommendations),
            "applied_count": len(applied_changes),
            "changes": applied_changes,
            "optimization_level": self.optimization_level.value,
        }

    def _filter_recommendations_by_level(
        self, recommendations: List[ConfigRecommendation]
    ) -> List[ConfigRecommendation]:
        """Filter recommendations based on optimization level"""
        if self.optimization_level == ConfigOptimizationLevel.CONSERVATIVE:
            # Only high-confidence, low-impact changes
            return [
                r
                for r in recommendations
                if r.confidence >= 0.9 and r.impact_estimate == "low"
            ]
        elif self.optimization_level == ConfigOptimizationLevel.BALANCED:
            # Medium to high confidence changes
            return [r for r in recommendations if r.confidence >= 0.7]
        elif self.optimization_level == ConfigOptimizationLevel.AGGRESSIVE:
            # All recommendations with reasonable confidence
            return [r for r in recommendations if r.confidence >= 0.5]
        else:  # CUSTOM
            # Use all recommendations (let user decide)
            return recommendations

    def get_configuration_report(self) -> Dict[str, Any]:
        """Get comprehensive configuration report"""
        return {
            "current_config": self.current_config,
            "available_profiles": {
                name: profile.to_dict()
                for name, profile in self.built_in_profiles.items()
            },
            "optimization_history": list(self.optimization_history)[
                -10:
            ],  # Last 10 optimizations
            "auto_tuning_status": {
                "enabled": self.auto_tuning_enabled,
                "optimization_level": self.optimization_level.value,
                "performance_data_points": len(self.performance_history),
            },
            "system_recommendations": self._get_system_recommendations(),
            "config_health_score": self._calculate_config_health_score(),
        }

    def _get_system_recommendations(self) -> List[str]:
        """Get general system recommendations"""
        recommendations = []

        # Check if auto-tuning should be enabled
        if not self.auto_tuning_enabled and len(self.performance_history) >= 50:
            recommendations.append(
                "Consider enabling auto-tuning feature to optimize performance"
            )

        # Check optimization history
        if len(self.optimization_history) == 0:
            recommendations.append(
                "No configuration optimization performed yet, recommend running performance analysis"
            )

        # Check configuration age
        config_path = Path(self.config_file)
        if config_path.exists():
            config_age_days = (time.time() - config_path.stat().st_mtime) / 86400
            if config_age_days > 30:
                recommendations.append(
                    f"Configuration file has not been updated for {config_age_days:.0f} days, recommend checking if optimization is needed"
                )

        return recommendations

    def _calculate_config_health_score(self) -> float:
        """Calculate configuration health score (0-100)"""
        score = 100.0

        # Deduct points for suboptimal settings
        isolation_config = self.current_config.get("isolation", {})
        performance_config = self.current_config.get("performance", {})

        # Check token budget efficiency
        token_budget = isolation_config.get("token_budget", 16000)
        if token_budget < 8000:
            score -= 10  # Too low
        elif token_budget > 50000:
            score -= 15  # Too high, wasteful

        # Check concurrent sessions
        max_sessions = isolation_config.get("max_concurrent_sessions", 4)
        if max_sessions < 2:
            score -= 5  # Too conservative
        elif max_sessions > 10:
            score -= 10  # Potentially overwhelming

        # Check performance thresholds
        perf_threshold = performance_config.get("performance_threshold", 0.3)
        if perf_threshold > 0.5:
            score -= 15  # Too lenient
        elif perf_threshold < 0.1:
            score -= 10  # Too strict

        # Bonus for auto-tuning
        if self.auto_tuning_enabled:
            score += 5

        # Bonus for recent optimizations
        if len(self.optimization_history) > 0:
            last_optimization = self.optimization_history[-1]["timestamp"]
            days_since = (time.time() - last_optimization) / 86400
            if days_since < 7:
                score += 5

        return max(0, min(100, score))

    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def enable_auto_tuning(
        self,
        optimization_level: ConfigOptimizationLevel = ConfigOptimizationLevel.BALANCED,
    ):
        """Enable automatic configuration tuning"""
        self.auto_tuning_enabled = True
        self.optimization_level = optimization_level

        # Update config
        if "optimization" not in self.current_config:
            self.current_config["optimization"] = {}
        self.current_config["optimization"]["auto_tuning_enabled"] = True
        self.current_config["optimization"][
            "optimization_level"
        ] = optimization_level.value

        self._save_config(self.current_config)
        logger.info(f"Auto-tuning enabled with {optimization_level.value} level")

    def disable_auto_tuning(self):
        """Disable automatic configuration tuning"""
        self.auto_tuning_enabled = False

        # Update config
        if "optimization" in self.current_config:
            self.current_config["optimization"]["auto_tuning_enabled"] = False

        self._save_config(self.current_config)
        logger.info("Auto-tuning disabled")

    def reset_to_defaults(self):
        """Reset configuration to intelligent defaults"""
        self.current_config = self._generate_intelligent_defaults()
        self._save_config(self.current_config)
        logger.info("Configuration reset to intelligent defaults")

    def export_config_backup(self, backup_file: Optional[str] = None) -> str:
        """Export configuration backup"""
        backup_file = (
            backup_file
            or f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        backup_data = {
            "config": self.current_config,
            "optimization_history": list(self.optimization_history),
            "backup_timestamp": datetime.now().isoformat(),
            "auto_tuning_enabled": self.auto_tuning_enabled,
            "optimization_level": self.optimization_level.value,
        }

        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Configuration backup exported to {backup_file}")
        return backup_file


# Global configuration optimizer instance
_global_config_optimizer: Optional[ConfigurationOptimizer] = None


def get_config_optimizer() -> ConfigurationOptimizer:
    """Get global configuration optimizer instance"""
    global _global_config_optimizer
    if _global_config_optimizer is None:
        _global_config_optimizer = ConfigurationOptimizer()
    return _global_config_optimizer


def reset_config_optimizer():
    """Reset global configuration optimizer (for testing)"""
    global _global_config_optimizer
    _global_config_optimizer = None
