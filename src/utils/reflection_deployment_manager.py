# -*- coding: utf-8 -*-
"""
Reflection Deployment Manager

This module implements the deployment strategy for Phase 5 reflection integration,
including progressive enablement, A/B testing, and rollback capabilities.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class DeploymentPhase(Enum):
    """Deployment phases for reflection system."""

    DISABLED = "disabled"
    TESTING = "testing"  # Phase 1: Default off, test environment only
    LIMITED = "limited"  # Phase 2: Specific user groups
    GRADUAL = "gradual"  # Phase 3: Gradual expansion
    FULL = "full"  # Phase 4: Full deployment


class RollbackTrigger(Enum):
    """Triggers for automatic rollback."""

    ERROR_RATE = "error_rate"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    USER_COMPLAINTS = "user_complaints"
    MANUAL = "manual"


@dataclass
class DeploymentConfig:
    """Configuration for reflection deployment."""

    # Current deployment phase
    current_phase: DeploymentPhase = DeploymentPhase.DISABLED

    # User group configurations
    test_user_groups: Set[str] = field(default_factory=set)
    limited_user_groups: Set[str] = field(default_factory=set)
    gradual_rollout_percentage: float = 0.0

    # Safety thresholds
    max_error_rate: float = 0.05
    max_performance_overhead: float = 0.25
    min_user_satisfaction: float = 0.7

    # Rollback settings
    auto_rollback_enabled: bool = True
    rollback_cooldown_hours: int = 24

    # A/B testing
    ab_testing_enabled: bool = False
    ab_test_percentage: float = 0.1
    ab_test_duration_days: int = 7

    # Monitoring
    monitoring_enabled: bool = True
    health_check_interval_minutes: int = 5

    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class DeploymentMetrics:
    """Metrics for deployment monitoring."""

    phase: DeploymentPhase
    enabled_users: int = 0
    total_users: int = 0
    error_rate: float = 0.0
    performance_overhead: float = 0.0
    user_satisfaction: float = 0.0
    reflection_usage_rate: float = 0.0
    rollback_count: int = 0
    last_rollback: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.now)


class ReflectionDeploymentManager:
    """Manages progressive deployment of reflection system."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "reflection_deployment_config.json"
        self.config = self._load_config()
        self.metrics_history: List[DeploymentMetrics] = []
        self.deployment_lock = threading.Lock()

        # Callbacks for deployment events
        self.deployment_callbacks: List[callable] = []
        self.rollback_callbacks: List[callable] = []

        # Health monitoring
        self.last_health_check = datetime.now()
        self.health_check_enabled = True

        logger.info(
            f"Reflection Deployment Manager initialized with phase: {self.config.current_phase}"
        )

    def _load_config(self) -> DeploymentConfig:
        """Load deployment configuration from file."""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, "r") as f:
                    data = json.load(f)

                config = DeploymentConfig(
                    current_phase=DeploymentPhase(
                        data.get("current_phase", "disabled")
                    ),
                    test_user_groups=set(data.get("test_user_groups", [])),
                    limited_user_groups=set(data.get("limited_user_groups", [])),
                    gradual_rollout_percentage=data.get(
                        "gradual_rollout_percentage", 0.0
                    ),
                    max_error_rate=data.get("max_error_rate", 0.05),
                    max_performance_overhead=data.get("max_performance_overhead", 0.25),
                    min_user_satisfaction=data.get("min_user_satisfaction", 0.7),
                    auto_rollback_enabled=data.get("auto_rollback_enabled", True),
                    rollback_cooldown_hours=data.get("rollback_cooldown_hours", 24),
                    ab_testing_enabled=data.get("ab_testing_enabled", False),
                    ab_test_percentage=data.get("ab_test_percentage", 0.1),
                    ab_test_duration_days=data.get("ab_test_duration_days", 7),
                    monitoring_enabled=data.get("monitoring_enabled", True),
                    health_check_interval_minutes=data.get(
                        "health_check_interval_minutes", 5
                    ),
                )

                logger.info(f"Loaded deployment config from {self.config_file}")
                return config

        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_file}: {e}")

        # Return default config
        return DeploymentConfig()

    def _save_config(self) -> None:
        """Save deployment configuration to file."""
        try:
            data = {
                "current_phase": self.config.current_phase.value,
                "test_user_groups": list(self.config.test_user_groups),
                "limited_user_groups": list(self.config.limited_user_groups),
                "gradual_rollout_percentage": self.config.gradual_rollout_percentage,
                "max_error_rate": self.config.max_error_rate,
                "max_performance_overhead": self.config.max_performance_overhead,
                "min_user_satisfaction": self.config.min_user_satisfaction,
                "auto_rollback_enabled": self.config.auto_rollback_enabled,
                "rollback_cooldown_hours": self.config.rollback_cooldown_hours,
                "ab_testing_enabled": self.config.ab_testing_enabled,
                "ab_test_percentage": self.config.ab_test_percentage,
                "ab_test_duration_days": self.config.ab_test_duration_days,
                "monitoring_enabled": self.config.monitoring_enabled,
                "health_check_interval_minutes": self.config.health_check_interval_minutes,
                "last_updated": self.config.last_updated.isoformat(),
            }

            with open(self.config_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved deployment config to {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to save config to {self.config_file}: {e}")

    def is_reflection_enabled_for_user(
        self, user_id: str, user_group: Optional[str] = None
    ) -> bool:
        """Check if reflection is enabled for a specific user."""
        with self.deployment_lock:
            phase = self.config.current_phase

            # Disabled phase
            if phase == DeploymentPhase.DISABLED:
                return False

            # Testing phase - only test users
            elif phase == DeploymentPhase.TESTING:
                return (
                    user_group in self.config.test_user_groups if user_group else False
                )

            # Limited phase - specific user groups
            elif phase == DeploymentPhase.LIMITED:
                return (
                    (
                        user_group in self.config.test_user_groups
                        or user_group in self.config.limited_user_groups
                    )
                    if user_group
                    else False
                )

            # Gradual phase - percentage-based rollout
            elif phase == DeploymentPhase.GRADUAL:
                # Always enable for test and limited groups
                if user_group and (
                    user_group in self.config.test_user_groups
                    or user_group in self.config.limited_user_groups
                ):
                    return True

                # Percentage-based for others
                user_hash = hash(user_id) % 100
                return user_hash < (self.config.gradual_rollout_percentage * 100)

            # Full phase - enabled for everyone
            elif phase == DeploymentPhase.FULL:
                return True

            return False

    def advance_deployment_phase(
        self, target_phase: DeploymentPhase, force: bool = False
    ) -> Dict[str, Any]:
        """Advance to the next deployment phase."""
        with self.deployment_lock:
            current_phase = self.config.current_phase

            # Validate phase transition
            if not force and not self._is_valid_phase_transition(
                current_phase, target_phase
            ):
                return {
                    "success": False,
                    "error": f"Invalid phase transition from {current_phase} to {target_phase}",
                }

            # Check safety conditions
            if not force and not self._check_safety_conditions():
                return {
                    "success": False,
                    "error": "Safety conditions not met for phase advancement",
                }

            # Perform phase transition
            old_phase = self.config.current_phase
            self.config.current_phase = target_phase
            self.config.last_updated = datetime.now()

            # Save configuration
            self._save_config()

            # Notify callbacks
            self._notify_deployment_callbacks(
                "phase_advanced",
                {
                    "old_phase": old_phase.value,
                    "new_phase": target_phase.value,
                    "forced": force,
                },
            )

            logger.info(f"Advanced deployment phase from {old_phase} to {target_phase}")

            return {
                "success": True,
                "old_phase": old_phase.value,
                "new_phase": target_phase.value,
                "timestamp": datetime.now().isoformat(),
            }

    def _is_valid_phase_transition(
        self, current: DeploymentPhase, target: DeploymentPhase
    ) -> bool:
        """Check if phase transition is valid."""
        valid_transitions = {
            DeploymentPhase.DISABLED: [DeploymentPhase.TESTING],
            DeploymentPhase.TESTING: [
                DeploymentPhase.LIMITED,
                DeploymentPhase.DISABLED,
            ],
            DeploymentPhase.LIMITED: [DeploymentPhase.GRADUAL, DeploymentPhase.TESTING],
            DeploymentPhase.GRADUAL: [DeploymentPhase.FULL, DeploymentPhase.LIMITED],
            DeploymentPhase.FULL: [DeploymentPhase.GRADUAL],
        }

        return target in valid_transitions.get(current, [])

    def _check_safety_conditions(self) -> bool:
        """Check if safety conditions are met for deployment."""
        try:
            # Get recent metrics
            recent_metrics = self._get_recent_metrics()
            if not recent_metrics:
                return True  # No metrics available, assume safe

            latest_metrics = recent_metrics[-1]

            # Check error rate
            if latest_metrics.error_rate > self.config.max_error_rate:
                logger.warning(
                    f"Error rate {latest_metrics.error_rate} exceeds threshold {self.config.max_error_rate}"
                )
                return False

            # Check performance overhead
            if (
                latest_metrics.performance_overhead
                > self.config.max_performance_overhead
            ):
                logger.warning(
                    f"Performance overhead {latest_metrics.performance_overhead} exceeds threshold {self.config.max_performance_overhead}"
                )
                return False

            # Check user satisfaction
            if latest_metrics.user_satisfaction < self.config.min_user_satisfaction:
                logger.warning(
                    f"User satisfaction {latest_metrics.user_satisfaction} below threshold {self.config.min_user_satisfaction}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking safety conditions: {e}")
            return False

    def trigger_rollback(self, trigger: RollbackTrigger, reason: str) -> Dict[str, Any]:
        """Trigger deployment rollback."""
        with self.deployment_lock:
            current_phase = self.config.current_phase

            # Determine rollback target
            rollback_target = self._get_rollback_target(current_phase)

            if rollback_target is None:
                return {
                    "success": False,
                    "error": f"No rollback target available from phase {current_phase}",
                }

            # Check rollback cooldown
            if self._is_in_rollback_cooldown():
                return {"success": False, "error": "Rollback is in cooldown period"}

            # Perform rollback
            old_phase = self.config.current_phase
            self.config.current_phase = rollback_target
            self.config.last_updated = datetime.now()

            # Update metrics
            if self.metrics_history:
                self.metrics_history[-1].rollback_count += 1
                self.metrics_history[-1].last_rollback = datetime.now()

            # Save configuration
            self._save_config()

            # Notify callbacks
            self._notify_rollback_callbacks(
                trigger,
                {
                    "old_phase": old_phase.value,
                    "new_phase": rollback_target.value,
                    "trigger": trigger.value,
                    "reason": reason,
                },
            )

            logger.warning(
                f"Rolled back deployment from {old_phase} to {rollback_target} due to {trigger.value}: {reason}"
            )

            return {
                "success": True,
                "old_phase": old_phase.value,
                "new_phase": rollback_target.value,
                "trigger": trigger.value,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            }

    def _get_rollback_target(
        self, current_phase: DeploymentPhase
    ) -> Optional[DeploymentPhase]:
        """Get rollback target for current phase."""
        rollback_map = {
            DeploymentPhase.TESTING: DeploymentPhase.DISABLED,
            DeploymentPhase.LIMITED: DeploymentPhase.TESTING,
            DeploymentPhase.GRADUAL: DeploymentPhase.LIMITED,
            DeploymentPhase.FULL: DeploymentPhase.GRADUAL,
        }

        return rollback_map.get(current_phase)

    def _is_in_rollback_cooldown(self) -> bool:
        """Check if system is in rollback cooldown period."""
        if not self.metrics_history:
            return False

        latest_metrics = self.metrics_history[-1]
        if latest_metrics.last_rollback is None:
            return False

        cooldown_end = latest_metrics.last_rollback + timedelta(
            hours=self.config.rollback_cooldown_hours
        )
        return datetime.now() < cooldown_end

    def update_gradual_rollout_percentage(self, percentage: float) -> Dict[str, Any]:
        """Update gradual rollout percentage."""
        if not (0.0 <= percentage <= 1.0):
            return {"success": False, "error": "Percentage must be between 0.0 and 1.0"}

        with self.deployment_lock:
            old_percentage = self.config.gradual_rollout_percentage
            self.config.gradual_rollout_percentage = percentage
            self.config.last_updated = datetime.now()

            self._save_config()

            logger.info(
                f"Updated gradual rollout percentage from {old_percentage} to {percentage}"
            )

            return {
                "success": True,
                "old_percentage": old_percentage,
                "new_percentage": percentage,
                "timestamp": datetime.now().isoformat(),
            }

    def add_user_group(self, group_name: str, group_type: str) -> Dict[str, Any]:
        """Add user group to deployment configuration."""
        if group_type not in ["test", "limited"]:
            return {"success": False, "error": "Group type must be 'test' or 'limited'"}

        with self.deployment_lock:
            if group_type == "test":
                self.config.test_user_groups.add(group_name)
            else:
                self.config.limited_user_groups.add(group_name)

            self.config.last_updated = datetime.now()
            self._save_config()

            logger.info(f"Added {group_type} user group: {group_name}")

            return {
                "success": True,
                "group_name": group_name,
                "group_type": group_type,
                "timestamp": datetime.now().isoformat(),
            }

    def remove_user_group(self, group_name: str, group_type: str) -> Dict[str, Any]:
        """Remove user group from deployment configuration."""
        with self.deployment_lock:
            removed = False

            if group_type == "test" and group_name in self.config.test_user_groups:
                self.config.test_user_groups.remove(group_name)
                removed = True
            elif (
                group_type == "limited"
                and group_name in self.config.limited_user_groups
            ):
                self.config.limited_user_groups.remove(group_name)
                removed = True

            if removed:
                self.config.last_updated = datetime.now()
                self._save_config()
                logger.info(f"Removed {group_type} user group: {group_name}")

            return {
                "success": removed,
                "group_name": group_name,
                "group_type": group_type,
                "timestamp": datetime.now().isoformat(),
            }

    def record_deployment_metrics(self, metrics: DeploymentMetrics) -> None:
        """Record deployment metrics."""
        with self.deployment_lock:
            self.metrics_history.append(metrics)

            # Keep only recent metrics (last 1000 entries)
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]

            # Check for automatic rollback conditions
            if self.config.auto_rollback_enabled:
                self._check_auto_rollback_conditions(metrics)

    def _check_auto_rollback_conditions(self, metrics: DeploymentMetrics) -> None:
        """Check if automatic rollback should be triggered."""
        try:
            # Error rate check
            if metrics.error_rate > self.config.max_error_rate:
                self.trigger_rollback(
                    RollbackTrigger.ERROR_RATE,
                    f"Error rate {metrics.error_rate} exceeds threshold {self.config.max_error_rate}",
                )
                return

            # Performance degradation check
            if metrics.performance_overhead > self.config.max_performance_overhead:
                self.trigger_rollback(
                    RollbackTrigger.PERFORMANCE_DEGRADATION,
                    f"Performance overhead {metrics.performance_overhead} exceeds threshold {self.config.max_performance_overhead}",
                )
                return

            # User satisfaction check
            if metrics.user_satisfaction < self.config.min_user_satisfaction:
                self.trigger_rollback(
                    RollbackTrigger.USER_COMPLAINTS,
                    f"User satisfaction {metrics.user_satisfaction} below threshold {self.config.min_user_satisfaction}",
                )
                return

        except Exception as e:
            logger.error(f"Error checking auto rollback conditions: {e}")

    def _get_recent_metrics(self, hours: int = 24) -> List[DeploymentMetrics]:
        """Get recent deployment metrics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        with self.deployment_lock:
            recent_metrics = self._get_recent_metrics(hours=1)
            latest_metrics = recent_metrics[-1] if recent_metrics else None

            status = {
                "current_phase": self.config.current_phase.value,
                "configuration": {
                    "test_user_groups": list(self.config.test_user_groups),
                    "limited_user_groups": list(self.config.limited_user_groups),
                    "gradual_rollout_percentage": self.config.gradual_rollout_percentage,
                    "auto_rollback_enabled": self.config.auto_rollback_enabled,
                    "ab_testing_enabled": self.config.ab_testing_enabled,
                },
                "safety_thresholds": {
                    "max_error_rate": self.config.max_error_rate,
                    "max_performance_overhead": self.config.max_performance_overhead,
                    "min_user_satisfaction": self.config.min_user_satisfaction,
                },
                "recent_metrics": latest_metrics.__dict__ if latest_metrics else None,
                "rollback_status": {
                    "in_cooldown": self._is_in_rollback_cooldown(),
                    "last_rollback": (
                        latest_metrics.last_rollback.isoformat()
                        if latest_metrics and latest_metrics.last_rollback
                        else None
                    ),
                    "rollback_count": (
                        latest_metrics.rollback_count if latest_metrics else 0
                    ),
                },
                "health_status": self._check_safety_conditions(),
                "last_updated": self.config.last_updated.isoformat(),
            }

            return status

    def add_deployment_callback(self, callback: callable) -> None:
        """Add callback for deployment events."""
        self.deployment_callbacks.append(callback)

    def add_rollback_callback(self, callback: callable) -> None:
        """Add callback for rollback events."""
        self.rollback_callbacks.append(callback)

    def _notify_deployment_callbacks(
        self, event_type: str, data: Dict[str, Any]
    ) -> None:
        """Notify deployment callbacks."""
        for callback in self.deployment_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in deployment callback: {e}")

    def _notify_rollback_callbacks(
        self, trigger: RollbackTrigger, data: Dict[str, Any]
    ) -> None:
        """Notify rollback callbacks."""
        for callback in self.rollback_callbacks:
            try:
                callback(trigger, data)
            except Exception as e:
                logger.error(f"Error in rollback callback: {e}")


# Global deployment manager instance
_deployment_manager: Optional[ReflectionDeploymentManager] = None


def get_deployment_manager(
    config_file: Optional[str] = None,
) -> ReflectionDeploymentManager:
    """Get or create deployment manager instance."""
    global _deployment_manager

    if _deployment_manager is None:
        _deployment_manager = ReflectionDeploymentManager(config_file)

    return _deployment_manager


def is_reflection_enabled_for_user(
    user_id: str, user_group: Optional[str] = None
) -> bool:
    """Convenience function to check if reflection is enabled for user."""
    manager = get_deployment_manager()
    return manager.is_reflection_enabled_for_user(user_id, user_group)
