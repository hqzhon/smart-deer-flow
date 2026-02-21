"""System infrastructure utilities.

This module contains utilities for system health checking,
rate limiting, circuit breaker, and dependency injection.
"""

from .health_check import HealthCheck, HealthStatus, HealthReport, SystemMetrics
from .rate_limiter import RateLimiter
from .error_recovery import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitState,
    RecoveryStrategy,
    RecoveryConfig,
    ErrorStats,
    global_circuit_breaker_manager,
    with_circuit_breaker,
)
from .callback_safety import SafeCallbackManager
from .dependency_injection import DependencyInjectionContainer

__all__ = [
    "HealthCheck",
    "HealthStatus",
    "HealthReport",
    "SystemMetrics",
    "RateLimiter",
    "CircuitBreaker",
    "CircuitBreakerManager",
    "CircuitState",
    "RecoveryStrategy",
    "RecoveryConfig",
    "ErrorStats",
    "global_circuit_breaker_manager",
    "with_circuit_breaker",
    "SafeCallbackManager",
    "DependencyInjectionContainer",
]
