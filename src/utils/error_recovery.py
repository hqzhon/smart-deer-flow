# -*- coding: utf-8 -*-
"""
Enhanced error recovery mechanism
Provides intelligent error recovery strategies including circuit breaker pattern, adaptive retry and degradation handling
"""

import logging
import time
import asyncio
from typing import Callable, Optional, Any, Dict, List, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import random

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker state"""
    CLOSED = "closed"      # Normal state
    OPEN = "open"          # Circuit open state
    HALF_OPEN = "half_open" # Half-open state


class RecoveryStrategy(Enum):
    """Recovery strategy"""
    IMMEDIATE = "immediate"        # Immediate retry
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential backoff
    LINEAR_BACKOFF = "linear_backoff"            # Linear backoff
    JITTERED_BACKOFF = "jittered_backoff"        # Jittered backoff
    CIRCUIT_BREAKER = "circuit_breaker"          # Circuit breaker pattern
    DEGRADED_SERVICE = "degraded_service"        # Degraded service


@dataclass
class RecoveryConfig:
    """Recovery configuration"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_range: float = 0.1
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 60.0
    circuit_success_threshold: int = 3
    enable_degraded_service: bool = True
    degraded_service_timeout: float = 5.0


@dataclass
class ErrorStats:
    """Error statistics"""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    error_types: Dict[str, int] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, config: RecoveryConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.stats = ErrorStats()
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if current_time - self.last_failure_time >= self.config.circuit_recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self):
        """Record success"""
        self.failure_count = 0
        self.stats.consecutive_failures = 0
        self.stats.consecutive_successes += 1
        self.stats.successful_attempts += 1
        self.stats.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.circuit_success_threshold:
                self.state = CircuitState.CLOSED
                logger.info("Circuit breaker transitioning to CLOSED")
    
    def record_failure(self, error_type: str = "unknown"):
        """Record failure"""
        self.failure_count += 1
        self.stats.consecutive_successes = 0
        self.stats.consecutive_failures += 1
        self.stats.failed_attempts += 1
        self.stats.last_failure_time = time.time()
        self.last_failure_time = time.time()
        
        # Update error type statistics
        self.stats.error_types[error_type] = self.stats.error_types.get(error_type, 0) + 1
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.circuit_failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker transitioning to OPEN after {self.failure_count} failures")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker transitioning back to OPEN")
    
    def get_state(self) -> CircuitState:
        """Get current state"""
        return self.state
    
    def reset(self):
        """Reset circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        logger.info("Circuit breaker reset")


class ErrorRecoveryManager:
    """Error recovery manager"""
    
    def __init__(self, config: RecoveryConfig = None):
        self.config = config or RecoveryConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.degraded_services: Dict[str, Callable] = {}
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
    
    def register_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Register circuit breaker"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(self.config)
        return self.circuit_breakers[service_name]
    
    def register_degraded_service(self, service_name: str, degraded_func: Callable):
        """Register degraded service"""
        self.degraded_services[service_name] = degraded_func
        logger.debug(f"Registered degraded service for {service_name}")
    
    def set_recovery_strategy(self, service_name: str, strategy: RecoveryStrategy):
        """Set recovery strategy"""
        self.recovery_strategies[service_name] = strategy
        logger.debug(f"Set recovery strategy {strategy.value} for {service_name}")
    
    def calculate_delay(self, attempt: int, strategy: RecoveryStrategy) -> float:
        """Calculate delay time"""
        if strategy == RecoveryStrategy.IMMEDIATE:
            return 0
        elif strategy == RecoveryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        elif strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
        elif strategy == RecoveryStrategy.JITTERED_BACKOFF:
            base_delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
            jitter = base_delay * self.config.jitter_range * (2 * random.random() - 1)
            delay = base_delay + jitter
        else:
            delay = self.config.base_delay
        
        return min(delay, self.config.max_delay)
    
    async def execute_with_recovery(self, 
                                   func: Callable, 
                                   service_name: str,
                                   *args, 
                                   **kwargs) -> Any:
        """Execute with recovery mechanism"""
        strategy = self.recovery_strategies.get(service_name, RecoveryStrategy.EXPONENTIAL_BACKOFF)
        circuit_breaker = self.circuit_breakers.get(service_name)
        
        # If using circuit breaker pattern
        if strategy == RecoveryStrategy.CIRCUIT_BREAKER and circuit_breaker:
            if not circuit_breaker.can_execute():
                logger.warning(f"Circuit breaker is OPEN for {service_name}, trying degraded service")
                return await self._try_degraded_service(service_name, *args, **kwargs)
        
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record success
                if circuit_breaker:
                    circuit_breaker.record_success()
                
                logger.debug(f"Service {service_name} executed successfully on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                
                # Record failure
                if circuit_breaker:
                    circuit_breaker.record_failure(error_type)
                
                logger.warning(f"Service {service_name} failed on attempt {attempt + 1}: {e}")
                
                # If this is the last attempt, don't retry
                if attempt >= self.config.max_retries:
                    break
                
                # Calculate delay
                delay = self.calculate_delay(attempt, strategy)
                if delay > 0:
                    logger.debug(f"Waiting {delay:.2f}s before retry {attempt + 2}")
                    await asyncio.sleep(delay)
        
        # All retries failed, try degraded service
        if self.config.enable_degraded_service:
            logger.info(f"All retries failed for {service_name}, trying degraded service")
            try:
                return await self._try_degraded_service(service_name, *args, **kwargs)
            except Exception as degraded_error:
                logger.error(f"Degraded service also failed for {service_name}: {degraded_error}")
        
        # Raise the last exception
        raise last_exception
    
    async def _try_degraded_service(self, service_name: str, *args, **kwargs) -> Any:
        """Try degraded service"""
        degraded_func = self.degraded_services.get(service_name)
        
        if degraded_func is None:
            raise RuntimeError(f"No degraded service available for {service_name}")
        
        try:
            # Set timeout
            if asyncio.iscoroutinefunction(degraded_func):
                result = await asyncio.wait_for(
                    degraded_func(*args, **kwargs),
                    timeout=self.config.degraded_service_timeout
                )
            else:
                result = degraded_func(*args, **kwargs)
            
            logger.info(f"Degraded service executed successfully for {service_name}")
            return result
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Degraded service timeout for {service_name}")
    
    def get_circuit_breaker_stats(self, service_name: str) -> Optional[ErrorStats]:
        """Get circuit breaker statistics"""
        circuit_breaker = self.circuit_breakers.get(service_name)
        return circuit_breaker.stats if circuit_breaker else None
    
    def reset_circuit_breaker(self, service_name: str):
        """Reset circuit breaker"""
        circuit_breaker = self.circuit_breakers.get(service_name)
        if circuit_breaker:
            circuit_breaker.reset()


# Global error recovery manager
global_recovery_manager = ErrorRecoveryManager()


def with_error_recovery(service_name: str, 
                       strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF,
                       config: RecoveryConfig = None):
    """Error recovery decorator"""
    def decorator(func: Callable) -> Callable:
        # Register service
        recovery_manager = global_recovery_manager
        if config:
            recovery_manager = ErrorRecoveryManager(config)
        
        recovery_manager.set_recovery_strategy(service_name, strategy)
        
        if strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            recovery_manager.register_circuit_breaker(service_name)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await recovery_manager.execute_with_recovery(func, service_name, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, create event loop to run async recovery logic
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(
                recovery_manager.execute_with_recovery(func, service_name, *args, **kwargs)
            )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator