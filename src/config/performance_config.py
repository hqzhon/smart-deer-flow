"""Performance optimization configuration for DeerFlow."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os


@dataclass
class ConnectionPoolConfig:
    """Connection pool configuration."""
    max_connections: int = 50
    initial_connections: int = 10
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    max_retries: int = 3


@dataclass
class BatchProcessingConfig:
    """Batch processing configuration."""
    batch_size: int = 10
    batch_timeout: float = 1.5
    max_queue_size: int = 1000
    priority_enabled: bool = True
    adaptive_sizing: bool = True


@dataclass
class CacheConfig:
    """Hierarchical cache configuration."""
    l1_size: int = 1000
    l2_size: int = 5000
    l3_size: int = 10000
    default_ttl: int = 3600  # 1 hour
    cleanup_interval: int = 300  # 5 minutes
    eviction_policy: str = "lru"  # lru, lfu, fifo


@dataclass
class RateLimitConfig:
    """Adaptive rate limiting configuration."""
    initial_rate: float = 10.0  # requests per second
    max_rate: float = 100.0
    min_rate: float = 1.0
    adaptation_factor: float = 1.2
    window_size: int = 60  # seconds
    time_window: int = 60  # seconds (alias for window_size for compatibility)
    burst_allowance: int = 20


@dataclass
class ErrorRecoveryConfig:
    """Smart error recovery configuration."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    jitter_enabled: bool = True


@dataclass
class ParallelExecutionConfig:
    """Advanced parallel execution configuration."""
    max_workers: int = 20
    queue_size: int = 1000
    priority_levels: int = 3
    load_balancing: bool = True
    worker_timeout: float = 300.0
    health_check_interval: float = 30.0


@dataclass
class MonitoringConfig:
    """Performance monitoring configuration."""
    metrics_enabled: bool = True
    detailed_logging: bool = True
    slow_request_threshold: float = 10.0  # seconds
    high_utilization_threshold: float = 0.8
    metrics_retention: int = 86400  # 24 hours
    export_interval: int = 60  # seconds


@dataclass
class PerformanceConfig:
    """Main performance configuration container."""
    connection_pool: ConnectionPoolConfig
    batch_processing: BatchProcessingConfig
    cache: CacheConfig
    rate_limit: RateLimitConfig
    error_recovery: ErrorRecoveryConfig
    parallel_execution: ParallelExecutionConfig
    monitoring: MonitoringConfig
    
    # Global settings
    enable_advanced_optimization: bool = True
    enable_collaboration: bool = True
    debug_mode: bool = False
    
    @classmethod
    def from_env(cls) -> 'PerformanceConfig':
        """Create configuration from environment variables."""
        return cls(
            connection_pool=ConnectionPoolConfig(
                max_connections=int(os.getenv('DEER_MAX_CONNECTIONS', 50)),
                initial_connections=int(os.getenv('DEER_INITIAL_CONNECTIONS', 10)),
                connection_timeout=float(os.getenv('DEER_CONNECTION_TIMEOUT', 30.0)),
                idle_timeout=float(os.getenv('DEER_IDLE_TIMEOUT', 300.0)),
                max_retries=int(os.getenv('DEER_CONNECTION_RETRIES', 3))
            ),
            batch_processing=BatchProcessingConfig(
                batch_size=int(os.getenv('DEER_BATCH_SIZE', 10)),
                batch_timeout=float(os.getenv('DEER_BATCH_TIMEOUT', 1.5)),
                max_queue_size=int(os.getenv('DEER_MAX_QUEUE_SIZE', 1000)),
                priority_enabled=os.getenv('DEER_PRIORITY_ENABLED', 'true').lower() == 'true',
                adaptive_sizing=os.getenv('DEER_ADAPTIVE_SIZING', 'true').lower() == 'true'
            ),
            cache=CacheConfig(
                l1_size=int(os.getenv('DEER_L1_CACHE_SIZE', 1000)),
                l2_size=int(os.getenv('DEER_L2_CACHE_SIZE', 5000)),
                l3_size=int(os.getenv('DEER_L3_CACHE_SIZE', 10000)),
                default_ttl=int(os.getenv('DEER_CACHE_TTL', 3600)),
                cleanup_interval=int(os.getenv('DEER_CACHE_CLEANUP_INTERVAL', 300)),
                eviction_policy=os.getenv('DEER_CACHE_EVICTION_POLICY', 'lru')
            ),
            rate_limit=RateLimitConfig(
                initial_rate=float(os.getenv('DEER_INITIAL_RATE', 10.0)),
                max_rate=float(os.getenv('DEER_MAX_RATE', 100.0)),
                min_rate=float(os.getenv('DEER_MIN_RATE', 1.0)),
                adaptation_factor=float(os.getenv('DEER_ADAPTATION_FACTOR', 1.2)),
                window_size=int(os.getenv('DEER_RATE_WINDOW_SIZE', 60)),
                time_window=int(os.getenv('DEER_RATE_TIME_WINDOW', os.getenv('DEER_RATE_WINDOW_SIZE', 60))),
                burst_allowance=int(os.getenv('DEER_BURST_ALLOWANCE', 20))
            ),
            error_recovery=ErrorRecoveryConfig(
                max_retries=int(os.getenv('DEER_MAX_RETRIES', 3)),
                base_delay=float(os.getenv('DEER_BASE_DELAY', 1.0)),
                max_delay=float(os.getenv('DEER_MAX_DELAY', 60.0)),
                exponential_base=float(os.getenv('DEER_EXPONENTIAL_BASE', 2.0)),
                circuit_breaker_threshold=int(os.getenv('DEER_CIRCUIT_THRESHOLD', 5)),
                circuit_breaker_timeout=float(os.getenv('DEER_CIRCUIT_TIMEOUT', 60.0)),
                jitter_enabled=os.getenv('DEER_JITTER_ENABLED', 'true').lower() == 'true'
            ),
            parallel_execution=ParallelExecutionConfig(
                max_workers=int(os.getenv('DEER_MAX_WORKERS', 20)),
                queue_size=int(os.getenv('DEER_PARALLEL_QUEUE_SIZE', 1000)),
                priority_levels=int(os.getenv('DEER_PRIORITY_LEVELS', 3)),
                load_balancing=os.getenv('DEER_LOAD_BALANCING', 'true').lower() == 'true',
                worker_timeout=float(os.getenv('DEER_WORKER_TIMEOUT', 300.0)),
                health_check_interval=float(os.getenv('DEER_HEALTH_CHECK_INTERVAL', 30.0))
            ),
            monitoring=MonitoringConfig(
                metrics_enabled=os.getenv('DEER_METRICS_ENABLED', 'true').lower() == 'true',
                detailed_logging=os.getenv('DEER_DETAILED_LOGGING', 'true').lower() == 'true',
                slow_request_threshold=float(os.getenv('DEER_SLOW_REQUEST_THRESHOLD', 10.0)),
                high_utilization_threshold=float(os.getenv('DEER_HIGH_UTILIZATION_THRESHOLD', 0.8)),
                metrics_retention=int(os.getenv('DEER_METRICS_RETENTION', 86400)),
                export_interval=int(os.getenv('DEER_EXPORT_INTERVAL', 60))
            ),
            enable_advanced_optimization=os.getenv('DEER_ADVANCED_OPTIMIZATION', 'true').lower() == 'true',
            enable_collaboration=os.getenv('DEER_COLLABORATION', 'true').lower() == 'true',
            debug_mode=os.getenv('DEER_DEBUG_MODE', 'false').lower() == 'true'
        )
    
    @classmethod
    def default(cls) -> 'PerformanceConfig':
        """Create default configuration."""
        return cls(
            connection_pool=ConnectionPoolConfig(),
            batch_processing=BatchProcessingConfig(),
            cache=CacheConfig(),
            rate_limit=RateLimitConfig(),
            error_recovery=ErrorRecoveryConfig(),
            parallel_execution=ParallelExecutionConfig(),
            monitoring=MonitoringConfig()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'connection_pool': self.connection_pool.__dict__,
            'batch_processing': self.batch_processing.__dict__,
            'cache': self.cache.__dict__,
            'rate_limit': self.rate_limit.__dict__,
            'error_recovery': self.error_recovery.__dict__,
            'parallel_execution': self.parallel_execution.__dict__,
            'monitoring': self.monitoring.__dict__,
            'enable_advanced_optimization': self.enable_advanced_optimization,
            'enable_collaboration': self.enable_collaboration,
            'debug_mode': self.debug_mode
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for section, values in config_dict.items():
            if hasattr(self, section) and isinstance(values, dict):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
            elif hasattr(self, section):
                setattr(self, section, values)


# Global configuration instance
performance_config = PerformanceConfig.from_env()


def get_performance_config() -> PerformanceConfig:
    """Get the global performance configuration."""
    return performance_config


def update_performance_config(config_dict: Dict[str, Any]) -> None:
    """Update the global performance configuration."""
    global performance_config
    performance_config.update_from_dict(config_dict)


def reset_performance_config() -> None:
    """Reset to default configuration."""
    global performance_config
    performance_config = PerformanceConfig.default()