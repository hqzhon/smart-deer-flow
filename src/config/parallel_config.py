#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel execution and rate limiting configuration
"""

from dataclasses import dataclass
from typing import Dict, Optional
import os


@dataclass
class ParallelExecutionConfig:
    """Parallel execution configuration"""
    enable_parallel_execution: bool = True
    max_parallel_tasks: int = 3
    enable_adaptive_scheduling: bool = True
    task_timeout: float = 300.0  # 5 minutes
    max_retries_per_task: int = 2
    
    @classmethod
    def from_env(cls) -> 'ParallelExecutionConfig':
        """Load configuration from environment variables"""
        return cls(
            enable_parallel_execution=os.getenv('ENABLE_PARALLEL_EXECUTION', 'true').lower() == 'true',
            max_parallel_tasks=int(os.getenv('MAX_PARALLEL_TASKS', '3')),
            enable_adaptive_scheduling=os.getenv('ENABLE_ADAPTIVE_SCHEDULING', 'true').lower() == 'true',
            task_timeout=float(os.getenv('TASK_TIMEOUT', '300.0')),
            max_retries_per_task=int(os.getenv('MAX_RETRIES_PER_TASK', '2'))
        )


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    enable_rate_limiting: bool = True
    requests_per_minute: int = 60
    burst_capacity: int = 10
    adaptive_adjustment: bool = True
    min_requests_per_minute: int = 10
    max_requests_per_minute: int = 120
    
    # Default configurations for different model providers
    provider_configs: Dict[str, Dict[str, int]] = None
    
    def __post_init__(self):
        if self.provider_configs is None:
            self.provider_configs = {
                'openai': {
                    'requests_per_minute': 60,
                    'burst_capacity': 10
                },
                'claude': {
                    'requests_per_minute': 50,
                    'burst_capacity': 8
                },
                'zhipu': {
                    'requests_per_minute': 30,
                    'burst_capacity': 5
                },
                'deepseek': {
                    'requests_per_minute': 40,
                    'burst_capacity': 6
                },
                'qwen': {
                    'requests_per_minute': 45,
                    'burst_capacity': 7
                },
                'doubao': {
                    'requests_per_minute': 35,
                    'burst_capacity': 5
                },
                'moonshot': {
                    'requests_per_minute': 25,
                    'burst_capacity': 4
                },
                'yi': {
                    'requests_per_minute': 30,
                    'burst_capacity': 5
                }
            }
    
    @classmethod
    def from_env(cls) -> 'RateLimitConfig':
        """Load configuration from environment variables"""
        return cls(
            enable_rate_limiting=os.getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true',
            requests_per_minute=int(os.getenv('REQUESTS_PER_MINUTE', '60')),
            burst_capacity=int(os.getenv('BURST_CAPACITY', '10')),
            adaptive_adjustment=os.getenv('ADAPTIVE_ADJUSTMENT', 'true').lower() == 'true',
            min_requests_per_minute=int(os.getenv('MIN_REQUESTS_PER_MINUTE', '10')),
            max_requests_per_minute=int(os.getenv('MAX_REQUESTS_PER_MINUTE', '120'))
        )
    
    def get_provider_config(self, provider: str) -> Dict[str, int]:
        """Get configuration for specific provider"""
        provider_lower = provider.lower()
        if provider_lower in self.provider_configs:
            return self.provider_configs[provider_lower]
        
        # Return default configuration
        return {
            'requests_per_minute': self.requests_per_minute,
            'burst_capacity': self.burst_capacity
        }


@dataclass
class OptimizationConfig:
    """Performance optimization configuration"""
    # Error handling
    max_consecutive_failures: int = 5
    failure_cooldown_seconds: int = 30
    
    # Monitoring and logging
    enable_performance_monitoring: bool = True
    log_execution_stats: bool = True
    stats_report_interval: int = 60  # seconds
    
    # Resource management
    memory_limit_mb: Optional[int] = None
    cpu_limit_percent: Optional[int] = None
    
    @classmethod
    def from_env(cls) -> 'OptimizationConfig':
        """Load configuration from environment variables"""
        memory_limit = os.getenv('MEMORY_LIMIT_MB')
        cpu_limit = os.getenv('CPU_LIMIT_PERCENT')
        
        return cls(
            max_consecutive_failures=int(os.getenv('MAX_CONSECUTIVE_FAILURES', '5')),
            failure_cooldown_seconds=int(os.getenv('FAILURE_COOLDOWN_SECONDS', '30')),
            enable_performance_monitoring=os.getenv('ENABLE_PERFORMANCE_MONITORING', 'true').lower() == 'true',
            log_execution_stats=os.getenv('LOG_EXECUTION_STATS', 'true').lower() == 'true',
            stats_report_interval=int(os.getenv('STATS_REPORT_INTERVAL', '60')),
            memory_limit_mb=int(memory_limit) if memory_limit else None,
            cpu_limit_percent=int(cpu_limit) if cpu_limit else None
        )


@dataclass
class CombinedConfig:
    """Combined configuration"""
    parallel: ParallelExecutionConfig
    rate_limit: RateLimitConfig
    optimization: OptimizationConfig
    
    @classmethod
    def from_env(cls) -> 'CombinedConfig':
        """Load all configurations from environment variables"""
        return cls(
            parallel=ParallelExecutionConfig.from_env(),
            rate_limit=RateLimitConfig.from_env(),
            optimization=OptimizationConfig.from_env()
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'CombinedConfig':
        """Load configuration from dictionary"""
        parallel_config = ParallelExecutionConfig(
            **config_dict.get('parallel_execution', {})
        )
        
        rate_limit_config = RateLimitConfig(
            **config_dict.get('rate_limiting', {})
        )
        
        optimization_config = OptimizationConfig(
            **config_dict.get('optimization', {})
        )
        
        return cls(
            parallel=parallel_config,
            rate_limit=rate_limit_config,
            optimization=optimization_config
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'parallel_execution': {
                'enable_parallel_execution': self.parallel.enable_parallel_execution,
                'max_parallel_tasks': self.parallel.max_parallel_tasks,
                'enable_adaptive_scheduling': self.parallel.enable_adaptive_scheduling,
                'task_timeout': self.parallel.task_timeout,
                'max_retries_per_task': self.parallel.max_retries_per_task
            },
            'rate_limiting': {
                'enable_rate_limiting': self.rate_limit.enable_rate_limiting,
                'requests_per_minute': self.rate_limit.requests_per_minute,
                'burst_capacity': self.rate_limit.burst_capacity,
                'adaptive_adjustment': self.rate_limit.adaptive_adjustment,
                'min_requests_per_minute': self.rate_limit.min_requests_per_minute,
                'max_requests_per_minute': self.rate_limit.max_requests_per_minute,
                'provider_configs': self.rate_limit.provider_configs
            },
            'optimization': {
                'max_consecutive_failures': self.optimization.max_consecutive_failures,
                'failure_cooldown_seconds': self.optimization.failure_cooldown_seconds,
                'enable_performance_monitoring': self.optimization.enable_performance_monitoring,
                'log_execution_stats': self.optimization.log_execution_stats,
                'stats_report_interval': self.optimization.stats_report_interval,
                'memory_limit_mb': self.optimization.memory_limit_mb,
                'cpu_limit_percent': self.optimization.cpu_limit_percent
            }
        }


# Global configuration instance
_global_config: Optional[CombinedConfig] = None


def get_global_config() -> CombinedConfig:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = CombinedConfig.from_env()
    return _global_config


def set_global_config(config: CombinedConfig):
    """Set global configuration instance"""
    global _global_config
    _global_config = config


def load_config_from_file(config_path: str) -> CombinedConfig:
    """Load configuration from configuration file"""
    import yaml
    import json
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
        
        return CombinedConfig.from_dict(config_dict)
    
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load config from {config_path}: {e}, using environment defaults")
        return CombinedConfig.from_env()