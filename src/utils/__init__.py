# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Utility functions package

This package has been reorganized into submodules for better maintainability:
- context: Context management utilities
- researcher: Researcher-specific utilities
- performance: Performance optimization utilities
- tokens: Token and content management utilities
- system: System infrastructure utilities
- config: Configuration management utilities
- common: Common utilities and helpers

For backward compatibility, all modules are still available at the top level.
"""

# Import all submodules for backward compatibility
from .context import *
from .researcher import *
from .performance import *
from .tokens import *
from .system import *
from .common import *

# Re-export commonly used classes and functions
__all__ = [
    # Context management
    "AdvancedContextManager",
    "ContextPriority",
    "CompressionStrategy",
    "ExecutionContextManager",
    "ContextConfig",
    "ContextEvaluator",
    # Researcher utilities
    "ResearcherContextIsolator",
    "ResearcherContextConfig",
    "ResearcherContextExtension",
    "ResearcherConfigOptimizerPhase4",
    "ResearcherIsolationMetrics",
    "ResearcherIsolationMetricsPhase4",
    "ResearcherPhase4Integration",
    "ResearcherProgressiveEnablement",
    "ResearcherProgressiveEnablementPhase4",
    # Performance optimization
    "PerformanceOptimizer",
    "TaskPriority",
    "ParallelTask",
    "ParallelExecutor",
    "WorkflowOptimizer",
    "HierarchicalMemoryManager",
    "CacheLevel",
    "EvictionPolicy",
    # Token and content management
    "TokenManager",
    "TokenValidationResult",
    "TokenCounter",
    "ContentProcessor",
    "EnhancedMessageExtractor",
    # System infrastructure
    "HealthCheck",
    "HealthStatus",
    "HealthReport",
    "SystemMetrics",
    "RateLimiter",
    "ErrorRecovery",
    "CallbackSafety",
    "DependencyInjection",
    # Configuration management
    "# Removed - use get_settings() instead",
    "# Removed - use get_settings() instead",
    # Common utilities
    "safe_background_task",
    "retry_with_backoff",
    "JsonUtils",
    "get_logger",
    "EventType",
    "SearchResultFilter",
]
