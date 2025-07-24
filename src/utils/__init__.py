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

# Import specific modules for backward compatibility
try:
    from .context import (
        AdvancedContextManager,
        ContextPriority,
        CompressionStrategy,
        ExecutionContextManager,
        ContextConfig,
        ContextEvaluator,
    )
except ImportError:
    pass

try:
    from .researcher import (
        ResearcherContextIsolator,
        ResearcherContextConfig,
        ResearcherContextExtension,
        ResearcherConfigOptimizerPhase4,
        ResearcherIsolationMetrics,
        ResearcherIsolationMetricsPhase4,
        ResearcherPhase4Integration,
        ResearcherProgressiveEnablement,
        ResearcherProgressiveEnablementPhase4,
    )
except ImportError:
    pass

try:
    from .performance import (
        PerformanceOptimizer,
        TaskPriority,
        ParallelTask,
        ParallelExecutor,
        WorkflowOptimizer,
        HierarchicalMemoryManager,
        CacheLevel,
        EvictionPolicy,
    )
    from . import performance
    # Make parallel_executor available as a submodule
    from .performance import parallel_executor
except ImportError:
    pass

try:
    from .tokens import (
        TokenManager,
        TokenValidationResult,
        TokenCounter,
        ContentProcessor,
        EnhancedMessageExtractor,
    )
except ImportError:
    pass

try:
    from .system import (
        HealthCheck,
        HealthStatus,
        HealthReport,
        SystemMetrics,
        RateLimiter,
        ErrorRecovery,
        CallbackSafety,
        DependencyInjection,
    )
except ImportError:
    pass

try:
    from .common import (
        safe_background_task,
        retry_with_backoff,
        JsonUtils,
        get_logger,
        EventType,
        SearchResultFilter,
    )
except ImportError:
    pass

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
    "parallel_executor",
    "performance",
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
