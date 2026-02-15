"""Context management utilities.

This module contains utilities for managing execution context,
advanced context management, context evaluation, KV-Cache optimization,
and smart compression.
"""

from .advanced_context_manager import (
    AdvancedContextManager,
    ContextPriority,
    CompressionStrategy,
)
from .execution_context_manager import ExecutionContextManager, ContextConfig
from .context_evaluator import ContextStateEvaluator
from .kv_cache_optimizer import (
    KVCacheOptimizer,
    KVCacheManager,
    CacheState,
    CacheStats,
    get_global_cache_manager,
)
from .smart_compressor import (
    SmartContextCompressor,
    CompressionLevel,
    ContentType,
    CompressionResult,
    ContentAnalyzer,
    Artifact,
)

__all__ = [
    "AdvancedContextManager",
    "ContextPriority",
    "CompressionStrategy",
    "ExecutionContextManager",
    "ContextConfig",
    "ContextStateEvaluator",
    "KVCacheOptimizer",
    "KVCacheManager",
    "CacheState",
    "CacheStats",
    "get_global_cache_manager",
    "SmartContextCompressor",
    "CompressionLevel",
    "ContentType",
    "CompressionResult",
    "ContentAnalyzer",
    "Artifact",
]
