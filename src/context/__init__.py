"""Context Engineering Module for Deer Flow.

This module implements context engineering best practices inspired by Manus AI,
including:
- File-based external memory (three-file pattern)
- KV cache optimization
- Attention manipulation
- Error recovery with learning
- Diversity injection
- Recoverable compression
- Unified context engineering manager
"""

from .file_based_memory import (
    TaskPlan,
    Findings,
    Progress,
    FileBasedMemory,
)
from .research_memory_integration import ResearchMemoryManager
from .attention_manipulation import GoalTracker, AttentionManager
from .attention_injection import (
    AttentionInjector,
    AttentionInjectionConfig,
    InjectionPoint,
    inject_attention_to_messages,
)
from .diversity_injector import DiversityInjector, DiversityConfig
from .error_recovery import (
    ErrorRecoveryManager,
    ErrorRecord,
    ErrorSeverity,
    ToolExecutorWithErrorRecovery,
)
from .recoverable_compression import (
    RecoverableCompressor,
    CompressedContent,
    ContextManagerWithCompression,
)
from .metrics import ContextMetrics
from .context_engineering_manager import (
    ContextEngineeringManager,
    ContextEngineeringConfig,
    SessionMetrics,
    get_context_engineering_manager,
    create_session_context_manager,
)

__all__ = [
    "TaskPlan",
    "Findings",
    "Progress",
    "FileBasedMemory",
    "ResearchMemoryManager",
    "GoalTracker",
    "AttentionManager",
    "AttentionInjector",
    "AttentionInjectionConfig",
    "InjectionPoint",
    "inject_attention_to_messages",
    "DiversityInjector",
    "DiversityConfig",
    "ErrorRecoveryManager",
    "ErrorRecord",
    "ErrorSeverity",
    "ToolExecutorWithErrorRecovery",
    "RecoverableCompressor",
    "CompressedContent",
    "ContextManagerWithCompression",
    "ContextMetrics",
    "ContextEngineeringManager",
    "ContextEngineeringConfig",
    "SessionMetrics",
    "get_context_engineering_manager",
    "create_session_context_manager",
]
