"""Researcher-specific utilities.

This module contains utilities specifically designed for researcher nodes,
including context isolation, metrics, and progressive enablement features.
"""

from .researcher_context_isolator import (
    ResearcherContextIsolator,
    ResearcherContextConfig,
)
from .researcher_context_extension import ResearcherContextExtension
from .researcher_config_optimizer_phase4 import ConfigurationOptimizer
from .researcher_isolation_metrics import ResearcherIsolationMetrics
from .researcher_isolation_metrics_phase4 import AdvancedResearcherIsolationMetrics
from .researcher_phase4_integration import ResearcherPhase4System
from .researcher_progressive_enablement import ResearcherProgressiveEnabler
from .researcher_progressive_enablement_phase4 import (
    AdvancedResearcherProgressiveEnabler,
)

# Phase 1 Simplification: New management classes
from .isolation_config_manager import (
    IsolationConfigManager,
    UnifiedResearchConfig,
)
from .research_tool_manager import ResearchToolManager
from .mcp_client_manager import MCPClientManager
from .reflection_system_manager import ReflectionSystemManager
from .iterative_research_engine import IterativeResearchEngine
from .research_result_processor import ResearchResultProcessor

# Note: EnhancedResearcher has been successfully refactored into single-responsibility graph nodes
# The new nodes are: prepare_research_step_node, researcher_agent_node, reflection_node, update_plan_node
# See src/graph/nodes.py for the implementation

# Phase 5 Performance Optimization
from .performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    GlobalPerformanceTracker,
    global_performance_tracker,
)
from .config_cache_optimizer import (
    ConfigCacheOptimizer,
    SmartCache,
    CacheEntry,
    global_config_cache_optimizer,
)
from .concurrent_optimizer import (
    ConcurrentTaskManager,
    TaskInfo,
    TaskPriority,
    TaskStatus,
    AdaptiveSemaphore,
    concurrent_execution_context,
    global_concurrent_manager,
)

__all__ = [
    "ResearcherContextIsolator",
    "ResearcherContextConfig",
    "ResearcherContextExtension",
    "ConfigurationOptimizer",
    "ResearcherIsolationMetrics",
    "AdvancedResearcherIsolationMetrics",
    "ResearcherPhase4System",
    "ResearcherProgressiveEnabler",
    "AdvancedResearcherProgressiveEnabler",
    # Phase 1 Simplification exports
    "IsolationConfigManager",
    "UnifiedResearchConfig",
    "ResearchToolManager",
    "MCPClientManager",
    "ReflectionSystemManager",
    "IterativeResearchEngine",
    "ResearchResultProcessor",
    # Phase 5 Performance Optimization exports
    "PerformanceMonitor",
    "PerformanceMetrics",
    "GlobalPerformanceTracker",
    "global_performance_tracker",
    "ConfigCacheOptimizer",
    "SmartCache",
    "CacheEntry",
    "global_config_cache_optimizer",
    "ConcurrentTaskManager",
    "TaskInfo",
    "TaskPriority",
    "TaskStatus",
    "AdaptiveSemaphore",
    "concurrent_execution_context",
    "global_concurrent_manager",
]
