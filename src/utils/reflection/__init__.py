"""Reflection module for DeerFlow.

This module provides:
- Enhanced reflection agent for knowledge gap identification
- Reflexion agent with external knowledge retrieval
- Smart reflection controller for dynamic loop control
- Multilingual prompt management
- Workflow integration for automatic step generation
"""

from .models import ReflectionResult, BaseReflectionResult
from .reflection_prompt_manager import ReflectionPromptManager
from .enhanced_reflection import (
    EnhancedReflectionAgent,
    ReflectionContext,
    ReflectionConfig,
)
from .reflexion_models import (
    ReflexionResult,
    KnowledgeGap,
    KnowledgeGapCategory,
    ExternalKnowledge,
    ExternalKnowledgeSource,
    SmartReflectionControllerConfig,
)
from .reflexion_agent import ReflexionAgent
from .smart_reflection_controller import (
    SmartReflectionController,
    KnowledgeGapPrioritizer,
)
from .reflection_integration import (
    FollowUpStep,
    ReflectionStepGenerator,
    integrate_reflection_with_workflow,
    should_continue_research,
    create_step_dict_from_follow_up,
)

__all__ = [
    "BaseReflectionResult",
    "ReflectionResult",
    "ReflectionPromptManager",
    "EnhancedReflectionAgent",
    "ReflectionContext",
    "ReflectionConfig",
    "ReflexionResult",
    "KnowledgeGap",
    "KnowledgeGapCategory",
    "ExternalKnowledge",
    "ExternalKnowledgeSource",
    "SmartReflectionControllerConfig",
    "ReflexionAgent",
    "SmartReflectionController",
    "KnowledgeGapPrioritizer",
    "FollowUpStep",
    "ReflectionStepGenerator",
    "integrate_reflection_with_workflow",
    "should_continue_research",
    "create_step_dict_from_follow_up",
]
