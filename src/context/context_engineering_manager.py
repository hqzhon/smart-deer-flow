# -*- coding: utf-8 -*-
"""
Context Engineering Manager - Unified context management.

Integrates all context engineering components:
- FileBasedMemory (filesystem as context)
- AttentionManager (attention manipulation)
- KVCacheOptimizer (KV-Cache optimization)
- SmartContextCompressor (intelligent compression)

Follows Manus AI principles:
1. KV-Cache design with stable prefix
2. File system as unlimited external memory
3. Attention manipulation through restatement
4. Error preservation for learning
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from src.context.file_based_memory import FileBasedMemory
from src.context.attention_manipulation import AttentionManager
from src.context.attention_injection import (
    AttentionInjector,
    AttentionInjectionConfig,
    InjectionPoint,
)
from src.utils.context.kv_cache_optimizer import (
    KVCacheOptimizer,
    get_global_cache_manager,
)
from src.utils.context.smart_compressor import (
    SmartContextCompressor,
    ContentType,
)

logger = logging.getLogger(__name__)


@dataclass
class ContextEngineeringConfig:
    """Configuration for context engineering manager."""

    enable_file_memory: bool = True
    enable_attention_injection: bool = True
    enable_kv_cache: bool = True
    enable_smart_compression: bool = True

    max_context_tokens: int = 8000
    attention_injection_interval: int = 5
    max_injections_per_session: int = 10

    workspace_path: Optional[str] = None


@dataclass
class SessionMetrics:
    """Metrics for a context engineering session."""

    session_id: str
    start_time: datetime = field(default_factory=datetime.now)
    total_tokens_processed: int = 0
    total_tokens_saved: int = 0
    attention_injections: int = 0
    compression_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "total_tokens_processed": self.total_tokens_processed,
            "total_tokens_saved": self.total_tokens_saved,
            "attention_injections": self.attention_injections,
            "compression_operations": self.compression_operations,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                f"{self.cache_hits / (self.cache_hits + self.cache_misses):.2%}"
                if (self.cache_hits + self.cache_misses) > 0
                else "N/A"
            ),
        }


class ContextEngineeringManager:
    """Unified context engineering manager.

    This class provides a single entry point for all context engineering
    operations, integrating:
    - File-based external memory
    - Attention manipulation
    - KV-Cache optimization
    - Smart compression

    Usage:
        manager = ContextEngineeringManager(config)
        await manager.initialize_session(thread_id, research_topic, plan)
        context = manager.get_context_for_llm(state, max_tokens)
    """

    def __init__(self, config: Optional[ContextEngineeringConfig] = None):
        self.config = config or ContextEngineeringConfig()

        self._file_memory: Optional[FileBasedMemory] = None
        self._attention_manager: Optional[AttentionManager] = None
        self._attention_injector: Optional[AttentionInjector] = None
        self._kv_optimizer: Optional[KVCacheOptimizer] = None
        self._compressor: Optional[SmartContextCompressor] = None

        self._sessions: Dict[str, SessionMetrics] = {}
        self._initialized = False

    def initialize_session(
        self,
        thread_id: str,
        research_topic: str,
        initial_plan: Optional[Any] = None,
        locale: str = "en-US",
    ) -> None:
        """Initialize a new context engineering session.

        Args:
            thread_id: Unique session identifier
            research_topic: The research topic/goal
            initial_plan: Optional initial research plan
            locale: Language locale
        """
        logger.info(f"Initializing context engineering session: {thread_id}")

        self._sessions[thread_id] = SessionMetrics(session_id=thread_id)

        if self.config.enable_file_memory:
            workspace = Path(
                self.config.workspace_path or f".deer-flow-memory/{thread_id}"
            )
            self._file_memory = FileBasedMemory(workspace)
            self._file_memory.initialize(
                research_topic=research_topic,
                initial_plan=(
                    {"phases": self._plan_to_phases(initial_plan)}
                    if initial_plan
                    else {}
                ),
            )
            logger.info(f"File-based memory initialized at {workspace}")

        if self.config.enable_attention_injection:
            self._attention_manager = AttentionManager()
            self._attention_manager.create_goal_tracker(
                thread_id=thread_id,
                original_goal=research_topic,
            )
            self._attention_injector = AttentionInjector(
                AttentionInjectionConfig(
                    injection_interval=self.config.attention_injection_interval,
                    max_injections_per_session=self.config.max_injections_per_session,
                )
            )
            logger.info("Attention management initialized")

        if self.config.enable_kv_cache:
            cache_manager = get_global_cache_manager()
            self._kv_optimizer = cache_manager.get_cache(thread_id)

            stable_prefix = self._build_stable_prefix(research_topic, locale)
            self._kv_optimizer.set_stable_prefix(stable_prefix)
            logger.info("KV-Cache optimizer initialized")

        if self.config.enable_smart_compression:
            self._compressor = SmartContextCompressor()
            logger.info("Smart compressor initialized")

        self._initialized = True
        logger.info(f"Context engineering session initialized: {thread_id}")

    def get_context_for_llm(
        self,
        state: Dict[str, Any],
        max_tokens: Optional[int] = None,
        include_attention: bool = True,
    ) -> str:
        """Get optimized context for LLM input.

        Args:
            state: Current workflow state
            max_tokens: Maximum tokens allowed (uses config if not specified)
            include_attention: Whether to include attention reminder

        Returns:
            Optimized context string
        """
        max_tokens = max_tokens or self.config.max_context_tokens
        context_parts = []

        if self._kv_optimizer:
            context_parts.append(self._kv_optimizer.get_optimized_context())

        if self._file_memory:
            memory_summary = self._file_memory.get_context_summary()
            if memory_summary:
                context_parts.append(memory_summary)

        if include_attention and self._attention_injector:
            thread_id = state.get("thread_id", "default")
            attention_msg = self._attention_injector.create_attention_message(
                thread_id=thread_id,
                original_goal=state.get("research_topic", ""),
                current_focus=state.get("current_step", {}).get("title", "Research"),
                progress=(
                    state.get("current_step_index", 0),
                    state.get("total_steps", 1),
                ),
                injection_point=InjectionPoint.PERIODIC,
                locale=state.get("locale", "en-US"),
            )
            if attention_msg:
                context_parts.append(attention_msg.content)
                self._sessions[thread_id].attention_injections += 1

        full_context = "\n\n".join(context_parts)
        current_tokens = len(full_context) // 4

        self._sessions[
            state.get("thread_id", "default")
        ].total_tokens_processed += current_tokens

        if current_tokens > max_tokens and self._compressor:
            result = self._compressor.compress(
                full_context,
                max_tokens,
                ContentType.MIXED,
            )
            full_context = result.compressed_content
            self._sessions[
                state.get("thread_id", "default")
            ].total_tokens_saved += result.tokens_saved
            self._sessions[
                state.get("thread_id", "default")
            ].compression_operations += 1

        return full_context

    def record_finding(
        self,
        content: str,
        source: str,
        relevance: float = 1.0,
    ) -> None:
        """Record a research finding.

        Args:
            content: Finding content
            source: Source of the finding
            relevance: Relevance score
        """
        if self._file_memory:
            self._file_memory.findings.add_observation(
                content=content,
                source=source,
                relevance=relevance,
            )
            self._file_memory.persist()
            logger.debug(f"Recorded finding from {source}")

    def record_error(
        self,
        error: str,
        context: str,
        recovery: Optional[str] = None,
    ) -> None:
        """Record an error for learning.

        Args:
            error: Error message
            context: Error context
            recovery: Recovery action taken
        """
        if self._file_memory:
            self._file_memory.progress.log_error(
                error=error,
                context=context,
                recovery=recovery,
            )
            self._file_memory.persist()
            logger.debug(f"Recorded error: {error[:50]}...")

    def update_progress(
        self,
        phase_idx: int,
        step_idx: int,
        completed: bool,
        notes: Optional[str] = None,
    ) -> None:
        """Update research progress.

        Args:
            phase_idx: Phase index
            step_idx: Step index
            completed: Whether step is completed
            notes: Optional notes
        """
        if self._file_memory:
            self._file_memory.update_step_status(phase_idx, step_idx, completed)
            if notes:
                self._file_memory.progress.log_info(notes)
            self._file_memory.persist()

    def append_context(self, content: str) -> None:
        """Append content to the context buffer.

        Args:
            content: Content to append
        """
        if self._kv_optimizer:
            self._kv_optimizer.append_context(content)

    def check_goal_drift(
        self,
        thread_id: str,
        current_action: str,
    ) -> Optional[str]:
        """Check for potential goal drift.

        Args:
            thread_id: Session identifier
            current_action: Current action being performed

        Returns:
            Warning message if drift detected, None otherwise
        """
        if self._attention_injector:
            tracker = (
                self._attention_manager.get_goal_tracker(thread_id)
                if self._attention_manager
                else None
            )
            if tracker:
                return self._attention_injector.detect_goal_drift(
                    original_goal=tracker.original_goal,
                    current_action=current_action,
                )
        return None

    def get_session_metrics(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a session.

        Args:
            thread_id: Session identifier

        Returns:
            Session metrics dictionary
        """
        if thread_id in self._sessions:
            return self._sessions[thread_id].to_dict()
        return None

    def get_cache_stats(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get KV-Cache statistics for a session.

        Args:
            thread_id: Session identifier

        Returns:
            Cache statistics dictionary
        """
        if self._kv_optimizer:
            stats = self._kv_optimizer.get_stats()
            if thread_id in self._sessions:
                self._sessions[thread_id].cache_hits = stats.cache_hits
                self._sessions[thread_id].cache_misses = stats.cache_misses
            return stats.to_dict()
        return None

    def end_session(self, thread_id: str) -> Dict[str, Any]:
        """End a context engineering session.

        Args:
            thread_id: Session identifier

        Returns:
            Final session metrics
        """
        metrics = self.get_session_metrics(thread_id) or {}

        if self._file_memory:
            self._file_memory.persist()

        if thread_id in self._sessions:
            del self._sessions[thread_id]

        logger.info(f"Context engineering session ended: {thread_id}")
        return metrics

    def _build_stable_prefix(self, research_topic: str, locale: str) -> str:
        """Build the stable prefix for KV-Cache.

        Args:
            research_topic: Research topic
            locale: Language locale

        Returns:
            Stable prefix string
        """
        if locale.startswith("zh"):
            return f"""# 研究上下文

**研究主题**: {research_topic}

本上下文将用于整个研究过程，请保持对此主题的关注。
"""
        else:
            return f"""# Research Context

**Research Topic**: {research_topic}

This context will be maintained throughout the research process. Please keep focus on this topic.
"""

    def _plan_to_phases(self, plan: Any) -> List[Dict[str, Any]]:
        """Convert a plan to phases format for file memory.

        Args:
            plan: Research plan object

        Returns:
            List of phase dictionaries
        """
        if not plan:
            return []

        phases = []
        if hasattr(plan, "steps"):
            phases.append(
                {
                    "name": "Research Phase",
                    "steps": [
                        {"title": s.title, "description": s.description}
                        for s in plan.steps
                    ],
                }
            )
        elif isinstance(plan, dict) and "steps" in plan:
            phases.append(
                {
                    "name": "Research Phase",
                    "steps": [
                        {
                            "title": s.get("title", ""),
                            "description": s.get("description", ""),
                        }
                        for s in plan["steps"]
                    ],
                }
            )

        return phases


_global_manager: Optional[ContextEngineeringManager] = None


def get_context_engineering_manager(
    config: Optional[ContextEngineeringConfig] = None,
) -> ContextEngineeringManager:
    """Get the global context engineering manager instance.

    Args:
        config: Optional configuration (used only on first call)

    Returns:
        ContextEngineeringManager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = ContextEngineeringManager(config)
    return _global_manager


def create_session_context_manager(
    thread_id: str,
    research_topic: str,
    initial_plan: Optional[Any] = None,
    locale: str = "en-US",
) -> ContextEngineeringManager:
    """Create and initialize a context engineering manager for a session.

    Args:
        thread_id: Session identifier
        research_topic: Research topic
        initial_plan: Optional initial plan
        locale: Language locale

    Returns:
        Initialized ContextEngineeringManager
    """
    manager = get_context_engineering_manager()
    manager.initialize_session(thread_id, research_topic, initial_plan, locale)
    return manager
