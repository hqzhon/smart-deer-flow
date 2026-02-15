"""Research memory integration with LangGraph workflow.

This module integrates the file-based memory system with the
LangGraph research workflow, enabling persistent research state
across context resets.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import threading
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .file_based_memory import FileBasedMemory


@dataclass
class MemoryConfig:
    """Configuration for research memory."""

    base_dir: Path = field(default_factory=lambda: Path(".deer-flow-memory"))
    auto_persist: bool = True
    inject_attention_interval: int = 10
    max_observations_in_context: int = 20


class ResearchMemoryManager:
    """Research memory manager - integrates with LangGraph workflow.

    This class provides a singleton-like interface for managing
    research memory across the application. It supports:
    - Starting new research sessions
    - Loading existing sessions
    - Injecting context into LLM messages
    - Tracking progress and findings
    """

    _instance: Optional["ResearchMemoryManager"] = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[MemoryConfig] = None) -> "ResearchMemoryManager":
        """Singleton pattern for global memory management."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the memory manager."""
        if self._initialized:
            return

        self.config = config or MemoryConfig()
        self.config.base_dir.mkdir(parents=True, exist_ok=True)

        self._memories: Dict[str, FileBasedMemory] = {}
        self._active_thread_id: Optional[str] = None
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "ResearchMemoryManager":
        """Get the singleton instance."""
        if cls._instance is None:
            return cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def start_research(
        self,
        thread_id: Optional[str] = None,
        research_topic: str = "",
        initial_plan: Optional[Dict[str, Any]] = None,
    ) -> FileBasedMemory:
        """Start a new research session.

        Args:
            thread_id: Optional thread ID (generated if not provided)
            research_topic: The research topic/question
            initial_plan: Initial research plan with phases

        Returns:
            FileBasedMemory instance for the session
        """
        thread_id = thread_id or str(uuid.uuid4())

        workspace = self.config.base_dir / thread_id
        memory = FileBasedMemory(workspace)

        if research_topic:
            memory.initialize(research_topic, initial_plan or {})

        self._memories[thread_id] = memory
        self._active_thread_id = thread_id

        return memory

    def get_memory(self, thread_id: Optional[str] = None) -> FileBasedMemory:
        """Get or load research memory for a thread.

        Args:
            thread_id: Thread ID (uses active thread if not provided)

        Returns:
            FileBasedMemory instance
        """
        thread_id = thread_id or self._active_thread_id or "default"

        if thread_id not in self._memories:
            workspace = self.config.base_dir / thread_id
            memory = FileBasedMemory(workspace)
            memory.load_from_disk()
            self._memories[thread_id] = memory

        return self._memories[thread_id]

    def set_active_thread(self, thread_id: str) -> None:
        """Set the active thread ID."""
        self._active_thread_id = thread_id

    def get_active_thread_id(self) -> Optional[str]:
        """Get the active thread ID."""
        return self._active_thread_id

    def inject_context(
        self, state: Dict[str, Any], include_attention_reminder: bool = True
    ) -> str:
        """Inject file memory into context.

        Implements the "restate to manipulate attention" principle
        by including goal reminders in the context.

        Args:
            state: Current workflow state
            include_attention_reminder: Whether to include goal reminder

        Returns:
            Context string for injection
        """
        thread_id = state.get("thread_id", self._active_thread_id or "default")
        memory = self.get_memory(thread_id)

        context_parts = ["## 当前研究状态（来自持久化记忆）\n"]

        context_summary = memory.get_context_summary()
        if context_summary:
            context_parts.append(context_summary)

        if include_attention_reminder:
            attention_reminder = memory.get_attention_reminder()
            if attention_reminder:
                context_parts.append("\n" + attention_reminder)

        context_parts.append("\n---\n*请根据以上状态继续研究，确保目标一致性。*")

        return "\n".join(context_parts)

    def create_context_message(
        self, state: Dict[str, Any], message_type: str = "system"
    ) -> Optional[BaseMessage]:
        """Create a context message for LLM injection.

        Args:
            state: Current workflow state
            message_type: Type of message (system, human)

        Returns:
            Message object or None if no context to inject
        """
        context = self.inject_context(state)

        if (
            not context
            or context
            == "## 当前研究状态（来自持久化记忆）\n\n---\n*请根据以上状态继续研究，确保目标一致性。*"
        ):
            return None

        if message_type == "system":
            return SystemMessage(content=context)
        elif message_type == "human":
            return HumanMessage(content=context)

        return SystemMessage(content=context)

    def log_action(
        self,
        action: str,
        result: str,
        success: bool = True,
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an action to the progress file.

        Args:
            action: Action description
            result: Result of the action
            success: Whether the action succeeded
            thread_id: Thread ID (uses active if not provided)
            metadata: Optional metadata
        """
        memory = self.get_memory(thread_id)
        memory.progress.log_action(action, result, success, metadata)

        if self.config.auto_persist:
            memory.persist()

    def log_error(
        self,
        error: str,
        context: str,
        recovery: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Log an error for learning.

        Implements the "preserve error content" principle.

        Args:
            error: Error message
            context: Error context
            recovery: Recovery strategy used
            thread_id: Thread ID
        """
        memory = self.get_memory(thread_id)
        memory.progress.log_error(error, context, recovery)

        if self.config.auto_persist:
            memory.persist()

    def add_observation(
        self,
        content: str,
        source: str,
        relevance: float = 1.0,
        thread_id: Optional[str] = None,
    ) -> None:
        """Add an observation to findings.

        Args:
            content: Observation content
            source: Source of observation
            relevance: Relevance score
            thread_id: Thread ID
        """
        memory = self.get_memory(thread_id)
        memory.findings.add_observation(content, source, relevance)

        if self.config.auto_persist:
            memory.persist()

    def add_key_insight(self, insight: str, thread_id: Optional[str] = None) -> None:
        """Add a key insight.

        Args:
            insight: Key insight
            thread_id: Thread ID
        """
        memory = self.get_memory(thread_id)
        memory.findings.add_key_insight(insight)

        if self.config.auto_persist:
            memory.persist()

    def update_phase_status(
        self, phase_idx: int, completed: bool, thread_id: Optional[str] = None
    ) -> None:
        """Update phase completion status.

        Args:
            phase_idx: Phase index
            completed: Completion status
            thread_id: Thread ID
        """
        memory = self.get_memory(thread_id)
        memory.update_phase_status(phase_idx, completed)

    def should_inject_attention(self, state: Dict[str, Any]) -> bool:
        """Determine if attention reminder should be injected.

        Args:
            state: Current workflow state

        Returns:
            True if attention reminder should be injected
        """
        messages = state.get("messages", [])
        message_count = len(messages)

        if message_count == 0:
            return False

        if message_count % self.config.inject_attention_interval == 0:
            return True

        recent_messages = messages[-5:] if len(messages) > 5 else messages
        for msg in recent_messages:
            content = getattr(msg, "content", "") or ""
            if "error" in content.lower() or "失败" in content.lower():
                return True

        return False

    def get_error_context_for_llm(
        self, max_errors: int = 5, thread_id: Optional[str] = None
    ) -> str:
        """Get error context for LLM injection.

        Implements the "preserve error content" principle by
        keeping errors in context for model learning.

        Args:
            max_errors: Maximum errors to include
            thread_id: Thread ID

        Returns:
            Error context string
        """
        memory = self.get_memory(thread_id)
        errors = memory.progress.get_recent_errors(max_errors)

        if not errors:
            return ""

        lines = ["## ⚠️ 之前的错误尝试（请避免重复）\n"]

        for i, err in enumerate(errors, 1):
            lines.append(f"### 错误 {i}: {err.get('error_type', 'unknown')}")
            lines.append(f"- **消息**: {err.get('error', '')}")
            context = err.get("context", "")
            if len(context) > 200:
                context = context[:200] + "..."
            lines.append(f"- **上下文**: {context}")

            if err.get("recovery"):
                lines.append(f"- **恢复策略**: {err['recovery']}")
            lines.append("")

        return "\n".join(lines)

    def list_sessions(self) -> List[str]:
        """List all research session IDs.

        Returns:
            List of thread IDs with sessions
        """
        sessions = []
        if self.config.base_dir.exists():
            for path in self.config.base_dir.iterdir():
                if path.is_dir() and (path / "task_plan.md").exists():
                    sessions.append(path.name)
        return sessions

    def delete_session(self, thread_id: str) -> bool:
        """Delete a research session.

        Args:
            thread_id: Thread ID to delete

        Returns:
            True if deleted successfully
        """
        if thread_id in self._memories:
            memory = self._memories[thread_id]
            memory.clear()
            del self._memories[thread_id]

            if self._active_thread_id == thread_id:
                self._active_thread_id = None

            return True
        return False

    def export_session(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Export a session's state.

        Args:
            thread_id: Thread ID

        Returns:
            Session state dictionary or None
        """
        memory = self.get_memory(thread_id)
        return memory.export_state()

    def import_session(self, thread_id: str, state: Dict[str, Any]) -> FileBasedMemory:
        """Import a session's state.

        Args:
            thread_id: Thread ID
            state: Session state dictionary

        Returns:
            FileBasedMemory instance
        """
        memory = self.start_research(thread_id=thread_id)
        memory.import_state(state)
        return memory


def get_memory_manager() -> ResearchMemoryManager:
    """Get the global memory manager instance."""
    return ResearchMemoryManager.get_instance()
