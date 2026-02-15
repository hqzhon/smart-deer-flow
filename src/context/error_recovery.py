"""Error recovery with learning mechanism.

This module implements the "preserve error content" principle from Manus AI,
keeping failed attempts in context for model learning.

Key principles:
1. Keep failed attempts in context
2. Let model implicitly update internal beliefs
3. Reduce probability of repeating same errors
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import threading


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""

    error_type: str
    error_message: str
    context: str
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    recovery_attempted: bool = False
    recovery_strategy: Optional[str] = None
    recovery_successful: Optional[bool] = None
    learned_patterns: List[str] = field(default_factory=list)
    tool_name: Optional[str] = None
    args: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "context": self.context,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "recovery_attempted": self.recovery_attempted,
            "recovery_strategy": self.recovery_strategy,
            "recovery_successful": self.recovery_successful,
            "learned_patterns": self.learned_patterns,
            "tool_name": self.tool_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorRecord":
        """Create from dictionary."""
        return cls(
            error_type=data.get("error_type", "unknown"),
            error_message=data.get("error_message", ""),
            context=data.get("context", ""),
            severity=ErrorSeverity(data.get("severity", "medium")),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if data.get("timestamp")
                else datetime.now()
            ),
            recovery_attempted=data.get("recovery_attempted", False),
            recovery_strategy=data.get("recovery_strategy"),
            recovery_successful=data.get("recovery_successful"),
            learned_patterns=data.get("learned_patterns", []),
            tool_name=data.get("tool_name"),
        )


class ErrorRecoveryManager:
    """Error recovery manager - implements "preserve error content" principle.

    Keeps failed attempts in context so the model can learn from them
    and avoid repeating the same mistakes.
    """

    _instance: Optional["ErrorRecoveryManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ErrorRecoveryManager":
        """Singleton pattern for global error management."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the error recovery manager."""
        if self._initialized:
            return

        self.error_history: List[ErrorRecord] = []
        self.learned_patterns: Dict[str, List[str]] = {}
        self._max_history: int = 100
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "ErrorRecoveryManager":
        """Get the singleton instance."""
        if cls._instance is None:
            return cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def record_error(
        self,
        error_type: str,
        error_message: str,
        context: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        tool_name: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> ErrorRecord:
        """Record an error (preserve error content principle).

        Key principle: Keep failed attempts in context so the model
        can implicitly update its internal beliefs.

        Args:
            error_type: Type of error
            error_message: Error message
            context: Error context
            severity: Error severity
            tool_name: Tool that caused the error
            args: Tool arguments

        Returns:
            ErrorRecord instance
        """
        record = ErrorRecord(
            error_type=error_type,
            error_message=error_message,
            context=context,
            severity=severity,
            tool_name=tool_name,
            args=args,
        )

        if error_type in self.learned_patterns:
            record.learned_patterns = self.learned_patterns[error_type].copy()

        self.error_history.append(record)

        while len(self.error_history) > self._max_history:
            self.error_history.pop(0)

        return record

    def attempt_recovery(
        self, error: ErrorRecord, recovery_strategies: List[str]
    ) -> Optional[str]:
        """Attempt recovery from an error.

        Args:
            error: Error record
            recovery_strategies: List of strategies to try

        Returns:
            Successful strategy or None
        """
        error.recovery_attempted = True

        for strategy in recovery_strategies:
            error.recovery_strategy = strategy
            error.recovery_successful = True
            self._learn_from_recovery(error, strategy)
            return strategy

        error.recovery_successful = False
        return None

    def _learn_from_recovery(self, error: ErrorRecord, strategy: str) -> None:
        """Learn from a successful recovery.

        Args:
            error: Error record
            strategy: Successful strategy
        """
        if error.error_type not in self.learned_patterns:
            self.learned_patterns[error.error_type] = []

        if strategy not in self.learned_patterns[error.error_type]:
            self.learned_patterns[error.error_type].append(strategy)

    def get_error_context_for_llm(self, max_errors: int = 5) -> str:
        """Get error context for LLM injection.

        Keeps errors in context so the model can learn to avoid them.

        Args:
            max_errors: Maximum errors to include

        Returns:
            Error context string
        """
        if not self.error_history:
            return ""

        recent_errors = self.error_history[-max_errors:]

        lines = ["## ⚠️ 之前的错误尝试（请避免重复）\n"]

        for i, error in enumerate(recent_errors, 1):
            lines.append(f"### 错误 {i}: {error.error_type}")
            lines.append(f"- **消息**: {error.error_message}")

            context = error.context
            if len(context) > 200:
                context = context[:200] + "..."
            lines.append(f"- **上下文**: {context}")

            if error.recovery_successful and error.recovery_strategy:
                lines.append(f"- **恢复策略**: {error.recovery_strategy}")

            lines.append("")

        return "\n".join(lines)

    def get_recovery_suggestions(self, error_type: str) -> List[str]:
        """Get recovery suggestions for an error type.

        Args:
            error_type: Type of error

        Returns:
            List of suggested strategies
        """
        return self.learned_patterns.get(error_type, []).copy()

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics.

        Returns:
            Statistics dictionary
        """
        total = len(self.error_history)
        recovered = sum(1 for e in self.error_history if e.recovery_successful)

        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for error in self.error_history:
            by_type[error.error_type] = by_type.get(error.error_type, 0) + 1
            by_severity[error.severity.value] = (
                by_severity.get(error.severity.value, 0) + 1
            )

        return {
            "total_errors": total,
            "recovered_errors": recovered,
            "recovery_rate": recovered / total if total > 0 else 0,
            "by_type": by_type,
            "by_severity": by_severity,
            "learned_patterns_count": sum(
                len(v) for v in self.learned_patterns.values()
            ),
        }

    def clear_history(self) -> None:
        """Clear error history."""
        self.error_history.clear()

    def export_state(self) -> Dict[str, Any]:
        """Export state for persistence."""
        return {
            "error_history": [e.to_dict() for e in self.error_history],
            "learned_patterns": self.learned_patterns,
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """Import state from persistence."""
        self.error_history = [
            ErrorRecord.from_dict(e) for e in state.get("error_history", [])
        ]
        self.learned_patterns = state.get("learned_patterns", {})


class ToolExecutorWithErrorRecovery:
    """Tool executor with error recovery.

    Wraps tool execution with automatic error recording and
    recovery attempt functionality.
    """

    def __init__(
        self,
        recovery_manager: Optional[ErrorRecoveryManager] = None,
        max_retries: int = 3,
    ):
        self.recovery_manager = recovery_manager or ErrorRecoveryManager()
        self.max_retries = max_retries
        self._tool_registry: Dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable) -> None:
        """Register a tool function.

        Args:
            name: Tool name
            func: Tool function
        """
        self._tool_registry[name] = func

    async def execute_with_recovery(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        max_retries: Optional[int] = None,
    ) -> Any:
        """Execute a tool with recovery.

        Args:
            tool_name: Tool name
            tool_args: Tool arguments
            max_retries: Override max retries

        Returns:
            Tool result

        Raises:
            Exception: If all retries fail
        """
        max_retries = max_retries or self.max_retries
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                result = await self._execute_tool(tool_name, tool_args)
                return result

            except Exception as e:
                error_record = self.recovery_manager.record_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context=f"Tool: {tool_name}, Args: {tool_args}",
                    severity=ErrorSeverity.MEDIUM,
                    tool_name=tool_name,
                    args=tool_args,
                )

                suggestions = self.recovery_manager.get_recovery_suggestions(
                    type(e).__name__
                )

                if suggestions:
                    recovery = self.recovery_manager.attempt_recovery(
                        error_record, suggestions
                    )

                    if recovery:
                        tool_args = self._apply_recovery_strategy(tool_args, recovery)

                last_error = e

        raise last_error

    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool.

        Args:
            tool_name: Tool name
            args: Tool arguments

        Returns:
            Tool result
        """
        if tool_name not in self._tool_registry:
            raise ValueError(f"Unknown tool: {tool_name}")

        func = self._tool_registry[tool_name]

        if asyncio.iscoroutinefunction(func):
            return await func(**args)
        else:
            return func(**args)

    def _apply_recovery_strategy(
        self, args: Dict[str, Any], strategy: str
    ) -> Dict[str, Any]:
        """Apply a recovery strategy to arguments.

        Args:
            args: Current arguments
            strategy: Recovery strategy

        Returns:
            Modified arguments
        """
        args = args.copy()

        if strategy == "reduce_scope":
            if "max_results" in args:
                args["max_results"] = max(1, args["max_results"] // 2)
            if "limit" in args:
                args["limit"] = max(1, args["limit"] // 2)

        elif strategy == "add_filters":
            if "include_domains" not in args:
                args["include_domains"] = []

        elif strategy == "simplify_query":
            if "query" in args and len(args["query"]) > 100:
                args["query"] = args["query"][:100]

        elif strategy == "increase_timeout":
            if "timeout" in args:
                args["timeout"] = args.get("timeout", 30) * 2

        return args


def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager instance."""
    return ErrorRecoveryManager.get_instance()


def record_error(
    error_type: str,
    error_message: str,
    context: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
) -> ErrorRecord:
    """Convenience function to record an error.

    Args:
        error_type: Type of error
        error_message: Error message
        context: Error context
        severity: Error severity

    Returns:
        ErrorRecord instance
    """
    manager = get_error_recovery_manager()
    return manager.record_error(error_type, error_message, context, severity)


def get_error_context(max_errors: int = 5) -> str:
    """Convenience function to get error context.

    Args:
        max_errors: Maximum errors to include

    Returns:
        Error context string
    """
    manager = get_error_recovery_manager()
    return manager.get_error_context_for_llm(max_errors)
