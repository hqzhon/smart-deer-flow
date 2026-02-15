"""Context engineering metrics for monitoring.

This module provides metrics for monitoring context engineering
effectiveness, including KV cache hit rate, context utilization,
goal alignment, and error recovery rates.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class ContextMetrics:
    """Context engineering metrics.

    Tracks key indicators for context engineering effectiveness:
    - KV cache hit rate (most important single metric)
    - Context utilization
    - Goal alignment rate
    - Error recovery rate
    """

    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0

    total_tokens_used: int = 0
    effective_tokens: int = 0
    context_utilization: float = 0.0

    total_steps: int = 0
    goal_aligned_steps: int = 0
    goal_alignment_rate: float = 0.0

    total_errors: int = 0
    recovered_errors: int = 0
    error_recovery_rate: float = 0.0

    attention_reminders_injected: int = 0
    goal_drifts_detected: int = 0

    session_start: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def calculate_rates(self) -> None:
        """Calculate all rate metrics."""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            self.cache_hit_rate = self.cache_hits / total_cache_requests

        if self.total_tokens_used > 0:
            self.context_utilization = self.effective_tokens / self.total_tokens_used

        if self.total_steps > 0:
            self.goal_alignment_rate = self.goal_aligned_steps / self.total_steps

        if self.total_errors > 0:
            self.error_recovery_rate = self.recovered_errors / self.total_errors

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1
        self.last_updated = datetime.now()

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.cache_misses += 1
        self.last_updated = datetime.now()

    def record_token_usage(self, total: int, effective: Optional[int] = None) -> None:
        """Record token usage.

        Args:
            total: Total tokens used
            effective: Effective (useful) tokens
        """
        self.total_tokens_used += total
        if effective is not None:
            self.effective_tokens += effective
        else:
            self.effective_tokens += total
        self.last_updated = datetime.now()

    def record_step(self, aligned: bool = True) -> None:
        """Record a step execution.

        Args:
            aligned: Whether step was aligned with goal
        """
        self.total_steps += 1
        if aligned:
            self.goal_aligned_steps += 1
        self.last_updated = datetime.now()

    def record_error(self, recovered: bool = False) -> None:
        """Record an error.

        Args:
            recovered: Whether error was recovered
        """
        self.total_errors += 1
        if recovered:
            self.recovered_errors += 1
        self.last_updated = datetime.now()

    def record_attention_injection(self) -> None:
        """Record an attention reminder injection."""
        self.attention_reminders_injected += 1
        self.last_updated = datetime.now()

    def record_goal_drift(self) -> None:
        """Record a detected goal drift."""
        self.goal_drifts_detected += 1
        self.last_updated = datetime.now()

    def to_report(self) -> str:
        """Generate a metrics report."""
        self.calculate_rates()

        duration = datetime.now() - self.session_start
        duration_str = str(duration).split(".")[0]

        return f"""## 上下文工程指标报告

**会话时长**: {duration_str}

### KV 缓存效率
- 命中率: {self.cache_hit_rate:.2%}
- 命中次数: {self.cache_hits:,}
- 未命中次数: {self.cache_misses:,}

### 上下文利用
- 利用率: {self.context_utilization:.2%}
- 有效 Token: {self.effective_tokens:,}
- 总 Token: {self.total_tokens_used:,}

### 目标一致性
- 一致性率: {self.goal_alignment_rate:.2%}
- 目标一致步骤: {self.goal_aligned_steps}/{self.total_steps}

### 错误恢复
- 恢复率: {self.error_recovery_rate:.2%}
- 已恢复错误: {self.recovered_errors}/{self.total_errors}

### 注意力管理
- 注入提醒次数: {self.attention_reminders_injected}
- 检测到的目标偏离: {self.goal_drifts_detected}
"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        self.calculate_rates()
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "total_tokens_used": self.total_tokens_used,
            "effective_tokens": self.effective_tokens,
            "context_utilization": self.context_utilization,
            "total_steps": self.total_steps,
            "goal_aligned_steps": self.goal_aligned_steps,
            "goal_alignment_rate": self.goal_alignment_rate,
            "total_errors": self.total_errors,
            "recovered_errors": self.recovered_errors,
            "error_recovery_rate": self.error_recovery_rate,
            "attention_reminders_injected": self.attention_reminders_injected,
            "goal_drifts_detected": self.goal_drifts_detected,
            "session_start": self.session_start.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextMetrics":
        """Create from dictionary."""
        metrics = cls()
        metrics.cache_hits = data.get("cache_hits", 0)
        metrics.cache_misses = data.get("cache_misses", 0)
        metrics.cache_hit_rate = data.get("cache_hit_rate", 0.0)
        metrics.total_tokens_used = data.get("total_tokens_used", 0)
        metrics.effective_tokens = data.get("effective_tokens", 0)
        metrics.context_utilization = data.get("context_utilization", 0.0)
        metrics.total_steps = data.get("total_steps", 0)
        metrics.goal_aligned_steps = data.get("goal_aligned_steps", 0)
        metrics.goal_alignment_rate = data.get("goal_alignment_rate", 0.0)
        metrics.total_errors = data.get("total_errors", 0)
        metrics.recovered_errors = data.get("recovered_errors", 0)
        metrics.error_recovery_rate = data.get("error_recovery_rate", 0.0)
        metrics.attention_reminders_injected = data.get(
            "attention_reminders_injected", 0
        )
        metrics.goal_drifts_detected = data.get("goal_drifts_detected", 0)

        if data.get("session_start"):
            metrics.session_start = datetime.fromisoformat(data["session_start"])
        if data.get("last_updated"):
            metrics.last_updated = datetime.fromisoformat(data["last_updated"])

        return metrics

    def merge(self, other: "ContextMetrics") -> "ContextMetrics":
        """Merge with another metrics instance.

        Args:
            other: Another metrics instance

        Returns:
            New merged metrics instance
        """
        merged = ContextMetrics()
        merged.cache_hits = self.cache_hits + other.cache_hits
        merged.cache_misses = self.cache_misses + other.cache_misses
        merged.total_tokens_used = self.total_tokens_used + other.total_tokens_used
        merged.effective_tokens = self.effective_tokens + other.effective_tokens
        merged.total_steps = self.total_steps + other.total_steps
        merged.goal_aligned_steps = self.goal_aligned_steps + other.goal_aligned_steps
        merged.total_errors = self.total_errors + other.total_errors
        merged.recovered_errors = self.recovered_errors + other.recovered_errors
        merged.attention_reminders_injected = (
            self.attention_reminders_injected + other.attention_reminders_injected
        )
        merged.goal_drifts_detected = (
            self.goal_drifts_detected + other.goal_drifts_detected
        )
        merged.session_start = min(self.session_start, other.session_start)
        merged.last_updated = max(self.last_updated, other.last_updated)
        merged.calculate_rates()
        return merged


class MetricsCollector:
    """Collector for context engineering metrics.

    Provides a centralized way to collect and aggregate metrics
    across different components.
    """

    _instance: Optional["MetricsCollector"] = None

    def __new__(cls) -> "MetricsCollector":
        """Singleton pattern for global metrics collection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the metrics collector."""
        if self._initialized:
            return

        self._metrics: Dict[str, ContextMetrics] = {}
        self._global_metrics = ContextMetrics()
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        """Get the singleton instance."""
        if cls._instance is None:
            return cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance."""
        cls._instance = None

    def get_metrics(self, thread_id: str = "default") -> ContextMetrics:
        """Get metrics for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            ContextMetrics instance
        """
        if thread_id not in self._metrics:
            self._metrics[thread_id] = ContextMetrics()
        return self._metrics[thread_id]

    def record_cache_event(self, hit: bool, thread_id: str = "default") -> None:
        """Record a cache event.

        Args:
            hit: Whether it was a cache hit
            thread_id: Thread identifier
        """
        metrics = self.get_metrics(thread_id)
        if hit:
            metrics.record_cache_hit()
            self._global_metrics.record_cache_hit()
        else:
            metrics.record_cache_miss()
            self._global_metrics.record_cache_miss()

    def record_step(self, aligned: bool = True, thread_id: str = "default") -> None:
        """Record a step.

        Args:
            aligned: Whether step was goal-aligned
            thread_id: Thread identifier
        """
        metrics = self.get_metrics(thread_id)
        metrics.record_step(aligned)
        self._global_metrics.record_step(aligned)

    def get_global_metrics(self) -> ContextMetrics:
        """Get global aggregated metrics."""
        return self._global_metrics

    def get_all_metrics(self) -> Dict[str, ContextMetrics]:
        """Get all thread metrics."""
        return self._metrics.copy()

    def aggregate_metrics(self) -> ContextMetrics:
        """Aggregate all thread metrics.

        Returns:
            Aggregated metrics
        """
        result = ContextMetrics()
        for metrics in self._metrics.values():
            result = result.merge(metrics)
        return result

    def reset_thread_metrics(self, thread_id: str) -> None:
        """Reset metrics for a thread.

        Args:
            thread_id: Thread identifier
        """
        if thread_id in self._metrics:
            del self._metrics[thread_id]

    def reset_all_metrics(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._global_metrics = ContextMetrics()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return MetricsCollector.get_instance()


def record_cache_hit(thread_id: str = "default") -> None:
    """Convenience function to record a cache hit."""
    get_metrics_collector().record_cache_event(True, thread_id)


def record_cache_miss(thread_id: str = "default") -> None:
    """Convenience function to record a cache miss."""
    get_metrics_collector().record_cache_event(False, thread_id)
