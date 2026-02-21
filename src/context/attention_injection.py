# -*- coding: utf-8 -*-
"""
Attention Injection Module - Deep integration with workflow nodes.

Implements the "restate to manipulate attention" principle from Manus AI:
- Inject attention reminders at key workflow points
- Detect and warn about goal drift
- Maintain focus on original research objectives
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)


class InjectionPoint(str, Enum):
    """Points in the workflow where attention can be injected."""

    BEFORE_PLANNING = "before_planning"
    AFTER_PLANNING = "after_planning"
    BEFORE_RESEARCH = "before_research"
    AFTER_RESEARCH = "after_research"
    BEFORE_REFLECTION = "before_reflection"
    AFTER_REFLECTION = "after_reflection"
    BEFORE_REPORT = "before_report"
    ON_ERROR = "on_error"
    PERIODIC = "periodic"


@dataclass
class AttentionInjectionConfig:
    """Configuration for attention injection."""

    enabled: bool = True
    injection_interval: int = 5
    max_injections_per_session: int = 10
    inject_on_error: bool = True
    inject_before_critical_steps: bool = True
    detect_goal_drift: bool = True
    drift_threshold: float = 0.3


@dataclass
class InjectionRecord:
    """Record of an attention injection."""

    timestamp: datetime
    injection_point: InjectionPoint
    message_length: int
    thread_id: str
    was_injected: bool


class AttentionInjector:
    """Injects attention reminders at key workflow points.

    This class integrates with the workflow nodes to inject
    attention reminders that help maintain focus on the original
    research objectives.
    """

    def __init__(self, config: Optional[AttentionInjectionConfig] = None):
        self.config = config or AttentionInjectionConfig()
        self._injection_history: List[InjectionRecord] = []
        self._injection_counts: Dict[str, int] = {}
        self._last_injection_step: Dict[str, int] = {}

    def should_inject(
        self,
        thread_id: str,
        injection_point: InjectionPoint,
        current_step: int = 0,
        message_count: int = 0,
        has_error: bool = False,
    ) -> bool:
        """Determine if attention should be injected.

        Args:
            thread_id: Session/thread identifier
            injection_point: Where in the workflow
            current_step: Current step number
            message_count: Number of messages in context
            has_error: Whether there's an error condition

        Returns:
            True if attention should be injected
        """
        if not self.config.enabled:
            return False

        injection_count = self._injection_counts.get(thread_id, 0)
        if injection_count >= self.config.max_injections_per_session:
            logger.debug(f"Max injections reached for thread {thread_id}")
            return False

        if injection_point == InjectionPoint.ON_ERROR:
            return self.config.inject_on_error and has_error

        if injection_point in [
            InjectionPoint.BEFORE_PLANNING,
            InjectionPoint.BEFORE_REPORT,
        ]:
            return self.config.inject_before_critical_steps

        if injection_point == InjectionPoint.PERIODIC:
            last_step = self._last_injection_step.get(thread_id, 0)
            if current_step - last_step >= self.config.injection_interval:
                return True
            if (
                message_count > 0
                and message_count % self.config.injection_interval == 0
            ):
                return True

        return False

    def create_attention_message(
        self,
        thread_id: str,
        original_goal: str,
        current_focus: str,
        progress: tuple,
        injection_point: InjectionPoint,
        locale: str = "en-US",
    ) -> Optional[SystemMessage]:
        """Create an attention reminder message.

        Args:
            thread_id: Session identifier
            original_goal: The original research goal
            current_focus: Current focus area
            progress: Tuple of (completed, total)
            injection_point: Where in the workflow
            locale: Language locale

        Returns:
            SystemMessage with attention reminder, or None
        """
        if not self.should_inject(
            thread_id=thread_id,
            injection_point=injection_point,
        ):
            return None

        completed, total = progress

        if locale.startswith("zh"):
            reminder = self._create_chinese_reminder(
                original_goal, current_focus, completed, total, injection_point
            )
        else:
            reminder = self._create_english_reminder(
                original_goal, current_focus, completed, total, injection_point
            )

        self._record_injection(thread_id, injection_point, len(reminder))

        return SystemMessage(content=reminder)

    def _create_english_reminder(
        self,
        original_goal: str,
        current_focus: str,
        completed: int,
        total: int,
        injection_point: InjectionPoint,
    ) -> str:
        """Create English attention reminder."""
        progress_pct = f"{completed}/{total}" if total > 0 else "N/A"

        base = f"""## 🎯 Goal Reminder

**Original Goal**: {original_goal}

**Current Focus**: {current_focus}

**Progress**: {progress_pct} steps completed

---
*Please ensure all actions align with the original goal.*
"""

        if injection_point == InjectionPoint.ON_ERROR:
            base += "\n\n⚠️ **Note**: An error was encountered. Please refocus on the original goal while addressing the issue."
        elif injection_point == InjectionPoint.BEFORE_REPORT:
            base += "\n\n📝 **Note**: Preparing final report. Ensure all findings directly address the original goal."

        return base

    def _create_chinese_reminder(
        self,
        original_goal: str,
        current_focus: str,
        completed: int,
        total: int,
        injection_point: InjectionPoint,
    ) -> str:
        """Create Chinese attention reminder."""
        progress_pct = f"{completed}/{total}" if total > 0 else "未知"

        base = f"""## 🎯 目标提醒

**原始目标**: {original_goal}

**当前焦点**: {current_focus}

**进度**: {progress_pct} 步骤已完成

---
*请确保所有行动都与原始目标保持一致。*
"""

        if injection_point == InjectionPoint.ON_ERROR:
            base += (
                "\n\n⚠️ **注意**: 遇到错误。请在解决问题的同时，重新聚焦于原始目标。"
            )
        elif injection_point == InjectionPoint.BEFORE_REPORT:
            base += (
                "\n\n📝 **注意**: 正在准备最终报告。请确保所有发现都直接回应原始目标。"
            )

        return base

    def _record_injection(
        self,
        thread_id: str,
        injection_point: InjectionPoint,
        message_length: int,
    ) -> None:
        """Record an injection for tracking."""
        record = InjectionRecord(
            timestamp=datetime.now(),
            injection_point=injection_point,
            message_length=message_length,
            thread_id=thread_id,
            was_injected=True,
        )
        self._injection_history.append(record)

        self._injection_counts[thread_id] = self._injection_counts.get(thread_id, 0) + 1

    def detect_goal_drift(
        self,
        original_goal: str,
        current_action: str,
        threshold: Optional[float] = None,
    ) -> Optional[str]:
        """Detect potential goal drift.

        Args:
            original_goal: The original research goal
            current_action: Current action being performed
            threshold: Drift detection threshold (0.0-1.0)

        Returns:
            Warning message if drift detected, None otherwise
        """
        if not self.config.detect_goal_drift:
            return None

        threshold = threshold or self.config.drift_threshold

        goal_keywords = set(original_goal.lower().split())
        action_keywords = set(current_action.lower().split())

        goal_keywords = {w for w in goal_keywords if len(w) > 3}
        action_keywords = {w for w in action_keywords if len(w) > 3}

        if not goal_keywords:
            return None

        overlap = len(goal_keywords & action_keywords)
        relevance = overlap / len(goal_keywords)

        if relevance < threshold:
            return f"""⚠️ Potential Goal Drift Detected

**Original Goal**: {original_goal}
**Current Action**: {current_action}
**Relevance Score**: {relevance:.2f}

Please verify this action aligns with the research goal."""

        return None

    def get_injection_stats(self, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Get injection statistics."""
        if thread_id:
            return {
                "thread_id": thread_id,
                "injection_count": self._injection_counts.get(thread_id, 0),
                "last_injection_step": self._last_injection_step.get(thread_id, 0),
            }

        return {
            "total_injections": len(self._injection_history),
            "injections_by_thread": dict(self._injection_counts),
            "injections_by_point": {
                point.value: sum(
                    1 for r in self._injection_history if r.injection_point == point
                )
                for point in InjectionPoint
            },
        }

    def reset(self, thread_id: Optional[str] = None) -> None:
        """Reset injection tracking."""
        if thread_id:
            self._injection_counts.pop(thread_id, None)
            self._last_injection_step.pop(thread_id, None)
            self._injection_history = [
                r for r in self._injection_history if r.thread_id != thread_id
            ]
        else:
            self._injection_history.clear()
            self._injection_counts.clear()
            self._last_injection_step.clear()


def inject_attention_to_messages(
    messages: List[Any],
    thread_id: str,
    original_goal: str,
    current_focus: str,
    progress: tuple,
    injection_point: InjectionPoint,
    injector: Optional[AttentionInjector] = None,
    locale: str = "en-US",
) -> List[Any]:
    """Utility function to inject attention into message list.

    Args:
        messages: Current message list
        thread_id: Session identifier
        original_goal: Original research goal
        current_focus: Current focus area
        progress: Progress tuple (completed, total)
        injection_point: Where in the workflow
        injector: AttentionInjector instance (creates new if None)
        locale: Language locale

    Returns:
        Updated message list with attention reminder if injected
    """
    injector = injector or AttentionInjector()

    attention_msg = injector.create_attention_message(
        thread_id=thread_id,
        original_goal=original_goal,
        current_focus=current_focus,
        progress=progress,
        injection_point=injection_point,
        locale=locale,
    )

    if attention_msg:
        return messages + [attention_msg]

    return messages
