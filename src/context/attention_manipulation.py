"""Attention manipulation mechanism for goal consistency.

This module implements the "restate to manipulate attention" principle
from Manus AI, ensuring the model maintains focus on original goals
throughout long research tasks.

Key principles:
1. Restate goals at the end of context
2. Avoid "lost in the middle" problem
3. Reduce goal inconsistency
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import threading


@dataclass
class SubGoal:
    """A sub-goal within a larger goal."""

    goal: str
    priority: int = 0
    completed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def mark_completed(self) -> None:
        """Mark this sub-goal as completed."""
        self.completed = True
        self.completed_at = datetime.now().isoformat()


@dataclass
class GoalTracker:
    """Goal tracker - implements "restate to manipulate attention" principle.

    Tracks original goals and sub-goals, providing attention reminders
    that can be injected into context to maintain goal consistency.
    """

    original_goal: str
    sub_goals: List[SubGoal] = field(default_factory=list)
    attention_reminders: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_sub_goal(self, goal: str, priority: int = 0) -> None:
        """Add a sub-goal.

        Args:
            goal: Sub-goal description
            priority: Priority (higher = more important)
        """
        sub_goal = SubGoal(goal=goal, priority=priority)
        self.sub_goals.append(sub_goal)

    def complete_sub_goal(self, goal: str) -> bool:
        """Mark a sub-goal as completed.

        Args:
            goal: Sub-goal to complete

        Returns:
            True if found and completed
        """
        for sg in self.sub_goals:
            if sg.goal == goal and not sg.completed:
                sg.mark_completed()
                return True
        return False

    def get_current_focus(self) -> str:
        """Get the current focus (highest priority incomplete sub-goal).

        Returns:
            Current focus description
        """
        incomplete = [sg for sg in self.sub_goals if not sg.completed]
        if not incomplete:
            return "所有子目标已完成"

        sorted_goals = sorted(incomplete, key=lambda x: -x.priority)
        return sorted_goals[0].goal

    def get_progress(self) -> tuple[int, int]:
        """Get progress statistics.

        Returns:
            Tuple of (completed_count, total_count)
        """
        completed = sum(1 for sg in self.sub_goals if sg.completed)
        return completed, len(self.sub_goals)

    def generate_attention_reminder(self) -> str:
        """Generate attention reminder.

        Implements the "restate to manipulate attention" principle:
        - Restate goals at the end of context
        - Avoid "lost in the middle" problem
        - Reduce goal inconsistency

        Returns:
            Attention reminder string
        """
        current_focus = self.get_current_focus()
        completed_count, total_count = self.get_progress()

        reminder = f"""## 🎯 目标提醒

**原始目标**: {self.original_goal}

**当前焦点**: {current_focus}

**进度**: {completed_count}/{total_count} 子目标已完成

---
*请确保所有行动都与原始目标保持一致。*
"""
        self.attention_reminders.append(reminder)
        return reminder

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_goal": self.original_goal,
            "sub_goals": [
                {
                    "goal": sg.goal,
                    "priority": sg.priority,
                    "completed": sg.completed,
                    "created_at": sg.created_at,
                    "completed_at": sg.completed_at,
                }
                for sg in self.sub_goals
            ],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalTracker":
        """Create from dictionary."""
        tracker = cls(original_goal=data.get("original_goal", ""))
        tracker.created_at = data.get("created_at", datetime.now().isoformat())

        for sg_data in data.get("sub_goals", []):
            sub_goal = SubGoal(
                goal=sg_data.get("goal", ""),
                priority=sg_data.get("priority", 0),
                completed=sg_data.get("completed", False),
                created_at=sg_data.get("created_at", datetime.now().isoformat()),
                completed_at=sg_data.get("completed_at"),
            )
            tracker.sub_goals.append(sub_goal)

        return tracker


class AttentionManager:
    """Attention manager for goal consistency.

    Manages goal trackers across multiple threads/sessions and
    provides attention injection functionality.
    """

    _instance: Optional["AttentionManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AttentionManager":
        """Singleton pattern for global attention management."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the attention manager."""
        if self._initialized:
            return

        self.goal_trackers: Dict[str, GoalTracker] = {}
        self._injection_history: Dict[str, List[str]] = {}
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "AttentionManager":
        """Get the singleton instance."""
        if cls._instance is None:
            return cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def create_goal_tracker(self, thread_id: str, original_goal: str) -> GoalTracker:
        """Create a goal tracker for a thread.

        Args:
            thread_id: Thread identifier
            original_goal: Original goal description

        Returns:
            GoalTracker instance
        """
        tracker = GoalTracker(original_goal=original_goal)
        self.goal_trackers[thread_id] = tracker
        self._injection_history[thread_id] = []
        return tracker

    def get_goal_tracker(self, thread_id: str) -> Optional[GoalTracker]:
        """Get goal tracker for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            GoalTracker or None
        """
        return self.goal_trackers.get(thread_id)

    def add_sub_goal(self, thread_id: str, goal: str, priority: int = 0) -> bool:
        """Add a sub-goal to a tracker.

        Args:
            thread_id: Thread identifier
            goal: Sub-goal description
            priority: Priority level

        Returns:
            True if successful
        """
        tracker = self.goal_trackers.get(thread_id)
        if tracker:
            tracker.add_sub_goal(goal, priority)
            return True
        return False

    def complete_sub_goal(self, thread_id: str, goal: str) -> bool:
        """Complete a sub-goal.

        Args:
            thread_id: Thread identifier
            goal: Sub-goal to complete

        Returns:
            True if successful
        """
        tracker = self.goal_trackers.get(thread_id)
        if tracker:
            return tracker.complete_sub_goal(goal)
        return False

    def should_inject_attention(
        self, state: Dict[str, Any], interval: int = 10
    ) -> bool:
        """Determine if attention reminder should be injected.

        Args:
            state: Current workflow state
            interval: Injection interval (number of messages)

        Returns:
            True if should inject
        """
        messages = state.get("messages", [])
        message_count = len(messages)

        if message_count == 0:
            return False

        if message_count % interval == 0:
            return True

        recent_messages = messages[-5:] if len(messages) > 5 else messages
        for msg in recent_messages:
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str):
                if "error" in content.lower() or "失败" in content.lower():
                    return True

        return False

    def inject_attention_reminder(
        self, state: Dict[str, Any], force_inject: bool = False, interval: int = 10
    ) -> str:
        """Inject attention reminder into context.

        Args:
            state: Current workflow state
            force_inject: Force injection regardless of interval
            interval: Injection interval

        Returns:
            Attention reminder string (empty if not injecting)
        """
        thread_id = state.get("thread_id", "default")
        tracker = self.goal_trackers.get(thread_id)

        if not tracker:
            return ""

        should_inject = force_inject or self.should_inject_attention(state, interval)

        if should_inject:
            reminder = tracker.generate_attention_reminder()
            if thread_id not in self._injection_history:
                self._injection_history[thread_id] = []
            self._injection_history[thread_id].append(datetime.now().isoformat())
            return reminder

        return ""

    def get_injection_count(self, thread_id: str) -> int:
        """Get number of attention injections for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Number of injections
        """
        return len(self._injection_history.get(thread_id, []))

    def detect_goal_drift(self, thread_id: str, current_action: str) -> Optional[str]:
        """Detect potential goal drift.

        Args:
            thread_id: Thread identifier
            current_action: Current action description

        Returns:
            Warning message if drift detected, None otherwise
        """
        tracker = self.goal_trackers.get(thread_id)
        if not tracker:
            return None

        current_focus = tracker.get_current_focus()
        original_goal = tracker.original_goal.lower()
        action_lower = current_action.lower()

        keywords = original_goal.split()[:5]
        relevance_score = sum(1 for kw in keywords if kw in action_lower)

        if relevance_score == 0 and len(keywords) > 0:
            return f"""⚠️ 潜在目标偏离检测

**原始目标**: {tracker.original_goal}
**当前焦点**: {current_focus}
**当前行动**: {current_action}

请确认此行动是否与目标一致。"""

        return None

    def clear_tracker(self, thread_id: str) -> None:
        """Clear a goal tracker.

        Args:
            thread_id: Thread identifier
        """
        if thread_id in self.goal_trackers:
            del self.goal_trackers[thread_id]
        if thread_id in self._injection_history:
            del self._injection_history[thread_id]

    def export_tracker(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Export a tracker's state.

        Args:
            thread_id: Thread identifier

        Returns:
            Tracker state dictionary or None
        """
        tracker = self.goal_trackers.get(thread_id)
        if tracker:
            return tracker.to_dict()
        return None

    def import_tracker(self, thread_id: str, data: Dict[str, Any]) -> GoalTracker:
        """Import a tracker's state.

        Args:
            thread_id: Thread identifier
            data: Tracker state dictionary

        Returns:
            GoalTracker instance
        """
        tracker = GoalTracker.from_dict(data)
        self.goal_trackers[thread_id] = tracker
        self._injection_history[thread_id] = []
        return tracker


def get_attention_manager() -> AttentionManager:
    """Get the global attention manager instance."""
    return AttentionManager.get_instance()


def create_attention_reminder(
    original_goal: str, current_focus: str, progress: tuple[int, int]
) -> str:
    """Create an attention reminder without a tracker.

    Args:
        original_goal: Original goal description
        current_focus: Current focus
        progress: Tuple of (completed, total)

    Returns:
        Attention reminder string
    """
    completed, total = progress
    return f"""## 🎯 目标提醒

**原始目标**: {original_goal}

**当前焦点**: {current_focus}

**进度**: {completed}/{total} 子目标已完成

---
*请确保所有行动都与原始目标保持一致。*
"""
