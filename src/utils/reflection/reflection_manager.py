"""Unified reflection manager for controlling reflection execution count and avoiding duplicate triggers."""

import logging
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReflectionStats:
    """Reflection statistics information"""

    total_reflections: int = 0
    iteration_reflections: int = 0
    final_reflections: int = 0
    follow_up_reflections: int = 0
    max_allowed: int = 1
    start_time: float = 0.0

    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()

    @property
    def is_limit_reached(self) -> bool:
        """Check if reflection count limit is reached"""
        return self.total_reflections >= self.max_allowed

    @property
    def remaining_count(self) -> int:
        """Remaining available reflection count"""
        return max(0, self.max_allowed - self.total_reflections)


class ReflectionManager:
    """Unified reflection manager"""

    def __init__(
        self, max_reflections: Optional[int] = None, config: Optional[Any] = None
    ):
        # If no config provided, try to load from unified config system
        if config is None and max_reflections is None:
            try:
                from src.config.config_loader import get_settings

                app_settings = get_settings()
                reflection_config = app_settings.get_reflection_config()
                self.max_reflections = reflection_config.max_loops  # 默认为2
                self.config = reflection_config
                logger.debug(
                    f"Successfully loaded reflection config: max_loops={reflection_config.max_loops}"
                )
            except Exception as e:
                logger.warning(f"Failed to load reflection config: {e}")
                self.max_reflections = 2  # Fallback to 2 if config is not available
                self.config = None
        else:
            self.max_reflections = max_reflections or 1
            self.config = config

        self._session_stats: Dict[str, ReflectionStats] = {}
        self._global_stats = ReflectionStats(max_allowed=self.max_reflections)

    def get_session_stats(self, session_id: str = "default") -> ReflectionStats:
        """Get reflection statistics for a session"""
        if session_id not in self._session_stats:
            self._session_stats[session_id] = ReflectionStats(
                max_allowed=self.max_reflections
            )
        return self._session_stats[session_id]

    def can_execute_reflection(
        self, session_id: str = "default", reflection_type: str = "general"
    ) -> Tuple[bool, str]:
        """Check if reflection can be executed

        Args:
            session_id: Session ID
            reflection_type: Reflection type (iteration, final, follow_up, general)

        Returns:
            (can_execute, reason)
        """
        stats = self.get_session_stats(session_id)

        if stats.is_limit_reached:
            return (
                False,
                f"Maximum reflection count limit reached ({stats.max_allowed})",
            )

        # Special rule: if final reflection already exists, don't allow another final reflection
        if reflection_type == "final" and stats.final_reflections > 0:
            return False, "Final reflection analysis already executed"

        # Check follow-up reflection frequency limit
        if reflection_type == "follow_up" and stats.follow_up_reflections >= 2:
            return False, "Follow-up reflection count limit reached"

        return True, f"Can execute reflection (remaining {stats.remaining_count} times)"

    def record_reflection(
        self, session_id: str = "default", reflection_type: str = "general"
    ) -> bool:
        """Record reflection execution

        Args:
            session_id: Session ID
            reflection_type: Reflection type

        Returns:
            Whether successfully recorded
        """
        can_execute, reason = self.can_execute_reflection(session_id, reflection_type)

        if not can_execute:
            logger.warning(f"Reflection execution rejected: {reason}")
            return False

        stats = self.get_session_stats(session_id)
        stats.total_reflections += 1

        # Record by type
        if reflection_type == "iteration":
            stats.iteration_reflections += 1
        elif reflection_type == "final":
            stats.final_reflections += 1
        elif reflection_type == "follow_up":
            stats.follow_up_reflections += 1

        logger.info(
            f"Reflection recorded - Session: {session_id}, "
            f"Type: {reflection_type}, "
            f"Total: {stats.total_reflections}/{stats.max_allowed}"
        )

        return True

    def reset_session(self, session_id: str = "default"):
        """Reset reflection statistics for a session"""
        if session_id in self._session_stats:
            del self._session_stats[session_id]
        logger.info(f"Reset reflection statistics for session {session_id}")

    def get_summary(self, session_id: str = "default") -> Dict[str, Any]:
        """Get reflection execution summary"""
        stats = self.get_session_stats(session_id)

        return {
            "session_id": session_id,
            "total_reflections": stats.total_reflections,
            "max_allowed": stats.max_allowed,
            "remaining": stats.remaining_count,
            "is_limit_reached": stats.is_limit_reached,
            "breakdown": {
                "iteration": stats.iteration_reflections,
                "final": stats.final_reflections,
                "follow_up": stats.follow_up_reflections,
            },
            "duration_seconds": time.time() - stats.start_time,
        }


# Global reflection manager instance
_global_reflection_manager: Optional[ReflectionManager] = None


def get_reflection_manager(max_reflections: Optional[int] = None) -> ReflectionManager:
    """Get global reflection manager instance"""
    global _global_reflection_manager

    if _global_reflection_manager is None:
        _global_reflection_manager = ReflectionManager(max_reflections)

    return _global_reflection_manager


def reset_reflection_manager():
    """Reset global reflection manager"""
    global _global_reflection_manager
    _global_reflection_manager = None
