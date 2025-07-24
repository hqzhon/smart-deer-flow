"""统一的反射管理器，用于控制反射执行次数和避免重复触发。"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReflectionStats:
    """反射统计信息"""

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
        """检查是否达到反射次数限制"""
        return self.total_reflections >= self.max_allowed

    @property
    def remaining_count(self) -> int:
        """剩余可用反射次数"""
        return max(0, self.max_allowed - self.total_reflections)


class ReflectionManager:
    """统一的反射管理器"""

    def __init__(self, max_reflections: Optional[int] = None, config: Optional[Any] = None):
        # 如果没有提供配置，尝试从统一配置系统加载
        if config is None and max_reflections is None:
            try:
                from src.config.config_loader import get_settings
                app_settings = get_settings()
                reflection_config = app_settings.get_reflection_config()
                self.max_reflections = reflection_config.max_reflection_loops
                self.config = reflection_config
            except ImportError:
                self.max_reflections = 1
                self.config = None
        else:
            self.max_reflections = max_reflections or 1
            self.config = config
            
        self._session_stats: Dict[str, ReflectionStats] = {}
        self._global_stats = ReflectionStats(max_allowed=self.max_reflections)

    def get_session_stats(self, session_id: str = "default") -> ReflectionStats:
        """获取会话的反射统计信息"""
        if session_id not in self._session_stats:
            self._session_stats[session_id] = ReflectionStats(
                max_allowed=self.max_reflections
            )
        return self._session_stats[session_id]

    def can_execute_reflection(
        self, session_id: str = "default", reflection_type: str = "general"
    ) -> Tuple[bool, str]:
        """检查是否可以执行反射

        Args:
            session_id: 会话ID
            reflection_type: 反射类型 (iteration, final, follow_up, general)

        Returns:
            (can_execute, reason)
        """
        stats = self.get_session_stats(session_id)

        if stats.is_limit_reached:
            return False, f"已达到最大反射次数限制 ({stats.max_allowed})"

        # 特殊规则：如果已经有final反射，不允许再次执行final反射
        if reflection_type == "final" and stats.final_reflections > 0:
            return False, "已执行过最终反射分析"

        # 检查follow-up反射的频率限制
        if reflection_type == "follow_up" and stats.follow_up_reflections >= 2:
            return False, "Follow-up反射次数已达到限制"

        return True, f"可以执行反射 (剩余 {stats.remaining_count} 次)"

    def record_reflection(
        self, session_id: str = "default", reflection_type: str = "general"
    ) -> bool:
        """记录反射执行

        Args:
            session_id: 会话ID
            reflection_type: 反射类型

        Returns:
            是否成功记录
        """
        can_execute, reason = self.can_execute_reflection(session_id, reflection_type)

        if not can_execute:
            logger.warning(f"反射执行被拒绝: {reason}")
            return False

        stats = self.get_session_stats(session_id)
        stats.total_reflections += 1

        # 按类型记录
        if reflection_type == "iteration":
            stats.iteration_reflections += 1
        elif reflection_type == "final":
            stats.final_reflections += 1
        elif reflection_type == "follow_up":
            stats.follow_up_reflections += 1

        logger.info(
            f"反射执行记录: 类型={reflection_type}, 总次数={stats.total_reflections}/{stats.max_allowed}, "
            f"会话={session_id}"
        )

        return True

    def reset_session(self, session_id: str = "default"):
        """重置会话的反射统计"""
        if session_id in self._session_stats:
            del self._session_stats[session_id]
        logger.info(f"已重置会话 {session_id} 的反射统计")

    def get_summary(self, session_id: str = "default") -> Dict[str, Any]:
        """获取反射执行摘要"""
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


# 全局反射管理器实例
_global_reflection_manager: Optional[ReflectionManager] = None


def get_reflection_manager(max_reflections: Optional[int] = None) -> ReflectionManager:
    """获取全局反射管理器实例"""
    global _global_reflection_manager

    if _global_reflection_manager is None:
        _global_reflection_manager = ReflectionManager(max_reflections)

    return _global_reflection_manager


def reset_reflection_manager():
    """重置全局反射管理器"""
    global _global_reflection_manager
    _global_reflection_manager = None
