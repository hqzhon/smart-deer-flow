"""Token Budget Manager for Parallel Execution

Implements proactive token budget management to prevent token overflow
in parallel execution scenarios. This addresses the root cause of token
limits being exceeded by managing token allocation before message construction.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import threading
import time

from src.utils.content_processor import ContentProcessor
from src.config.configuration import Configuration

logger = logging.getLogger(__name__)


class BudgetPriority(Enum):
    """Token budget priority levels"""

    CRITICAL = 1  # System messages, current task
    HIGH = 2  # Recent context, tool descriptions
    MEDIUM = 3  # Historical context
    LOW = 4  # Auxiliary information


@dataclass
class TokenBudget:
    """Represents a token budget allocation"""

    total_budget: int
    system_budget: int
    context_budget: int
    task_budget: int
    tool_budget: int
    safety_margin: int

    @property
    def available_budget(self) -> int:
        """Calculate remaining available budget"""
        allocated = (
            self.system_budget
            + self.context_budget
            + self.task_budget
            + self.tool_budget
        )
        return max(0, self.total_budget - allocated - self.safety_margin)


@dataclass
class TaskTokenAllocation:
    """Token allocation for a specific parallel task"""

    task_id: str
    allocated_tokens: int
    used_tokens: int
    priority: BudgetPriority
    timestamp: float

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.allocated_tokens - self.used_tokens)


class TokenBudgetManager:
    """Manages token budgets for parallel execution tasks"""

    def __init__(
        self,
        content_processor: ContentProcessor,
        config: Configuration,
        safety_margin_ratio: float = 0.15,
    ):
        self.content_processor = content_processor
        self.config = config
        self.safety_margin_ratio = safety_margin_ratio

        # Thread-safe allocation tracking
        self._lock = threading.Lock()
        self._task_allocations: Dict[str, TaskTokenAllocation] = {}
        self._global_budget_used = 0

        # Budget distribution ratios
        self.budget_ratios = {
            "system": 0.15,  # 15% for system messages
            "tools": 0.20,  # 20% for tool descriptions
            "context": 0.45,  # 45% for context (flexible)
            "task": 0.20,  # 20% for current task
        }

        logger.info("Token Budget Manager initialized")

    def calculate_budget_for_model(
        self, model_name: str, parallel_tasks: int = 1
    ) -> TokenBudget:
        """Calculate token budget for a specific model and parallel task count"""
        limits = self.content_processor.get_model_limits(model_name)

        # Use safe input limit as base
        base_budget = limits.safe_input_limit

        # Adjust for parallel execution - distribute budget among tasks
        if parallel_tasks > 1:
            # Reserve some budget for coordination overhead
            coordination_overhead = int(base_budget * 0.1)
            available_for_tasks = base_budget - coordination_overhead
            per_task_budget = available_for_tasks // parallel_tasks
        else:
            per_task_budget = base_budget

        # Calculate safety margin
        safety_margin = int(per_task_budget * self.safety_margin_ratio)
        effective_budget = per_task_budget - safety_margin

        # Distribute budget according to ratios
        system_budget = int(effective_budget * self.budget_ratios["system"])
        tool_budget = int(effective_budget * self.budget_ratios["tools"])
        task_budget = int(effective_budget * self.budget_ratios["task"])
        context_budget = effective_budget - system_budget - tool_budget - task_budget

        budget = TokenBudget(
            total_budget=per_task_budget,
            system_budget=system_budget,
            context_budget=context_budget,
            task_budget=task_budget,
            tool_budget=tool_budget,
            safety_margin=safety_margin,
        )

        logger.info(
            f"Calculated budget for {model_name} (parallel_tasks={parallel_tasks}): "
            f"total={budget.total_budget}, context={budget.context_budget}, "
            f"system={budget.system_budget}, tools={budget.tool_budget}, "
            f"task={budget.task_budget}, safety={budget.safety_margin}"
        )

        return budget

    def allocate_task_budget(
        self,
        task_id: str,
        model_name: str,
        priority: BudgetPriority = BudgetPriority.MEDIUM,
        parallel_tasks: int = 1,
    ) -> TaskTokenAllocation:
        """Allocate token budget for a specific task"""
        with self._lock:
            budget = self.calculate_budget_for_model(model_name, parallel_tasks)

            allocation = TaskTokenAllocation(
                task_id=task_id,
                allocated_tokens=budget.total_budget,
                used_tokens=0,
                priority=priority,
                timestamp=time.time(),
            )

            self._task_allocations[task_id] = allocation

            logger.info(
                f"Allocated {allocation.allocated_tokens} tokens for task {task_id}"
            )
            return allocation

    def get_context_budget(self, task_id: str) -> int:
        """Get available context budget for a task"""
        with self._lock:
            allocation = self._task_allocations.get(task_id)
            if not allocation:
                logger.warning(f"No allocation found for task {task_id}")
                return 0

            # Calculate context budget based on remaining tokens
            budget = self.calculate_budget_for_model(
                "deepseek-chat"
            )  # Use default for calculation
            return min(budget.context_budget, allocation.remaining_tokens)

    def estimate_message_tokens(self, messages: List[Any]) -> int:
        """Estimate total tokens for a list of messages"""
        total_tokens = 0
        for msg in messages:
            content = msg.content if hasattr(msg, "content") else str(msg)
            total_tokens += self.content_processor.count_tokens_accurate(
                content, "deepseek-chat"
            ).total_tokens
        return total_tokens

    def validate_message_budget(
        self, task_id: str, messages: List[Any], model_name: str
    ) -> Tuple[bool, int, int]:
        """Validate if messages fit within task budget

        Returns:
            Tuple of (is_valid, estimated_tokens, available_tokens)
        """
        with self._lock:
            allocation = self._task_allocations.get(task_id)
            if not allocation:
                logger.warning(f"No allocation found for task {task_id}")
                return False, 0, 0

            estimated_tokens = self.estimate_message_tokens(messages)
            available_tokens = allocation.remaining_tokens

            is_valid = estimated_tokens <= available_tokens

            if not is_valid:
                logger.warning(
                    f"Task {task_id} budget exceeded: {estimated_tokens} > {available_tokens}"
                )

            return is_valid, estimated_tokens, available_tokens

    def update_token_usage(self, task_id: str, used_tokens: int) -> bool:
        """Update token usage for a task"""
        with self._lock:
            allocation = self._task_allocations.get(task_id)
            if not allocation:
                logger.warning(f"No allocation found for task {task_id}")
                return False

            allocation.used_tokens += used_tokens

            if allocation.used_tokens > allocation.allocated_tokens:
                logger.warning(
                    f"Task {task_id} exceeded allocated budget: "
                    f"{allocation.used_tokens} > {allocation.allocated_tokens}"
                )
                return False

            logger.debug(
                f"Updated token usage for task {task_id}: {allocation.used_tokens}/{allocation.allocated_tokens}"
            )
            return True

    def release_task_budget(self, task_id: str) -> bool:
        """Release budget allocation for a completed task"""
        with self._lock:
            allocation = self._task_allocations.pop(task_id, None)
            if allocation:
                logger.info(
                    f"Released budget for task {task_id}: "
                    f"used {allocation.used_tokens}/{allocation.allocated_tokens} tokens"
                )
                return True
            return False

    def get_budget_stats(self) -> Dict[str, Any]:
        """Get current budget allocation statistics"""
        with self._lock:
            total_allocated = sum(
                alloc.allocated_tokens for alloc in self._task_allocations.values()
            )
            total_used = sum(
                alloc.used_tokens for alloc in self._task_allocations.values()
            )

            return {
                "active_tasks": len(self._task_allocations),
                "total_allocated": total_allocated,
                "total_used": total_used,
                "utilization_rate": (
                    total_used / total_allocated if total_allocated > 0 else 0
                ),
                "task_details": {
                    task_id: {
                        "allocated": alloc.allocated_tokens,
                        "used": alloc.used_tokens,
                        "remaining": alloc.remaining_tokens,
                        "priority": alloc.priority.name,
                    }
                    for task_id, alloc in self._task_allocations.items()
                },
            }

    def optimize_message_for_budget(
        self, task_id: str, messages: List[Any], model_name: str
    ) -> List[Any]:
        """Optimize messages to fit within task budget"""
        is_valid, estimated_tokens, available_tokens = self.validate_message_budget(
            task_id, messages, model_name
        )

        if is_valid:
            return messages

        logger.info(
            f"Optimizing messages for task {task_id}: {estimated_tokens} -> {available_tokens} tokens"
        )

        # Apply aggressive truncation to fit budget
        from src.graph.nodes import check_and_truncate_messages

        optimized_messages = check_and_truncate_messages(
            messages, model_name=model_name, max_tokens=available_tokens
        )

        # Verify the optimization worked
        final_tokens = self.estimate_message_tokens(optimized_messages)
        if final_tokens > available_tokens:
            logger.warning(
                f"Message optimization failed for task {task_id}: "
                f"{final_tokens} still exceeds {available_tokens}"
            )

            # Emergency truncation - keep only the most important messages
            if len(optimized_messages) > 1:
                # Keep first (system) and last (current task) messages only
                optimized_messages = [optimized_messages[0], optimized_messages[-1]]

                # If still too long, truncate the content of the last message
                if self.estimate_message_tokens(optimized_messages) > available_tokens:
                    last_msg = optimized_messages[-1]
                    if hasattr(last_msg, "content"):
                        # Calculate how much content we can keep
                        first_msg_tokens = self.estimate_message_tokens(
                            [optimized_messages[0]]
                        )
                        remaining_for_last = available_tokens - first_msg_tokens

                        if remaining_for_last > 100:  # Minimum viable content
                            # Truncate content to fit
                            char_limit = remaining_for_last * 3  # Rough estimation
                            last_msg.content = (
                                last_msg.content[:char_limit]
                                + "\n\n[Content truncated to fit token budget]"
                            )

        final_tokens = self.estimate_message_tokens(optimized_messages)
        logger.info(
            f"Message optimization complete for task {task_id}: {final_tokens} tokens"
        )

        return optimized_messages


# Global instance for shared use
_global_budget_manager: Optional[TokenBudgetManager] = None


def get_global_budget_manager() -> Optional[TokenBudgetManager]:
    """Get the global token budget manager instance"""
    return _global_budget_manager


def initialize_global_budget_manager(
    content_processor: ContentProcessor, config: Configuration
) -> TokenBudgetManager:
    """Initialize the global token budget manager"""
    global _global_budget_manager
    _global_budget_manager = TokenBudgetManager(content_processor, config)
    return _global_budget_manager


def create_budget_manager(
    content_processor: ContentProcessor, config: Configuration
) -> TokenBudgetManager:
    """Create a new token budget manager instance"""
    return TokenBudgetManager(content_processor, config)
