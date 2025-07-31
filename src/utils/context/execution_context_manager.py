"""Unified Execution Context Manager

This module provides unified context processing logic for serial and parallel execution paths.
Solves previous context accumulation and inconsistency issues.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.messages import BaseMessage

from ..tokens.token_counter import count_tokens
from typing import NamedTuple
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Context processing configuration"""

    max_context_steps: int = 3
    max_step_content_length: int = 2000
    max_observations_length: int = 10000
    token_budget_ratio: float = 0.7
    enable_content_deduplication: bool = True
    enable_smart_truncation: bool = True


class TokenAllocation(NamedTuple):
    """Token allocation result"""

    allocated_tokens: int
    task_id: str
    model_name: str
    parallel_tasks: int


class ExecutionContextManager:
    """Unified Execution Context Manager

    Provides unified context processing logic for serial and parallel execution, including:
    - Intelligent step history management
    - Observation result compression and deduplication
    - Dynamic token budget management
    - Message history optimization
    - Planning context optimization
    - Advanced observation result management
    """

    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()
        self._task_allocations = {}  # Track task token allocations
        self._observation_cache = {}  # Observation result cache

    def prepare_context_for_execution(
        self,
        completed_steps: List[Dict[str, Any]],
        current_step: Dict[str, Any],
        agent_type: str = "researcher",
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Prepare optimized context for agent execution

        Args:
            completed_steps: List of completed steps
            current_step: Current step
            agent_type: Agent type

        Returns:
            Optimized step list and formatted context information
        """
        # 1. Apply step count limit
        limited_steps = self._limit_context_steps(completed_steps)

        # 2. Content deduplication and truncation
        if self.config.enable_content_deduplication:
            limited_steps = self._deduplicate_step_content(limited_steps)

        if self.config.enable_smart_truncation:
            limited_steps = self._truncate_step_content(limited_steps)

        # 3. Format context information
        context_info = self._format_context_info(limited_steps, current_step)

        logger.info(
            f"Context prepared for {agent_type}: {len(limited_steps)} steps, "
            f"{len(context_info)} chars"
        )

        return limited_steps, context_info

    def manage_observations(
        self, observations: List[str], new_observation: str
    ) -> List[str]:
        """Manage observation result list to prevent infinite accumulation

        Args:
            observations: Existing observation result list
            new_observation: New observation result

        Returns:
            Optimized observation result list
        """
        # Add new observation
        updated_observations = observations + [new_observation]

        # Calculate total length
        total_length = sum(len(obs) for obs in updated_observations)

        # If exceeds limit, perform compression
        if total_length > self.config.max_observations_length:
            updated_observations = self._compress_observations(updated_observations)

        return updated_observations

    def optimize_messages(
        self, messages: List[BaseMessage], token_limit: Optional[int] = None
    ) -> List[BaseMessage]:
        """Optimize message history to prevent token overflow

        Args:
            messages: Message list
            token_limit: Token limit

        Returns:
            Optimized message list
        """
        if not token_limit:
            return messages

        # Calculate current token usage
        current_tokens = sum(count_tokens(msg.content).total_tokens for msg in messages)

        if current_tokens <= token_limit:
            return messages

        # Use internal truncation strategy
        return self._simple_message_truncation(messages, token_limit)

    def _limit_context_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Limit context step count"""
        if len(steps) <= self.config.max_context_steps:
            return steps

        # Keep recent steps
        return steps[-self.config.max_context_steps :]

    def _deduplicate_step_content(
        self, steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate information from step content"""
        seen_content = set()
        deduplicated_steps = []

        for step in steps:
            execution_res = step.get("execution_res", "")

            # Simple content fingerprint
            content_hash = hash(
                execution_res[:200]
            )  # Use first 200 characters as fingerprint

            if content_hash not in seen_content:
                seen_content.add(content_hash)
                deduplicated_steps.append(step)
            else:
                # Keep step structure but mark as duplicate
                modified_step = step.copy()
                modified_step["execution_res"] = "[Duplicate content omitted]"
                deduplicated_steps.append(modified_step)

        return deduplicated_steps

    def _truncate_step_content(
        self, steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Intelligently truncate step content"""
        truncated_steps = []

        for step in steps:
            execution_res = step.get("execution_res", "")

            if len(execution_res) > self.config.max_step_content_length:
                # Keep beginning and end, use ellipsis in middle
                keep_length = self.config.max_step_content_length // 2 - 50
                truncated_content = (
                    execution_res[:keep_length]
                    + "\n\n[... Content truncated ...]\n\n"
                    + execution_res[-keep_length:]
                )

                modified_step = step.copy()
                modified_step["execution_res"] = truncated_content
                truncated_steps.append(modified_step)
            else:
                truncated_steps.append(step)

        return truncated_steps

    def _format_context_info(
        self, steps: List[Dict[str, Any]], current_step: Dict[str, Any]
    ) -> str:
        """Format context information"""
        if not steps:
            return "No completed steps."

        context_parts = []
        context_parts.append(f"Completed {len(steps)} research steps:\n")

        for i, step in enumerate(steps, 1):
            step_info = f"{i}. {step.get('step', 'Unknown step')}"
            execution_res = step.get("execution_res", "")

            if execution_res:
                # Limit the length of each step in context
                if len(execution_res) > 300:
                    execution_res = execution_res[:300] + "..."
                step_info += f"\n   Result: {execution_res}"

            context_parts.append(step_info)

        return "\n".join(context_parts)

    def _compress_observations(self, observations: List[str]) -> List[str]:
        """Compress observation list"""
        if not observations:
            return observations

        # Keep recent observations
        recent_count = max(1, len(observations) // 2)
        recent_observations = observations[-recent_count:]

        # If still too long, further truncate each observation
        compressed = []
        remaining_budget = self.config.max_observations_length

        for obs in reversed(recent_observations):
            if remaining_budget <= 0:
                break

            if len(obs) > remaining_budget:
                # Truncate this observation
                truncated_obs = obs[: remaining_budget - 50] + "...[truncated]"
                compressed.insert(0, truncated_obs)
                break
            else:
                compressed.insert(0, obs)
                remaining_budget -= len(obs)

        return compressed

    def _simple_message_truncation(
        self, messages: List[BaseMessage], token_limit: int
    ) -> List[BaseMessage]:
        """Simple message truncation strategy"""
        if not messages:
            return messages

        # Keep system messages and recent user/assistant messages
        system_messages = [msg for msg in messages if msg.type == "system"]
        other_messages = [msg for msg in messages if msg.type != "system"]

        # Keep starting from the latest messages
        truncated_messages = system_messages[:]
        current_tokens = sum(
            count_tokens(msg.content).total_tokens for msg in system_messages
        )

        for msg in reversed(other_messages):
            msg_tokens = count_tokens(msg.content).total_tokens
            if current_tokens + msg_tokens <= token_limit:
                truncated_messages.insert(
                    -len(system_messages) or len(truncated_messages), msg
                )
                current_tokens += msg_tokens
            else:
                break

        return truncated_messages

    def manage_observations_advanced(
        self, observations: List[str], optimization_level: str = "standard"
    ) -> List[str]:
        """高级观察结果管理，支持智能压缩和去重

        Args:
            observations: 观察结果列表
            optimization_level: 优化级别 ("minimal", "standard", "aggressive")

        Returns:
            优化后的观察结果列表
        """
        if not observations:
            return observations

        # 根据优化级别设置参数
        if optimization_level == "minimal":
            max_observations = len(observations)
            compression_ratio = 0.9
        elif optimization_level == "standard":
            max_observations = max(5, len(observations) // 2)
            compression_ratio = 0.7
        else:  # aggressive
            max_observations = max(3, len(observations) // 3)
            compression_ratio = 0.5

        # 1. 去重处理
        deduplicated = self._deduplicate_observations(observations)

        # 2. 重要性评分和排序
        scored_observations = self._score_observations(deduplicated)

        # 3. 选择最重要的观察结果
        selected = sorted(scored_observations, key=lambda x: x[1], reverse=True)[
            :max_observations
        ]
        selected_observations = [obs for obs, score in selected]

        # 4. 内容压缩
        compressed = self._compress_observation_content(
            selected_observations, compression_ratio
        )

        logger.info(
            f"Advanced observation management: {len(observations)} -> {len(compressed)} "
            f"(level: {optimization_level})"
        )

        return compressed

    def optimize_planning_context(
        self, messages: List[BaseMessage], observations: List[str], plan_iterations: int
    ) -> Tuple[List[BaseMessage], List[str]]:
        """规划阶段的上下文优化

        Args:
            messages: 消息历史
            observations: 观察结果
            plan_iterations: 规划迭代次数

        Returns:
            优化后的消息列表和观察结果列表
        """
        # 根据迭代次数调整优化强度
        if plan_iterations <= 2:
            optimization_level = "minimal"
        elif plan_iterations <= 5:
            optimization_level = "standard"
        else:
            optimization_level = "aggressive"

        # 优化消息历史
        optimized_messages = self._optimize_planning_messages(messages, plan_iterations)

        # 优化观察结果
        optimized_observations = self.manage_observations_advanced(
            observations, optimization_level
        )

        logger.info(
            f"Planning context optimization: messages {len(messages)} -> {len(optimized_messages)}, "
            f"observations {len(observations)} -> {len(optimized_observations)}"
        )

        return optimized_messages, optimized_observations

    def manage_token_budget(
        self, task_id: str, model_name: str, parallel_tasks: int = 1
    ) -> TokenAllocation:
        """集成的Token预算管理

        Args:
            task_id: 任务ID
            model_name: 模型名称
            parallel_tasks: 并行任务数量

        Returns:
            Token分配结果
        """
        # 基础token限制（根据模型类型）
        base_limits = {
            "deepseek-chat": 32000,
            "deepseek-reasoner": 64000,
            "gpt-4": 8000,
            "gpt-3.5-turbo": 4000,
        }

        base_limit = base_limits.get(model_name, 8000)

        # 根据并行任务数量和预算比例分配
        allocated_tokens = int(
            (base_limit * self.config.token_budget_ratio) / max(1, parallel_tasks)
        )

        allocation = TokenAllocation(
            allocated_tokens=allocated_tokens,
            task_id=task_id,
            model_name=model_name,
            parallel_tasks=parallel_tasks,
        )

        # Record allocation
        self._task_allocations[task_id] = allocation

        logger.info(
            f"Token budget allocated: {allocated_tokens} tokens for task {task_id} "
            f"(model: {model_name}, parallel: {parallel_tasks})"
        )

        return allocation

    def release_task_resources(self, task_id: str) -> None:
        """Release task resources

        Args:
            task_id: Task ID
        """
        if task_id in self._task_allocations:
            del self._task_allocations[task_id]
            logger.debug(f"Released resources for task {task_id}")

    def _deduplicate_observations(self, observations: List[str]) -> List[str]:
        """Deduplicate observations"""
        seen_hashes = set()
        deduplicated = []

        for obs in observations:
            # Use content hash for deduplication
            content_hash = hashlib.md5(str(obs).encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduplicated.append(obs)

        return deduplicated

    def _score_observations(self, observations: List[str]) -> List[Tuple[str, float]]:
        """Score observations"""
        scored = []

        for obs in observations:
            score = 0.0

            # Length scoring (moderate length gets higher score)
            length = len(obs)
            if 100 <= length <= 2000:
                score += 1.0
            elif length > 2000:
                score += 0.5
            else:
                score += 0.3

            # Keyword scoring
            keywords = [
                "conclusion",
                "finding",
                "important",
                "key",
                "recommendation",
                "summary",
                "analysis",
            ]
            for keyword in keywords:
                if keyword in obs:
                    score += 0.2

            # Structured content scoring
            if any(marker in obs for marker in ["##", "**", "1.", "2.", "-"]):
                score += 0.3

            scored.append((obs, score))

        return scored

    def _compress_observation_content(
        self, observations: List[str], compression_ratio: float
    ) -> List[str]:
        """Compress observation content"""
        compressed = []

        for obs in observations:
            target_length = int(len(obs) * compression_ratio)

            if len(obs) <= target_length:
                compressed.append(obs)
            else:
                # Keep beginning and end, compress middle
                keep_start = target_length // 2
                keep_end = target_length - keep_start - 20  # Leave space for ellipsis

                if keep_end > 0:
                    compressed_obs = (
                        obs[:keep_start]
                        + "\n[...content compressed...]\n"
                        + obs[-keep_end:]
                    )
                else:
                    compressed_obs = obs[:target_length] + "..."

                compressed.append(compressed_obs)

        return compressed

    def _optimize_planning_messages(
        self, messages: List[BaseMessage], plan_iterations: int
    ) -> List[BaseMessage]:
        """Optimize planning message history"""
        if len(messages) <= 8:
            return messages

        # Keep important messages
        important_messages = []
        regular_messages = []

        for msg in messages:
            content = msg.content.lower()

            # Identify important messages
            if any(
                keyword in content
                for keyword in [
                    "user request",
                    "task goal",
                    "error",
                    "failure",
                    "important",
                    "key",
                ]
            ):
                important_messages.append(msg)
            else:
                regular_messages.append(msg)

        # Decide number of regular messages to keep based on iterations
        if plan_iterations <= 3:
            keep_regular = len(regular_messages) // 2
        else:
            keep_regular = len(regular_messages) // 3

        # Keep recent regular messages
        kept_regular = regular_messages[-keep_regular:] if keep_regular > 0 else []

        # Merge and sort by time
        optimized = important_messages + kept_regular

        # Simple time sorting (assuming messages are in chronological order)
        return optimized

    def evaluate_and_optimize_context_before_call_sync(
        self, llm_func, args: tuple, kwargs: dict, operation_name: str, context: str
    ) -> tuple:
        """Synchronous version of context evaluation and optimization

        Args:
            llm_func: LLM function
            args: Positional arguments
            kwargs: Keyword arguments
            operation_name: Operation name
            context: Context information

        Returns:
            Optimized (args, kwargs) tuple
        """
        try:
            # Extract messages from arguments
            messages = self._extract_messages_from_args(args, kwargs)
            if not messages:
                return args, kwargs

            # Get model name
            self._extract_model_name(llm_func, kwargs)

            # Apply message optimization
            optimized_messages = self.optimize_messages(messages)

            # Update messages in arguments
            new_args, new_kwargs = self._update_args_with_messages(
                args, kwargs, optimized_messages
            )

            logger.debug(
                f"Context optimized for {operation_name}: "
                f"{len(messages)} -> {len(optimized_messages)} messages"
            )

            return new_args, new_kwargs

        except Exception as e:
            logger.warning(
                f"Context optimization failed in sync call: {e}, "
                "proceeding with original arguments"
            )
            return args, kwargs

    async def evaluate_and_optimize_context_before_call(
        self, llm_func, args: tuple, kwargs: dict, operation_name: str, context: str
    ) -> tuple:
        """Asynchronous version of context evaluation and optimization

        Args:
            llm_func: LLM function
            args: Positional arguments
            kwargs: Keyword arguments
            operation_name: Operation name
            context: Context information

        Returns:
            Optimized (args, kwargs) tuple
        """
        try:
            # Extract messages from arguments
            messages = self._extract_messages_from_args(args, kwargs)
            if not messages:
                return args, kwargs

            # Get model name
            self._extract_model_name(llm_func, kwargs)

            # Apply message optimization
            optimized_messages = self.optimize_messages(messages)

            # Update messages in arguments
            new_args, new_kwargs = self._update_args_with_messages(
                args, kwargs, optimized_messages
            )

            logger.debug(
                f"Context optimized for {operation_name}: "
                f"{len(messages)} -> {len(optimized_messages)} messages"
            )

            return new_args, new_kwargs

        except Exception as e:
            logger.warning(
                f"Context optimization failed in async call: {e}, "
                "proceeding with original arguments"
            )
            return args, kwargs

    def _extract_messages_from_args(
        self, args: tuple, kwargs: dict
    ) -> List[BaseMessage]:
        """Extract message list from function arguments"""
        messages = []

        # Check messages in positional arguments
        for arg in args:
            if isinstance(arg, list) and arg and isinstance(arg[0], BaseMessage):
                messages = arg
                break
            elif isinstance(arg, BaseMessage):
                messages = [arg]
                break

        # Check messages in keyword arguments
        if not messages:
            for key in ["messages", "input", "prompt"]:
                if key in kwargs:
                    value = kwargs[key]
                    if (
                        isinstance(value, list)
                        and value
                        and isinstance(value[0], BaseMessage)
                    ):
                        messages = value
                        break
                    elif isinstance(value, BaseMessage):
                        messages = [value]
                        break

        return messages

    def _extract_model_name(self, llm_func, kwargs: dict) -> str:
        """Extract model name from LLM function or parameters"""
        # Try to get from kwargs
        if "model" in kwargs:
            return kwargs["model"]

        # Try to get from llm_func attributes
        if hasattr(llm_func, "__self__"):
            llm_instance = llm_func.__self__
            if hasattr(llm_instance, "model_name"):
                return llm_instance.model_name
            elif hasattr(llm_instance, "model"):
                return llm_instance.model

        # Default value
        return "deepseek-chat"

    def _update_args_with_messages(
        self, args: tuple, kwargs: dict, optimized_messages: List[BaseMessage]
    ) -> tuple:
        """Update function arguments with optimized messages"""
        new_args = list(args)
        new_kwargs = kwargs.copy()

        # Update messages in positional arguments
        for i, arg in enumerate(args):
            if isinstance(arg, list) and arg and isinstance(arg[0], BaseMessage):
                new_args[i] = optimized_messages
                return tuple(new_args), new_kwargs
            elif isinstance(arg, BaseMessage):
                new_args[i] = optimized_messages[0] if optimized_messages else arg
                return tuple(new_args), new_kwargs

        # Update messages in keyword arguments
        for key in ["messages", "input", "prompt"]:
            if key in kwargs:
                value = kwargs[key]
                if (
                    isinstance(value, list)
                    and value
                    and isinstance(value[0], BaseMessage)
                ):
                    new_kwargs[key] = optimized_messages
                    return tuple(new_args), new_kwargs
                elif isinstance(value, BaseMessage):
                    new_kwargs[key] = (
                        optimized_messages[0] if optimized_messages else value
                    )
                    return tuple(new_args), new_kwargs

        return tuple(new_args), new_kwargs
