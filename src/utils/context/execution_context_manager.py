"""统一的执行上下文管理器

这个模块提供了统一的上下文处理逻辑，用于串行和并行执行路径。
解决了之前存在的上下文累积和不一致性问题。
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
    """上下文处理配置"""

    max_context_steps: int = 3
    max_step_content_length: int = 2000
    max_observations_length: int = 10000
    token_budget_ratio: float = 0.7
    enable_content_deduplication: bool = True
    enable_smart_truncation: bool = True


class TokenAllocation(NamedTuple):
    """Token分配结果"""

    allocated_tokens: int
    task_id: str
    model_name: str
    parallel_tasks: int


class ExecutionContextManager:
    """统一的执行上下文管理器

    提供串行和并行执行的统一上下文处理逻辑，包括：
    - 智能的步骤历史管理
    - 观察结果的压缩和去重
    - Token预算的动态管理
    - 消息历史的优化
    - 规划上下文优化
    - 高级观察结果管理
    """

    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()
        self._task_allocations = {}  # 跟踪任务的token分配
        self._observation_cache = {}  # 观察结果缓存

    def prepare_context_for_execution(
        self,
        completed_steps: List[Dict[str, Any]],
        current_step: Dict[str, Any],
        agent_type: str = "researcher",
    ) -> Tuple[List[Dict[str, Any]], str]:
        """为代理执行准备优化的上下文

        Args:
            completed_steps: 已完成的步骤列表
            current_step: 当前步骤
            agent_type: 代理类型

        Returns:
            优化后的步骤列表和格式化的上下文信息
        """
        # 1. 应用步骤数量限制
        limited_steps = self._limit_context_steps(completed_steps)

        # 2. 内容去重和截断
        if self.config.enable_content_deduplication:
            limited_steps = self._deduplicate_step_content(limited_steps)

        if self.config.enable_smart_truncation:
            limited_steps = self._truncate_step_content(limited_steps)

        # 3. 格式化上下文信息
        context_info = self._format_context_info(limited_steps, current_step)

        logger.info(
            f"Context prepared for {agent_type}: {len(limited_steps)} steps, "
            f"{len(context_info)} chars"
        )

        return limited_steps, context_info

    def manage_observations(
        self, observations: List[str], new_observation: str
    ) -> List[str]:
        """管理观察结果列表，防止无限累积

        Args:
            observations: 现有观察结果列表
            new_observation: 新的观察结果

        Returns:
            优化后的观察结果列表
        """
        # 添加新观察
        updated_observations = observations + [new_observation]

        # 计算总长度
        total_length = sum(len(obs) for obs in updated_observations)

        # 如果超过限制，进行压缩
        if total_length > self.config.max_observations_length:
            updated_observations = self._compress_observations(updated_observations)

        return updated_observations

    def optimize_messages(
        self, messages: List[BaseMessage], token_limit: Optional[int] = None
    ) -> List[BaseMessage]:
        """优化消息历史，防止token超限

        Args:
            messages: 消息列表
            token_limit: Token限制

        Returns:
            优化后的消息列表
        """
        if not token_limit:
            return messages

        # 计算当前token使用量
        current_tokens = sum(count_tokens(msg.content).total_tokens for msg in messages)

        if current_tokens <= token_limit:
            return messages

        # 使用内部的截断策略
        return self._simple_message_truncation(messages, token_limit)

    def _limit_context_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """限制上下文步骤数量"""
        if len(steps) <= self.config.max_context_steps:
            return steps

        # 保留最近的步骤
        return steps[-self.config.max_context_steps :]

    def _deduplicate_step_content(
        self, steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """去除步骤内容中的重复信息"""
        seen_content = set()
        deduplicated_steps = []

        for step in steps:
            execution_res = step.get("execution_res", "")

            # 简单的内容指纹
            content_hash = hash(execution_res[:200])  # 使用前200字符作为指纹

            if content_hash not in seen_content:
                seen_content.add(content_hash)
                deduplicated_steps.append(step)
            else:
                # 保留步骤结构但标记为重复
                modified_step = step.copy()
                modified_step["execution_res"] = "[重复内容已省略]"
                deduplicated_steps.append(modified_step)

        return deduplicated_steps

    def _truncate_step_content(
        self, steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """智能截断步骤内容"""
        truncated_steps = []

        for step in steps:
            execution_res = step.get("execution_res", "")

            if len(execution_res) > self.config.max_step_content_length:
                # 保留开头和结尾，中间用省略号
                keep_length = self.config.max_step_content_length // 2 - 50
                truncated_content = (
                    execution_res[:keep_length]
                    + "\n\n[... 内容已截断 ...]\n\n"
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
        """格式化上下文信息"""
        if not steps:
            return "没有已完成的步骤。"

        context_parts = []
        context_parts.append(f"已完成 {len(steps)} 个研究步骤：\n")

        for i, step in enumerate(steps, 1):
            step_info = f"{i}. {step.get('step', '未知步骤')}"
            execution_res = step.get("execution_res", "")

            if execution_res:
                # 限制每个步骤在上下文中的长度
                if len(execution_res) > 300:
                    execution_res = execution_res[:300] + "..."
                step_info += f"\n   结果: {execution_res}"

            context_parts.append(step_info)

        return "\n".join(context_parts)

    def _compress_observations(self, observations: List[str]) -> List[str]:
        """压缩观察结果列表"""
        if not observations:
            return observations

        # 保留最近的观察结果
        recent_count = max(1, len(observations) // 2)
        recent_observations = observations[-recent_count:]

        # 如果还是太长，进一步截断每个观察结果
        compressed = []
        remaining_budget = self.config.max_observations_length

        for obs in reversed(recent_observations):
            if remaining_budget <= 0:
                break

            if len(obs) > remaining_budget:
                # 截断这个观察结果
                truncated_obs = obs[: remaining_budget - 50] + "...[截断]"
                compressed.insert(0, truncated_obs)
                break
            else:
                compressed.insert(0, obs)
                remaining_budget -= len(obs)

        return compressed

    def _simple_message_truncation(
        self, messages: List[BaseMessage], token_limit: int
    ) -> List[BaseMessage]:
        """简单的消息截断策略"""
        if not messages:
            return messages

        # 保留系统消息和最近的用户/助手消息
        system_messages = [msg for msg in messages if msg.type == "system"]
        other_messages = [msg for msg in messages if msg.type != "system"]

        # 从最新消息开始保留
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

        # 记录分配
        self._task_allocations[task_id] = allocation

        logger.info(
            f"Token budget allocated: {allocated_tokens} tokens for task {task_id} "
            f"(model: {model_name}, parallel: {parallel_tasks})"
        )

        return allocation

    def release_task_resources(self, task_id: str) -> None:
        """释放任务资源

        Args:
            task_id: 任务ID
        """
        if task_id in self._task_allocations:
            del self._task_allocations[task_id]
            logger.debug(f"Released resources for task {task_id}")

    def _deduplicate_observations(self, observations: List[str]) -> List[str]:
        """去重观察结果"""
        seen_hashes = set()
        deduplicated = []

        for obs in observations:
            # 使用内容哈希进行去重
            content_hash = hashlib.md5(obs.encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduplicated.append(obs)

        return deduplicated

    def _score_observations(self, observations: List[str]) -> List[Tuple[str, float]]:
        """为观察结果评分"""
        scored = []

        for obs in observations:
            score = 0.0

            # 长度评分（适中长度得分更高）
            length = len(obs)
            if 100 <= length <= 2000:
                score += 1.0
            elif length > 2000:
                score += 0.5
            else:
                score += 0.3

            # 关键词评分
            keywords = ["结论", "发现", "重要", "关键", "建议", "总结", "分析"]
            for keyword in keywords:
                if keyword in obs:
                    score += 0.2

            # 结构化内容评分
            if any(marker in obs for marker in ["##", "**", "1.", "2.", "-"]):
                score += 0.3

            scored.append((obs, score))

        return scored

    def _compress_observation_content(
        self, observations: List[str], compression_ratio: float
    ) -> List[str]:
        """压缩观察结果内容"""
        compressed = []

        for obs in observations:
            target_length = int(len(obs) * compression_ratio)

            if len(obs) <= target_length:
                compressed.append(obs)
            else:
                # 保留开头和结尾，中间压缩
                keep_start = target_length // 2
                keep_end = target_length - keep_start - 20  # 为省略号留空间

                if keep_end > 0:
                    compressed_obs = (
                        obs[:keep_start] + "\n[...内容已压缩...]\n" + obs[-keep_end:]
                    )
                else:
                    compressed_obs = obs[:target_length] + "..."

                compressed.append(compressed_obs)

        return compressed

    def _optimize_planning_messages(
        self, messages: List[BaseMessage], plan_iterations: int
    ) -> List[BaseMessage]:
        """优化规划消息历史"""
        if len(messages) <= 8:
            return messages

        # 保留重要消息
        important_messages = []
        regular_messages = []

        for msg in messages:
            content = msg.content.lower()

            # 识别重要消息
            if any(
                keyword in content
                for keyword in ["用户请求", "任务目标", "错误", "失败", "重要", "关键"]
            ):
                important_messages.append(msg)
            else:
                regular_messages.append(msg)

        # 根据迭代次数决定保留的常规消息数量
        if plan_iterations <= 3:
            keep_regular = len(regular_messages) // 2
        else:
            keep_regular = len(regular_messages) // 3

        # 保留最近的常规消息
        kept_regular = regular_messages[-keep_regular:] if keep_regular > 0 else []

        # 合并并按时间排序
        optimized = important_messages + kept_regular

        # 简单的时间排序（假设消息按时间顺序）
        return optimized

    def evaluate_and_optimize_context_before_call_sync(
        self, llm_func, args: tuple, kwargs: dict, operation_name: str, context: str
    ) -> tuple:
        """同步版本的上下文评估和优化

        Args:
            llm_func: LLM函数
            args: 位置参数
            kwargs: 关键字参数
            operation_name: 操作名称
            context: 上下文信息

        Returns:
            优化后的(args, kwargs)元组
        """
        try:
            # 从参数中提取消息
            messages = self._extract_messages_from_args(args, kwargs)
            if not messages:
                return args, kwargs

            # 获取模型名称
            self._extract_model_name(llm_func, kwargs)

            # 应用消息优化
            optimized_messages = self.optimize_messages(messages)

            # 更新参数中的消息
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
        """异步版本的上下文评估和优化

        Args:
            llm_func: LLM函数
            args: 位置参数
            kwargs: 关键字参数
            operation_name: 操作名称
            context: 上下文信息

        Returns:
            优化后的(args, kwargs)元组
        """
        try:
            # 从参数中提取消息
            messages = self._extract_messages_from_args(args, kwargs)
            if not messages:
                return args, kwargs

            # 获取模型名称
            self._extract_model_name(llm_func, kwargs)

            # 应用消息优化
            optimized_messages = self.optimize_messages(messages)

            # 更新参数中的消息
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
        """从函数参数中提取消息列表"""
        messages = []

        # 检查位置参数中的消息
        for arg in args:
            if isinstance(arg, list) and arg and isinstance(arg[0], BaseMessage):
                messages = arg
                break
            elif isinstance(arg, BaseMessage):
                messages = [arg]
                break

        # 检查关键字参数中的消息
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
        """从LLM函数或参数中提取模型名称"""
        # 尝试从kwargs中获取
        if "model" in kwargs:
            return kwargs["model"]

        # 尝试从llm_func的属性中获取
        if hasattr(llm_func, "__self__"):
            llm_instance = llm_func.__self__
            if hasattr(llm_instance, "model_name"):
                return llm_instance.model_name
            elif hasattr(llm_instance, "model"):
                return llm_instance.model

        # 默认值
        return "deepseek-chat"

    def _update_args_with_messages(
        self, args: tuple, kwargs: dict, optimized_messages: List[BaseMessage]
    ) -> tuple:
        """用优化后的消息更新函数参数"""
        new_args = list(args)
        new_kwargs = kwargs.copy()

        # 更新位置参数中的消息
        for i, arg in enumerate(args):
            if isinstance(arg, list) and arg and isinstance(arg[0], BaseMessage):
                new_args[i] = optimized_messages
                return tuple(new_args), new_kwargs
            elif isinstance(arg, BaseMessage):
                new_args[i] = optimized_messages[0] if optimized_messages else arg
                return tuple(new_args), new_kwargs

        # 更新关键字参数中的消息
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
