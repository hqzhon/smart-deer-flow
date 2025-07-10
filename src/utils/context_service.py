import logging
import traceback
from typing import Any, Optional, Dict, List
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from src.graph.types import State

from src.utils.advanced_context_manager import AdvancedContextManager, CompressionStrategy
from src.utils.token_budget_manager import TokenBudgetManager
from src.utils.content_processor import ContentProcessor
from src.config.configuration import Configuration

logger = logging.getLogger(__name__)

@dataclass
class ContextOptimizationResult:
    """Result of context optimization operation."""
    optimized_content: str
    total_tokens: int
    max_tokens: int
    compression_ratio: float
    strategy_used: str
    success: bool
    error_message: Optional[str] = None

class ContextService:
    """Dedicated service for LLM context preparation and optimization.
    
    This service encapsulates all context management logic, providing a clean
    separation of concerns from the core node functionality.
    """
    
    def __init__(
        self, 
        configurable: Configuration, 
        content_processor: Optional[ContentProcessor] = None, 
        budget_manager: Optional[TokenBudgetManager] = None
    ):
        """Initialize the ContextService.
        
        Args:
            configurable: Configuration object containing model settings
            content_processor: Optional content processor for token management
            budget_manager: Optional budget manager for token allocation
        """
        self.configurable = configurable
        self.content_processor = content_processor or self._create_default_content_processor()
        self.budget_manager = budget_manager
        self._context_manager_cache: Optional[AdvancedContextManager] = None
        
    def _create_default_content_processor(self) -> ContentProcessor:
        """Create a default content processor if none provided."""
        try:
            return ContentProcessor(self.configurable.model_token_limits)
        except Exception as e:
            logger.warning(f"Failed to create content processor: {e}")
            # Return a minimal content processor with default limits
            from src.utils.content_processor import ModelTokenLimits
            default_limits = {
                "deepseek-chat": ModelTokenLimits(
                    input_limit=65536,
                    output_limit=8192,
                    context_window=65536,
                    safety_margin=0.8
                )
            }
            return ContentProcessor(default_limits)
    
    def _get_context_manager(self) -> AdvancedContextManager:
        """Get or create a cached AdvancedContextManager instance."""
        if self._context_manager_cache is None:
            self._context_manager_cache = AdvancedContextManager(
                self.configurable, 
                self.content_processor
            )
        return self._context_manager_cache
    
    def _determine_compression_strategy(
        self, 
        completed_steps: List[Any], 
        context_budget: Optional[int] = None,
        disable_context_parallel: bool = False
    ) -> CompressionStrategy:
        """Determine the optimal compression strategy based on context.
        
        Args:
            completed_steps: List of completed steps
            context_budget: Available token budget
            disable_context_parallel: Whether to disable context sharing
            
        Returns:
            Optimal compression strategy
        """
        if disable_context_parallel:
            return CompressionStrategy.NONE
            
        if context_budget and context_budget < 1000:
            logger.info(f"Using hierarchical compression due to limited budget: {context_budget}")
            return CompressionStrategy.HIERARCHICAL
            
        if len(completed_steps) > 3:
            return CompressionStrategy.HIERARCHICAL
        elif len(completed_steps) > 1:
            return CompressionStrategy.SLIDING_WINDOW
        else:
            return CompressionStrategy.ADAPTIVE
    
    def _apply_budget_constraints(
        self, 
        context_manager: AdvancedContextManager,
        model_name: str,
        context_budget: Optional[int]
    ) -> None:
        """Apply budget constraints to the context manager.
        
        Args:
            context_manager: The context manager to configure
            model_name: Name of the model being used
            context_budget: Available token budget
        """
        if not self.budget_manager or not context_budget:
            return
            
        try:
            limits = self.content_processor.get_model_limits(model_name)
            budget_ratio = min(0.6, context_budget / limits.safe_input_limit)
            context_manager.max_context_ratio = budget_ratio
            logger.info(f"Adjusted context ratio to {budget_ratio:.2f} based on budget constraints")
        except Exception as e:
            logger.warning(f"Failed to apply budget constraints: {e}")
    
    def _limit_completed_steps_for_parallel(
        self, 
        completed_steps: List[Any], 
        max_steps: int = 2
    ) -> List[Any]:
        """Limit completed steps for parallel execution to prevent token explosion.
        
        Args:
            completed_steps: List of completed steps
            max_steps: Maximum number of steps to keep
            
        Returns:
            Limited list of completed steps
        """
        if len(completed_steps) > max_steps:
            limited_steps = completed_steps[-max_steps:]
            logger.info(f"Limited context to last {max_steps} completed steps for parallel execution")
            return limited_steps
        elif len(completed_steps) > 1 and max_steps == 1:
            limited_steps = completed_steps[-1:]
            logger.info("Limited context to last 1 completed step for parallel execution")
            return limited_steps
        return completed_steps

    def get_optimized_llm_context(
        self,
        state: State,
        step: Any,
        current_plan: Any,
        agent_type: str,
        config: RunnableConfig,
        model_name: str,
        task_id: str,
        parallel_tasks: int
    ) -> ContextOptimizationResult:
        """Get optimized LLM context with comprehensive error handling and recovery.
        
        Args:
            state: Current workflow state
            step: Current step being executed
            current_plan: The current execution plan
            agent_type: Type of agent requesting context
            config: Runnable configuration
            model_name: Name of the LLM model
            task_id: Unique task identifier
            parallel_tasks: Number of parallel tasks
            
        Returns:
            ContextOptimizationResult containing optimized content and metadata
        """
        try:
            return self._optimize_context_internal(
                state, step, current_plan, agent_type, config,
                model_name, task_id, parallel_tasks
            )
        except Exception as e:
            logger.error(f"Context optimization failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Return minimal context as fallback
            minimal_content = self._create_minimal_context(step, state)
            return ContextOptimizationResult(
                optimized_content=minimal_content,
                total_tokens=len(minimal_content.split()),  # Rough estimate
                max_tokens=0,
                compression_ratio=0.0,
                strategy_used="minimal_fallback",
                success=False,
                error_message=str(e)
            )
    
    def _optimize_context_internal(
        self,
        state: State,
        step: Any,
        current_plan: Any,
        agent_type: str,
        config: RunnableConfig,
        model_name: str,
        task_id: str,
        parallel_tasks: int
    ) -> ContextOptimizationResult:
        # Extract completed steps
        completed_steps = [
            s for s in current_plan.steps if s.execution_res and s.title != step.title
        ]
        
        # Limit completed steps for parallel execution
        completed_steps = self._limit_completed_steps_for_parallel(completed_steps)
        
        # Get context budget if available
        context_budget = None
        if self.budget_manager:
            context_budget = self.budget_manager.get_context_budget(task_id)
            logger.info(f"Available context budget for task {task_id}: {context_budget} tokens")
        
        # Check if context sharing should be disabled
        disable_context_parallel = getattr(self.configurable, "disable_context_parallel", False)
        
        # Aggressively disable context sharing if needed
        if disable_context_parallel or len(completed_steps) > 0:
            logger.info("Context sharing disabled/limited for parallel execution to prevent token overflow")
            completed_steps = []  # Completely disable context sharing
        
        # Determine compression strategy
        strategy = self._determine_compression_strategy(
            completed_steps, context_budget, disable_context_parallel
        )
        
        # Create current task description
        current_task = f"Title: {step.title}\n\nDescription: {step.description}\n\nLocale: {state.get('locale', 'en-US')}"
        
        # Get context manager and apply budget constraints
        context_manager = self._get_context_manager()
        self._apply_budget_constraints(context_manager, model_name, context_budget)

        try:
            # Optimize context using the context manager
            context_window = context_manager.optimize_context_for_parallel(
                completed_steps=completed_steps,
                current_task=current_task,
                model_name=model_name,
                strategy=strategy,
            )
            
            # Format optimized context
            main_content = context_manager.format_context_window(context_window)
            
            # Get optimization statistics
            stats = context_manager.get_optimization_stats(context_window)
            logger.info(
                f"Context optimization stats: {stats['total_tokens']}/{stats['max_tokens']} tokens, "
                f"compression: {stats['compression_ratio']:.2%}, strategy: {stats['strategy_used']}"
            )
            
            return ContextOptimizationResult(
                optimized_content=main_content,
                total_tokens=stats['total_tokens'],
                max_tokens=stats['max_tokens'],
                compression_ratio=stats['compression_ratio'],
                strategy_used=stats['strategy_used'],
                success=True
            )

        except Exception as e:
            logger.error(f"Advanced context management failed: {e}, attempting recovery")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Attempt recovery with conservative settings
            return self._attempt_context_recovery(
                step, state, completed_steps, current_task, model_name
            )
    
    def _attempt_context_recovery(
        self,
        step: Any,
        state: State,
        completed_steps: List[Any],
        current_task: str,
        model_name: str
    ) -> ContextOptimizationResult:
        """Attempt to recover from context optimization failure.
        
        Args:
            step: Current step
            state: Current state
            completed_steps: List of completed steps
            current_task: Current task description
            model_name: Model name
            
        Returns:
            ContextOptimizationResult with recovery attempt
        """
        try:
            # Create a new context manager with very conservative settings
            recovery_manager = AdvancedContextManager(self.configurable, self.content_processor)
            recovery_manager.max_context_ratio = 0.3  # Very conservative
            recovery_manager.sliding_window_size = 2  # Minimal window
            recovery_manager.compression_threshold = 0.5  # Early compression
            
            # Force hierarchical compression for maximum token reduction
            strategy = CompressionStrategy.HIERARCHICAL
            
            # Limit completed steps more aggressively
            if len(completed_steps) > 2:
                completed_steps = completed_steps[-2:]
            
            logger.info("Retrying context optimization with conservative settings")
            
            context_window = recovery_manager.optimize_context_for_parallel(
                completed_steps=completed_steps,
                current_task=current_task,
                model_name=model_name,
                strategy=strategy,
            )
            
            main_content = recovery_manager.format_context_window(context_window)
            stats = recovery_manager.get_optimization_stats(context_window)
            
            logger.info(
                f"Recovery context stats: {stats['total_tokens']}/{stats['max_tokens']} tokens, "
                f"compression: {stats['compression_ratio']:.2%}, strategy: {stats['strategy_used']}"
            )
            
            return ContextOptimizationResult(
                optimized_content=main_content,
                total_tokens=stats['total_tokens'],
                max_tokens=stats['max_tokens'],
                compression_ratio=stats['compression_ratio'],
                strategy_used=f"recovery_{stats['strategy_used']}",
                success=True
            )
            
        except Exception as recovery_error:
            logger.error(f"Context management recovery also failed: {recovery_error}")
            logger.error(f"Recovery traceback: {traceback.format_exc()}")
            
            # Last resort: minimal context
            minimal_content = self._create_minimal_context(step, state)
            return ContextOptimizationResult(
                optimized_content=minimal_content,
                total_tokens=len(minimal_content.split()),  # Rough estimate
                max_tokens=0,
                compression_ratio=0.0,
                strategy_used="minimal_recovery",
                success=False,
                error_message=str(recovery_error)
            )
    
    def _create_minimal_context(self, step: Any, state: State) -> str:
        """Create minimal context as last resort fallback.
        
        Args:
            step: Current step
            state: Current state
            
        Returns:
            Minimal context string
        """
        return f"Title: {step.title}\n\nDescription: {step.description}\n\nLocale: {state.get('locale', 'en-US')}"
    
    def optimize_messages_for_budget(
        self,
        messages: List[BaseMessage],
        model_name: str,
        task_id: str,
        operation_name: str = "LLM Call",
        fallback_to_truncation: bool = True
    ) -> List[BaseMessage]:
        """Optimize messages for budget constraints with fallback options.
        
        Args:
            messages: List of messages to optimize
            model_name: Name of the model
            task_id: Task identifier
            operation_name: Name of the operation
            fallback_to_truncation: Whether to fallback to legacy truncation
            
        Returns:
            Optimized list of messages
        """
        if not self.budget_manager:
            if fallback_to_truncation:
                return self._apply_legacy_truncation(messages, model_name)
            return messages
            
        try:
            return self.budget_manager.optimize_message_for_budget(
                task_id=task_id,
                messages=messages,
                model_name=model_name
            )
        except Exception as e:
            logger.warning(f"Budget optimization failed: {e}")
            if fallback_to_truncation:
                logger.info("Falling back to legacy truncation")
                return self._apply_legacy_truncation(messages, model_name)
            return messages
    
    def _apply_legacy_truncation(
        self, 
        messages: List[BaseMessage], 
        model_name: str,
        max_messages: int = 10,
        aggressive_mode: bool = True
    ) -> List[BaseMessage]:
        """Apply legacy message truncation as fallback.
        
        Args:
            messages: List of messages to truncate
            model_name: Name of the model
            max_messages: Maximum number of messages
            aggressive_mode: Whether to use aggressive truncation
            
        Returns:
            Truncated list of messages
        """
        try:
            from src.graph.nodes import check_and_truncate_messages
            
            limits = self.content_processor.get_model_limits(model_name)
            max_tokens = int(limits.safe_input_limit * 0.6)  # Conservative limit
            
            return check_and_truncate_messages(
                messages,
                model_name=model_name,
                max_messages=max_messages,
                max_tokens=max_tokens,
                aggressive_mode=aggressive_mode
            )
        except Exception as e:
            logger.error(f"Legacy truncation failed: {e}")
            # Return original messages as last resort
            return messages
    
    def validate_message_budget(
        self,
        messages: List[BaseMessage],
        model_name: str,
        task_id: str
    ) -> tuple[bool, int, int]:
        """Validate message budget.
        
        Args:
            messages: List of messages to validate
            model_name: Name of the model
            task_id: Task identifier
            
        Returns:
            Tuple of (is_valid, estimated_tokens, available_tokens)
        """
        if not self.budget_manager:
            return True, 0, 0
            
        try:
            return self.budget_manager.validate_message_budget(
                task_id, messages, model_name
            )
        except Exception as e:
            logger.warning(f"Budget validation failed: {e}")
            return True, 0, 0
    
    def prepare_agent_messages(
        self,
        state: State,
        step: Any,
        current_plan: Any,
        agent_type: str,
        config: RunnableConfig,
        model_name: str,
        task_id: str,
        parallel_tasks: int,
        additional_messages: Optional[List[BaseMessage]] = None
    ) -> tuple[List[BaseMessage], ContextOptimizationResult]:
        """Complete message preparation pipeline for agents.
        
        This method encapsulates the entire message preparation process,
        including context optimization, budget management, and validation.
        
        Args:
            state: Current workflow state
            step: Current step being executed
            current_plan: The current execution plan
            agent_type: Type of agent requesting context
            config: Runnable configuration
            model_name: Name of the LLM model
            task_id: Unique task identifier
            parallel_tasks: Number of parallel tasks
            additional_messages: Optional additional messages to include
            
        Returns:
            Tuple of (optimized_messages, context_result)
        """
        try:
            # Step 1: Get optimized context
            context_result = self.get_optimized_llm_context(
                state=state,
                step=step,
                current_plan=current_plan,
                agent_type=agent_type,
                config=config,
                model_name=model_name,
                task_id=task_id,
                parallel_tasks=parallel_tasks
            )
            
            # Step 2: Create initial messages
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=context_result.optimized_content)]
            
            # Step 3: Add additional messages if provided
            if additional_messages:
                messages.extend(additional_messages)
            
            # Step 4: Apply budget optimization
            optimized_messages = self.optimize_messages_for_budget(
                messages=messages,
                model_name=model_name,
                task_id=task_id,
                operation_name=f"{agent_type} executor",
                fallback_to_truncation=True
            )
            
            # Step 5: Validate budget
            is_valid, estimated_tokens, available_tokens = self.validate_message_budget(
                messages=optimized_messages,
                model_name=model_name,
                task_id=task_id
            )
            
            # Step 6: Log results
            if available_tokens > 0:  # Only log if we have budget tracking
                if not is_valid:
                    logger.error(
                        f"Message budget validation failed for task {task_id}: "
                        f"{estimated_tokens} > {available_tokens}"
                    )
                else:
                    logger.info(
                        f"Message budget validated for task {task_id}: "
                        f"{estimated_tokens}/{available_tokens} tokens"
                    )
            
            return optimized_messages, context_result
            
        except Exception as e:
            logger.error(f"Message preparation failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Fallback to minimal context
            from langchain_core.messages import HumanMessage
            minimal_content = self._create_minimal_context(step, state)
            fallback_messages = [HumanMessage(content=minimal_content)]
            
            fallback_result = ContextOptimizationResult(
                optimized_content=minimal_content,
                total_tokens=len(minimal_content.split()),
                max_tokens=0,
                compression_ratio=0.0,
                strategy_used="fallback_preparation",
                success=False,
                error_message=str(e)
            )
            
            return fallback_messages, fallback_result
    
    def update_token_usage_tracking(
        self,
        task_id: str,
        input_messages: List[BaseMessage],
        result: Any,
        model_name: str
    ) -> None:
        """Update token usage tracking after LLM call.
        
        Args:
            task_id: Task identifier
            input_messages: Input messages sent to LLM
            result: Result from LLM call
            model_name: Name of the model used
        """
        if not self.budget_manager:
            return
            
        try:
            # Estimate input tokens
            input_tokens = self.budget_manager.estimate_message_tokens(input_messages)
            
            # Estimate output tokens
            if hasattr(result, "get") and "messages" in result:
                output_content = result["messages"][-1].content
            else:
                output_content = result.content if hasattr(result, "content") else str(result)
            
            output_tokens = self.content_processor.estimate_tokens(output_content)
            total_used = input_tokens + output_tokens
            
            # Update usage
            self.budget_manager.update_token_usage(task_id, total_used)
            
            # Log statistics
            stats = self.budget_manager.get_budget_stats()
            logger.info(
                f"Token usage updated for task {task_id}: {total_used} tokens. "
                f"Overall utilization: {stats['utilization_rate']:.2%}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to update token usage tracking: {e}")
            logger.warning(f"Full traceback: {traceback.format_exc()}")
    
    def release_task_resources(self, task_id: str) -> None:
        """Release resources allocated to a task.
        
        Args:
            task_id: Task identifier
        """
        if self.budget_manager:
            try:
                self.budget_manager.release_task_budget(task_id)
                logger.debug(f"Released resources for task {task_id}")
            except Exception as e:
                logger.warning(f"Failed to release task resources: {e}")
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get context service statistics.
        
        Returns:
            Dictionary containing service statistics
        """
        stats = {
            "service_initialized": True,
            "has_budget_manager": self.budget_manager is not None,
            "has_content_processor": self.content_processor is not None,
            "context_manager_cached": self._context_manager_cache is not None
        }
        
        if self._context_manager_cache:
            try:
                # Add context manager specific stats if available
                stats["context_manager_settings"] = {
                    "max_context_ratio": self._context_manager_cache.max_context_ratio,
                    "sliding_window_size": self._context_manager_cache.sliding_window_size,
                    "compression_threshold": self._context_manager_cache.compression_threshold
                }
            except Exception as e:
                logger.warning(f"Failed to get context manager stats: {e}")
        
        if self.budget_manager:
            try:
                budget_stats = self.budget_manager.get_budget_stats()
                stats["budget_statistics"] = budget_stats
            except Exception as e:
                logger.warning(f"Failed to get budget stats: {e}")
        
        return stats
