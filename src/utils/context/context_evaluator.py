"""Context State Evaluator for Dynamic LLM Call Optimization

Provides dynamic context evaluation and optimization before every LLM call.
Ensures optimal token usage and prevents context overflow errors.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .advanced_context_manager import (
    AdvancedContextManager,
    ContextPriority,
    CompressionStrategy,
)
from ..tokens.content_processor import ContentProcessor
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.config.configuration import Configuration
from ..common.structured_logging import get_logger

logger = get_logger(__name__)


class ContextEvaluationResult(Enum):
    """Context evaluation results"""

    OPTIMAL = "optimal"  # Context is within optimal range
    NEEDS_COMPRESSION = "needs_compression"  # Context needs compression
    NEEDS_TRUNCATION = "needs_truncation"  # Context needs truncation
    CRITICAL_OVERFLOW = "critical_overflow"  # Context severely exceeds limits
    REQUIRES_CHUNKING = "requires_chunking"  # Content needs to be chunked


@dataclass
class ContextEvaluationMetrics:
    """Metrics from context evaluation"""

    def __init__(self, current_tokens: int, max_tokens: int, utilization_ratio: float,
                 compression_needed: bool, recommended_strategy: CompressionStrategy,
                 estimated_savings: int, evaluation_result: ContextEvaluationResult,
                 message_count: int, avg_message_length: int, critical_content_ratio: float,
                 optimized_messages: Optional[List[BaseMessage]] = None):
        self.current_tokens = current_tokens
        self.max_tokens = max_tokens
        self.utilization_ratio = utilization_ratio
        self.compression_needed = compression_needed
        self.recommended_strategy = recommended_strategy
        self.estimated_savings = estimated_savings
        self.evaluation_result = evaluation_result
        self.message_count = message_count
        self.avg_message_length = avg_message_length
        self.critical_content_ratio = critical_content_ratio
        self.optimized_messages = optimized_messages  # Store optimized messages if any


class ContextStateEvaluator:
    """Dynamic context state evaluator for LLM calls"""

    def __init__(
        self,
        config: "Configuration",
        content_processor: Optional[ContentProcessor] = None,
        context_manager: Optional[AdvancedContextManager] = None,
    ):
        self.config = config
        self.content_processor = content_processor or ContentProcessor(
            config.model_token_limits
        )
        self.context_manager = context_manager or AdvancedContextManager(
            config, self.content_processor
        )

        # Evaluation thresholds
        self.optimal_threshold = 0.6  # 60% utilization is optimal
        self.compression_threshold = 0.75  # Start compression at 75%
        self.truncation_threshold = 0.85  # Start truncation at 85%
        self.critical_threshold = 0.95  # Critical overflow at 95%

        # Recursion protection
        self.max_recursion_depth = 3  # Maximum recursion depth for ADAPTIVE strategy
        self._current_recursion_depth = 0
        self._evaluation_cache = {}  # Cache to prevent duplicate evaluations

        # Performance tracking
        self.evaluation_history = []
        self.optimization_stats = {
            "total_evaluations": 0,
            "optimizations_applied": 0,
            "tokens_saved": 0,
            "compression_count": 0,
            "truncation_count": 0,
            "recursion_prevented": 0,
        }

        logger.info("Context State Evaluator initialized with recursion protection")

    def evaluate_context_before_llm_call(
        self, messages: List[BaseMessage], model_name: str, operation_context: str = ""
    ) -> ContextEvaluationMetrics:
        """Evaluate context state before LLM call

        Args:
            messages: List of messages to be sent to LLM
            model_name: Target model name
            operation_context: Context about the operation being performed

        Returns:
            ContextEvaluationMetrics with evaluation results and recommendations
        """
        start_time = time.time()
        self.optimization_stats["total_evaluations"] += 1

        # Create cache key for duplicate evaluation detection
        cache_key = self._create_evaluation_cache_key(messages, model_name, operation_context)
        
        # Check for recent duplicate evaluation
        if cache_key in self._evaluation_cache:
            cached_result = self._evaluation_cache[cache_key]
            if time.time() - cached_result['timestamp'] < 5.0:  # 5 second cache
                logger.debug(f"Using cached evaluation result for {model_name}")
                return cached_result['metrics']

        logger.info(f"Evaluating context for {model_name} call: {operation_context}")

        # Get model limits
        limits = self.content_processor.get_model_limits(model_name)
        max_tokens = int(limits.safe_input_limit)

        # Calculate current token usage
        current_tokens = self._calculate_message_tokens(messages, model_name)
        utilization_ratio = current_tokens / max_tokens

        # If current context severely exceeds limits, apply immediate optimization
        if utilization_ratio > 2.0:  # More than 200% of limit
            logger.warning(
                f"Context severely exceeds limits ({utilization_ratio:.1%}). "
                f"Applying immediate optimization before evaluation."
            )
            # Apply emergency truncation to bring context to manageable size
            messages = self._apply_emergency_truncation_to_messages(messages, model_name)
            current_tokens = self._calculate_message_tokens(messages, model_name)
            utilization_ratio = current_tokens / max_tokens
            logger.info(
                f"Emergency optimization applied: reduced to {current_tokens} tokens "
                f"({utilization_ratio:.1%})"
            )

        # Analyze message characteristics
        message_count = len(messages)
        avg_message_length = current_tokens // max(message_count, 1)
        critical_content_ratio = self._calculate_critical_content_ratio(messages)

        # Determine evaluation result and recommended strategy
        evaluation_result, recommended_strategy = self._determine_optimization_strategy(
            utilization_ratio, current_tokens, max_tokens, messages
        )

        # Estimate potential savings
        estimated_savings = self._estimate_optimization_savings(
            messages, recommended_strategy, current_tokens, max_tokens
        )

        metrics = ContextEvaluationMetrics(
            current_tokens=current_tokens,
            max_tokens=max_tokens,
            utilization_ratio=utilization_ratio,
            compression_needed=(
                evaluation_result
                in [
                    ContextEvaluationResult.NEEDS_COMPRESSION,
                    ContextEvaluationResult.CRITICAL_OVERFLOW,
                ]
            ),
            recommended_strategy=recommended_strategy,
            estimated_savings=estimated_savings,
            evaluation_result=evaluation_result,
            message_count=message_count,
            avg_message_length=avg_message_length,
            critical_content_ratio=critical_content_ratio,
            optimized_messages=messages  # Include optimized messages if emergency truncation was applied
        )

        # Log evaluation results
        evaluation_time = time.time() - start_time
        logger.info(
            f"Context evaluation complete: {current_tokens}/{max_tokens} tokens "
            f"({utilization_ratio:.1%}), result: {evaluation_result.value}, "
            f"strategy: {recommended_strategy.value}, time: {evaluation_time:.3f}s"
        )

        # Store evaluation history with intelligent compression
        history_entry = {
            "timestamp": time.time(),
            "model_name": model_name,
            "operation_context": operation_context,
            "metrics": metrics,
            "evaluation_time": evaluation_time,
        }
        
        # Calculate token size of the history entry
        entry_tokens = self._calculate_history_entry_tokens(history_entry, model_name)
        logger.info(f"pre compression Calculated history entry tokens: {entry_tokens}")
        
        # If entry is too large, compress it instead of skipping
        if entry_tokens > max_tokens:
            original_tokens = entry_tokens
            history_entry = self._compress_history_entry(history_entry, max_tokens, model_name)
            entry_tokens = self._calculate_history_entry_tokens(history_entry, model_name)
            logger.info(
                f"Compressed evaluation history entry to {entry_tokens} tokens (was {original_tokens} tokens)"
            )
        
        # Always add the entry (compressed if necessary)
        self.evaluation_history.append(history_entry)
        
        # Keep only recent history and ensure total doesn't exceed token limit
        self._trim_evaluation_history_by_tokens(max_tokens, model_name)

        # Cache the evaluation result
        self._evaluation_cache[cache_key] = {
            'timestamp': time.time(),
            'metrics': metrics
        }
        
        # Clean old cache entries (keep only last 10 entries)
        if len(self._evaluation_cache) > 10:
            oldest_key = min(self._evaluation_cache.keys(), 
                           key=lambda k: self._evaluation_cache[k]['timestamp'])
            del self._evaluation_cache[oldest_key]

        return metrics

    def optimize_context_for_llm_call(
        self,
        messages: List[BaseMessage],
        model_name: str,
        strategy: CompressionStrategy,
        operation_context: str = "",
    ) -> Tuple[List[BaseMessage], Dict[str, Any]]:
        """Optimize context based on evaluation results

        Args:
            messages: Original messages
            model_name: Target model name
            strategy: Compression strategy to apply
            operation_context: Context about the operation

        Returns:
            Tuple of (optimized_messages, optimization_info)
        """
        start_time = time.time()
        self.optimization_stats["optimizations_applied"] += 1

        logger.info(f"Optimizing context with strategy: {strategy.value}")

        original_tokens = self._calculate_message_tokens(messages, model_name)

        # Apply optimization strategy
        if strategy == CompressionStrategy.NONE:
            optimized_messages = messages
            optimization_info = {"strategy": "none", "tokens_saved": 0}

        elif strategy == CompressionStrategy.TRUNCATE:
            optimized_messages = self._apply_message_truncation(messages, model_name)
            self.optimization_stats["truncation_count"] += 1

        elif strategy == CompressionStrategy.SLIDING_WINDOW:
            optimized_messages = self._apply_sliding_window_to_messages(
                messages, model_name
            )

        elif strategy == CompressionStrategy.HIERARCHICAL:
            optimized_messages = self._apply_hierarchical_message_selection(
                messages, model_name
            )

        elif strategy == CompressionStrategy.SUMMARIZE:
            optimized_messages = self._apply_message_summarization(messages, model_name)
            self.optimization_stats["compression_count"] += 1

        else:  # ADAPTIVE
            # Prevent infinite recursion
            if self._current_recursion_depth >= self.max_recursion_depth:
                logger.warning(
                    f"Maximum recursion depth ({self.max_recursion_depth}) reached for ADAPTIVE strategy. "
                    f"Falling back to TRUNCATE strategy."
                )
                self.optimization_stats["recursion_prevented"] += 1
                optimized_messages = self._apply_message_truncation(messages, model_name)
                self.optimization_stats["truncation_count"] += 1
            else:
                # Increment recursion depth
                self._current_recursion_depth += 1
                try:
                    # Re-evaluate and select best strategy
                    metrics = self.evaluate_context_before_llm_call(
                        messages, model_name, operation_context
                    )
                    # Prevent recursive ADAPTIVE calls
                    if metrics.recommended_strategy == CompressionStrategy.ADAPTIVE:
                        logger.warning("ADAPTIVE strategy recommended ADAPTIVE again. Using TRUNCATE instead.")
                        recommended_strategy = CompressionStrategy.TRUNCATE
                    else:
                        recommended_strategy = metrics.recommended_strategy
                    
                    return self.optimize_context_for_llm_call(
                        messages, model_name, recommended_strategy, operation_context
                    )
                finally:
                    # Always decrement recursion depth
                    self._current_recursion_depth -= 1

        # Calculate optimization results
        optimized_tokens = self._calculate_message_tokens(
            optimized_messages, model_name
        )
        tokens_saved = original_tokens - optimized_tokens
        self.optimization_stats["tokens_saved"] += tokens_saved

        optimization_info = {
            "strategy": strategy.value,
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": tokens_saved,
            "compression_ratio": (
                tokens_saved / original_tokens if original_tokens > 0 else 0
            ),
            "optimization_time": time.time() - start_time,
            "message_count_change": len(messages) - len(optimized_messages),
        }

        logger.info(
            f"Context optimization complete: {original_tokens} -> {optimized_tokens} tokens "
            f"({tokens_saved} saved, {optimization_info['compression_ratio']:.1%} reduction)"
        )

        return optimized_messages, optimization_info

    def _calculate_message_tokens(
        self, messages: List[BaseMessage], model_name: str
    ) -> int:
        """Calculate total tokens for message list"""
        total_tokens = 0
        for message in messages:
            content = message.content if hasattr(message, "content") else str(message)
            total_tokens += self.content_processor.count_tokens_accurate(
                content, model_name
            ).total_tokens
        return total_tokens

    def _calculate_critical_content_ratio(self, messages: List[BaseMessage]) -> float:
        """Calculate ratio of critical content in messages"""
        if not messages:
            return 0.0

        critical_count = 0
        for message in messages:
            if isinstance(message, SystemMessage):
                critical_count += 1
            elif hasattr(message, "content"):
                content = message.content.lower()
                if any(
                    keyword in content
                    for keyword in ["error", "critical", "important", "required"]
                ):
                    critical_count += 1

        return critical_count / len(messages)

    def _calculate_history_entry_tokens(self, entry: Dict[str, Any], model_name: str) -> int:
        """Calculate token count for a history entry"""
        total_tokens = 0
        
        # Convert entry to string representation for token counting
        entry_str = ""
        
        # Add basic fields
        entry_str += f"timestamp: {entry.get('timestamp', '')}"
        entry_str += f"model_name: {entry.get('model_name', '')}"
        entry_str += f"operation_context: {entry.get('operation_context', '')}"
        entry_str += f"evaluation_time: {entry.get('evaluation_time', '')}"
        
        # Add metrics information (simplified to avoid circular references)
        metrics = entry.get('metrics')
        if metrics:
            entry_str += f"current_tokens: {getattr(metrics, 'current_tokens', 0)}"
            entry_str += f"max_tokens: {getattr(metrics, 'max_tokens', 0)}"
            entry_str += f"utilization_ratio: {getattr(metrics, 'utilization_ratio', 0)}"
            entry_str += f"evaluation_result: {getattr(metrics, 'evaluation_result', '')}"
            entry_str += f"recommended_strategy: {getattr(metrics, 'recommended_strategy', '')}"
            entry_str += f"message_count: {getattr(metrics, 'message_count', 0)}"
            entry_str += f"avg_message_length: {getattr(metrics, 'avg_message_length', 0)}"
        
        # Calculate tokens for the string representation
        total_tokens = self.content_processor.count_tokens_accurate(
            entry_str, model_name
        ).total_tokens
        
        return total_tokens

    def _compress_history_entry(self, entry: Dict[str, Any], target_tokens: int, model_name: str) -> Dict[str, Any]:
        """Compress a history entry to fit within target token limit while preserving key information"""
        compressed_entry = entry.copy()
        original_tokens = self._calculate_history_entry_tokens(compressed_entry, model_name)
        logger.info(f"pre Compressed evaluation history entry tokens: {original_tokens}")
        
        # If already within target, return as is
        if original_tokens <= target_tokens:
            return compressed_entry
        
        # Progressive compression levels
        compression_levels = [
            self._apply_level1_compression,
            self._apply_level2_compression,
            self._apply_level3_compression,
            self._apply_level4_compression
        ]
        
        # Apply compression levels progressively until target is met
        for level, compress_func in enumerate(compression_levels, 1):
            compressed_entry = compress_func(entry)
            current_tokens = self._calculate_history_entry_tokens(compressed_entry, model_name)
            
            if current_tokens <= target_tokens:
                logger.debug(f"History entry compressed to level {level}: {original_tokens} -> {current_tokens} tokens")
                return compressed_entry
        
        # If still too large after all compression levels, apply emergency truncation
        logger.warning(f"Emergency truncation applied to history entry: {original_tokens} -> {current_tokens} tokens")
        return self._apply_emergency_truncation(compressed_entry, target_tokens, model_name)
    
    def _apply_level1_compression(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Level 1: Basic compression - truncate long operation context"""
        compressed = entry.copy()
        operation_context = entry.get('operation_context', '')
        if len(operation_context) > 100:
            compressed['operation_context'] = operation_context[:100] + '...'
        return compressed
    
    def _apply_level2_compression(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Level 2: Compress metrics by removing less critical fields"""
        compressed = self._apply_level1_compression(entry)
        metrics = entry.get('metrics')
        if metrics:
            compressed_metrics = {
                'current_tokens': getattr(metrics, 'current_tokens', 0),
                'max_tokens': getattr(metrics, 'max_tokens', 0),
                'utilization_ratio': round(getattr(metrics, 'utilization_ratio', 0), 3),
                'evaluation_result': str(getattr(metrics, 'evaluation_result', '')),
                'recommended_strategy': str(getattr(metrics, 'recommended_strategy', '')),
                'message_count': getattr(metrics, 'message_count', 0)
            }
            compressed['metrics'] = type('CompressedMetrics', (), compressed_metrics)()
        return compressed
    
    def _apply_level3_compression(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Level 3: Further reduce operation context and simplify metrics"""
        compressed = entry.copy()
        operation_context = entry.get('operation_context', '')
        if len(operation_context) > 50:
            compressed['operation_context'] = operation_context[:50] + '...'
        
        metrics = entry.get('metrics')
        if metrics:
            compressed_metrics = {
                'current_tokens': getattr(metrics, 'current_tokens', 0),
                'utilization_ratio': round(getattr(metrics, 'utilization_ratio', 0), 2),
                'evaluation_result': str(getattr(metrics, 'evaluation_result', ''))[:15],
                'recommended_strategy': str(getattr(metrics, 'recommended_strategy', ''))[:15]
            }
            compressed['metrics'] = type('Level3Metrics', (), compressed_metrics)()
        return compressed
    
    def _apply_level4_compression(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Level 4: Minimal information - only critical fields"""
        metrics = entry.get('metrics')
        minimal_entry = {
            'timestamp': entry.get('timestamp'),
            'model_name': entry.get('model_name'),
            'evaluation_time': entry.get('evaluation_time'),
            'operation_context': entry.get('operation_context', '')[:20] + '...' if len(entry.get('operation_context', '')) > 20 else entry.get('operation_context', ''),
        }
        
        if metrics:
            minimal_entry['metrics'] = type('MinimalMetrics', (), {
                'current_tokens': getattr(metrics, 'current_tokens', 0),
                'utilization_ratio': round(getattr(metrics, 'utilization_ratio', 0), 1),
                'result': str(getattr(metrics, 'evaluation_result', ''))[:10]
            })()
        
        return minimal_entry
    
    def _apply_emergency_truncation(self, entry: Dict[str, Any], target_tokens: int, model_name: str) -> Dict[str, Any]:
        """Emergency truncation when all compression levels fail"""
        # Keep only absolute essentials
        emergency_entry = {
            'timestamp': entry.get('timestamp'),
            'model_name': entry.get('model_name'),
            'tokens': getattr(entry.get('metrics'), 'current_tokens', 0) if entry.get('metrics') else 0
        }
        
        # Check if even this is too large (shouldn't happen, but safety check)
        current_tokens = self._calculate_history_entry_tokens(emergency_entry, model_name)
        if current_tokens > target_tokens:
            # Last resort: just timestamp and model
            emergency_entry = {
                'timestamp': entry.get('timestamp'),
                'model_name': entry.get('model_name')[:10] if entry.get('model_name') else 'unknown'
            }
        
        return emergency_entry

    def _apply_emergency_truncation_to_messages(self, messages: List[BaseMessage], model_name: str) -> List[BaseMessage]:
        """Apply emergency truncation to message list when context severely exceeds limits"""
        limits = self.content_processor.get_model_limits(model_name)
        # Target 60% of safe limit for emergency situations
        target_tokens = int(limits.safe_input_limit * 0.6)
        
        # Always keep system messages as they contain critical instructions
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        # Start with system messages
        result_messages = system_messages.copy()
        used_tokens = self._calculate_message_tokens(result_messages, model_name)
        
        # If system messages alone exceed target, truncate them too
        if used_tokens > target_tokens and system_messages:
            logger.warning("System messages exceed emergency target. Applying aggressive truncation.")
            # Keep only the most recent system message
            result_messages = [system_messages[-1]] if system_messages else []
            used_tokens = self._calculate_message_tokens(result_messages, model_name)
        
        # Add other messages from most recent, but be very conservative
        max_additional_messages = min(3, len(other_messages))  # Maximum 3 additional messages
        for message in reversed(other_messages[-max_additional_messages:]):
            message_tokens = self.content_processor.count_tokens_accurate(
                message.content if hasattr(message, "content") else str(message),
                model_name,
            ).total_tokens
            
            if used_tokens + message_tokens <= target_tokens:
                result_messages.insert(
                    -len([msg for msg in result_messages if isinstance(msg, SystemMessage)]) 
                    if any(isinstance(msg, SystemMessage) for msg in result_messages) else 0, 
                    message
                )
                used_tokens += message_tokens
            else:
                break
        
        logger.info(
            f"Emergency truncation: {len(messages)} -> {len(result_messages)} messages, "
            f"tokens reduced to {used_tokens} (target: {target_tokens})"
        )
        
        return result_messages

    def _trim_evaluation_history_by_tokens(self, max_tokens: int, model_name: str) -> None:
        """Trim evaluation history to stay within token limits using compression first"""
        if not self.evaluation_history:
            return
            
        # Calculate total tokens for current history
        total_tokens = 0
        for entry in self.evaluation_history:
            total_tokens += self._calculate_history_entry_tokens(entry, model_name)
        
        # If within limits, just apply count-based trimming
        if total_tokens <= max_tokens:
            if len(self.evaluation_history) > 100:
                self.evaluation_history = self.evaluation_history[-50:]
            return
        
        # First try to compress older entries before removing them
        compressed_count = 0
        for i in range(len(self.evaluation_history)):
            if total_tokens <= max_tokens:
                break
                
            entry = self.evaluation_history[i]
            original_tokens = self._calculate_history_entry_tokens(entry, model_name)
            
            # Try to compress the entry to half its size
            target_tokens = max(50, original_tokens // 2)  # Minimum 50 tokens
            compressed_entry = self._compress_history_entry(entry, target_tokens, model_name)
            compressed_tokens = self._calculate_history_entry_tokens(compressed_entry, model_name)
            
            if compressed_tokens < original_tokens:
                self.evaluation_history[i] = compressed_entry
                total_tokens = total_tokens - original_tokens + compressed_tokens
                compressed_count += 1
        
        # If still over limit, remove oldest entries
        removed_count = 0
        while self.evaluation_history and total_tokens > max_tokens:
            removed_entry = self.evaluation_history.pop(0)
            removed_tokens = self._calculate_history_entry_tokens(removed_entry, model_name)
            total_tokens -= removed_tokens
            removed_count += 1
            
        if compressed_count > 0 or removed_count > 0:
            logger.info(
                f"History management: compressed {compressed_count} entries, "
                f"removed {removed_count} entries. "
                f"Final: {len(self.evaluation_history)} entries ({total_tokens} tokens)"
            )

    def _determine_optimization_strategy(
        self,
        utilization_ratio: float,
        current_tokens: int,
        max_tokens: int,
        messages: List[BaseMessage],
    ) -> Tuple[ContextEvaluationResult, CompressionStrategy]:
        """Determine the best optimization strategy based on context state"""

        if utilization_ratio <= self.optimal_threshold:
            return ContextEvaluationResult.OPTIMAL, CompressionStrategy.NONE

        elif utilization_ratio <= self.compression_threshold:
            return ContextEvaluationResult.OPTIMAL, CompressionStrategy.NONE

        elif utilization_ratio <= self.truncation_threshold:
            # Choose between compression strategies
            if len(messages) > 10:  # Many messages - use sliding window
                return (
                    ContextEvaluationResult.NEEDS_COMPRESSION,
                    CompressionStrategy.SLIDING_WINDOW,
                )
            else:  # Few messages - use hierarchical
                return (
                    ContextEvaluationResult.NEEDS_COMPRESSION,
                    CompressionStrategy.HIERARCHICAL,
                )

        elif utilization_ratio <= self.critical_threshold:
            return (
                ContextEvaluationResult.NEEDS_TRUNCATION,
                CompressionStrategy.TRUNCATE,
            )

        else:
            # Critical overflow - need aggressive optimization
            if utilization_ratio > 1.5:  # Severe overflow
                return (
                    ContextEvaluationResult.REQUIRES_CHUNKING,
                    CompressionStrategy.SUMMARIZE,
                )
            else:
                return (
                    ContextEvaluationResult.CRITICAL_OVERFLOW,
                    CompressionStrategy.SUMMARIZE,
                )

    def _estimate_optimization_savings(
        self,
        messages: List[BaseMessage],
        strategy: CompressionStrategy,
        current_tokens: int,
        max_tokens: int,
    ) -> int:
        """Estimate potential token savings from optimization"""

        if strategy == CompressionStrategy.NONE:
            return 0

        elif strategy == CompressionStrategy.TRUNCATE:
            # Estimate 20-40% reduction
            return int(current_tokens * 0.3)

        elif strategy == CompressionStrategy.SLIDING_WINDOW:
            # Estimate based on message count reduction
            if len(messages) > 5:
                reduction_ratio = 1 - (5 / len(messages))
                return int(current_tokens * reduction_ratio)
            return int(current_tokens * 0.2)

        elif strategy == CompressionStrategy.HIERARCHICAL:
            # Estimate 15-30% reduction
            return int(current_tokens * 0.25)

        elif strategy == CompressionStrategy.SUMMARIZE:
            # Estimate 40-70% reduction
            return int(current_tokens * 0.55)

        else:
            return int(current_tokens * 0.3)  # Default estimate

    def _apply_message_truncation(
        self, messages: List[BaseMessage], model_name: str
    ) -> List[BaseMessage]:
        """Apply truncation to message list"""
        limits = self.content_processor.get_model_limits(model_name)
        target_tokens = int(limits.safe_input_limit * 0.8)  # Target 80% of safe limit

        # Keep system messages and recent messages
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]

        # Start with system messages
        result_messages = system_messages.copy()
        used_tokens = self._calculate_message_tokens(result_messages, model_name)

        # Add other messages from most recent
        for message in reversed(other_messages):
            message_tokens = self.content_processor.count_tokens_accurate(
                message.content if hasattr(message, "content") else str(message),
                model_name,
            ).total_tokens

            if used_tokens + message_tokens <= target_tokens:
                result_messages.insert(
                    -len(system_messages) if system_messages else 0, message
                )
                used_tokens += message_tokens
            else:
                break

        return result_messages

    def _create_evaluation_cache_key(self, messages: List[BaseMessage], model_name: str, operation_context: str) -> str:
        """Create a cache key for evaluation results"""
        # Create a simple hash based on message count, total length, and context
        message_count = len(messages)
        total_length = sum(len(msg.content if hasattr(msg, 'content') else str(msg)) for msg in messages)
        context_hash = hash(operation_context) % 10000  # Simple hash to avoid long keys
        
        return f"{model_name}_{message_count}_{total_length}_{context_hash}"

    def _apply_sliding_window_to_messages(
        self, messages: List[BaseMessage], model_name: str
    ) -> List[BaseMessage]:
        """Apply sliding window to message list"""
        # Keep system messages and last N messages
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]

        window_size = min(self.context_manager.sliding_window_size, len(other_messages))
        recent_messages = other_messages[-window_size:] if other_messages else []

        return system_messages + recent_messages

    def _apply_hierarchical_message_selection(
        self, messages: List[BaseMessage], model_name: str
    ) -> List[BaseMessage]:
        """Apply hierarchical selection to messages"""
        limits = self.content_processor.get_model_limits(model_name)
        target_tokens = int(limits.safe_input_limit * 0.8)

        # Prioritize messages
        prioritized_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                priority = ContextPriority.CRITICAL
            elif hasattr(message, "content") and any(
                keyword in message.content.lower()
                for keyword in ["error", "important", "critical"]
            ):
                priority = ContextPriority.HIGH
            else:
                priority = ContextPriority.MEDIUM

            prioritized_messages.append((message, priority))

        # Sort by priority and select
        prioritized_messages.sort(key=lambda x: x[1].value)

        selected_messages = []
        used_tokens = 0

        for message, priority in prioritized_messages:
            message_tokens = self.content_processor.count_tokens_accurate(
                message.content if hasattr(message, "content") else str(message),
                model_name,
            ).total_tokens

            if used_tokens + message_tokens <= target_tokens:
                selected_messages.append(message)
                used_tokens += message_tokens
            else:
                break

        return selected_messages

    def _apply_message_summarization(
        self, messages: List[BaseMessage], model_name: str
    ) -> List[BaseMessage]:
        """Apply summarization to messages"""
        # Keep system messages, summarize others
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]

        if len(other_messages) <= 2:
            return messages  # Too few to summarize

        # Combine content for summarization
        combined_content = "\n\n".join(
            [
                msg.content if hasattr(msg, "content") else str(msg)
                for msg in other_messages
            ]
        )

        # Create summarized message
        summary_content = f"[Summary of {len(other_messages)} previous messages]\n{combined_content[:500]}..."
        summary_message = HumanMessage(content=summary_content)

        return system_messages + [summary_message]

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        stats = self.optimization_stats.copy()

        if stats["total_evaluations"] > 0:
            stats["optimization_rate"] = (
                stats["optimizations_applied"] / stats["total_evaluations"]
            )
            stats["avg_tokens_saved"] = stats["tokens_saved"] / max(
                stats["optimizations_applied"], 1
            )
        else:
            stats["optimization_rate"] = 0.0
            stats["avg_tokens_saved"] = 0

        return stats

    def reset_stats(self):
        """Reset optimization statistics"""
        self.optimization_stats = {
            "total_evaluations": 0,
            "optimizations_applied": 0,
            "tokens_saved": 0,
            "compression_count": 0,
            "truncation_count": 0,
        }
        self.evaluation_history.clear()
        logger.info("Context evaluator statistics reset")


# Global context evaluator instance
_global_context_evaluator: Optional[ContextStateEvaluator] = None


def get_global_context_evaluator(
    config: Optional["Configuration"] = None,
) -> ContextStateEvaluator:
    """Get or create global context evaluator instance"""
    global _global_context_evaluator

    if _global_context_evaluator is None:
        if config is None:
            from src.config.config_loader import config_loader

            config = config_loader.create_configuration()

        _global_context_evaluator = ContextStateEvaluator(config)
        logger.info("Global context evaluator created")

    return _global_context_evaluator


def reset_global_context_evaluator():
    """Reset global context evaluator"""
    global _global_context_evaluator
    _global_context_evaluator = None
    logger.info("Global context evaluator reset")
