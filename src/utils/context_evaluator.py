"""Context State Evaluator for Dynamic LLM Call Optimization

Provides dynamic context evaluation and optimization before every LLM call.
Ensures optimal token usage and prevents context overflow errors.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from src.utils.advanced_context_manager import AdvancedContextManager, CompressionStrategy, ContextPriority
from src.utils.content_processor import ContentProcessor
from src.config.configuration import Configuration
from src.utils.structured_logging import get_logger

logger = get_logger(__name__)

class ContextEvaluationResult(Enum):
    """Context evaluation results"""
    OPTIMAL = "optimal"                    # Context is within optimal range
    NEEDS_COMPRESSION = "needs_compression" # Context needs compression
    NEEDS_TRUNCATION = "needs_truncation"   # Context needs truncation
    CRITICAL_OVERFLOW = "critical_overflow" # Context severely exceeds limits
    REQUIRES_CHUNKING = "requires_chunking" # Content needs to be chunked

@dataclass
class ContextEvaluationMetrics:
    """Metrics from context evaluation"""
    current_tokens: int
    max_tokens: int
    utilization_ratio: float
    compression_needed: bool
    recommended_strategy: CompressionStrategy
    estimated_savings: int
    evaluation_result: ContextEvaluationResult
    message_count: int
    avg_message_length: int
    critical_content_ratio: float

class ContextStateEvaluator:
    """Dynamic context state evaluator for LLM calls"""
    
    def __init__(self, 
                 config: Configuration,
                 content_processor: Optional[ContentProcessor] = None,
                 context_manager: Optional[AdvancedContextManager] = None):
        self.config = config
        self.content_processor = content_processor or ContentProcessor(config.model_token_limits)
        self.context_manager = context_manager or AdvancedContextManager(config, self.content_processor)
        
        # Evaluation thresholds
        self.optimal_threshold = 0.6      # 60% utilization is optimal
        self.compression_threshold = 0.75  # Start compression at 75%
        self.truncation_threshold = 0.85   # Start truncation at 85%
        self.critical_threshold = 0.95     # Critical overflow at 95%
        
        # Performance tracking
        self.evaluation_history = []
        self.optimization_stats = {
            'total_evaluations': 0,
            'optimizations_applied': 0,
            'tokens_saved': 0,
            'compression_count': 0,
            'truncation_count': 0
        }
        
        logger.info("Context State Evaluator initialized")
    
    def evaluate_context_before_llm_call(self, 
                                        messages: List[BaseMessage],
                                        model_name: str,
                                        operation_context: str = "") -> ContextEvaluationMetrics:
        """Evaluate context state before LLM call
        
        Args:
            messages: List of messages to be sent to LLM
            model_name: Target model name
            operation_context: Context about the operation being performed
            
        Returns:
            ContextEvaluationMetrics with evaluation results and recommendations
        """
        start_time = time.time()
        self.optimization_stats['total_evaluations'] += 1
        
        logger.info(f"Evaluating context for {model_name} call: {operation_context}")
        
        # Get model limits
        limits = self.content_processor.get_model_limits(model_name)
        max_tokens = int(limits.safe_input_limit)
        
        # Calculate current token usage
        current_tokens = self._calculate_message_tokens(messages, model_name)
        utilization_ratio = current_tokens / max_tokens
        
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
            compression_needed=(evaluation_result in [ContextEvaluationResult.NEEDS_COMPRESSION, 
                                                    ContextEvaluationResult.CRITICAL_OVERFLOW]),
            recommended_strategy=recommended_strategy,
            estimated_savings=estimated_savings,
            evaluation_result=evaluation_result,
            message_count=message_count,
            avg_message_length=avg_message_length,
            critical_content_ratio=critical_content_ratio
        )
        
        # Log evaluation results
        evaluation_time = time.time() - start_time
        logger.info(f"Context evaluation complete: {current_tokens}/{max_tokens} tokens "
                   f"({utilization_ratio:.1%}), result: {evaluation_result.value}, "
                   f"strategy: {recommended_strategy.value}, time: {evaluation_time:.3f}s")
        
        # Store evaluation history
        self.evaluation_history.append({
            'timestamp': time.time(),
            'model_name': model_name,
            'operation_context': operation_context,
            'metrics': metrics,
            'evaluation_time': evaluation_time
        })
        
        # Keep only recent history
        if len(self.evaluation_history) > 100:
            self.evaluation_history = self.evaluation_history[-50:]
        
        return metrics
    
    def optimize_context_for_llm_call(self, 
                                     messages: List[BaseMessage],
                                     model_name: str,
                                     strategy: CompressionStrategy,
                                     operation_context: str = "") -> Tuple[List[BaseMessage], Dict[str, Any]]:
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
        self.optimization_stats['optimizations_applied'] += 1
        
        logger.info(f"Optimizing context with strategy: {strategy.value}")
        
        original_tokens = self._calculate_message_tokens(messages, model_name)
        
        # Apply optimization strategy
        if strategy == CompressionStrategy.NONE:
            optimized_messages = messages
            optimization_info = {'strategy': 'none', 'tokens_saved': 0}
        
        elif strategy == CompressionStrategy.TRUNCATE:
            optimized_messages = self._apply_message_truncation(messages, model_name)
            self.optimization_stats['truncation_count'] += 1
        
        elif strategy == CompressionStrategy.SLIDING_WINDOW:
            optimized_messages = self._apply_sliding_window_to_messages(messages, model_name)
        
        elif strategy == CompressionStrategy.HIERARCHICAL:
            optimized_messages = self._apply_hierarchical_message_selection(messages, model_name)
        
        elif strategy == CompressionStrategy.SUMMARIZE:
            optimized_messages = self._apply_message_summarization(messages, model_name)
            self.optimization_stats['compression_count'] += 1
        
        else:  # ADAPTIVE
            # Re-evaluate and select best strategy
            metrics = self.evaluate_context_before_llm_call(messages, model_name, operation_context)
            return self.optimize_context_for_llm_call(
                messages, model_name, metrics.recommended_strategy, operation_context
            )
        
        # Calculate optimization results
        optimized_tokens = self._calculate_message_tokens(optimized_messages, model_name)
        tokens_saved = original_tokens - optimized_tokens
        self.optimization_stats['tokens_saved'] += tokens_saved
        
        optimization_info = {
            'strategy': strategy.value,
            'original_tokens': original_tokens,
            'optimized_tokens': optimized_tokens,
            'tokens_saved': tokens_saved,
            'compression_ratio': tokens_saved / original_tokens if original_tokens > 0 else 0,
            'optimization_time': time.time() - start_time,
            'message_count_change': len(messages) - len(optimized_messages)
        }
        
        logger.info(f"Context optimization complete: {original_tokens} -> {optimized_tokens} tokens "
                   f"({tokens_saved} saved, {optimization_info['compression_ratio']:.1%} reduction)")
        
        return optimized_messages, optimization_info
    
    def _calculate_message_tokens(self, messages: List[BaseMessage], model_name: str) -> int:
        """Calculate total tokens for message list"""
        total_tokens = 0
        for message in messages:
            content = message.content if hasattr(message, 'content') else str(message)
            total_tokens += self.content_processor.estimate_tokens(content, model_name)
        return total_tokens
    
    def _calculate_critical_content_ratio(self, messages: List[BaseMessage]) -> float:
        """Calculate ratio of critical content in messages"""
        if not messages:
            return 0.0
        
        critical_count = 0
        for message in messages:
            if isinstance(message, SystemMessage):
                critical_count += 1
            elif hasattr(message, 'content'):
                content = message.content.lower()
                if any(keyword in content for keyword in ['error', 'critical', 'important', 'required']):
                    critical_count += 1
        
        return critical_count / len(messages)
    
    def _determine_optimization_strategy(self, 
                                       utilization_ratio: float,
                                       current_tokens: int,
                                       max_tokens: int,
                                       messages: List[BaseMessage]) -> Tuple[ContextEvaluationResult, CompressionStrategy]:
        """Determine the best optimization strategy based on context state"""
        
        if utilization_ratio <= self.optimal_threshold:
            return ContextEvaluationResult.OPTIMAL, CompressionStrategy.NONE
        
        elif utilization_ratio <= self.compression_threshold:
            return ContextEvaluationResult.OPTIMAL, CompressionStrategy.NONE
        
        elif utilization_ratio <= self.truncation_threshold:
            # Choose between compression strategies
            if len(messages) > 10:  # Many messages - use sliding window
                return ContextEvaluationResult.NEEDS_COMPRESSION, CompressionStrategy.SLIDING_WINDOW
            else:  # Few messages - use hierarchical
                return ContextEvaluationResult.NEEDS_COMPRESSION, CompressionStrategy.HIERARCHICAL
        
        elif utilization_ratio <= self.critical_threshold:
            return ContextEvaluationResult.NEEDS_TRUNCATION, CompressionStrategy.TRUNCATE
        
        else:
            # Critical overflow - need aggressive optimization
            if utilization_ratio > 1.5:  # Severe overflow
                return ContextEvaluationResult.REQUIRES_CHUNKING, CompressionStrategy.SUMMARIZE
            else:
                return ContextEvaluationResult.CRITICAL_OVERFLOW, CompressionStrategy.SUMMARIZE
    
    def _estimate_optimization_savings(self, 
                                     messages: List[BaseMessage],
                                     strategy: CompressionStrategy,
                                     current_tokens: int,
                                     max_tokens: int) -> int:
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
    
    def _apply_message_truncation(self, messages: List[BaseMessage], model_name: str) -> List[BaseMessage]:
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
            message_tokens = self.content_processor.estimate_tokens(
                message.content if hasattr(message, 'content') else str(message), model_name
            )
            
            if used_tokens + message_tokens <= target_tokens:
                result_messages.insert(-len(system_messages) if system_messages else 0, message)
                used_tokens += message_tokens
            else:
                break
        
        return result_messages
    
    def _apply_sliding_window_to_messages(self, messages: List[BaseMessage], model_name: str) -> List[BaseMessage]:
        """Apply sliding window to message list"""
        # Keep system messages and last N messages
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        window_size = min(self.context_manager.sliding_window_size, len(other_messages))
        recent_messages = other_messages[-window_size:] if other_messages else []
        
        return system_messages + recent_messages
    
    def _apply_hierarchical_message_selection(self, messages: List[BaseMessage], model_name: str) -> List[BaseMessage]:
        """Apply hierarchical selection to messages"""
        limits = self.content_processor.get_model_limits(model_name)
        target_tokens = int(limits.safe_input_limit * 0.8)
        
        # Prioritize messages
        prioritized_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                priority = ContextPriority.CRITICAL
            elif hasattr(message, 'content') and any(keyword in message.content.lower() 
                                                    for keyword in ['error', 'important', 'critical']):
                priority = ContextPriority.HIGH
            else:
                priority = ContextPriority.MEDIUM
            
            prioritized_messages.append((message, priority))
        
        # Sort by priority and select
        prioritized_messages.sort(key=lambda x: x[1].value)
        
        selected_messages = []
        used_tokens = 0
        
        for message, priority in prioritized_messages:
            message_tokens = self.content_processor.estimate_tokens(
                message.content if hasattr(message, 'content') else str(message), model_name
            )
            
            if used_tokens + message_tokens <= target_tokens:
                selected_messages.append(message)
                used_tokens += message_tokens
            else:
                break
        
        return selected_messages
    
    def _apply_message_summarization(self, messages: List[BaseMessage], model_name: str) -> List[BaseMessage]:
        """Apply summarization to messages"""
        # Keep system messages, summarize others
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        if len(other_messages) <= 2:
            return messages  # Too few to summarize
        
        # Combine content for summarization
        combined_content = "\n\n".join([
            msg.content if hasattr(msg, 'content') else str(msg) 
            for msg in other_messages
        ])
        
        # Create summarized message
        summary_content = f"[Summary of {len(other_messages)} previous messages]\n{combined_content[:500]}..."
        summary_message = HumanMessage(content=summary_content)
        
        return system_messages + [summary_message]
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        stats = self.optimization_stats.copy()
        
        if stats['total_evaluations'] > 0:
            stats['optimization_rate'] = stats['optimizations_applied'] / stats['total_evaluations']
            stats['avg_tokens_saved'] = stats['tokens_saved'] / max(stats['optimizations_applied'], 1)
        else:
            stats['optimization_rate'] = 0.0
            stats['avg_tokens_saved'] = 0
        
        return stats
    
    def reset_stats(self):
        """Reset optimization statistics"""
        self.optimization_stats = {
            'total_evaluations': 0,
            'optimizations_applied': 0,
            'tokens_saved': 0,
            'compression_count': 0,
            'truncation_count': 0
        }
        self.evaluation_history.clear()
        logger.info("Context evaluator statistics reset")


# Global context evaluator instance
_global_context_evaluator: Optional[ContextStateEvaluator] = None

def get_global_context_evaluator(config: Optional[Configuration] = None) -> ContextStateEvaluator:
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