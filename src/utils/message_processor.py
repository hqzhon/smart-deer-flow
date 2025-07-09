"""Message processor module for handling context evaluation during message assembly.

This module provides functionality to evaluate and optimize message context
before LLM calls, moving the context evaluation from the LLM call stage
to the message assembly stage for better performance and clearer separation of concerns.
"""

import logging
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


def prepare_messages_with_context_evaluation(messages: list,
                                           model_name: str = "deepseek-chat",
                                           operation_name: str = "LLM Call",
                                           context: str = "") -> list:
    """Prepare messages with context evaluation and optimization
    
    This function moves context evaluation from LLM call stage to message assembly stage
    for better performance and separation of concerns.
    
    Args:
        messages: List of messages to be sent to LLM
        model_name: Name of the LLM model
        operation_name: Name of the operation for logging
        context: Context information for logging
        
    Returns:
        Optimized list of messages ready for LLM call
    """
    if not messages:
        return messages
    
    try:
        from src.utils.context_evaluator import get_global_context_evaluator
        from src.utils.content_processor import ContentProcessor
        
        # Get context evaluator
        evaluator = get_global_context_evaluator()
        
        logger.debug(f"Evaluating context for {operation_name}: {len(messages)} messages, model: {model_name}")
        
        # Evaluate context state
        metrics = evaluator.evaluate_context_before_llm_call(
            messages, model_name, f"{operation_name} - {context}"
        )
        
        # Apply optimization if needed
        if metrics.compression_needed or metrics.evaluation_result.value in [
            'needs_compression', 'needs_truncation', 'critical_overflow', 'requires_chunking'
        ]:
            logger.info(f"Context optimization needed: {metrics.evaluation_result.value}, "
                       f"applying {metrics.recommended_strategy.value}")
            
            optimized_messages, optimization_info = evaluator.optimize_context_for_llm_call(
                messages, model_name, metrics.recommended_strategy, f"{operation_name} - {context}"
            )
            
            logger.info(f"Context optimized: {optimization_info['original_tokens']} -> "
                       f"{optimization_info['optimized_tokens']} tokens "
                       f"({optimization_info['tokens_saved']} saved)")
            
            # Final token validation with proper model limits from configuration
            try:
                processor = ContentProcessor()
                model_limits = processor.get_model_limits(model_name)
                
                # Calculate total tokens in optimized messages
                total_content = ""
                for msg in optimized_messages:
                    if hasattr(msg, 'content'):
                        total_content += str(msg.content) + " "
                
                estimated_tokens = processor.estimate_tokens(total_content, model_name)
                model_limit = model_limits.safe_input_limit  # Use configured limit instead of hardcoded
                
                if estimated_tokens > model_limit * 0.75:  # 75% threshold for final check (reduced from 90%)
                    logger.warning(f"Final token check: {estimated_tokens} tokens exceeds 75% of limit ({model_limit}) for model {model_name}")
                    # Emergency truncation with precise token counting
                    target_tokens = int(model_limit * 0.7)  # Target 70% of safe limit (more conservative)
                    
                    # Truncate messages from the end while preserving system messages
                    from langchain_core.messages import SystemMessage
                    system_messages = [msg for msg in optimized_messages if isinstance(msg, SystemMessage)]
                    other_messages = [msg for msg in optimized_messages if not isinstance(msg, SystemMessage)]
                    
                    # Calculate tokens for system messages using accurate counting
                    system_tokens = 0
                    for msg in system_messages:
                        if hasattr(msg, 'content'):
                            system_tokens += processor.estimate_tokens(str(msg.content), model_name)
                    
                    # Allocate remaining tokens to other messages
                    remaining_tokens = target_tokens - system_tokens
                    truncated_messages = system_messages.copy()
                    current_tokens = system_tokens
                    
                    for msg in other_messages:
                        if hasattr(msg, 'content'):
                            msg_tokens = processor.estimate_tokens(str(msg.content), model_name)
                            if current_tokens + msg_tokens <= remaining_tokens:
                                truncated_messages.append(msg)
                                current_tokens += msg_tokens
                            else:
                                # Precise token-based truncation for this message
                                available_tokens = remaining_tokens - current_tokens
                                if available_tokens > 50:  # Only if we have reasonable space
                                    # Use binary search for precise truncation
                                    content = str(msg.content)
                                    left, right = 0, len(content)
                                    best_content = ""
                                    
                                    while left <= right:
                                        mid = (left + right) // 2
                                        test_content = content[:mid] + "...[truncated]"
                                        test_tokens = processor.estimate_tokens(test_content, model_name)
                                        
                                        if test_tokens <= available_tokens:
                                            best_content = test_content
                                            left = mid + 1
                                        else:
                                            right = mid - 1
                                    
                                    if best_content and len(best_content) > 50:  # Minimum meaningful content
                                        msg.content = best_content
                                        truncated_messages.append(msg)
                                break
                    
                    optimized_messages = truncated_messages
                    logger.info(f"Emergency truncation applied: {len(messages)} -> {len(optimized_messages)} messages")
                            
            except Exception as e:
                logger.warning(f"Final token validation failed: {e}")
            
            return optimized_messages
        else:
            logger.debug(f"Context is optimal: {metrics.current_tokens}/{metrics.max_tokens} tokens "
                       f"({metrics.utilization_ratio:.1%})")
            return messages
        
    except Exception as e:
        logger.warning(f"Context evaluation failed: {e}, proceeding with original messages")
        return messages