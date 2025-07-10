from typing import Any, Callable, Optional
from langchain_core.messages import HumanMessage, BaseMessage

from src.utils.structured_logging import get_logger
from src.utils.content_processor import ContentProcessor
from src.utils.token_manager import TokenManager
from src.config.config_loader import config_loader
from src.utils.context_evaluator import get_global_context_evaluator
from src.llms.llm import get_llm_by_type
from src.utils.enhanced_message_extractor import get_global_message_extractor

logger = get_logger(__name__)


class ErrorHandlerConfig:
    """Configuration constants for error handler to avoid hardcoded values"""
    
    # Binary search and iteration limits
    MAX_ITERATIONS = 50
    
    # Token to character ratio estimates
    CHAR_TO_TOKEN_RATIO_CONSERVATIVE = 2.5  # More conservative estimate
    CHAR_TO_TOKEN_RATIO_ROUGH = 3.0  # Rough estimate: 1 token ≈ 3-4 characters
    
    # Token multipliers for different strategies
    TOKEN_SAFETY_MULTIPLIER = 2  # For checking if content is extremely long
    TOKEN_AGGRESSIVE_MULTIPLIER = 3  # For aggressive truncation fallback
    
    # Optimization and safety thresholds
    OPTIMIZATION_THRESHOLD = 0.9  # 90% threshold for optimization
    SAFETY_MARGIN = 0.8  # 80% safety margin
    
    # Truncation messages
    TRUNCATION_SUFFIX = "...[truncated]"
    TRUNCATION_NOTE = "\n\n[Note: Content has been significantly truncated due to length constraints. Please provide more specific queries for detailed analysis.]"
    SIMPLE_TRUNCATION_NOTE = "\n\n[Content truncated due to length constraints]"


# Global TokenManager instance for reuse
_token_manager = None

def _get_token_manager(processor=None):
    """Get or create a global TokenManager instance."""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager(processor)
    return _token_manager

def _apply_precise_token_truncation_internal(
    content: str, max_tokens: int, processor, model_name: str
) -> str:
    """Apply precise token-based truncation using TokenManager.
    
    Args:
        content: Content to truncate
        max_tokens: Maximum token limit
        processor: ContentProcessor instance
        model_name: Model name for token calculation
        
    Returns:
        Truncated content that fits within token limit
    """
    if not content:
        return ErrorHandlerConfig.TRUNCATION_SUFFIX
        
    # Use TokenManager for more robust truncation
    token_manager = _get_token_manager(processor)
    return token_manager.truncate_to_token_limit(
        content, max_tokens, model_name, preserve_start=True, preserve_end=False
    )

def _prepare_content_truncation(
    args: tuple, kwargs: dict, is_async: bool = False
) -> tuple[Any, str, list, Any, ContentProcessor]:
    """Prepare content truncation by extracting common logic
    
    Returns:
        Tuple of (config, model_name, messages, llm, processor)
    """
    
    # Load configuration
    config = config_loader.create_configuration()
    
    # Smart chunking is now always enabled
    logger.info("Smart chunking is enabled by default")
    
    # Create content processor
    processor = ContentProcessor(config.model_token_limits)
    
    # Use enhanced message extractor for consistent extraction
    message_extractor = get_global_message_extractor()
    extracted_messages, model_info = message_extractor.extract_messages_and_model(args, kwargs)
    
    messages = extracted_messages.messages
    model_name = model_info.model_name
    llm = model_info.llm_instance
    
    # Log extraction results
    if extracted_messages.messages:
        logger.debug(
            f"Content truncation extraction: {len(extracted_messages.messages)} messages, "
            f"pattern: {extracted_messages.pattern.value if extracted_messages.pattern else 'unknown'}, "
            f"model: {model_name}"
        )
    else:
        logger.warning("Content truncation: message extraction failed")
    
    # Fallback LLM detection for cases where extractor didn't find it
    if llm is None:
        for arg in args:
            if hasattr(arg, "invoke" if not is_async else "ainvoke"):  # LLM object
                llm = arg
                # Update model name if not already extracted
                if model_name == "deepseek-chat":
                    if hasattr(arg, "model_name"):
                        model_name = arg.model_name
                    elif hasattr(arg, "model"):
                        model_name = arg.model
                break
    
    return config, model_name, messages, llm, processor


def _handle_content_too_long_error(
    llm_func: Callable, error: Exception, *args, **kwargs
) -> Any:
    """Smart processing function for handling content too long errors"""
    try:
        config, model_name, messages, llm, processor = _prepare_content_truncation(
            args, kwargs, is_async=False
        )

        # Messages and model information already extracted by _prepare_content_truncation

        if not messages:
            logger.warning("Could not extract messages from function arguments")
            raise error

        # Extract text content
        content_parts = []
        for msg in messages:
            if hasattr(msg, "content"):
                content_parts.append(str(msg.content))

        combined_content = "\n\n".join(content_parts)

        # Get model limits for the specific model
        model_limits = processor.get_model_limits(model_name)
        max_tokens = int(model_limits.input_limit * model_limits.safety_margin)

        # If content is extremely long, use aggressive chunking
        if (
            processor.count_tokens_accurate(combined_content, model_name).total_tokens
            > max_tokens * ErrorHandlerConfig.TOKEN_SAFETY_MULTIPLIER
        ):
            logger.info("Content is extremely long, using aggressive chunking")
            # Use aggressive chunking strategy for extremely long content
            chunks = processor.smart_chunk_content(
                combined_content, model_name, "aggressive"
            )
            if chunks:
                # Use only the first chunk for extremely long content
                truncated_content = chunks[0]

                # Double-check the chunk size and further truncate if needed
                if (
                    processor.count_tokens_accurate(
                        truncated_content, model_name
                    ).total_tokens
                    > max_tokens
                ):
                    # Emergency truncation: use character-based limit
                    # Use precise token-based truncation instead of character estimation
                    # Reuse existing processor instance

                    # Use precise token-based truncation
                    truncated_content = _apply_precise_token_truncation_internal(
                        truncated_content, max_tokens, processor, model_name
                    )

                    # Add a note about truncation
                    truncated_content += ErrorHandlerConfig.TRUNCATION_NOTE
            else:
                # Fallback: take first portion based on character count
                # Use precise token-based truncation
                # Reuse existing processor instance

                # Use precise token-based truncation
                truncated_content = _apply_precise_token_truncation_internal(
                    combined_content, max_tokens, processor, model_name
                )
        else:
            # Try summarization first if LLM is available
            if config.enable_content_summarization and llm:
                logger.info("Attempting to summarize content to fit token limits")
                try:
                    basic_llm = get_llm_by_type("basic")
                    summarized_content = processor.summarize_content(
                        combined_content, basic_llm, model_name, config.summary_type
                    )
                    truncated_content = summarized_content
                except (ImportError, AttributeError, ValueError, RuntimeError) as summarize_error:
                    logger.warning(
                        f"Summarization failed: {summarize_error}, falling back to chunking"
                    )
                except Exception as summarize_error:
                    logger.error(
                        f"Unexpected error during summarization: {type(summarize_error).__name__}: {summarize_error}"
                    )
                    # Re-raise unexpected errors after logging
                    raise
            else:
                # Use chunking with aggressive strategy for very long content
                logger.info("Attempting smart content chunking")
                # Use aggressive strategy if content is extremely long
                strategy = (
                    "aggressive"
                    if processor.count_tokens_accurate(
                        combined_content, model_name
                    ).total_tokens
                    > max_tokens * ErrorHandlerConfig.TOKEN_SAFETY_MULTIPLIER
                    else "auto"
                )
                chunks = processor.smart_chunk_content(
                    combined_content, model_name, strategy
                )
                if chunks:
                    truncated_content = chunks[0]
                    # Ensure the chunk fits within limits
                    if (
                        processor.count_tokens_accurate(
                            truncated_content, model_name
                        ).total_tokens
                        > max_tokens
                    ):
                        # Use precise token-based truncation
                        # Reuse existing processor instance

                        # Use precise token-based truncation
                        truncated_content = _apply_precise_token_truncation_internal(
                            truncated_content, max_tokens, processor, model_name
                        )
                else:
                    truncated_content = combined_content[: max_tokens * ErrorHandlerConfig.TOKEN_AGGRESSIVE_MULTIPLIER]

        # Create new message list with truncated content
        new_messages = []
        for msg in messages[:-1]:  # Keep all messages except the last one
            new_messages.append(msg)

        # Replace the last message with truncated content
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                if isinstance(last_msg, BaseMessage):
                    # Use the same message type as the original
                    new_msg = type(last_msg)(content=truncated_content)
                else:
                    # Default to HumanMessage for non-BaseMessage objects
                    new_msg = HumanMessage(content=truncated_content)
                new_messages.append(new_msg)

        # Update parameters
        new_args = list(args)
        new_kwargs = kwargs.copy()

        # Update messages in the appropriate location
        if "messages" in kwargs:
            new_kwargs["messages"] = new_messages
        else:
            # Update in args
            for i, arg in enumerate(new_args):
                if arg is messages:
                    new_args[i] = new_messages
                    break

        # Re-invoke the function
        logger.info(
            f"Retrying with truncated content (reduced from {processor.count_tokens_accurate(combined_content, model_name).total_tokens} to ~{processor.count_tokens_accurate(truncated_content, model_name).total_tokens} tokens)"
        )
        result = llm_func(*new_args, **new_kwargs)
        return result

    except (ValueError, TypeError, AttributeError, KeyError) as e:
        logger.error(f"Smart content processing failed due to data/type error: {type(e).__name__}: {e}")
        # Create a new exception with more context information
        enhanced_error = Exception(f"Smart processing failed: {e}. Original error: {error}")
        enhanced_error.__cause__ = e
        raise enhanced_error
    except Exception as e:
        logger.error(f"Unexpected error in smart content processing: {type(e).__name__}: {e}")
        # For unexpected errors, preserve the original error
        raise error from e


async def _handle_content_too_long_error_async(
    llm_func: Callable, error: Exception, *args, **kwargs
) -> Any:
    """Async smart processing function for handling content too long errors"""
    try:
        config, model_name, messages, llm, processor = _prepare_content_truncation(
            args, kwargs, is_async=True
        )

        if not messages:
            logger.warning("Could not extract messages from function arguments")
            raise error

        # Extract text content
        content_parts = []
        for msg in messages:
            if hasattr(msg, "content"):
                content_parts.append(str(msg.content))

        combined_content = "\n\n".join(content_parts)

        # Get model limits for the specific model
        model_limits = processor.get_model_limits(model_name)
        max_tokens = int(model_limits.input_limit * model_limits.safety_margin)

        # If content is extremely long, use aggressive chunking
        if (
            processor.count_tokens_accurate(combined_content, model_name).total_tokens
            > max_tokens * 2
        ):
            logger.info("Content is extremely long, using aggressive chunking")
            # Use aggressive chunking strategy for extremely long content
            chunks = processor.smart_chunk_content(
                combined_content, model_name, "aggressive"
            )
            if chunks:
                # Use only the first chunk for extremely long content
                truncated_content = chunks[0]

                # Double-check the chunk size and further truncate if needed
                if (
                    processor.count_tokens_accurate(
                        truncated_content, model_name
                    ).total_tokens
                    > max_tokens
                ):
                    # Emergency truncation: use character-based limit
                    # Use precise token-based truncation instead of character estimation
                    # Reuse existing processor instance

                    # Binary search for precise truncation
                    content = truncated_content
                    left, right = 0, len(content)
                    best_content = ""
                    max_iterations = ErrorHandlerConfig.MAX_ITERATIONS  # Prevent infinite loops
                    iteration_count = 0

                    # Handle edge case: empty content
                    if not content:
                        best_content = ErrorHandlerConfig.TRUNCATION_SUFFIX
                    else:
                        while left <= right and iteration_count < max_iterations:
                            iteration_count += 1
                            mid = (left + right) // 2
                            
                            # Handle edge case: mid is 0
                            if mid == 0:
                                test_content = ErrorHandlerConfig.TRUNCATION_SUFFIX
                            else:
                                test_content = content[:mid] + ErrorHandlerConfig.TRUNCATION_SUFFIX
                            
                            test_tokens = processor.estimate_tokens(
                                test_content, model_name
                            )

                            if test_tokens <= max_tokens:
                                best_content = test_content
                                left = mid + 1
                            else:
                                right = mid - 1
                        
                        # Log if we hit max iterations
                        if iteration_count >= max_iterations:
                            logger.warning(f"Binary search hit max iterations ({max_iterations}) for content truncation")

                    if best_content:
                        truncated_content = best_content
                    else:
                        # Fallback to conservative character limit if binary search fails
                        char_limit = int(max_tokens * ErrorHandlerConfig.CHAR_TO_TOKEN_RATIO_CONSERVATIVE)  # More conservative estimate
                        truncated_content = truncated_content[:char_limit]
                        logger.warning(
                            f"Applied emergency character-based truncation to {char_limit} characters"
                        )

                # Add a note about truncation
                truncated_content += ErrorHandlerConfig.TRUNCATION_NOTE
            else:
                # Fallback: take first portion based on character count
                char_limit = max_tokens * ErrorHandlerConfig.TOKEN_AGGRESSIVE_MULTIPLIER  # Rough estimate: 1 token ≈ 3-4 characters
                truncated_content = (
                    combined_content[:char_limit]
                    + ErrorHandlerConfig.SIMPLE_TRUNCATION_NOTE
                )
        else:
            # Try summarization first if LLM is available
            if config.enable_content_summarization and llm:
                logger.info("Attempting to summarize content to fit token limits")
                try:
                    basic_llm = get_llm_by_type("basic")
                    summarized_content = processor.summarize_content(
                        combined_content, basic_llm, model_name, config.summary_type
                    )
                    truncated_content = summarized_content
                except (ImportError, AttributeError, ValueError, RuntimeError) as summarize_error:
                    logger.warning(
                        f"Summarization failed: {summarize_error}, falling back to chunking"
                    )
                except Exception as summarize_error:
                    logger.error(
                        f"Unexpected error during async summarization: {type(summarize_error).__name__}: {summarize_error}"
                    )
                    # Re-raise unexpected errors after logging
                    raise
            else:
                # Use chunking with aggressive strategy for very long content
                logger.info("Attempting smart content chunking")
                # Use aggressive strategy if content is extremely long
                strategy = (
                    "aggressive"
                    if processor.count_tokens_accurate(
                        combined_content, model_name
                    ).total_tokens
                    > max_tokens * ErrorHandlerConfig.TOKEN_SAFETY_MULTIPLIER
                    else "auto"
                )
                chunks = processor.smart_chunk_content(
                    combined_content, model_name, strategy
                )
                if chunks:
                    truncated_content = chunks[0]
                    # Ensure the chunk fits within limits
                    if (
                        processor.count_tokens_accurate(
                            truncated_content, model_name
                        ).total_tokens
                        > max_tokens
                    ):
                        char_limit = int(max_tokens * ErrorHandlerConfig.TOKEN_AGGRESSIVE_MULTIPLIER)
                        truncated_content = truncated_content[:char_limit]
                else:
                    truncated_content = combined_content[: max_tokens * ErrorHandlerConfig.TOKEN_AGGRESSIVE_MULTIPLIER]

        # Create new message list with truncated content
        new_messages = []
        for msg in messages[:-1]:  # Keep all messages except the last one
            new_messages.append(msg)

        # Replace the last message with truncated content
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                if isinstance(last_msg, BaseMessage):
                    # Use the same message type as the original
                    new_msg = type(last_msg)(content=truncated_content)
                else:
                    # Default to HumanMessage for non-BaseMessage objects
                    new_msg = HumanMessage(content=truncated_content)
                new_messages.append(new_msg)

        # Update parameters
        new_args = list(args)
        new_kwargs = kwargs.copy()

        # Update messages in the appropriate location
        if "input" in kwargs and "messages" in kwargs["input"]:
            new_input = kwargs["input"].copy()
            new_input["messages"] = new_messages
            new_kwargs["input"] = new_input
        elif "messages" in kwargs:
            new_kwargs["messages"] = new_messages
        else:
            # Update in args
            for i, arg in enumerate(new_args):
                if arg is messages:
                    new_args[i] = new_messages
                    break

        # Re-invoke the function
        logger.info(
            f"Retrying with truncated content (reduced from {processor.count_tokens_accurate(combined_content, model_name).total_tokens} to ~{processor.count_tokens_accurate(truncated_content, model_name).total_tokens} tokens)"
        )
        result = await llm_func(*new_args, **new_kwargs)
        return result

    except (ValueError, TypeError, AttributeError, KeyError) as e:
        logger.error(f"Async smart content processing failed due to data/type error: {type(e).__name__}: {e}")
        # Create a new exception with more context information
        enhanced_error = Exception(f"Async smart processing failed: {e}. Original error: {error}")
        enhanced_error.__cause__ = e
        raise enhanced_error
    except Exception as e:
        logger.error(f"Unexpected error in async smart content processing: {type(e).__name__}: {e}")
        # For unexpected errors, preserve the original error
        raise error from e


class LLMContextOptimizer:
    """Handles LLM context evaluation and optimization, including content truncation."""

    def __init__(self):
        pass # No specific initialization needed for now, as it uses global evaluators/configs

    async def evaluate_and_optimize_context_before_call(
        self, llm_func: Callable, args: tuple, kwargs: dict, operation_name: str, context: str
    ) -> tuple:
        """Evaluate and optimize context before LLM call

        Args:
            llm_func: LLM function to be called
            args: Function arguments
            kwargs: Function keyword arguments
            operation_name: Name of the operation
            context: Context information

        Returns:
            Tuple of (optimized_args, optimized_kwargs)
        """
        try:
            # Get context evaluator and message extractor
            evaluator = get_global_context_evaluator()
            message_extractor = get_global_message_extractor()

            # Extract messages and model information using enhanced extractor
            extracted_messages, model_info = message_extractor.extract_messages_and_model(args, kwargs)
            
            # Use extracted information
            messages = extracted_messages.messages
            model_name = model_info.model_name
            
            # Log extraction results for debugging
            if extracted_messages.messages:
                logger.debug(
                    f"Enhanced extraction successful: {len(extracted_messages.messages)} messages, "
                    f"pattern: {extracted_messages.pattern.value if extracted_messages.pattern else 'unknown'}, "
                    f"model: {model_name}, source: {model_info.source}"
                )
            else:
                logger.warning("Enhanced message extraction failed - no valid messages found")
                
            # Fallback to LLM function inspection if model name is still default
            if model_name == "deepseek-chat" and hasattr(llm_func, "__self__"):
                llm_instance = llm_func.__self__
                if hasattr(llm_instance, "model_name"):
                    model_name = llm_instance.model_name
                elif hasattr(llm_instance, "model"):
                    model_name = llm_instance.model

            # If we found messages, evaluate and optimize
            if messages and isinstance(messages, list) and messages:
                logger.debug(
                    f"Evaluating context for {operation_name}: {len(messages)} messages, model: {model_name}"
                )

                # Evaluate context state
                metrics = evaluator.evaluate_context_before_llm_call(
                    messages, model_name, f"{operation_name} - {context}"
                )
                logger.info(f"metrics.compression_needed: {metrics.compression_needed}, metrics.evaluation_result.value: {metrics.evaluation_result.value}")
                # Apply optimization if needed
                if metrics.compression_needed or metrics.evaluation_result.value in [
                    "needs_compression",
                    "needs_truncation",
                    "critical_overflow",
                    "requires_chunking",
                ]:
                    logger.info(
                        f"Context optimization needed: {metrics.evaluation_result.value}, "
                        f"applying {metrics.recommended_strategy.value}"
                    )

                    optimized_messages, optimization_info = (
                        evaluator.optimize_context_for_llm_call(
                            messages,
                            model_name,
                            metrics.recommended_strategy,
                            f"{operation_name} - {context}",
                        )
                    )

                    # Update the arguments with optimized messages
                    new_args = list(args)
                    new_kwargs = kwargs.copy()

                    # Update messages in the appropriate location
                    if (
                        "input" in kwargs
                        and isinstance(kwargs["input"], dict)
                        and "messages" in kwargs["input"]
                    ):
                        new_input = kwargs["input"].copy()
                        new_input["messages"] = optimized_messages
                        new_kwargs["input"] = new_input
                    elif "messages" in kwargs:
                        new_kwargs["messages"] = optimized_messages
                    else:
                        # Update in args
                        for i, arg in enumerate(new_args):
                            if arg is messages:
                                new_args[i] = optimized_messages
                                break

                    logger.info(
                        f"Context optimized: {optimization_info['original_tokens']} -> "
                        f"{optimization_info['optimized_tokens']} tokens "
                        f"({optimization_info['tokens_saved']} saved)"
                    )

                    # Final token validation before returning
                    try:
                        config = config_loader.create_configuration()
                        processor = ContentProcessor(config.model_token_limits)

                        # Calculate total tokens in optimized messages
                        total_content = ""
                        for msg in optimized_messages:
                            if hasattr(msg, "content"):
                                total_content += str(msg.content) + " "

                        estimated_tokens = processor.estimate_tokens(
                            total_content, model_name
                        )
                        model_limits = processor.get_model_limits(model_name)
                        model_limit = model_limits.safe_input_limit  # Use configured limit

                        if (
                            estimated_tokens > model_limit * ErrorHandlerConfig.OPTIMIZATION_THRESHOLD
                        ):  # 90% threshold for final check
                            logger.warning(
                                f"Final token check: {estimated_tokens} tokens exceeds 90% of limit ({model_limit})"
                            )
                            # Emergency truncation
                            max_chars = int(
                                model_limit * ErrorHandlerConfig.SAFETY_MARGIN * ErrorHandlerConfig.TOKEN_AGGRESSIVE_MULTIPLIER
                            )  # Conservative character limit
                            if len(total_content) > max_chars:
                                truncated_content = (
                                    total_content[:max_chars] + ErrorHandlerConfig.TRUNCATION_SUFFIX
                                )
                                # Update the last message with truncated content
                                if optimized_messages and hasattr(
                                    optimized_messages[-1], "content"
                                ):
                                    optimized_messages[-1].content = truncated_content

                                    # Update the arguments again with emergency truncation
                                    if (
                                        "input" in new_kwargs
                                        and isinstance(new_kwargs["input"], dict)
                                        and "messages" in new_kwargs["input"]
                                    ):
                                        new_kwargs["input"]["messages"] = optimized_messages
                                    elif "messages" in new_kwargs:
                                        new_kwargs["messages"] = optimized_messages
                                    else:
                                        for i, arg in enumerate(new_args):
                                            if (
                                                isinstance(arg, list)
                                                and arg
                                                and hasattr(arg[0], "content")
                                            ):
                                                new_args[i] = optimized_messages
                                                break

                    except Exception as e:
                        logger.warning(f"Final token validation failed: {e}")

                    return tuple(new_args), new_kwargs
                else:
                    logger.debug(
                        f"Context is optimal: {metrics.current_tokens}/{metrics.max_tokens} tokens "
                        f"({metrics.utilization_ratio:.1%})"
                    )

            # Return original arguments if no optimization needed
            return args, kwargs

        except Exception as e:
            logger.warning(
                f"Context evaluation failed: {e}, proceeding with original arguments"
            )
            return args, kwargs

    def _handle_content_too_long_error(
        self, llm_func: Callable, error: Exception, *args, **kwargs
    ) -> Any:
        """Smart processing function for handling content too long errors"""
        return _handle_content_too_long_error(llm_func, error, *args, **kwargs)

    async def _handle_content_too_long_error_async(
        self, llm_func: Callable, error: Exception, *args, **kwargs
    ) -> Any:
        """Async smart processing function for handling content too long errors"""
        return await _handle_content_too_long_error_async(llm_func, error, *args, **kwargs)

    def evaluate_and_optimize_context_before_call_sync(
        self, llm_func: Callable, args: tuple, kwargs: dict, operation_name: str, context: str
    ) -> tuple:
        """Synchronous version of context evaluation and optimization

        Args:
            llm_func: LLM function to be called
            args: Function arguments
            kwargs: Function keyword arguments
            operation_name: Name of the operation
            context: Context information

        Returns:
            Tuple of (optimized_args, optimized_kwargs)
        """
        try:
            # Get context evaluator and message extractor
            evaluator = get_global_context_evaluator()
            message_extractor = get_global_message_extractor()

            # Extract messages and model information using enhanced extractor
            extracted_messages, model_info = message_extractor.extract_messages_and_model(args, kwargs)
            
            # Use extracted information
            messages = extracted_messages.messages
            model_name = model_info.model_name
            
            # Log extraction results for debugging
            if extracted_messages.messages:
                logger.debug(
                    f"Enhanced extraction successful: {len(extracted_messages.messages)} messages, "
                    f"pattern: {extracted_messages.pattern.value if extracted_messages.pattern else 'unknown'}, "
                    f"model: {model_name}, source: {model_info.source}"
                )
            else:
                logger.warning("Enhanced message extraction failed - no valid messages found")
                
            # Fallback to LLM function inspection if model name is still default
            if model_name == "deepseek-chat" and hasattr(llm_func, "__self__"):
                llm_instance = llm_func.__self__
                if hasattr(llm_instance, "model_name"):
                    model_name = llm_instance.model_name
                elif hasattr(llm_instance, "model"):
                    model_name = llm_instance.model

            # If we found messages, evaluate and optimize
            if messages and isinstance(messages, list) and messages:
                logger.debug(
                    f"Evaluating context for {operation_name}: {len(messages)} messages, model: {model_name}"
                )

                # Evaluate context state
                metrics = evaluator.evaluate_context_before_llm_call(
                    messages, model_name, f"{operation_name} - {context}"
                )

                logger.info(f"metrics.compression_needed: {metrics.compression_needed}, metrics.evaluation_result.value: {metrics.evaluation_result.value}")
                # Apply optimization if needed
                if metrics.compression_needed or metrics.evaluation_result.value in [
                    "needs_compression",
                    "needs_truncation",
                    "critical_overflow",
                    "requires_chunking",
                ]:
                    logger.info(
                        f"Context optimization needed: {metrics.evaluation_result.value}, "
                        f"applying {metrics.recommended_strategy.value}"
                    )

                    optimized_messages, optimization_info = (
                        evaluator.optimize_context_for_llm_call(
                            messages,
                            model_name,
                            metrics.recommended_strategy,
                            f"{operation_name} - {context}",
                        )
                    )

                    # Update the arguments with optimized messages
                    new_args = list(args)
                    new_kwargs = kwargs.copy()

                    # Update messages in the appropriate location
                    if (
                        "input" in kwargs
                        and isinstance(kwargs["input"], dict)
                        and "messages" in kwargs["input"]
                    ):
                        new_input = kwargs["input"].copy()
                        new_input["messages"] = optimized_messages
                        new_kwargs["input"] = new_input
                    elif "messages" in kwargs:
                        new_kwargs["messages"] = optimized_messages
                    else:
                        # Update in args
                        for i, arg in enumerate(new_args):
                            if arg is messages:
                                new_args[i] = optimized_messages
                                break

                    logger.info(
                        f"Context optimized: {optimization_info['original_tokens']} -> "
                        f"{optimization_info['optimized_tokens']} tokens "
                        f"({optimization_info['tokens_saved']} saved)"
                    )

                    # Final token validation before returning
                    try:
                        config = config_loader.create_configuration()
                        processor = ContentProcessor(config.model_token_limits)

                        # Calculate total tokens in optimized messages
                        total_content = ""
                        for msg in optimized_messages:
                            if hasattr(msg, "content"):
                                total_content += str(msg.content) + " "

                        estimated_tokens = processor.estimate_tokens(
                            total_content, model_name
                        )
                        model_limits = processor.get_model_limits(model_name)
                        model_limit = model_limits.safe_input_limit  # Use configured limit

                        if (
                            estimated_tokens > model_limit * ErrorHandlerConfig.OPTIMIZATION_THRESHOLD
                        ):  # 90% threshold for final check
                            logger.warning(
                                f"Final token check: {estimated_tokens} tokens exceeds 90% of limit ({model_limit})"
                            )
                            # Emergency truncation
                            max_chars = int(
                                model_limit * ErrorHandlerConfig.SAFETY_MARGIN * ErrorHandlerConfig.TOKEN_AGGRESSIVE_MULTIPLIER
                            )  # Conservative character limit
                            if len(total_content) > max_chars:
                                truncated_content = (
                                    total_content[:max_chars] + ErrorHandlerConfig.TRUNCATION_SUFFIX
                                )
                                # Update the last message with truncated content
                                if optimized_messages and hasattr(
                                    optimized_messages[-1], "content"
                                ):
                                    optimized_messages[-1].content = truncated_content

                                    # Update the arguments again with emergency truncation
                                    if (
                                        "input" in new_kwargs
                                        and isinstance(new_kwargs["input"], dict)
                                        and "messages" in new_kwargs["input"]
                                    ):
                                        new_kwargs["input"]["messages"] = optimized_messages
                                    elif "messages" in new_kwargs:
                                        new_kwargs["messages"] = optimized_messages
                                    else:
                                        for i, arg in enumerate(new_args):
                                            if (
                                                isinstance(arg, list)
                                                and arg
                                                and hasattr(arg[0], "content")
                                            ):
                                                new_args[i] = optimized_messages
                                                break

                    except Exception as e:
                        logger.warning(f"Final token validation failed: {e}")

                    return tuple(new_args), new_kwargs
                else:
                    logger.debug(
                        f"Context is optimal: {metrics.current_tokens}/{metrics.max_tokens} tokens "
                        f"({metrics.utilization_ratio:.1%})"
                    )

            # Return original arguments if no optimization needed
            return args, kwargs

        except Exception as e:
            logger.warning(
                f"Context evaluation failed: {e}, proceeding with original arguments"
            )
            return args, kwargs