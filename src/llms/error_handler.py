"""LLM Error Handling Module

Provides unified LLM error handling mechanisms with support for different error handling strategies.
"""

import asyncio
import time
from typing import Any, Callable, Optional
from functools import wraps
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

# Import new tool modules
from src.utils.structured_logging import (
    get_logger,
)

logger = get_logger(__name__)


class LLMErrorType:
    """LLM error type constants"""

    DATA_INSPECTION_FAILED = "data_inspection_failed"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID_API_KEY = "invalid_api_key"
    MODEL_NOT_FOUND = "model_not_found"
    CONTENT_TOO_LONG = "content_too_long"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    BAD_REQUEST = "bad_request"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    CONCURRENT_LIMIT_EXCEEDED = "concurrent_limit_exceeded"
    UNKNOWN_ERROR = "unknown_error"


class LLMErrorHandler:
    """LLM error handler"""

    def __init__(self):
        self.error_patterns = {
            LLMErrorType.DATA_INSPECTION_FAILED: [
                "data_inspection_failed",
                "content safety",
                "content policy",
                "inappropriate content",
                "content filter",
                "safety filter",
            ],
            LLMErrorType.RATE_LIMIT_EXCEEDED: [
                "rate limit",
                "too many requests",
                "rate_limit_exceeded",
                "requests per minute",
                "requests per second",
                "throttled",
            ],
            LLMErrorType.QUOTA_EXCEEDED: [
                "quota exceeded",
                "insufficient quota",
                "quota_exceeded",
                "billing quota",
                "usage limit",
            ],
            LLMErrorType.INVALID_API_KEY: [
                "invalid api key",
                "api key",
                "invalid key",
                "bad api key",
            ],
            LLMErrorType.AUTHENTICATION_ERROR: [
                "authentication failed",
                "unauthorized",
                "auth failed",
                "authentication error",
                "invalid credentials",
                "access denied",
            ],
            LLMErrorType.PERMISSION_ERROR: [
                "permission denied",
                "forbidden",
                "access forbidden",
                "insufficient permissions",
                "not authorized",
            ],
            LLMErrorType.MODEL_NOT_FOUND: [
                "model not found",
                "model does not exist",
                "invalid model",
                "unknown model",
                "model unavailable",
            ],
            LLMErrorType.CONTENT_TOO_LONG: [
                "content too long",
                "token limit exceeded",
                "maximum context length",
                "range of input length should be",
                "input length should be",
                "invalidparameter: range of input length",
                "context length exceeded",
                "input too long",
            ],
            LLMErrorType.NETWORK_ERROR: [
                "network error",
                "connection error",
                "connection failed",
                "network unreachable",
                "dns resolution failed",
                "connection refused",
            ],
            LLMErrorType.TIMEOUT_ERROR: [
                "timeout",
                "request timeout",
                "read timeout",
                "connection timeout",
                "operation timeout",
                "deadline exceeded",
            ],
            LLMErrorType.SERVICE_UNAVAILABLE: [
                "service unavailable",
                "server unavailable",
                "temporarily unavailable",
                "maintenance mode",
                "service down",
            ],
            LLMErrorType.INTERNAL_SERVER_ERROR: [
                "internal server error",
                "server error",
                "500 error",
                "internal error",
                "server fault",
            ],
            LLMErrorType.BAD_REQUEST: [
                "bad request",
                "invalid request",
                "malformed request",
                "400 error",
                "invalid parameter",
            ],
            LLMErrorType.RESOURCE_EXHAUSTED: [
                "resource exhausted",
                "out of resources",
                "capacity exceeded",
                "resource limit",
                "memory exhausted",
            ],
            LLMErrorType.CONCURRENT_LIMIT_EXCEEDED: [
                "concurrent limit",
                "too many concurrent",
                "concurrency limit",
                "parallel requests limit",
                "simultaneous requests",
            ],
        }

        self.fallback_responses = {
            LLMErrorType.DATA_INSPECTION_FAILED: "Unable to process this request due to content safety restrictions. Please try with different content.",
            LLMErrorType.RATE_LIMIT_EXCEEDED: "Request rate limit exceeded, please try again later.",
            LLMErrorType.QUOTA_EXCEEDED: "API quota exhausted, please check your account quota.",
            LLMErrorType.INVALID_API_KEY: "Invalid API key, please check your configuration.",
            LLMErrorType.AUTHENTICATION_ERROR: "Authentication failed, please check your credentials.",
            LLMErrorType.PERMISSION_ERROR: "Permission denied, please check your access rights.",
            LLMErrorType.MODEL_NOT_FOUND: "Specified model does not exist, please check model configuration.",
            LLMErrorType.CONTENT_TOO_LONG: "Content too long, please shorten the input.",
            LLMErrorType.NETWORK_ERROR: "Network connection error, please check your network connection.",
            LLMErrorType.TIMEOUT_ERROR: "Request timeout, please try again later.",
            LLMErrorType.SERVICE_UNAVAILABLE: "Service temporarily unavailable, please try again later.",
            LLMErrorType.INTERNAL_SERVER_ERROR: "Internal server error occurred, please try again later.",
            LLMErrorType.BAD_REQUEST: "Invalid request format, please check your input.",
            LLMErrorType.RESOURCE_EXHAUSTED: "System resources exhausted, please try again later.",
            LLMErrorType.CONCURRENT_LIMIT_EXCEEDED: "Too many concurrent requests, please try again later.",
            LLMErrorType.UNKNOWN_ERROR: "Unknown error occurred, please try again later.",
        }

        self.skip_errors = {
            LLMErrorType.DATA_INSPECTION_FAILED,  # Content safety errors, skip and continue
        }

        self.retry_errors = {
            LLMErrorType.RATE_LIMIT_EXCEEDED,
            LLMErrorType.NETWORK_ERROR,
            LLMErrorType.TIMEOUT_ERROR,
            LLMErrorType.SERVICE_UNAVAILABLE,
            LLMErrorType.INTERNAL_SERVER_ERROR,
            LLMErrorType.RESOURCE_EXHAUSTED,
            LLMErrorType.CONCURRENT_LIMIT_EXCEEDED,
        }

        self.fatal_errors = {
            LLMErrorType.INVALID_API_KEY,
            LLMErrorType.AUTHENTICATION_ERROR,
            LLMErrorType.PERMISSION_ERROR,
            LLMErrorType.QUOTA_EXCEEDED,
            LLMErrorType.MODEL_NOT_FOUND,
            LLMErrorType.BAD_REQUEST,
        }

        # Error types that need smart processing
        self.smart_processing_errors = {
            LLMErrorType.CONTENT_TOO_LONG,
        }

    def classify_error(self, error_message) -> str:
        """Classify error type based on error message with priority"""
        # Handle both string and Exception objects
        if isinstance(error_message, Exception):
            error_message = str(error_message)
        error_message_lower = error_message.lower()
        
        # Define priority order - more specific errors have higher priority
        priority_order = [
            LLMErrorType.INVALID_API_KEY,           # Most specific authentication issue
            LLMErrorType.AUTHENTICATION_ERROR,      # General authentication issue
            LLMErrorType.PERMISSION_ERROR,          # Access permission issue
            LLMErrorType.QUOTA_EXCEEDED,            # Specific quota issue
            LLMErrorType.RATE_LIMIT_EXCEEDED,       # Specific rate limiting
            LLMErrorType.CONCURRENT_LIMIT_EXCEEDED, # Specific concurrency issue
            LLMErrorType.CONTENT_TOO_LONG,          # Content length issue
            LLMErrorType.MODEL_NOT_FOUND,           # Model availability issue
            LLMErrorType.DATA_INSPECTION_FAILED,    # Content safety issue
            LLMErrorType.BAD_REQUEST,               # Request format issue
            LLMErrorType.TIMEOUT_ERROR,             # Specific timeout issue
            LLMErrorType.NETWORK_ERROR,             # Network connectivity issue
            LLMErrorType.RESOURCE_EXHAUSTED,        # Resource availability issue
            LLMErrorType.SERVICE_UNAVAILABLE,       # Service availability issue
            LLMErrorType.INTERNAL_SERVER_ERROR,     # Server-side issue
        ]
        
        # Check patterns in priority order
        for error_type in priority_order:
            if error_type in self.error_patterns:
                for pattern in self.error_patterns[error_type]:
                    if pattern in error_message_lower:
                        return error_type
        
        return LLMErrorType.UNKNOWN_ERROR

    def should_skip_error(self, error_type: str) -> bool:
        """Determine if this error should be skipped"""
        return error_type in self.skip_errors

    def should_retry_error(self, error_type: str) -> bool:
        """Determine if this error should be retried"""
        return error_type in self.retry_errors

    def is_fatal_error(self, error_type: str) -> bool:
        """Determine if this is a fatal error"""
        return error_type in self.fatal_errors

    def get_fallback_response(self, error_type: str, context: str = "") -> AIMessage:
        """Get fallback response for error"""
        fallback_text = self.fallback_responses.get(
            error_type, self.fallback_responses[LLMErrorType.UNKNOWN_ERROR]
        )

        if context:
            fallback_text = f"{fallback_text} (Context: {context})"

        return AIMessage(content=fallback_text)

    def needs_smart_processing(self, error_type: str) -> bool:
        """Determine if smart processing is needed"""
        return error_type in self.smart_processing_errors

    def handle_error(
        self, error: Exception, context: str = "", operation_name: str = "LLM Operation"
    ) -> tuple[bool, Optional[AIMessage], bool]:
        """Handle LLM error

        Args:
            error: Exception object
            context: Error context
            operation_name: Operation name

        Returns:
            tuple: (should skip error, fallback response, needs smart processing)
        """
        error_message = str(error)
        error_type = self.classify_error(error_message)

        logger.warning(
            f"{operation_name} execution error: {error_message} (Error type: {error_type})"
        )

        # Check if smart processing is needed
        if self.needs_smart_processing(error_type):
            logger.info(
                f"Detected {error_type} error, triggering smart content processing"
            )
            return False, None, True

        elif self.should_skip_error(error_type):
            logger.info(
                f"Skipping {error_type} error, using fallback response to continue execution"
            )
            fallback_response = self.get_fallback_response(error_type, context)
            return True, fallback_response, False

        elif self.should_retry_error(error_type):
            logger.info(f"Detected retryable error: {error_type}")
            return False, None, False

        elif self.is_fatal_error(error_type):
            logger.error(f"Detected fatal error: {error_type}, stopping execution")
            raise error

        else:
            logger.error(f"Unknown error type: {error_type}, re-raising exception")
            raise error


def _handle_content_too_long_error(
    llm_func: Callable, error: Exception, *args, **kwargs
) -> Any:
    """Smart processing function for handling content too long errors"""
    from src.config.config_loader import config_loader
    from src.utils.content_processor import ContentProcessor

    try:
        # Load configuration
        config = config_loader.create_configuration()

        # Smart chunking is now always enabled
        logger.info("Smart chunking is enabled by default")

        # Create content processor
        processor = ContentProcessor(config.model_token_limits)

        # Try to extract messages and model information from parameters
        messages = None
        model_name = "deepseek-chat"  # Default fallback instead of 'unknown'
        llm = None

        # Check messages in args
        for arg in args:
            if hasattr(arg, "__iter__") and not isinstance(arg, str):
                try:
                    # Check if it's a message list
                    if all(hasattr(item, "content") for item in arg):
                        messages = arg
                        break
                except (TypeError, AttributeError, ValueError) as e:
                    logger.debug(f"Failed to check message list format: {e}")
                    continue
            elif hasattr(arg, "invoke"):  # LLM object
                llm = arg
                if hasattr(arg, "model_name"):
                    model_name = arg.model_name
                elif hasattr(arg, "model"):
                    model_name = arg.model

        # Check messages and model information in kwargs
        if "messages" in kwargs:
            messages = kwargs["messages"]
        if "model" in kwargs:
            model_name = kwargs["model"]

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
                    max_iterations = 50  # Prevent infinite loops
                    iteration_count = 0

                    # Handle edge case: empty content
                    if not content:
                        best_content = "...[truncated]"
                    else:
                        while left <= right and iteration_count < max_iterations:
                            iteration_count += 1
                            mid = (left + right) // 2
                            
                            # Handle edge case: mid is 0
                            if mid == 0:
                                test_content = "...[truncated]"
                            else:
                                test_content = content[:mid] + "...[truncated]"
                            
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
                        char_limit = int(max_tokens * 2.5)  # More conservative estimate
                        truncated_content = truncated_content[:char_limit]
                        logger.warning(
                            f"Applied emergency character-based truncation to {char_limit} characters"
                        )

                    # Add a note about truncation
                    truncated_content += "\n\n[Note: Content has been significantly truncated due to length constraints. Please provide more specific queries for detailed analysis.]"
            else:
                # Fallback: take first portion based on character count
                # Use precise token-based truncation
                # Reuse existing processor instance

                # Binary search for precise truncation
                left, right = 0, len(combined_content)
                best_content = ""
                max_iterations = 50  # Prevent infinite loops
                iteration_count = 0

                # Handle edge case: empty content
                if not combined_content:
                    best_content = "...[truncated]"
                else:
                    while left <= right and iteration_count < max_iterations:
                        iteration_count += 1
                        mid = (left + right) // 2
                        
                        # Handle edge case: mid is 0
                        if mid == 0:
                            test_content = "...[truncated]"
                        else:
                            test_content = combined_content[:mid] + "...[truncated]"
                        
                        test_tokens = processor.estimate_tokens(test_content, model_name)

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
                    # Fallback to conservative character limit
                    char_limit = int(max_tokens * 2.5)  # More conservative estimate
                    truncated_content = (
                        combined_content[:char_limit]
                        + "\n\n[Content truncated due to length constraints]"
                    )
        else:
            # Try summarization first if LLM is available
            if config.enable_content_summarization and llm:
                logger.info("Attempting to summarize content to fit token limits")
                try:
                    # Use a basic model for summarization to avoid recursive errors
                    from src.llms.llm import get_llm_by_type

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
                    chunks = processor.smart_chunk_content(
                        combined_content, model_name, "auto"
                    )
                    truncated_content = (
                        chunks[0] if chunks else combined_content[: max_tokens * 3]
                    )
            else:
                # Use chunking with aggressive strategy for very long content
                logger.info("Attempting smart content chunking")
                # Use aggressive strategy if content is extremely long
                strategy = (
                    "aggressive"
                    if processor.count_tokens_accurate(
                        combined_content, model_name
                    ).total_tokens
                    > max_tokens * 2
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

                        # Binary search for precise truncation
                        content = truncated_content
                        left, right = 0, len(content)
                        best_content = ""
                        max_iterations = 50  # Prevent infinite loops
                        iteration_count = 0

                        # Handle edge case: empty content
                        if not content:
                            best_content = "...[truncated]"
                        else:
                            while left <= right and iteration_count < max_iterations:
                                iteration_count += 1
                                mid = (left + right) // 2
                                
                                # Handle edge case: mid is 0
                                if mid == 0:
                                    test_content = "...[truncated]"
                                else:
                                    test_content = content[:mid] + "...[truncated]"
                                
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
                            # Fallback to conservative character limit
                            char_limit = int(
                                max_tokens * 2.5
                            )  # More conservative estimate
                        truncated_content = truncated_content[:char_limit]
                else:
                    truncated_content = combined_content[: max_tokens * 3]

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

        # If all smart processing fails, re-raise the original error
        raise error

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
    from src.config.config_loader import config_loader
    from src.utils.content_processor import ContentProcessor

    try:
        # Load configuration
        config = config_loader.create_configuration()

        # Smart chunking is now always enabled
        logger.info("Smart chunking is enabled by default")

        # Create content processor
        processor = ContentProcessor(config.model_token_limits)

        # Try to extract messages and model information from parameters
        messages = None
        model_name = "deepseek-chat"  # Default fallback instead of 'unknown'
        llm = None

        # Check messages in args
        for arg in args:
            if hasattr(arg, "__iter__") and not isinstance(arg, str):
                try:
                    # Check if it's a message list
                    if all(hasattr(item, "content") for item in arg):
                        messages = arg
                        break
                except Exception:
                    continue
            elif hasattr(arg, "ainvoke"):  # Async LLM object
                llm = arg
                if hasattr(arg, "model_name"):
                    model_name = arg.model_name
                elif hasattr(arg, "model"):
                    model_name = arg.model

        # Check messages and model information in kwargs
        if "input" in kwargs and "messages" in kwargs["input"]:
            messages = kwargs["input"]["messages"]
        elif "messages" in kwargs:
            messages = kwargs["messages"]
        if "model" in kwargs:
            model_name = kwargs["model"]

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
                    max_iterations = 50  # Prevent infinite loops
                    iteration_count = 0

                    # Handle edge case: empty content
                    if not content:
                        best_content = "...[truncated]"
                    else:
                        while left <= right and iteration_count < max_iterations:
                            iteration_count += 1
                            mid = (left + right) // 2
                            
                            # Handle edge case: mid is 0
                            if mid == 0:
                                test_content = "...[truncated]"
                            else:
                                test_content = content[:mid] + "...[truncated]"
                            
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
                        char_limit = int(max_tokens * 2.5)  # More conservative estimate
                        truncated_content = truncated_content[:char_limit]
                        logger.warning(
                            f"Applied emergency character-based truncation to {char_limit} characters"
                        )

                # Add a note about truncation
                truncated_content += "\n\n[Note: Content has been significantly truncated due to length constraints. Please provide more specific queries for detailed analysis.]"
            else:
                # Fallback: take first portion based on character count
                char_limit = max_tokens * 3  # Rough estimate: 1 token ≈ 3-4 characters
                truncated_content = (
                    combined_content[:char_limit]
                    + "\n\n[Content truncated due to length constraints]"
                )
        else:
            # Try summarization first if LLM is available
            if config.enable_content_summarization and llm:
                logger.info("Attempting to summarize content to fit token limits")
                try:
                    # Use a basic model for summarization to avoid recursive errors
                    from src.llms.llm import get_llm_by_type

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
                    chunks = processor.smart_chunk_content(
                        combined_content, model_name, "auto"
                    )
                    truncated_content = (
                        chunks[0] if chunks else combined_content[: max_tokens * 3]
                    )
            else:
                # Use chunking with aggressive strategy for very long content
                logger.info("Attempting smart content chunking")
                # Use aggressive strategy if content is extremely long
                strategy = (
                    "aggressive"
                    if processor.count_tokens_accurate(
                        combined_content, model_name
                    ).total_tokens
                    > max_tokens * 2
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
                        char_limit = int(max_tokens * 3)
                        truncated_content = truncated_content[:char_limit]
                else:
                    truncated_content = combined_content[: max_tokens * 3]

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


# Global error handler instance
error_handler = LLMErrorHandler()


async def _evaluate_and_optimize_context_before_call(
    llm_func: Callable, args: tuple, kwargs: dict, operation_name: str, context: str
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
    from src.utils.context_evaluator import get_global_context_evaluator

    try:
        # Get context evaluator
        evaluator = get_global_context_evaluator()

        # Extract messages and model information
        messages = None
        model_name = "deepseek-chat"  # Default fallback instead of 'unknown'

        # Look for messages in kwargs
        if (
            "input" in kwargs
            and isinstance(kwargs["input"], dict)
            and "messages" in kwargs["input"]
        ):
            messages = kwargs["input"]["messages"]
        elif "messages" in kwargs:
            messages = kwargs["messages"]

        # Look for messages in args
        if messages is None:
            for arg in args:
                if isinstance(arg, list) and arg and hasattr(arg[0], "content"):
                    messages = arg
                    break
                elif hasattr(arg, "__iter__") and not isinstance(arg, str):
                    try:
                        if all(hasattr(item, "content") for item in arg):
                            messages = list(arg)
                            break
                    except (TypeError, AttributeError, ValueError) as e:
                        logger.debug(f"Failed to check async message list format: {e}")
                        continue

        # Try to extract model name from LLM function or arguments
        if hasattr(llm_func, "__self__"):
            llm_instance = llm_func.__self__
            if hasattr(llm_instance, "model_name"):
                model_name = llm_instance.model_name
            elif hasattr(llm_instance, "model"):
                model_name = llm_instance.model

        # Look for model name in kwargs
        if model_name == "deepseek-chat":  # Check if still using default
            if "model" in kwargs:
                model_name = kwargs["model"]
            elif (
                "config" in kwargs
                and isinstance(kwargs["config"], dict)
                and "model" in kwargs["config"]
            ):
                model_name = kwargs["config"]["model"]

        # If we found messages, evaluate and optimize
        if messages and isinstance(messages, list) and messages:
            logger.debug(
                f"Evaluating context for {operation_name}: {len(messages)} messages, model: {model_name}"
            )

            # Evaluate context state
            metrics = evaluator.evaluate_context_before_llm_call(
                messages, model_name, f"{operation_name} - {context}"
            )

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
                    from src.utils.content_processor import ContentProcessor
                    from src.config import config_loader
                    
                    # Create processor with proper configuration
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
                        estimated_tokens > model_limit * 0.9
                    ):  # 90% threshold for final check
                        logger.warning(
                            f"Final token check: {estimated_tokens} tokens exceeds 90% of limit ({model_limit})"
                        )
                        # Emergency truncation
                        max_chars = int(
                            model_limit * 0.8 * 3
                        )  # Conservative character limit
                        if len(total_content) > max_chars:
                            truncated_content = (
                                total_content[:max_chars] + "...[truncated]"
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


def _evaluate_and_optimize_context_before_call_sync(
    llm_func: Callable, args: tuple, kwargs: dict, operation_name: str, context: str
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
    from src.utils.context_evaluator import get_global_context_evaluator

    try:
        # Get context evaluator
        evaluator = get_global_context_evaluator()

        # Extract messages and model information (same logic as async version)
        messages = None
        model_name = "deepseek-chat"  # Default fallback instead of 'unknown'

        # Look for messages in kwargs
        if (
            "input" in kwargs
            and isinstance(kwargs["input"], dict)
            and "messages" in kwargs["input"]
        ):
            messages = kwargs["input"]["messages"]
        elif "messages" in kwargs:
            messages = kwargs["messages"]

        # Look for messages in args
        if messages is None:
            for arg in args:
                if isinstance(arg, list) and arg and hasattr(arg[0], "content"):
                    messages = arg
                    break
                elif hasattr(arg, "__iter__") and not isinstance(arg, str):
                    try:
                        if all(hasattr(item, "content") for item in arg):
                            messages = list(arg)
                            break
                    except Exception:
                        continue

        # Try to extract model name
        if hasattr(llm_func, "__self__"):
            llm_instance = llm_func.__self__
            if hasattr(llm_instance, "model_name"):
                model_name = llm_instance.model_name
            elif hasattr(llm_instance, "model"):
                model_name = llm_instance.model

        if model_name == "deepseek-chat":  # Check if still using default
            if "model" in kwargs:
                model_name = kwargs["model"]
            elif (
                "config" in kwargs
                and isinstance(kwargs["config"], dict)
                and "model" in kwargs["config"]
            ):
                model_name = kwargs["config"]["model"]

        # If we found messages, evaluate and optimize
        if messages and isinstance(messages, list) and messages:
            logger.debug(
                f"Evaluating context for {operation_name}: {len(messages)} messages, model: {model_name}"
            )

            # Evaluate context state
            metrics = evaluator.evaluate_context_before_llm_call(
                messages, model_name, f"{operation_name} - {context}"
            )

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
                    from src.utils.content_processor import ContentProcessor
                    from src.config import config_loader
                    
                    # Create processor with proper configuration
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
                        estimated_tokens > model_limit * 0.9
                    ):  # 90% threshold for final check
                        logger.warning(
                            f"Final token check: {estimated_tokens} tokens exceeds 90% of limit ({model_limit})"
                        )
                        # Emergency truncation
                        max_chars = int(
                            model_limit * 0.8 * 3
                        )  # Conservative character limit
                        if len(total_content) > max_chars:
                            truncated_content = (
                                total_content[:max_chars] + "...[truncated]"
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


def handle_llm_errors(operation_name: str = "LLM Operation", context: str = ""):
    """LLM error handling decorator

    Args:
        operation_name: Operation name for logging
        context: Error context information
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                should_skip, fallback_response, needs_smart_processing = (
                    error_handler.handle_error(e, context, operation_name)
                )
                if should_skip and fallback_response:
                    return fallback_response
                elif needs_smart_processing:
                    # For errors that need smart processing, re-raise for upper layer handling
                    raise
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                should_skip, fallback_response, needs_smart_processing = (
                    error_handler.handle_error(e, context, operation_name)
                )
                if should_skip and fallback_response:
                    return fallback_response
                elif needs_smart_processing:
                    # For errors that need smart processing, re-raise for upper layer handling
                    raise
                raise

        # Choose appropriate wrapper based on whether function is coroutine

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def safe_llm_call(
    llm_func: Callable,
    *args,
    operation_name: str = "LLM Call",
    context: str = "",
    max_retries: int = 3,
    enable_smart_processing: bool = True,
    **kwargs,
) -> Any:
    """Safe LLM call function with retry mechanism and smart content processing

    Context evaluation is automatically enabled for all calls to ensure token limits are respected.

    Args:
        llm_func: LLM call function
        *args: Positional arguments
        operation_name: Operation name
        context: Context information
        max_retries: Maximum retry attempts
        enable_smart_processing: Whether to enable smart content processing
        **kwargs: Keyword arguments

    Returns:
        LLM call result or fallback response
    """

    # Apply context evaluation before the call
    try:
        args, kwargs = _evaluate_and_optimize_context_before_call_sync(
            llm_func, args, kwargs, operation_name, context
        )
    except Exception as e:
        logger.warning(
            f"Context evaluation failed: {e}, proceeding with original arguments"
        )

    for attempt in range(max_retries + 1):
        try:
            return llm_func(*args, **kwargs)
        except Exception as e:
            should_skip, fallback_response, needs_smart_processing = (
                error_handler.handle_error(e, context, operation_name)
            )

            if should_skip and fallback_response:
                return fallback_response

            # If smart processing is needed and enabled
            if needs_smart_processing and enable_smart_processing:
                logger.info("Attempting smart content processing for token limit error")
                try:
                    return _handle_content_too_long_error(llm_func, e, *args, **kwargs)
                except Exception as smart_error:
                    logger.error(f"Smart content processing failed: {smart_error}")
                    # Create a new exception with more context information
                    enhanced_error = Exception(f"Smart processing failed: {smart_error}. Original error: {e}")
                    enhanced_error.__cause__ = smart_error
                    # If smart processing fails, continue with original error handling flow
                    # Store the enhanced error for potential debugging
                    logger.debug(f"Enhanced error details: {enhanced_error}")

            error_type = error_handler.classify_error(str(e))

            if error_handler.should_retry_error(error_type) and attempt < max_retries:
                wait_time = 2**attempt  # Exponential backoff
                logger.info(
                    f"Retry {attempt + 1} failed, retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
                continue

            # If not a retryable error or max retries reached, re-raise exception
            raise


async def safe_llm_call_async(
    llm_func: Callable,
    *args,
    operation_name: str = "Async LLM Call",
    context: str = "",
    max_retries: int = 3,
    enable_smart_processing: bool = True,
    **kwargs,
) -> Any:
    """Safe async LLM call function with retry mechanism and smart content processing

    Context evaluation is automatically enabled for all calls to ensure token limits are respected.

    Args:
        llm_func: Async LLM call function
        *args: Positional arguments
        operation_name: Operation name
        context: Context information
        max_retries: Maximum retry attempts
        enable_smart_processing: Whether to enable smart content processing
        **kwargs: Keyword arguments

    Returns:
        LLM call result or fallback response
    """

    # Apply context evaluation before the call
    try:
        args, kwargs = await _evaluate_and_optimize_context_before_call(
            llm_func, args, kwargs, operation_name, context
        )
    except Exception as e:
        logger.warning(
            f"Context evaluation failed: {e}, proceeding with original arguments"
        )

    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(llm_func):
                return await llm_func(*args, **kwargs)
            else:
                return llm_func(*args, **kwargs)
        except Exception as e:
            should_skip, fallback_response, needs_smart_processing = (
                error_handler.handle_error(e, context, operation_name)
            )

            if should_skip and fallback_response:
                return fallback_response

            # If smart processing is needed and enabled
            if needs_smart_processing and enable_smart_processing:
                logger.info("Attempting smart content processing for token limit error")
                try:
                    if asyncio.iscoroutinefunction(llm_func):
                        return await _handle_content_too_long_error_async(
                            llm_func, e, *args, **kwargs
                        )
                    else:
                        return _handle_content_too_long_error(
                            llm_func, e, *args, **kwargs
                        )
                except Exception as smart_error:
                    logger.error(f"Smart content processing failed: {smart_error}")
                    # Create a new exception with more context information
                    enhanced_error = Exception(f"Async smart processing failed: {smart_error}. Original error: {e}")
                    enhanced_error.__cause__ = smart_error
                    # If smart processing fails, continue with original error handling flow
                    # Store the enhanced error for potential debugging
                    logger.debug(f"Enhanced error details: {enhanced_error}")

            error_type = error_handler.classify_error(str(e))

            if error_handler.should_retry_error(error_type) and attempt < max_retries:
                wait_time = 2**attempt  # Exponential backoff
                logger.info(
                    f"Retry {attempt + 1} failed, retrying in {wait_time} seconds..."
                )
                await asyncio.sleep(wait_time)
                continue

            # If not a retryable error or max retries reached, re-raise exception
            raise
