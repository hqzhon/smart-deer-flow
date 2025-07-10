"""LLM Error Handling Module

Provides unified LLM error handling mechanisms with support for different error handling strategies.
"""

import asyncio
import time
from typing import Any, Callable, Optional
from functools import wraps
from langchain_core.messages import AIMessage

from src.utils.structured_logging import get_logger
from src.llms.base_error_handler import BaseLLMErrorHandler
from src.llms.llm_context_optimizer import LLMContextOptimizer

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


class LLMErrorHandler(BaseLLMErrorHandler):
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


def handle_llm_errors(operation_name: str = "LLM Operation", context: str = "", error_handler_instance: Optional[BaseLLMErrorHandler] = None):
    """LLM error handling decorator

    Args:
        operation_name: Operation name for logging
        context: Error context information
        error_handler_instance: Optional, an instance of BaseLLMErrorHandler. If None, a default LLMErrorHandler will be used.
    """
    if error_handler_instance is None:
        error_handler_instance = LLMErrorHandler()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                should_skip, fallback_response, needs_smart_processing = (
                    error_handler_instance.handle_error(e, context, operation_name)
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
                    error_handler_instance.handle_error(e, context, operation_name)
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


def _execute_safe_llm_call_sync(
    llm_func: Callable,
    args: tuple,
    kwargs: dict,
    operation_name: str,
    context: str,
    max_retries: int,
    enable_smart_processing: bool,
    error_handler_instance: BaseLLMErrorHandler,
    context_optimizer_instance: LLMContextOptimizer
) -> Any:
    """Synchronous version of safe LLM call with error handling"""
    # Apply context evaluation before the call
    try:
        args, kwargs = context_optimizer_instance.evaluate_and_optimize_context_before_call_sync(
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
                error_handler_instance.handle_error(e, context, operation_name)
            )

            if should_skip and fallback_response:
                return fallback_response

            # If smart processing is needed and enabled
            if needs_smart_processing and enable_smart_processing:
                logger.info("Attempting smart content processing for token limit error")
                try:
                    return context_optimizer_instance._handle_content_too_long_error(llm_func, e, *args, **kwargs)
                except Exception as smart_error:
                    logger.error(f"Smart content processing failed: {smart_error}")
                    # Create a new exception with more context information
                    enhanced_error = Exception(f"Smart processing failed: {smart_error}. Original error: {e}")
                    enhanced_error.__cause__ = smart_error
                    # If smart processing fails, continue with original error handling flow
                    # Store the enhanced error for potential debugging
                    logger.debug(f"Enhanced error details: {enhanced_error}")

            error_type = error_handler_instance.classify_error(str(e))

            if error_handler_instance.should_retry_error(error_type) and attempt < max_retries:
                wait_time = 2**attempt  # Exponential backoff
                logger.info(
                    f"Retry {attempt + 1} failed, retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
                continue

            # If not a retryable error or max retries reached, re-raise exception
            raise


async def _execute_safe_llm_call_async(
    llm_func: Callable,
    args: tuple,
    kwargs: dict,
    operation_name: str,
    context: str,
    max_retries: int,
    enable_smart_processing: bool,
    error_handler_instance: BaseLLMErrorHandler,
    context_optimizer_instance: LLMContextOptimizer
) -> Any:
    """Asynchronous version of safe LLM call with error handling"""
    # Apply context evaluation before the call
    try:
        args, kwargs = await context_optimizer_instance.evaluate_and_optimize_context_before_call(
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
                error_handler_instance.handle_error(e, context, operation_name)
            )

            if should_skip and fallback_response:
                return fallback_response

            # If smart processing is needed and enabled
            if needs_smart_processing and enable_smart_processing:
                logger.info("Attempting smart content processing for token limit error")
                try:
                    if asyncio.iscoroutinefunction(llm_func):
                        return await context_optimizer_instance._handle_content_too_long_error_async(llm_func, e, *args, **kwargs)
                    else:
                        return context_optimizer_instance._handle_content_too_long_error(llm_func, e, *args, **kwargs)
                except Exception as smart_error:
                    logger.error(f"Async smart content processing failed: {smart_error}")
                    # Create a new exception with more context information
                    enhanced_error = Exception(f"Async smart processing failed: {smart_error}. Original error: {e}")
                    enhanced_error.__cause__ = smart_error
                    # If smart processing fails, continue with original error handling flow
                    # Store the enhanced error for potential debugging
                    logger.debug(f"Enhanced error details: {enhanced_error}")

            error_type = error_handler_instance.classify_error(str(e))

            if error_handler_instance.should_retry_error(error_type) and attempt < max_retries:
                wait_time = 2**attempt  # Exponential backoff
                logger.info(
                    f"Retry {attempt + 1} failed, retrying in {wait_time} seconds..."
                )
                await asyncio.sleep(wait_time)
                continue

            # If not a retryable error or max retries reached, re-raise exception
            raise


def safe_llm_call(
    llm_func: Callable,
    *args,
    operation_name: str = "LLM Call",
    context: str = "",
    max_retries: int = 3,
    enable_smart_processing: bool = True,
    error_handler_instance: Optional[BaseLLMErrorHandler] = None,
    context_optimizer_instance: Optional[LLMContextOptimizer] = None,
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
        error_handler_instance: Optional, an instance of BaseLLMErrorHandler. If None, a default LLMErrorHandler will be used.
        context_optimizer_instance: Optional, an instance of LLMContextOptimizer. If None, a default LLMContextOptimizer will be used.
        **kwargs: Keyword arguments

    Returns:
        LLM call result or fallback response
    """
    if error_handler_instance is None:
        error_handler_instance = LLMErrorHandler()
    if context_optimizer_instance is None:
        context_optimizer_instance = LLMContextOptimizer()

    return _execute_safe_llm_call_sync(
        llm_func,
        args,
        kwargs,
        operation_name,
        context,
        max_retries,
        enable_smart_processing,
        error_handler_instance,
        context_optimizer_instance
    )


async def safe_llm_call_async(
    llm_func: Callable,
    *args,
    operation_name: str = "Async LLM Call",
    context: str = "",
    max_retries: int = 3,
    enable_smart_processing: bool = True,
    error_handler_instance: Optional[BaseLLMErrorHandler] = None,
    context_optimizer_instance: Optional[LLMContextOptimizer] = None,
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
        error_handler_instance: Optional, an instance of BaseLLMErrorHandler. If None, a default LLMErrorHandler will be used.
        context_optimizer_instance: Optional, an instance of LLMContextOptimizer. If None, a default LLMContextOptimizer will be used.
        **kwargs: Keyword arguments

    Returns:
        LLM call result or fallback response
    """
    if error_handler_instance is None:
        error_handler_instance = LLMErrorHandler()
    if context_optimizer_instance is None:
        context_optimizer_instance = LLMContextOptimizer()

    return await _execute_safe_llm_call_async(
        llm_func,
        args,
        kwargs,
        operation_name,
        context,
        max_retries,
        enable_smart_processing,
        error_handler_instance,
        context_optimizer_instance
    )


# Create a global instance for easy importing
error_handler = LLMErrorHandler()