"""LLM Error Handling Module

Provides unified LLM error handling mechanisms with support for different error handling strategies.
"""

import logging
import asyncio
import time
from typing import Any, Callable, Optional, Dict, Union
from functools import wraps
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

# Import new tool modules
from src.utils.callback_safety import SafeCallbackManager, global_callback_manager
from src.utils.error_recovery import ErrorRecoveryManager, CircuitBreaker, RecoveryStrategy
from src.utils.structured_logging import get_logger, EventType, PerformanceMetrics, log_performance

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
    UNKNOWN_ERROR = "unknown_error"


class LLMErrorHandler:
    """LLM error handler"""
    
    def __init__(self):
        self.error_patterns = {
            LLMErrorType.DATA_INSPECTION_FAILED: [
                "data_inspection_failed",
                "content safety",
                "content policy",
                "inappropriate content"
            ],
            LLMErrorType.RATE_LIMIT_EXCEEDED: [
                "rate limit",
                "too many requests",
                "rate_limit_exceeded"
            ],
            LLMErrorType.QUOTA_EXCEEDED: [
                "quota exceeded",
                "insufficient quota",
                "quota_exceeded"
            ],
            LLMErrorType.INVALID_API_KEY: [
                "invalid api key",
                "authentication failed",
                "unauthorized"
            ],
            LLMErrorType.MODEL_NOT_FOUND: [
                "model not found",
                "model does not exist",
                "invalid model"
            ],
            LLMErrorType.CONTENT_TOO_LONG: [
                "content too long",
                "token limit exceeded",
                "maximum context length",
                "range of input length should be",
                "input length should be",
                "invalidparameter: range of input length"
            ],
            LLMErrorType.NETWORK_ERROR: [
                "network error",
                "connection error",
                "timeout",
                "connection timeout"
            ],
            LLMErrorType.TIMEOUT_ERROR: [
                "timeout",
                "request timeout",
                "read timeout"
            ]
        }
        
        self.fallback_responses = {
            LLMErrorType.DATA_INSPECTION_FAILED: "Unable to process this request due to content safety restrictions. Please try with different content.",
            LLMErrorType.RATE_LIMIT_EXCEEDED: "Request rate limit exceeded, please try again later.",
            LLMErrorType.QUOTA_EXCEEDED: "API quota exhausted, please check your account quota.",
            LLMErrorType.INVALID_API_KEY: "Invalid API key, please check your configuration.",
            LLMErrorType.MODEL_NOT_FOUND: "Specified model does not exist, please check model configuration.",
            LLMErrorType.CONTENT_TOO_LONG: "Content too long, please shorten the input.",
            LLMErrorType.NETWORK_ERROR: "Network connection error, please check your network connection.",
            LLMErrorType.TIMEOUT_ERROR: "Request timeout, please try again later.",
            LLMErrorType.UNKNOWN_ERROR: "Unknown error occurred, please try again later."
        }
        
        self.skip_errors = {
            LLMErrorType.DATA_INSPECTION_FAILED,  # Content safety errors, skip and continue
        }
        
        self.retry_errors = {
            LLMErrorType.RATE_LIMIT_EXCEEDED,
            LLMErrorType.NETWORK_ERROR,
            LLMErrorType.TIMEOUT_ERROR,
        }
        
        self.fatal_errors = {
            LLMErrorType.INVALID_API_KEY,
            LLMErrorType.QUOTA_EXCEEDED,
            LLMErrorType.MODEL_NOT_FOUND,
        }
        
        # Error types that need smart processing
        self.smart_processing_errors = {
            LLMErrorType.CONTENT_TOO_LONG,
        }
    
    def classify_error(self, error_message) -> str:
        """Classify error type based on error message"""
        # Handle both string and Exception objects
        if isinstance(error_message, Exception):
            error_message = str(error_message)
        error_message_lower = error_message.lower()
        
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
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
        fallback_text = self.fallback_responses.get(error_type, self.fallback_responses[LLMErrorType.UNKNOWN_ERROR])
        
        if context:
            fallback_text = f"{fallback_text} (Context: {context})"
        
        return AIMessage(content=fallback_text)
    
    def needs_smart_processing(self, error_type: str) -> bool:
        """Determine if smart processing is needed"""
        return error_type in self.smart_processing_errors
    
    def handle_error(self, 
                    error: Exception, 
                    context: str = "",
                    operation_name: str = "LLM Operation") -> tuple[bool, Optional[AIMessage], bool]:
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
        
        logger.warning(f"{operation_name} execution error: {error_message} (Error type: {error_type})")
        
        # Check if smart processing is needed
        if self.needs_smart_processing(error_type):
            logger.info(f"Detected {error_type} error, triggering smart content processing")
            return False, None, True
        
        elif self.should_skip_error(error_type):
            logger.info(f"Skipping {error_type} error, using fallback response to continue execution")
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


def _handle_content_too_long_error(llm_func: Callable, error: Exception, *args, **kwargs) -> Any:
    """Smart processing function for handling content too long errors"""
    from src.config.config_loader import config_loader
    from src.utils.content_processor import ContentProcessor
    
    try:
        # Load configuration
        config = config_loader.create_configuration()
        
        if not config.enable_smart_chunking:
            logger.info("Smart chunking is disabled, re-raising original error")
            raise error
        
        # Create content processor
        processor = ContentProcessor(config.model_token_limits)
        
        # Try to extract messages and model information from parameters
        messages = None
        model_name = "unknown"
        llm = None
        
        # Check messages in args
        for arg in args:
            if hasattr(arg, '__iter__') and not isinstance(arg, str):
                try:
                    # Check if it's a message list
                    if all(hasattr(item, 'content') for item in arg):
                        messages = arg
                        break
                except:
                    continue
            elif hasattr(arg, 'invoke'):  # LLM object
                llm = arg
                if hasattr(arg, 'model_name'):
                    model_name = arg.model_name
                elif hasattr(arg, 'model'):
                    model_name = arg.model
        
        # Check messages and model information in kwargs
        if 'messages' in kwargs:
            messages = kwargs['messages']
        if 'model' in kwargs:
            model_name = kwargs['model']
        
        if not messages:
            logger.warning("Could not extract messages from function arguments")
            raise error
        
        # Extract text content
        content_parts = []
        for msg in messages:
            if hasattr(msg, 'content'):
                content_parts.append(str(msg.content))
        
        combined_content = "\n\n".join(content_parts)
        
        # If content summarization is enabled
        if config.enable_content_summarization and llm:
            logger.info("Attempting to summarize content to fit token limits")
            summarized_content = processor.summarize_content(
                combined_content, llm, model_name, config.summary_type
            )
            
            # Create new message list, replace with summarized content
            new_messages = []
            for msg in messages[:-1]:  # Keep all messages except the last one
                new_messages.append(msg)
            
            # Replace the last message with summarized content
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, 'content'):
                    new_content = summarized_content
                    if hasattr(last_msg, '__class__'):
                        new_msg = last_msg.__class__(content=new_content)
                    else:
                        new_msg = HumanMessage(content=new_content)
                    new_messages.append(new_msg)
            
            # Update parameters
            new_args = list(args)
            for i, arg in enumerate(new_args):
                if arg is messages:
                    new_args[i] = new_messages
                    break
            
            if 'messages' in kwargs:
                kwargs['messages'] = new_messages
            
            # Re-invoke the function
            return llm_func(*new_args, **kwargs)
        
        else:
            # If no LLM or summarization not enabled, try simple truncation
            logger.info("Attempting simple content truncation")
            chunks = processor.smart_chunk_content(combined_content, model_name, config.chunk_strategy)
            
            if chunks:
                # Use the first chunk
                truncated_content = chunks[0]
                
                # Update messages
                new_messages = []
                for msg in messages[:-1]:
                    new_messages.append(msg)
                
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, '__class__'):
                        new_msg = last_msg.__class__(content=truncated_content)
                    else:
                        new_msg = HumanMessage(content=truncated_content)
                    new_messages.append(new_msg)
                
                # Update parameters
                new_args = list(args)
                for i, arg in enumerate(new_args):
                    if arg is messages:
                        new_args[i] = new_messages
                        break
                
                if 'messages' in kwargs:
                    kwargs['messages'] = new_messages
                
                # Re-invoke the function
                return llm_func(*new_args, **kwargs)
        
        # If all smart processing fails, re-raise the original error
        raise error
        
    except Exception as e:
        logger.error(f"Smart content processing failed: {e}")
        raise error


async def _handle_content_too_long_error_async(llm_func: Callable, error: Exception, *args, **kwargs) -> Any:
    """Async smart processing function for handling content too long errors"""
    import asyncio
    
    # For async functions, we can reuse the logic from the sync version
    # because content processing itself doesn't require async operations
    try:
        result = _handle_content_too_long_error(llm_func, error, *args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
    except Exception as e:
        raise error


# Global error handler instance
error_handler = LLMErrorHandler()


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
                should_skip, fallback_response, needs_smart_processing = error_handler.handle_error(
                    e, context, operation_name
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
                should_skip, fallback_response, needs_smart_processing = error_handler.handle_error(
                    e, context, operation_name
                )
                if should_skip and fallback_response:
                    return fallback_response
                elif needs_smart_processing:
                    # For errors that need smart processing, re-raise for upper layer handling
                    raise
                raise
        
        # Choose appropriate wrapper based on whether function is coroutine
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def safe_llm_call(llm_func: Callable, 
                  *args, 
                  operation_name: str = "LLM Call",
                  context: str = "",
                  max_retries: int = 3,
                  enable_smart_processing: bool = True,
                  **kwargs) -> Any:
    """Safe LLM call function with retry mechanism and smart content processing
    
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
    import time
    
    for attempt in range(max_retries + 1):
        try:
            return llm_func(*args, **kwargs)
        except Exception as e:
            should_skip, fallback_response, needs_smart_processing = error_handler.handle_error(
                e, context, operation_name
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
                    # If smart processing fails, continue with original error handling flow
            
            error_type = error_handler.classify_error(str(e))
            
            if error_handler.should_retry_error(error_type) and attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retry {attempt + 1} failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            
            # If not a retryable error or max retries reached, re-raise exception
            raise


async def safe_llm_call_async(llm_func: Callable, 
                             *args, 
                             operation_name: str = "Async LLM Call",
                             context: str = "",
                             max_retries: int = 3,
                             enable_smart_processing: bool = True,
                             **kwargs) -> Any:
    """Safe async LLM call function with retry mechanism and smart content processing
    
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
    import asyncio
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(llm_func):
                return await llm_func(*args, **kwargs)
            else:
                return llm_func(*args, **kwargs)
        except Exception as e:
            should_skip, fallback_response, needs_smart_processing = error_handler.handle_error(
                e, context, operation_name
            )
            
            if should_skip and fallback_response:
                return fallback_response
            
            # If smart processing is needed and enabled
            if needs_smart_processing and enable_smart_processing:
                logger.info("Attempting smart content processing for token limit error")
                try:
                    if asyncio.iscoroutinefunction(llm_func):
                        return await _handle_content_too_long_error_async(llm_func, e, *args, **kwargs)
                    else:
                        return _handle_content_too_long_error(llm_func, e, *args, **kwargs)
                except Exception as smart_error:
                    logger.error(f"Smart content processing failed: {smart_error}")
                    # If smart processing fails, continue with original error handling flow
            
            error_type = error_handler.classify_error(str(e))
            
            if error_handler.should_retry_error(error_type) and attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retry {attempt + 1} failed, retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue
            
            # If not a retryable error or max retries reached, re-raise exception
            raise