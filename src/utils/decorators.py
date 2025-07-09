# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""
统一装饰器模块
提供项目中常用的装饰器，避免重复定义
"""

import asyncio
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)


def safe_background_task(func: Callable) -> Callable:
    """Decorator to safely handle exceptions in background tasks.
    
    This decorator ensures that any exceptions in background tasks are properly
    caught, logged, and don't cause silent failures or resource leaks.
    
    Args:
        func: The function to wrap
        
    Returns:
        Wrapped function that handles exceptions safely
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            # Log the exception with full context
            logger.error(
                f"Exception in background task '{func.__name__}': {e}",
                exc_info=True,
                extra={
                    "task_name": func.__name__,
                    "args": str(args)[:200],  # Limit args length for logging
                    "kwargs": str(kwargs)[:200],  # Limit kwargs length for logging
                    "error_type": type(e).__name__
                }
            )
            # Don't re-raise to prevent background task failures from affecting main flow
            return None
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the exception with full context
            logger.error(
                f"Exception in background task '{func.__name__}': {e}",
                exc_info=True,
                extra={
                    "task_name": func.__name__,
                    "args": str(args)[:200],  # Limit args length for logging
                    "kwargs": str(kwargs)[:200],  # Limit kwargs length for logging
                    "error_type": type(e).__name__
                }
            )
            # Don't re-raise to prevent background task failures from affecting main flow
            return None
    
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """Decorator to retry function execution on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )
            
            # Re-raise the last exception if all retries failed
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )
            
            # Re-raise the last exception if all retries failed
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_execution_time(operation_name: str = None):
    """Decorator to log function execution time.
    
    Args:
        operation_name: Custom name for the operation (defaults to function name)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            op_name = operation_name or func.__name__
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                logger.info(
                    f"Operation '{op_name}' completed in {execution_time:.3f}s",
                    extra={
                        "operation": op_name,
                        "execution_time": execution_time,
                        "status": "success"
                    }
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Operation '{op_name}' failed after {execution_time:.3f}s: {e}",
                    extra={
                        "operation": op_name,
                        "execution_time": execution_time,
                        "status": "error",
                        "error_type": type(e).__name__
                    }
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            op_name = operation_name or func.__name__
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    f"Operation '{op_name}' completed in {execution_time:.3f}s",
                    extra={
                        "operation": op_name,
                        "execution_time": execution_time,
                        "status": "success"
                    }
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Operation '{op_name}' failed after {execution_time:.3f}s: {e}",
                    extra={
                        "operation": op_name,
                        "execution_time": execution_time,
                        "status": "error",
                        "error_type": type(e).__name__
                    }
                )
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator