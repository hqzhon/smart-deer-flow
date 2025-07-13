# -*- coding: utf-8 -*-
"""
Callback Safety Mode Implementation
Provides safe callback handling mechanism to prevent errors caused by None callback functions
"""

import logging
import asyncio
from typing import Callable, Optional, Any, Dict, List
from functools import wraps
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CallbackResult:
    """Callback execution result"""

    success: bool
    result: Any = None
    error: Optional[Exception] = None
    callback_name: str = "unknown"


class SafeCallbackManager:
    """Safe callback manager"""

    def __init__(self):
        self.callbacks: Dict[str, List[Callable]] = {}
        self.default_callbacks: Dict[str, Callable] = {}
        self.callback_stats: Dict[str, Dict[str, int]] = {}

    def register_callback(
        self, event_type: str, callback: Optional[Callable], callback_name: str = None
    ):
        """Register callback function"""
        if callback is None:
            logger.warning(
                f"Attempted to register None callback for event {event_type}"
            )
            return

        if not callable(callback):
            logger.error(
                f"Attempted to register non-callable object for event {event_type}"
            )
            return

        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
            self.callback_stats[event_type] = {"success": 0, "failure": 0}

        self.callbacks[event_type].append(callback)
        logger.debug(
            f"Registered callback {callback_name or 'anonymous'} for event {event_type}"
        )

    def register_default_callback(self, event_type: str, callback: Callable):
        """Register default callback function"""
        if callback is None or not callable(callback):
            logger.error(f"Invalid default callback for event {event_type}")
            return

        self.default_callbacks[event_type] = callback
        logger.debug(f"Registered default callback for event {event_type}")

    def safe_call(self, event_type: str, *args, **kwargs) -> List[CallbackResult]:
        """Safely call callback functions"""
        results = []

        # Get registered callback functions
        callbacks = self.callbacks.get(event_type, [])

        if not callbacks:
            # If no callbacks are registered, try to use default callback
            default_callback = self.default_callbacks.get(event_type)
            if default_callback:
                callbacks = [default_callback]
                logger.debug(f"Using default callback for event {event_type}")
            else:
                logger.debug(f"No callbacks registered for event {event_type}")
                return results

        # Execute all callbacks
        for i, callback in enumerate(callbacks):
            result = self._execute_callback(callback, event_type, i, *args, **kwargs)
            results.append(result)

            # Update statistics
            if result.success:
                self.callback_stats[event_type]["success"] += 1
            else:
                self.callback_stats[event_type]["failure"] += 1

        return results

    async def safe_call_async(
        self, event_type: str, *args, **kwargs
    ) -> List[CallbackResult]:
        """Safely call callback functions asynchronously"""
        results = []

        # Get registered callback functions
        callbacks = self.callbacks.get(event_type, [])

        if not callbacks:
            # If no callbacks are registered, try to use default callback
            default_callback = self.default_callbacks.get(event_type)
            if default_callback:
                callbacks = [default_callback]
                logger.debug(f"Using default callback for event {event_type}")
            else:
                logger.debug(f"No callbacks registered for event {event_type}")
                return results

        # Execute all callbacks
        for i, callback in enumerate(callbacks):
            result = await self._execute_callback_async(
                callback, event_type, i, *args, **kwargs
            )
            results.append(result)

            # Update statistics
            if result.success:
                self.callback_stats[event_type]["success"] += 1
            else:
                self.callback_stats[event_type]["failure"] += 1

        return results

    def _execute_callback(
        self, callback: Callable, event_type: str, index: int, *args, **kwargs
    ) -> CallbackResult:
        """Execute single callback function"""
        callback_name = f"{event_type}_{index}"

        try:
            if callback is None:
                raise ValueError("Callback is None")

            if not callable(callback):
                raise ValueError("Callback is not callable")

            result = callback(*args, **kwargs)

            logger.debug(f"Callback {callback_name} executed successfully")
            return CallbackResult(
                success=True, result=result, callback_name=callback_name
            )

        except Exception as e:
            logger.error(f"Callback {callback_name} failed: {e}")
            return CallbackResult(success=False, error=e, callback_name=callback_name)

    async def _execute_callback_async(
        self, callback: Callable, event_type: str, index: int, *args, **kwargs
    ) -> CallbackResult:
        """Execute single callback function asynchronously"""
        callback_name = f"{event_type}_{index}"

        try:
            if callback is None:
                raise ValueError("Callback is None")

            if not callable(callback):
                raise ValueError("Callback is not callable")

            if asyncio.iscoroutinefunction(callback):
                result = await callback(*args, **kwargs)
            else:
                result = callback(*args, **kwargs)

            logger.debug(f"Async callback {callback_name} executed successfully")
            return CallbackResult(
                success=True, result=result, callback_name=callback_name
            )

        except Exception as e:
            logger.error(f"Async callback {callback_name} failed: {e}")
            return CallbackResult(success=False, error=e, callback_name=callback_name)

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get callback statistics"""
        return self.callback_stats.copy()

    def clear_callbacks(self, event_type: Optional[str] = None):
        """Clear callback functions"""
        if event_type:
            if event_type in self.callbacks:
                del self.callbacks[event_type]
            if event_type in self.callback_stats:
                del self.callback_stats[event_type]
        else:
            self.callbacks.clear()
            self.callback_stats.clear()


# Global callback manager instance
global_callback_manager = SafeCallbackManager()


def safe_callback(event_type: str, callback_name: str = None):
    """Safe callback decorator"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if func is None:
                    logger.warning(
                        f"Attempted to call None function for event {event_type}"
                    )
                    return None

                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback {callback_name or func.__name__} failed: {e}")
                return None

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                if func is None:
                    logger.warning(
                        f"Attempted to call None function for event {event_type}"
                    )
                    return None

                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Async callback {callback_name or func.__name__} failed: {e}"
                )
                return None

        # Choose appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator


def ensure_callback_safety(
    callback: Optional[Callable], default_callback: Optional[Callable] = None
) -> Optional[Callable]:
    """Ensure callback function safety"""
    if callback is None:
        if default_callback is not None:
            logger.debug("Using default callback as fallback")
            return default_callback
        else:
            logger.debug("No callback provided and no default available")
            return None

    if not callable(callback):
        logger.error("Provided callback is not callable")
        if default_callback is not None:
            logger.debug("Using default callback as fallback")
            return default_callback
        return None

    return callback
