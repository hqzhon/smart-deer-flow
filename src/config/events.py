# -*- coding: utf-8 -*-
"""
Configuration Events
Provides event system for configuration changes
"""

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigEventType(Enum):
    """Configuration event types"""
    LOADED = "loaded"
    CHANGED = "changed"
    RELOADED = "reloaded"
    VALIDATED = "validated"
    VALIDATION_FAILED = "validation_failed"
    CACHE_CLEARED = "cache_cleared"
    ERROR = "error"


@dataclass
class ConfigEvent:
    """Configuration event data"""
    event_type: ConfigEventType
    timestamp: datetime
    source: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class IConfigEventHandler(ABC):
    """Interface for configuration event handlers"""
    
    @abstractmethod
    def handle_event(self, event: ConfigEvent) -> None:
        """Handle configuration event"""
        pass
    
    @abstractmethod
    def get_supported_events(self) -> Set[ConfigEventType]:
        """Get supported event types"""
        pass


class ConfigEventBus:
    """Event bus for configuration events"""
    
    def __init__(self):
        self._handlers: Dict[ConfigEventType, List[IConfigEventHandler]] = {}
        self._callbacks: Dict[ConfigEventType, List[Callable[[ConfigEvent], None]]] = {}
        self._lock = threading.RLock()
        self._event_history: List[ConfigEvent] = []
        self._max_history = 100
    
    def register_handler(self, handler: IConfigEventHandler) -> None:
        """Register event handler"""
        with self._lock:
            supported_events = handler.get_supported_events()
            for event_type in supported_events:
                if event_type not in self._handlers:
                    self._handlers[event_type] = []
                self._handlers[event_type].append(handler)
            
            logger.debug(f"Registered handler {handler.__class__.__name__} for events: {supported_events}")
    
    def unregister_handler(self, handler: IConfigEventHandler) -> None:
        """Unregister event handler"""
        with self._lock:
            for event_type, handlers in self._handlers.items():
                if handler in handlers:
                    handlers.remove(handler)
            
            logger.debug(f"Unregistered handler {handler.__class__.__name__}")
    
    def register_callback(self, event_type: ConfigEventType, 
                         callback: Callable[[ConfigEvent], None]) -> None:
        """Register event callback function"""
        with self._lock:
            if event_type not in self._callbacks:
                self._callbacks[event_type] = []
            self._callbacks[event_type].append(callback)
            
            logger.debug(f"Registered callback for event type: {event_type}")
    
    def unregister_callback(self, event_type: ConfigEventType, 
                           callback: Callable[[ConfigEvent], None]) -> None:
        """Unregister event callback function"""
        with self._lock:
            if event_type in self._callbacks and callback in self._callbacks[event_type]:
                self._callbacks[event_type].remove(callback)
                logger.debug(f"Unregistered callback for event type: {event_type}")
    
    def publish(self, event: ConfigEvent) -> None:
        """Publish configuration event"""
        with self._lock:
            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
            
            logger.debug(f"Publishing event: {event.event_type} from {event.source}")
            
            # Notify handlers
            handlers = self._handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    handler.handle_event(event)
                except Exception as e:
                    logger.error(f"Error in event handler {handler.__class__.__name__}: {e}")
            
            # Notify callbacks
            callbacks = self._callbacks.get(event.event_type, [])
            for callback in callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
    
    def get_event_history(self, event_type: Optional[ConfigEventType] = None, 
                         limit: Optional[int] = None) -> List[ConfigEvent]:
        """Get event history"""
        with self._lock:
            events = self._event_history
            
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            if limit:
                events = events[-limit:]
            
            return events.copy()
    
    def clear_history(self) -> None:
        """Clear event history"""
        with self._lock:
            self._event_history.clear()
            logger.debug("Cleared event history")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        with self._lock:
            event_counts = {}
            for event in self._event_history:
                event_type = event.event_type.value
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            return {
                'total_events': len(self._event_history),
                'event_counts': event_counts,
                'registered_handlers': sum(len(handlers) for handlers in self._handlers.values()),
                'registered_callbacks': sum(len(callbacks) for callbacks in self._callbacks.values())
            }


class LoggingEventHandler(IConfigEventHandler):
    """Event handler that logs configuration events"""
    
    def __init__(self, log_level: int = logging.INFO):
        self._log_level = log_level
    
    def handle_event(self, event: ConfigEvent) -> None:
        """Handle configuration event by logging"""
        message = f"Config event: {event.event_type.value} from {event.source}"
        
        if event.data:
            message += f" with data: {event.data}"
        
        if event.error:
            message += f" with error: {event.error}"
            logger.error(message)
        else:
            logger.log(self._log_level, message)
    
    def get_supported_events(self) -> Set[ConfigEventType]:
        """Get all supported event types"""
        return set(ConfigEventType)


class CacheInvalidationHandler(IConfigEventHandler):
    """Event handler that invalidates cache on configuration changes"""
    
    def __init__(self, cache_invalidator: Callable[[], None]):
        self._cache_invalidator = cache_invalidator
    
    def handle_event(self, event: ConfigEvent) -> None:
        """Handle configuration event by invalidating cache"""
        if event.event_type in {ConfigEventType.CHANGED, ConfigEventType.RELOADED}:
            try:
                self._cache_invalidator()
                logger.debug("Cache invalidated due to configuration change")
            except Exception as e:
                logger.error(f"Error invalidating cache: {e}")
    
    def get_supported_events(self) -> Set[ConfigEventType]:
        """Get supported event types"""
        return {ConfigEventType.CHANGED, ConfigEventType.RELOADED}


class ValidationEventHandler(IConfigEventHandler):
    """Event handler that triggers validation on configuration changes"""
    
    def __init__(self, validator: Callable[[], bool]):
        self._validator = validator
    
    def handle_event(self, event: ConfigEvent) -> None:
        """Handle configuration event by triggering validation"""
        if event.event_type in {ConfigEventType.LOADED, ConfigEventType.CHANGED, ConfigEventType.RELOADED}:
            try:
                is_valid = self._validator()
                validation_event = ConfigEvent(
                    event_type=ConfigEventType.VALIDATED if is_valid else ConfigEventType.VALIDATION_FAILED,
                    timestamp=datetime.now(),
                    source="ValidationEventHandler",
                    data={'original_event': event.event_type.value, 'is_valid': is_valid}
                )
                # Note: We don't publish the validation event here to avoid circular dependencies
                # The validator should handle publishing if needed
            except Exception as e:
                logger.error(f"Error during configuration validation: {e}")
    
    def get_supported_events(self) -> Set[ConfigEventType]:
        """Get supported event types"""
        return {ConfigEventType.LOADED, ConfigEventType.CHANGED, ConfigEventType.RELOADED}


# Global event bus instance
_event_bus = ConfigEventBus()


def get_config_event_bus() -> ConfigEventBus:
    """Get global configuration event bus"""
    return _event_bus


def publish_config_event(event_type: ConfigEventType, source: str, 
                        data: Optional[Dict[str, Any]] = None, 
                        error: Optional[Exception] = None) -> None:
    """Publish configuration event"""
    event = ConfigEvent(
        event_type=event_type,
        timestamp=datetime.now(),
        source=source,
        data=data,
        error=error
    )
    get_config_event_bus().publish(event)


def register_config_event_handler(handler: IConfigEventHandler) -> None:
    """Register configuration event handler"""
    get_config_event_bus().register_handler(handler)


def register_config_event_callback(event_type: ConfigEventType, 
                                  callback: Callable[[ConfigEvent], None]) -> None:
    """Register configuration event callback"""
    get_config_event_bus().register_callback(event_type, callback)