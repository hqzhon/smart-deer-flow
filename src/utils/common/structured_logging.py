# -*- coding: utf-8 -*-
"""
Structured logging system
Provides unified log format, context tracking and performance monitoring capabilities
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from functools import wraps
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import threading


class LogLevel(Enum):
    """Log level"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """Event type"""

    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SECURITY = "security"
    SYSTEM = "system"


@dataclass
class LogContext:
    """Log context"""

    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PerformanceMetrics:
    """Performance metrics"""

    duration_ms: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    operation_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class LogEntry:
    """Log entry"""

    timestamp: str
    level: str
    message: str
    event_type: str
    context: Dict[str, Any]
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


# Context variable
log_context: ContextVar[Optional[LogContext]] = ContextVar("log_context", default=None)


class StructuredLogger:
    """Structured logger"""

    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        self.name = name
        self.level = level
        self._logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        """Setup logger"""
        # Clear existing handlers
        self._logger.handlers.clear()

        # Create handler
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())

        # Set level
        level_mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }

        self._logger.setLevel(level_mapping[self.level])
        handler.setLevel(level_mapping[self.level])

        self._logger.addHandler(handler)
        self._logger.propagate = False

    def _create_log_entry(
        self,
        level: LogLevel,
        message: str,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        performance: Optional[PerformanceMetrics] = None,
        tags: Optional[List[str]] = None,
    ) -> LogEntry:
        """Create log entry"""
        # Get current context
        context = log_context.get()
        context_dict = context.to_dict() if context else {}

        # Add thread information
        context_dict["thread_id"] = threading.current_thread().ident
        context_dict["thread_name"] = threading.current_thread().name

        # Handle error information
        error_dict = None
        if error:
            error_dict = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }

        # Handle performance metrics
        performance_dict = None
        if performance:
            performance_dict = performance.to_dict()

        return LogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=level.value,
            message=message,
            event_type=event_type.value,
            context=context_dict,
            data=data,
            error=error_dict,
            performance=performance_dict,
            tags=tags,
        )

    def debug(
        self,
        message: str,
        event_type: EventType = EventType.SYSTEM,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Debug log"""
        entry = self._create_log_entry(
            LogLevel.DEBUG, message, event_type, data=data, tags=tags
        )
        self._logger.debug(entry.to_json())

    def info(
        self,
        message: str,
        event_type: EventType = EventType.BUSINESS,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Info log"""
        entry = self._create_log_entry(
            LogLevel.INFO, message, event_type, data=data, tags=tags
        )
        self._logger.info(entry.to_json())

    def warning(
        self,
        message: str,
        event_type: EventType = EventType.SYSTEM,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Warning log"""
        entry = self._create_log_entry(
            LogLevel.WARNING, message, event_type, data=data, tags=tags
        )
        self._logger.warning(entry.to_json())

    def error(
        self,
        message: str,
        error: Optional[Exception] = None,
        event_type: EventType = EventType.ERROR,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Error log"""
        entry = self._create_log_entry(
            LogLevel.ERROR, message, event_type, data=data, error=error, tags=tags
        )
        self._logger.error(entry.to_json())

    def critical(
        self,
        message: str,
        error: Optional[Exception] = None,
        event_type: EventType = EventType.ERROR,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Critical error log"""
        entry = self._create_log_entry(
            LogLevel.CRITICAL, message, event_type, data=data, error=error, tags=tags
        )
        self._logger.critical(entry.to_json())

    def performance(
        self,
        message: str,
        metrics: PerformanceMetrics,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Performance log"""
        entry = self._create_log_entry(
            LogLevel.INFO,
            message,
            EventType.PERFORMANCE,
            data=data,
            performance=metrics,
            tags=tags,
        )
        self._logger.info(entry.to_json())

    def request(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Request log"""
        entry = self._create_log_entry(
            LogLevel.INFO, message, EventType.REQUEST, data=data, tags=tags
        )
        self._logger.info(entry.to_json())

    def response(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Response log"""
        entry = self._create_log_entry(
            LogLevel.INFO, message, EventType.RESPONSE, data=data, tags=tags
        )
        self._logger.info(entry.to_json())

    def security(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Security log"""
        entry = self._create_log_entry(
            LogLevel.WARNING, message, EventType.SECURITY, data=data, tags=tags
        )
        self._logger.warning(entry.to_json())


class StructuredFormatter(logging.Formatter):
    """Structured log formatter"""

    def format(self, record):
        # If message is already in JSON format, return directly
        try:
            json.loads(record.getMessage())
            return record.getMessage()
        except (json.JSONDecodeError, ValueError):
            # If not JSON, create simple structured format
            entry = LogEntry(
                timestamp=datetime.utcnow().isoformat() + "Z",
                level=record.levelname,
                message=record.getMessage(),
                event_type=EventType.SYSTEM.value,
                context={
                    "logger": record.name,
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                },
            )
            return entry.to_json()


class LogContextManager:
    """Log context manager"""

    def __init__(self, context: LogContext):
        self.context = context
        self.token = None

    def __enter__(self):
        self.token = log_context.set(self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            log_context.reset(self.token)


def with_log_context(
    request_id: str = None,
    user_id: str = None,
    session_id: str = None,
    operation: str = None,
    component: str = None,
    trace_id: str = None,
):
    """Log context decorator"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = LogContext(
                request_id=request_id or str(uuid.uuid4()),
                user_id=user_id,
                session_id=session_id,
                operation=operation or func.__name__,
                component=component or func.__module__,
                trace_id=trace_id or str(uuid.uuid4()),
            )

            with LogContextManager(context):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = LogContext(
                request_id=request_id or str(uuid.uuid4()),
                user_id=user_id,
                session_id=session_id,
                operation=operation or func.__name__,
                component=component or func.__module__,
                trace_id=trace_id or str(uuid.uuid4()),
            )

            with LogContextManager(context):
                return await func(*args, **kwargs)

        if hasattr(func, "__call__") and hasattr(func, "__await__"):
            return async_wrapper
        else:
            return wrapper

    return decorator


def log_performance(
    operation: str = None, include_memory: bool = False, include_cpu: bool = False
):
    """Performance monitoring decorator"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = None
            start_cpu = None

            # Get initial resource usage
            if include_memory or include_cpu:
                try:
                    import psutil

                    process = psutil.Process()
                    if include_memory:
                        start_memory = process.memory_info().rss / 1024 / 1024  # MB
                    if include_cpu:
                        start_cpu = process.cpu_percent()
                except ImportError:
                    pass

            try:
                result = func(*args, **kwargs)

                # Calculate performance metrics
                duration = (time.time() - start_time) * 1000  # ms

                memory_usage = None
                cpu_usage = None

                if include_memory and start_memory is not None:
                    try:
                        import psutil

                        process = psutil.Process()
                        end_memory = process.memory_info().rss / 1024 / 1024  # MB
                        memory_usage = end_memory - start_memory
                    except ImportError:
                        pass

                if include_cpu and start_cpu is not None:
                    try:
                        import psutil

                        process = psutil.Process()
                        cpu_usage = process.cpu_percent()
                    except ImportError:
                        pass

                metrics = PerformanceMetrics(
                    duration_ms=duration,
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=cpu_usage,
                )

                logger = get_logger(func.__module__)
                logger.performance(
                    f"Operation completed: {operation or func.__name__}",
                    metrics,
                    data={"function": func.__name__, "module": func.__module__},
                )

                return result

            except Exception as e:
                duration = (time.time() - start_time) * 1000  # ms
                metrics = PerformanceMetrics(duration_ms=duration)

                logger = get_logger(func.__module__)
                logger.error(
                    f"Operation failed: {operation or func.__name__}",
                    error=e,
                    data={"function": func.__name__, "module": func.__module__},
                )
                logger.performance(
                    f"Operation failed: {operation or func.__name__}",
                    metrics,
                    data={
                        "function": func.__name__,
                        "module": func.__module__,
                        "status": "failed",
                    },
                )

                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = None
            start_cpu = None

            # Get initial resource usage
            if include_memory or include_cpu:
                try:
                    import psutil

                    process = psutil.Process()
                    if include_memory:
                        start_memory = process.memory_info().rss / 1024 / 1024  # MB
                    if include_cpu:
                        start_cpu = process.cpu_percent()
                except ImportError:
                    pass

            try:
                result = await func(*args, **kwargs)

                # Calculate performance metrics
                duration = (time.time() - start_time) * 1000  # ms

                memory_usage = None
                cpu_usage = None

                if include_memory and start_memory is not None:
                    try:
                        import psutil

                        process = psutil.Process()
                        end_memory = process.memory_info().rss / 1024 / 1024  # MB
                        memory_usage = end_memory - start_memory
                    except ImportError:
                        pass

                if include_cpu and start_cpu is not None:
                    try:
                        import psutil

                        process = psutil.Process()
                        cpu_usage = process.cpu_percent()
                    except ImportError:
                        pass

                metrics = PerformanceMetrics(
                    duration_ms=duration,
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=cpu_usage,
                )

                logger = get_logger(func.__module__)
                logger.performance(
                    f"Operation completed: {operation or func.__name__}",
                    metrics,
                    data={"function": func.__name__, "module": func.__module__},
                )

                return result

            except Exception as e:
                duration = (time.time() - start_time) * 1000  # ms
                metrics = PerformanceMetrics(duration_ms=duration)

                logger = get_logger(func.__module__)
                logger.error(
                    f"Operation failed: {operation or func.__name__}",
                    error=e,
                    data={"function": func.__name__, "module": func.__module__},
                )
                logger.performance(
                    f"Operation failed: {operation or func.__name__}",
                    metrics,
                    data={
                        "function": func.__name__,
                        "module": func.__module__,
                        "status": "failed",
                    },
                )

                raise

        if hasattr(func, "__call__") and hasattr(func, "__await__"):
            return async_wrapper
        else:
            return wrapper

    return decorator


# Global logger cache
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str, level: LogLevel = LogLevel.INFO) -> StructuredLogger:
    """Get structured logger"""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, level)
    return _loggers[name]


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    format_json: bool = True,
    include_context: bool = True,
):
    """Configure global logging"""
    # Set root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    if format_json:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(handler)

    level_mapping = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.CRITICAL: logging.CRITICAL,
    }

    root_logger.setLevel(level_mapping[level])


# Default logger
default_logger = get_logger(__name__)
