"""Performance monitoring module - Collect and track performance metrics for researcher nodes"""

import psutil
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""

    # Time metrics
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Memory metrics
    memory_start_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_end_mb: float = 0.0
    memory_delta_mb: float = 0.0

    # CPU metrics
    cpu_percent_avg: float = 0.0
    cpu_percent_peak: float = 0.0

    # Task metrics
    task_count: int = 0
    success_count: int = 0
    error_count: int = 0

    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "memory_start_mb": self.memory_start_mb,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_end_mb": self.memory_end_mb,
            "memory_delta_mb": self.memory_delta_mb,
            "cpu_percent_avg": self.cpu_percent_avg,
            "cpu_percent_peak": self.cpu_percent_peak,
            "task_count": self.task_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "custom_metrics": self.custom_metrics,
        }


class PerformanceMonitor:
    """Performance monitor - Real-time collection and analysis of performance data"""

    def __init__(self, name: str = "researcher_performance"):
        self.name = name
        self.metrics = PerformanceMetrics()
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None

        logger.debug(f"Performance monitor '{name}' initialized")

    def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return

        self.monitoring_active = True
        self.metrics.start_time = datetime.now()
        self.metrics.memory_start_mb = self._get_memory_usage_mb()

        logger.info(f"Performance monitoring started for '{self.name}'")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.monitoring_active:
            logger.warning("Performance monitoring not active")
            return

        self.monitoring_active = False
        self.metrics.end_time = datetime.now()
        self.metrics.duration_seconds = (
            self.metrics.end_time - self.metrics.start_time
        ).total_seconds()

        # Calculate final metrics
        self.metrics.memory_end_mb = self._get_memory_usage_mb()
        self.metrics.memory_delta_mb = (
            self.metrics.memory_end_mb - self.metrics.memory_start_mb
        )

        if self.cpu_samples:
            self.metrics.cpu_percent_avg = sum(self.cpu_samples) / len(self.cpu_samples)
            self.metrics.cpu_percent_peak = max(self.cpu_samples)

        if self.memory_samples:
            self.metrics.memory_peak_mb = max(self.memory_samples)

        logger.info(f"Performance monitoring stopped for '{self.name}'")
        logger.info(
            f"Duration: {self.metrics.duration_seconds:.2f}s, "
            f"Memory delta: {self.metrics.memory_delta_mb:.2f}MB"
        )

    async def start_async_monitoring(self, sample_interval: float = 1.0):
        """Start asynchronous performance monitoring"""
        if self._monitor_task and not self._monitor_task.done():
            logger.warning("Async monitoring already running")
            return

        self.start_monitoring()
        self._monitor_task = asyncio.create_task(
            self._async_monitor_loop(sample_interval)
        )

        logger.debug(f"Async monitoring started with {sample_interval}s interval")

    async def stop_async_monitoring(self):
        """Stop asynchronous performance monitoring"""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        self.stop_monitoring()
        logger.debug("Async monitoring stopped")

    async def _async_monitor_loop(self, interval: float):
        """Asynchronous monitoring loop"""
        try:
            while self.monitoring_active:
                # Sample CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_mb = self._get_memory_usage_mb()

                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append(memory_mb)

                # Limit sample count to avoid memory leaks
                if len(self.cpu_samples) > 1000:
                    self.cpu_samples = self.cpu_samples[-500:]
                if len(self.memory_samples) > 1000:
                    self.memory_samples = self.memory_samples[-500:]

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("Monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")

    def record_task_start(self):
        """Record task start"""
        self.metrics.task_count += 1
        logger.debug(f"Task started, total count: {self.metrics.task_count}")

    def record_task_success(self):
        """Record task success"""
        self.metrics.success_count += 1
        logger.debug(f"Task succeeded, success count: {self.metrics.success_count}")

    def record_task_error(self):
        """Record task error"""
        self.metrics.error_count += 1
        logger.debug(f"Task failed, error count: {self.metrics.error_count}")

    def add_custom_metric(self, key: str, value: Any):
        """Add custom metric"""
        self.metrics.custom_metrics[key] = value
        logger.debug(f"Custom metric added: {key} = {value}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        current_metrics = self.metrics.to_dict()

        # If monitoring is active, calculate current duration
        if self.monitoring_active and not self.metrics.end_time:
            current_time = datetime.now()
            current_metrics["current_duration_seconds"] = (
                current_time - self.metrics.start_time
            ).total_seconds()
            current_metrics["current_memory_mb"] = self._get_memory_usage_mb()

        return current_metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        metrics = self.get_current_metrics()

        # Calculate efficiency metrics
        success_rate = 0.0
        if self.metrics.task_count > 0:
            success_rate = self.metrics.success_count / self.metrics.task_count

        tasks_per_second = 0.0
        if self.metrics.duration_seconds > 0:
            tasks_per_second = self.metrics.task_count / self.metrics.duration_seconds

        return {
            "monitor_name": self.name,
            "performance_summary": {
                "success_rate": success_rate,
                "tasks_per_second": tasks_per_second,
                "memory_efficiency_mb_per_task": (
                    self.metrics.memory_delta_mb / max(self.metrics.task_count, 1)
                ),
                "avg_cpu_usage": self.metrics.cpu_percent_avg,
                "peak_memory_mb": self.metrics.memory_peak_mb,
            },
            "raw_metrics": metrics,
        }

    @staticmethod
    def _get_memory_usage_mb() -> float:
        """Get current memory usage (MB)"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    @asynccontextmanager
    async def monitor_context(self, sample_interval: float = 1.0):
        """Performance monitoring context manager"""
        await self.start_async_monitoring(sample_interval)
        try:
            yield self
        finally:
            await self.stop_async_monitoring()

    def __enter__(self):
        """Synchronous context manager entry"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Synchronous context manager exit"""
        self.stop_monitoring()

        # If there's an exception, record error
        if exc_type is not None:
            self.record_task_error()
            logger.error(
                f"Exception in monitored context: {exc_type.__name__}: {exc_val}"
            )

        return False  # Don't suppress exceptions


class GlobalPerformanceTracker:
    """Global performance tracker - Manage multiple performance monitors"""

    def __init__(self):
        self.monitors: Dict[str, PerformanceMonitor] = {}
        self.session_start_time = datetime.now()

        logger.info("Global performance tracker initialized")

    def create_monitor(self, name: str) -> PerformanceMonitor:
        """Create new performance monitor"""
        if name in self.monitors:
            logger.warning(f"Monitor '{name}' already exists, returning existing one")
            return self.monitors[name]

        monitor = PerformanceMonitor(name)
        self.monitors[name] = monitor

        logger.debug(f"Created performance monitor: {name}")
        return monitor

    def get_monitor(self, name: str) -> Optional[PerformanceMonitor]:
        """Get performance monitor"""
        return self.monitors.get(name)

    def get_all_summaries(self) -> Dict[str, Any]:
        """Get summaries of all monitors"""
        summaries = {}
        for name, monitor in self.monitors.items():
            summaries[name] = monitor.get_performance_summary()

        # Add global statistics
        total_tasks = sum(m.metrics.task_count for m in self.monitors.values())
        total_successes = sum(m.metrics.success_count for m in self.monitors.values())
        total_errors = sum(m.metrics.error_count for m in self.monitors.values())

        global_summary = {
            "session_duration_seconds": (
                (datetime.now() - self.session_start_time).total_seconds()
            ),
            "total_monitors": len(self.monitors),
            "total_tasks": total_tasks,
            "total_successes": total_successes,
            "total_errors": total_errors,
            "global_success_rate": total_successes / max(total_tasks, 1),
        }

        return {
            "global_summary": global_summary,
            "monitor_summaries": summaries,
        }

    def cleanup_inactive_monitors(self, max_age_hours: int = 24):
        """Clean up inactive monitors"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        inactive_monitors = []

        for name, monitor in self.monitors.items():
            if monitor.metrics.end_time and monitor.metrics.end_time < cutoff_time:
                inactive_monitors.append(name)

        for name in inactive_monitors:
            del self.monitors[name]
            logger.debug(f"Cleaned up inactive monitor: {name}")

        if inactive_monitors:
            logger.info(f"Cleaned up {len(inactive_monitors)} inactive monitors")


# Global performance tracker instance
global_performance_tracker = GlobalPerformanceTracker()
