"""Concurrency optimization module - Implements intelligent concurrency control and resource management"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from threading import RLock, Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority enumeration"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task status enumeration"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """Task information data class"""

    task_id: str
    name: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_duration(self) -> Optional[float]:
        """Get task execution duration (seconds)"""
        if self.started_at is None:
            return None

        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()

    def get_wait_time(self) -> float:
        """Get task wait time (seconds)"""
        start_time = self.started_at or datetime.now()
        return (start_time - self.created_at).total_seconds()


class AdaptiveSemaphore:
    """Adaptive semaphore - Dynamically adjusts concurrency based on system load"""

    def __init__(self, initial_value: int = 5, min_value: int = 1, max_value: int = 20):
        self.current_value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self._semaphore = Semaphore(initial_value)
        self._lock = RLock()

        # Performance monitoring data
        self._task_times: List[float] = []
        self._last_adjustment = datetime.now()
        self._adjustment_interval = timedelta(seconds=30)  # Adjust every 30 seconds

        logger.debug(
            f"Adaptive semaphore initialized: {initial_value} (range: {min_value}-{max_value})"
        )

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire semaphore"""
        return self._semaphore.acquire(blocking, timeout)

    def release(self, task_duration: Optional[float] = None):
        """Release semaphore and record task duration"""
        if task_duration is not None:
            with self._lock:
                self._task_times.append(task_duration)
                # Keep only the duration of the last 100 tasks
                if len(self._task_times) > 100:
                    self._task_times = self._task_times[-100:]

        self._semaphore.release()
        self._maybe_adjust_capacity()

    def _maybe_adjust_capacity(self):
        """Adjust concurrency capacity based on performance data"""
        now = datetime.now()
        if now - self._last_adjustment < self._adjustment_interval:
            return

        with self._lock:
            if len(self._task_times) < 10:  # Insufficient data, no adjustment
                return

            # Calculate average task duration and coefficient of variation
            avg_time = sum(self._task_times) / len(self._task_times)
            variance = sum((t - avg_time) ** 2 for t in self._task_times) / len(
                self._task_times
            )
            std_dev = variance**0.5
            cv = std_dev / avg_time if avg_time > 0 else 0  # Coefficient of variation

            # Adjustment strategy
            new_value = self.current_value

            if avg_time < 2.0 and cv < 0.5:  # Tasks are fast and stable, can increase concurrency
                new_value = min(self.max_value, self.current_value + 1)
            elif avg_time > 10.0 or cv > 1.0:  # Tasks are slow or unstable, reduce concurrency
                new_value = max(self.min_value, self.current_value - 1)

            if new_value != self.current_value:
                self._adjust_capacity(new_value)
                logger.info(
                    f"Adaptive semaphore adjusted: {self.current_value} -> {new_value} "
                    f"(avg_time: {avg_time:.2f}s, cv: {cv:.2f})"
                )

            self._last_adjustment = now

    def _adjust_capacity(self, new_value: int):
        """Adjust semaphore capacity"""
        if new_value > self.current_value:
            # Increase capacity
            for _ in range(new_value - self.current_value):
                self._semaphore.release()
        elif new_value < self.current_value:
            # Reduce capacity
            for _ in range(self.current_value - new_value):
                self._semaphore.acquire(blocking=False)

        self.current_value = new_value

    def get_stats(self) -> Dict[str, Any]:
        """Get semaphore statistics"""
        with self._lock:
            avg_time = (
                sum(self._task_times) / len(self._task_times) if self._task_times else 0
            )

            return {
                "current_capacity": self.current_value,
                "min_capacity": self.min_value,
                "max_capacity": self.max_value,
                "avg_task_time": avg_time,
                "task_samples": len(self._task_times),
                "available_permits": self._semaphore._value,
            }


class ConcurrentTaskManager:
    """Concurrent task manager - Manages execution and scheduling of asynchronous tasks"""

    def __init__(self, max_concurrent: int = 10, thread_pool_size: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = AdaptiveSemaphore(
            max_concurrent, min_value=1, max_value=max_concurrent * 2
        )
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)

        # Task tracking
        self.tasks: Dict[str, TaskInfo] = {}
        self.running_tasks: Set[str] = set()
        self._task_counter = 0
        self._lock = RLock()

        # Performance statistics
        self._stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "cancelled_tasks": 0,
            "total_execution_time": 0.0,
            "total_wait_time": 0.0,
        }

        logger.info(
            f"Concurrent task manager initialized: max_concurrent={max_concurrent}, "
            f"thread_pool_size={thread_pool_size}"
        )

    def _generate_task_id(self) -> str:
        """Generate unique task ID"""
        with self._lock:
            self._task_counter += 1
            return f"task_{self._task_counter}_{int(time.time() * 1000)}"

    async def execute_async_task(
        self,
        coro: Awaitable[Any],
        name: str = "async_task",
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskInfo:
        """Execute asynchronous task"""
        task_id = self._generate_task_id()
        task_info = TaskInfo(
            task_id=task_id, name=name, priority=priority, metadata=metadata or {}
        )

        with self._lock:
            self.tasks[task_id] = task_info
            self._stats["total_tasks"] += 1

        try:
            # Acquire semaphore
            if not self.semaphore.acquire(timeout=timeout):
                task_info.status = TaskStatus.FAILED
                task_info.error = TimeoutError(
                    "Failed to acquire semaphore within timeout"
                )
                return task_info

            try:
                # Start executing task
                task_info.status = TaskStatus.RUNNING
                task_info.started_at = datetime.now()

                with self._lock:
                    self.running_tasks.add(task_id)
                    self._stats["total_wait_time"] += task_info.get_wait_time()

                # Execute task (with timeout)
                if timeout:
                    task_info.result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    task_info.result = await coro

                # Task completed
                task_info.status = TaskStatus.COMPLETED
                task_info.completed_at = datetime.now()

                with self._lock:
                    self._stats["completed_tasks"] += 1
                    self._stats["total_execution_time"] += task_info.get_duration() or 0

                logger.debug(f"Async task completed: {name} ({task_id})")

            finally:
                # Release semaphore
                self.semaphore.release(task_info.get_duration())

                with self._lock:
                    self.running_tasks.discard(task_id)

        except asyncio.CancelledError:
            task_info.status = TaskStatus.CANCELLED
            with self._lock:
                self._stats["cancelled_tasks"] += 1
            logger.debug(f"Async task cancelled: {name} ({task_id})")

        except Exception as e:
            task_info.status = TaskStatus.FAILED
            task_info.error = e
            task_info.completed_at = datetime.now()

            with self._lock:
                self._stats["failed_tasks"] += 1

            logger.error(f"Async task failed: {name} ({task_id}): {e}")

        return task_info

    def execute_sync_task(
        self,
        func: Callable[..., Any],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        name: str = "sync_task",
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskInfo:
        """Execute synchronous task (in thread pool)"""
        task_id = self._generate_task_id()
        task_info = TaskInfo(
            task_id=task_id, name=name, priority=priority, metadata=metadata or {}
        )

        with self._lock:
            self.tasks[task_id] = task_info
            self._stats["total_tasks"] += 1

        try:
            # Submit to thread pool
            future = self.thread_pool.submit(
                self._execute_sync_task_wrapper, func, args, kwargs or {}, task_info
            )

            # Wait for result
            task_info = future.result(timeout=timeout)

        except Exception as e:
            task_info.status = TaskStatus.FAILED
            task_info.error = e
            task_info.completed_at = datetime.now()

            with self._lock:
                self._stats["failed_tasks"] += 1

            logger.error(f"Sync task failed: {name} ({task_id}): {e}")

        return task_info

    def _execute_sync_task_wrapper(
        self, func: Callable, args: tuple, kwargs: Dict[str, Any], task_info: TaskInfo
    ) -> TaskInfo:
        """Synchronous task execution wrapper"""
        try:
            # Acquire semaphore
            if not self.semaphore.acquire(timeout=30.0):  # 30 second timeout
                task_info.status = TaskStatus.FAILED
                task_info.error = TimeoutError("Failed to acquire semaphore")
                return task_info

            try:
                # Start executing task
                task_info.status = TaskStatus.RUNNING
                task_info.started_at = datetime.now()

                with self._lock:
                    self.running_tasks.add(task_info.task_id)
                    self._stats["total_wait_time"] += task_info.get_wait_time()

                # Execute task
                task_info.result = func(*args, **kwargs)

                # Task completed
                task_info.status = TaskStatus.COMPLETED
                task_info.completed_at = datetime.now()

                with self._lock:
                    self._stats["completed_tasks"] += 1
                    self._stats["total_execution_time"] += task_info.get_duration() or 0

                logger.debug(
                    f"Sync task completed: {task_info.name} ({task_info.task_id})"
                )

            finally:
                # Release semaphore
                self.semaphore.release(task_info.get_duration())

                with self._lock:
                    self.running_tasks.discard(task_info.task_id)

        except Exception as e:
            task_info.status = TaskStatus.FAILED
            task_info.error = e
            task_info.completed_at = datetime.now()

            with self._lock:
                self._stats["failed_tasks"] += 1

            logger.error(
                f"Sync task failed: {task_info.name} ({task_info.task_id}): {e}"
            )

        return task_info

    async def execute_batch_async(
        self,
        coros: List[Awaitable[Any]],
        names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        return_exceptions: bool = True,
    ) -> List[TaskInfo]:
        """Execute asynchronous tasks in batch"""
        if names is None:
            names = [f"batch_task_{i}" for i in range(len(coros))]
        elif len(names) != len(coros):
            raise ValueError("Names list length must match coroutines list length")

        # Create tasks
        tasks = []
        for i, (coro, name) in enumerate(zip(coros, names)):
            task = self.execute_async_task(coro, name=name, timeout=timeout)
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

        # Return task info list
        return [
            (
                result
                if isinstance(result, TaskInfo)
                else TaskInfo(
                    task_id=f"error_{i}",
                    name=names[i],
                    status=TaskStatus.FAILED,
                    error=result if isinstance(result, Exception) else None,
                )
            )
            for i, result in enumerate(results)
        ]

    def execute_batch_sync(
        self,
        funcs: List[Callable],
        args_list: Optional[List[tuple]] = None,
        kwargs_list: Optional[List[Dict[str, Any]]] = None,
        names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> List[TaskInfo]:
        """Execute synchronous tasks in batch"""
        if args_list is None:
            args_list = [()] * len(funcs)
        if kwargs_list is None:
            kwargs_list = [{}] * len(funcs)
        if names is None:
            names = [f"batch_sync_task_{i}" for i in range(len(funcs))]

        # Validate parameter lengths
        if not (len(funcs) == len(args_list) == len(kwargs_list) == len(names)):
            raise ValueError("All parameter lists must have the same length")

        # Submit all tasks to thread pool
        futures = []
        for func, args, kwargs, name in zip(funcs, args_list, kwargs_list, names):
            future = self.thread_pool.submit(
                self.execute_sync_task,
                func,
                args,
                kwargs,
                name,
                TaskPriority.NORMAL,
                timeout,
            )
            futures.append(future)

        # Wait for all tasks to complete
        results = []
        for future in as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Create failed task info
                error_task = TaskInfo(
                    task_id=f"error_{len(results)}",
                    name="batch_sync_error",
                    status=TaskStatus.FAILED,
                    error=e,
                )
                results.append(error_task)

        return results

    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """Get task information"""
        return self.tasks.get(task_id)

    def get_running_tasks(self) -> List[TaskInfo]:
        """Get running tasks"""
        with self._lock:
            return [
                self.tasks[task_id]
                for task_id in self.running_tasks
                if task_id in self.tasks
            ]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel task (only effective for unstarted tasks)"""
        task_info = self.tasks.get(task_id)
        if task_info and task_info.status == TaskStatus.PENDING:
            task_info.status = TaskStatus.CANCELLED
            with self._lock:
                self._stats["cancelled_tasks"] += 1
            logger.debug(f"Task cancelled: {task_id}")
            return True
        return False

    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up old completed tasks"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        with self._lock:
            tasks_to_remove = []
            for task_id, task_info in self.tasks.items():
                if (
                    task_info.status
                    in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                    and task_info.created_at < cutoff_time
                ):
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del self.tasks[task_id]

            if tasks_to_remove:
                logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")

            return len(tasks_to_remove)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            stats = self._stats.copy()

            # Calculate averages
            if stats["completed_tasks"] > 0:
                stats["avg_execution_time"] = (
                    stats["total_execution_time"] / stats["completed_tasks"]
                )
                stats["avg_wait_time"] = (
                    stats["total_wait_time"] / stats["completed_tasks"]
                )
            else:
                stats["avg_execution_time"] = 0.0
                stats["avg_wait_time"] = 0.0

            # Add current status
            stats["current_running_tasks"] = len(self.running_tasks)
            stats["total_tracked_tasks"] = len(self.tasks)
            stats["semaphore_stats"] = self.semaphore.get_stats()

            # Calculate success rate
            if stats["total_tasks"] > 0:
                stats["success_rate"] = stats["completed_tasks"] / stats["total_tasks"]
            else:
                stats["success_rate"] = 0.0

            return stats

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """Shutdown task manager"""
        logger.info("Shutting down concurrent task manager...")

        # Shutdown thread pool (ThreadPoolExecutor.shutdown() does not accept timeout parameter)
        self.thread_pool.shutdown(wait=wait)

        # Cancel all pending tasks
        with self._lock:
            cancelled_count = 0
            for task_id, task_info in self.tasks.items():
                if task_info.status == TaskStatus.PENDING:
                    task_info.status = TaskStatus.CANCELLED
                    cancelled_count += 1

            if cancelled_count > 0:
                logger.info(
                    f"Cancelled {cancelled_count} pending tasks during shutdown"
                )

        logger.info("Concurrent task manager shutdown completed")


@asynccontextmanager
async def concurrent_execution_context(max_concurrent: int = 5):
    """Concurrent execution context manager"""
    manager = ConcurrentTaskManager(max_concurrent=max_concurrent)
    try:
        yield manager
    finally:
        manager.shutdown(wait=True)


# Global concurrent task manager instance
global_concurrent_manager = ConcurrentTaskManager(max_concurrent=10, thread_pool_size=5)
