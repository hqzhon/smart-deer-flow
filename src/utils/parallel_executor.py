#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel execution manager
Provides intelligent parallel task execution with rate limiting, error handling and adaptive scheduling
"""

import asyncio
import time
import logging
from typing import List, Callable, Any, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from src.utils.rate_limiter import get_global_rate_limiter, RateLimiter
from src.llms.error_handler import error_handler, LLMErrorType

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    RETRYING = "retrying"


@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    retry_count: int = 0
    rate_limit_delays: int = 0
    total_delay_time: float = 0.0


@dataclass
class ParallelTask:
    """Parallel task definition"""
    task_id: str
    func: Callable
    args: tuple = ()
    kwargs: dict = None
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    timeout: Optional[float] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.dependencies is None:
            self.dependencies = []


class ParallelExecutor:
    """Parallel execution manager"""
    
    def __init__(self, 
                 max_concurrent_tasks: int = 3,
                 rate_limiter: Optional[RateLimiter] = None,
                 enable_adaptive_scheduling: bool = True,
                 task_timeout: float = 300.0):
        
        self.max_concurrent_tasks = max_concurrent_tasks
        self.rate_limiter = rate_limiter or get_global_rate_limiter()
        self.enable_adaptive_scheduling = enable_adaptive_scheduling
        self.task_timeout = task_timeout
        
        # Task management
        self.pending_tasks: List[ParallelTask] = []
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        
        # Statistics
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.rate_limited_tasks = 0
        
        # Adaptive scheduling parameters
        self.current_success_rate = 1.0
        self.recent_failures = []
        self.last_adjustment = time.time()
        
        logger.info(f"Parallel executor initialized with max_concurrent_tasks={max_concurrent_tasks}")
    
    def add_task(self, task: ParallelTask):
        """Add task to execution queue"""
        self.pending_tasks.append(task)
        self.total_tasks += 1
        logger.debug(f"Added task {task.task_id} to queue (priority: {task.priority.name})")
    
    def add_tasks(self, tasks: List[ParallelTask]):
        """Batch add tasks"""
        for task in tasks:
            self.add_task(task)
    
    async def execute_all(self) -> Dict[str, TaskResult]:
        """Execute all tasks"""
        logger.info(f"Starting execution of {len(self.pending_tasks)} tasks")
        start_time = time.time()
        
        # Sort tasks by priority and dependencies
        self._sort_tasks_by_priority_and_dependencies()
        
        while self.pending_tasks or self.running_tasks:
            # Adaptive adjustment of concurrency
            if self.enable_adaptive_scheduling:
                self._adaptive_adjust_concurrency()
            
            # Start new tasks
            await self._start_available_tasks()
            
            # Wait for task completion
            if self.running_tasks:
                await self._wait_for_task_completion()
            
            # Brief rest to avoid busy waiting
            await asyncio.sleep(0.01)
        
        execution_time = time.time() - start_time
        self._log_execution_summary(execution_time)
        
        return self.completed_tasks.copy()
    
    def _sort_tasks_by_priority_and_dependencies(self):
        """Sort tasks by priority and dependencies"""
        # First sort by priority (high priority first)
        self.pending_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        
        # Then handle dependencies
        sorted_tasks = []
        remaining_tasks = self.pending_tasks.copy()
        
        while remaining_tasks:
            # Find tasks with no incomplete dependencies
            ready_tasks = []
            for task in remaining_tasks:
                if all(dep_id in self.completed_tasks for dep_id in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # If no ready tasks, possible circular dependency, take first task
                logger.warning("Possible circular dependency detected, taking first task")
                ready_tasks = [remaining_tasks[0]]
            
            # Add ready tasks and remove from remaining tasks
            for task in ready_tasks:
                sorted_tasks.append(task)
                remaining_tasks.remove(task)
        
        self.pending_tasks = sorted_tasks
    
    async def _start_available_tasks(self):
        """Start available tasks"""
        available_slots = self.max_concurrent_tasks - len(self.running_tasks)
        
        if available_slots <= 0:
            return
        
        # Find tasks that can be started (no incomplete dependencies)
        ready_tasks = []
        for task in self.pending_tasks:
            if len(ready_tasks) >= available_slots:
                break
            
            # Check if dependencies are satisfied
            if all(dep_id in self.completed_tasks and 
                  self.completed_tasks[dep_id].status == TaskStatus.COMPLETED 
                  for dep_id in task.dependencies):
                ready_tasks.append(task)
        
        # Start ready tasks
        for task in ready_tasks:
            self.pending_tasks.remove(task)
            asyncio_task = asyncio.create_task(self._execute_single_task(task))
            self.running_tasks[task.task_id] = asyncio_task
            logger.debug(f"Started task {task.task_id}")
    
    async def _execute_single_task(self, task: ParallelTask) -> TaskResult:
        """Execute single task"""
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING
        )
        
        start_time = time.time()
        
        for attempt in range(task.max_retries + 1):
            try:
                # Rate limiting
                delay_time = await self.rate_limiter.acquire()
                if delay_time > 0:
                    result.rate_limit_delays += 1
                    result.total_delay_time += delay_time
                    logger.debug(f"Task {task.task_id} delayed {delay_time:.2f}s by rate limiter")
                
                # Execute task
                if task.timeout:
                    task_result = await asyncio.wait_for(
                        self._call_task_function(task),
                        timeout=task.timeout
                    )
                else:
                    task_result = await self._call_task_function(task)
                
                # Task successful
                result.status = TaskStatus.COMPLETED
                result.result = task_result
                result.execution_time = time.time() - start_time
                
                self.rate_limiter.record_success()
                self.successful_tasks += 1
                
                logger.debug(f"Task {task.task_id} completed successfully in {result.execution_time:.2f}s")
                break
                
            except asyncio.TimeoutError:
                logger.error(f"Task {task.task_id} timed out after {task.timeout}s")
                result.status = TaskStatus.FAILED
                result.error = TimeoutError(f"Task timed out after {task.timeout}s")
                self.failed_tasks += 1
                break
                
            except Exception as e:
                result.retry_count = attempt
                error_type = error_handler.classify_error(str(e))
                
                # Record rate limiting errors
                if error_type == LLMErrorType.RATE_LIMIT_EXCEEDED:
                    self.rate_limiter.record_failure(str(e))
                    self.rate_limited_tasks += 1
                    result.status = TaskStatus.RATE_LIMITED
                    logger.warning(f"Task {task.task_id} hit rate limit (attempt {attempt + 1})")
                else:
                    logger.error(f"Task {task.task_id} failed (attempt {attempt + 1}): {e}")
                
                # Check if should retry
                if attempt < task.max_retries and error_handler.should_retry_error(error_type):
                    result.status = TaskStatus.RETRYING
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying task {task.task_id} in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    result.total_delay_time += wait_time
                    continue
                else:
                    # No more retries
                    result.status = TaskStatus.FAILED
                    result.error = e
                    self.failed_tasks += 1
                    break
        
        # Record task completion
        self.completed_tasks[task.task_id] = result
        if task.task_id in self.running_tasks:
            del self.running_tasks[task.task_id]
        
        return result
    
    async def _call_task_function(self, task: ParallelTask) -> Any:
        """Call task function"""
        if asyncio.iscoroutinefunction(task.func):
            return await task.func(*task.args, **task.kwargs)
        else:
            # For synchronous functions, execute in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                return await loop.run_in_executor(executor, 
                                                lambda: task.func(*task.args, **task.kwargs))
    
    async def _wait_for_task_completion(self):
        """Wait for at least one task to complete"""
        if not self.running_tasks:
            return
        
        # Wait for any one task to complete
        done, pending = await asyncio.wait(
            self.running_tasks.values(),
            return_when=asyncio.FIRST_COMPLETED,
            timeout=1.0  # Wait at most 1 second
        )
        
        # Process completed tasks
        for task in done:
            try:
                await task  # Get result or exception
            except Exception as e:
                logger.error(f"Unexpected error in task: {e}")
    
    def _adaptive_adjust_concurrency(self):
        """Adaptive adjustment of concurrency"""
        now = time.time()
        
        # Adjust every 30 seconds
        if now - self.last_adjustment < 30:
            return
        
        # Calculate recent success rate
        total_recent = self.successful_tasks + self.failed_tasks
        if total_recent > 0:
            self.current_success_rate = self.successful_tasks / total_recent
        
        # Adjust concurrency based on success rate and rate limiting
        if self.current_success_rate > 0.9 and self.rate_limited_tasks == 0:
            # High success rate and no rate limiting, can increase concurrency
            new_concurrency = min(self.max_concurrent_tasks + 1, 10)
            if new_concurrency != self.max_concurrent_tasks:
                self.max_concurrent_tasks = new_concurrency
                logger.info(f"Increased concurrency to {self.max_concurrent_tasks}")
        
        elif self.current_success_rate < 0.7 or self.rate_limited_tasks > 0:
            # Low success rate or rate limiting, reduce concurrency
            new_concurrency = max(self.max_concurrent_tasks - 1, 1)
            if new_concurrency != self.max_concurrent_tasks:
                self.max_concurrent_tasks = new_concurrency
                logger.info(f"Decreased concurrency to {self.max_concurrent_tasks}")
        
        self.last_adjustment = now
    
    def _log_execution_summary(self, execution_time: float):
        """Log execution summary"""
        rate_limiter_stats = self.rate_limiter.get_stats() if self.rate_limiter else {}
        
        logger.info(f"Parallel execution completed in {execution_time:.2f}s")
        logger.info(f"Tasks: {self.total_tasks} total, {self.successful_tasks} successful, {self.failed_tasks} failed")
        
        # Safely get rate limiting statistics
        total_delays = rate_limiter_stats.get('total_delays', 0)
        average_delay = rate_limiter_stats.get('average_delay', 0.0)
        
        logger.info(f"Rate limiting: {self.rate_limited_tasks} rate limited, {total_delays} delays")
        logger.info(f"Average delay: {average_delay:.2f}s")
        
        if self.total_tasks > 0:
            success_rate = self.successful_tasks / self.total_tasks * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        rate_limiter_stats = self.rate_limiter.get_stats() if self.rate_limiter else {}
        
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "rate_limited_tasks": self.rate_limited_tasks,
            "current_concurrency": self.max_concurrent_tasks,
            "success_rate": self.successful_tasks / max(1, self.total_tasks),
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "rate_limiter_stats": rate_limiter_stats
        }


def create_parallel_executor(max_concurrent_tasks: Optional[int] = None,
                           enable_rate_limiting: bool = True,
                           enable_adaptive_scheduling: bool = True) -> ParallelExecutor:
    """Create parallel executor"""
    # Load default values from config
    from src.config.config_loader import config_loader
    
    try:
        config_data = config_loader.load_config()
        if max_concurrent_tasks is None:
            max_concurrent_tasks = config_data.get('max_parallel_tasks', 3)
            # Ensure max_concurrent_tasks is integer type
            if isinstance(max_concurrent_tasks, str):
                max_concurrent_tasks = int(max_concurrent_tasks)
    except Exception as e:
        logger.warning(f"Failed to load config: {e}, using defaults")
        if max_concurrent_tasks is None:
            max_concurrent_tasks = 3
    
    # Final type safety check
    if not isinstance(max_concurrent_tasks, int):
        logger.warning(f"max_concurrent_tasks is not an integer: {type(max_concurrent_tasks)}, converting to int")
        try:
            max_concurrent_tasks = int(max_concurrent_tasks)
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert max_concurrent_tasks to int: {e}, using default value 3")
            max_concurrent_tasks = 3
    
    rate_limiter = get_global_rate_limiter() if enable_rate_limiting else None
    
    return ParallelExecutor(
        max_concurrent_tasks=max_concurrent_tasks,
        rate_limiter=rate_limiter,
        enable_adaptive_scheduling=enable_adaptive_scheduling
    )