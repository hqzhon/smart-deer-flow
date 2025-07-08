# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# Performance optimization utilities for DeerFlow

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import heapq
from collections import defaultdict, deque

# Try to import psutil, use mock version if failed
try:
    import psutil
except ImportError:
    # Create a mock version if psutil is not installed
    class MockPsutil:
        @staticmethod
        def cpu_percent(interval=0.1):
            return 50.0  # Simulate 50% CPU usage
        
        @staticmethod
        def cpu_count():
            return 4  # Simulate 4-core CPU
        
        @staticmethod
        def virtual_memory():
            class MockMemory:
                total = 8 * 1024 * 1024 * 1024  # 8GB
                available = 4 * 1024 * 1024 * 1024  # 4GB available
            return MockMemory()
    
    psutil = MockPsutil()

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels for execution scheduling."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ParallelTask:
    """Parallel task definition"""
    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout: float = 60.0
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class TaskComplexity:
    """Task complexity analysis result"""
    complexity_score: float  # 0.0 - 1.0
    cpu_intensive: bool
    memory_intensive: bool
    io_intensive: bool
    estimated_duration: float
    resource_requirements: Dict[str, float]


class IntelligentTaskDecomposer:
    """Intelligent task decomposer that breaks down complex research tasks into parallelizable subtasks"""
    
    def __init__(self):
        self.task_patterns = {
            'research': ['web_search', 'data_analysis', 'source_verification'],
            'analysis': ['data_processing', 'pattern_recognition', 'insight_extraction'],
            'writing': ['outline_generation', 'content_creation', 'quality_review']
        }
        self.complexity_analyzer = TaskComplexityAnalyzer()
    
    async def decompose_research_task(self, user_query: str, task_type: str = 'research') -> List[ParallelTask]:
        """Decompose user query into parallel research tasks"""
        try:
            # Analyze query type and complexity
            query_analysis = await self._analyze_query_complexity(user_query)
            
            # Generate parallel tasks
            tasks = []
            
            if task_type == 'research' or 'research' in task_type.lower():
                # 1. Parallel web search tasks
                search_tasks = await self._create_search_tasks(query_analysis)
                tasks.extend(search_tasks)
                
                # 2. Parallel data analysis tasks
                analysis_tasks = await self._create_analysis_tasks(query_analysis)
                tasks.extend(analysis_tasks)
            
            if task_type == 'writing' or 'writing' in task_type.lower():
                # 3. Parallel content generation tasks
                content_tasks = await self._create_content_tasks(query_analysis)
                tasks.extend(content_tasks)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            return []
    
    async def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity"""
        # Simplified complexity analysis
        words = query.split()
        complexity = {
            'keywords': self._extract_keywords(query),
            'complexity_score': min(len(words) / 20.0, 1.0),
            'requires_deep_analysis': len(words) > 30,
            'domain': self._detect_domain(query),
            'estimated_subtasks': max(2, len(words) // 10)
        }
        return complexity
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords"""
        # Simplified keyword extraction
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word.lower().strip('.,!?;:') for word in query.split()]
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _detect_domain(self, query: str) -> str:
        """Detect query domain"""
        domain_keywords = {
            'finance': ['financial', 'money', 'investment', 'market', 'stock', 'economy'],
            'technology': ['technology', 'software', 'AI', 'machine learning', 'programming'],
            'health': ['health', 'medical', 'disease', 'treatment', 'medicine'],
            'science': ['research', 'study', 'experiment', 'analysis', 'data']
        }
        
        query_lower = query.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        return 'general'
    
    async def _create_search_tasks(self, query_analysis: Dict) -> List[ParallelTask]:
        """Create parallel search tasks"""
        keywords = query_analysis.get('keywords', [])
        tasks = []
        
        # Group keywords for parallel search
        keyword_groups = self._group_keywords(keywords)
        
        for i, keyword_group in enumerate(keyword_groups):
            task = ParallelTask(
                task_id=f"search_task_{i}",
                func=self._parallel_web_search,
                args=(keyword_group,),
                priority=TaskPriority.HIGH,
                timeout=30.0,
                estimated_duration=15.0,
                resource_requirements={'cpu': 0.3, 'memory': 0.2, 'network': 0.8}
            )
            tasks.append(task)
        
        return tasks
    
    def _group_keywords(self, keywords: List[str], group_size: int = 3) -> List[List[str]]:
        """Group keywords"""
        groups = []
        for i in range(0, len(keywords), group_size):
            groups.append(keywords[i:i + group_size])
        return groups
    
    async def _create_analysis_tasks(self, query_analysis: Dict) -> List[ParallelTask]:
        """Create parallel analysis tasks"""
        tasks = []
        
        if query_analysis.get('requires_deep_analysis', False):
            # Deep analysis task
            task = ParallelTask(
                task_id="deep_analysis_task",
                func=self._parallel_deep_analysis,
                args=(query_analysis,),
                priority=TaskPriority.MEDIUM,
                timeout=60.0,
                estimated_duration=45.0,
                resource_requirements={'cpu': 0.7, 'memory': 0.5, 'network': 0.3}
            )
            tasks.append(task)
        
        return tasks
    
    async def _create_content_tasks(self, query_analysis: Dict) -> List[ParallelTask]:
        """Create parallel content generation tasks"""
        tasks = []
        
        # Outline generation task
        outline_task = ParallelTask(
            task_id="outline_generation",
            func=self._parallel_outline_generation,
            args=(query_analysis,),
            priority=TaskPriority.HIGH,
            timeout=30.0,
            estimated_duration=20.0,
            resource_requirements={'cpu': 0.5, 'memory': 0.3, 'network': 0.2}
        )
        tasks.append(outline_task)
        
        return tasks
    
    async def _parallel_web_search(self, keywords: List[str]) -> Dict[str, Any]:
        """Parallel web search"""
        # Simulate search implementation
        await asyncio.sleep(1)  # Simulate network delay
        return {
            'keywords': keywords,
            'results': [f"Result for {kw}" for kw in keywords],
            'timestamp': time.time()
        }
    
    async def _parallel_deep_analysis(self, query_analysis: Dict) -> Dict[str, Any]:
        """Parallel deep analysis"""
        # Simulate analysis implementation
        await asyncio.sleep(2)  # Simulate computation time
        return {
            'analysis_type': 'deep',
            'insights': ['Insight 1', 'Insight 2'],
            'confidence': 0.85
        }
    
    async def _parallel_outline_generation(self, query_analysis: Dict) -> Dict[str, Any]:
        """Parallel outline generation"""
        # Simulate outline generation
        await asyncio.sleep(1.5)
        return {
            'outline': ['Section 1', 'Section 2', 'Section 3'],
            'structure': 'hierarchical'
        }


class TaskComplexityAnalyzer:
    """Task complexity analyzer"""
    
    def __init__(self):
        self.complexity_cache = {}
    
    async def analyze(self, task: ParallelTask) -> TaskComplexity:
        """Analyze task complexity"""
        # Use task ID as cache key
        cache_key = f"{task.task_id}_{hash(str(task.func))}"
        
        if cache_key in self.complexity_cache:
            return self.complexity_cache[cache_key]
        
        # Analyze complexity based on task type and parameters
        complexity = await self._calculate_complexity(task)
        
        # Cache result
        self.complexity_cache[cache_key] = complexity
        
        return complexity
    
    async def _calculate_complexity(self, task: ParallelTask) -> TaskComplexity:
        """Calculate task complexity"""
        # Simplified complexity calculation based on task name and parameters
        func_name = task.func.__name__ if hasattr(task.func, '__name__') else str(task.func)
        
        # Default complexity
        complexity_score = 0.5
        cpu_intensive = False
        memory_intensive = False
        io_intensive = False
        estimated_duration = task.estimated_duration or 30.0
        
        # Adjust complexity based on function name
        if 'search' in func_name.lower():
            io_intensive = True
            complexity_score = 0.3
            estimated_duration = 15.0
        elif 'analysis' in func_name.lower():
            cpu_intensive = True
            complexity_score = 0.7
            estimated_duration = 45.0
        elif 'generation' in func_name.lower():
            cpu_intensive = True
            memory_intensive = True
            complexity_score = 0.6
            estimated_duration = 30.0
        
        # Adjust based on parameter count
        arg_count = len(task.args) + len(task.kwargs)
        complexity_score += min(arg_count * 0.1, 0.3)
        
        return TaskComplexity(
            complexity_score=min(complexity_score, 1.0),
            cpu_intensive=cpu_intensive,
            memory_intensive=memory_intensive,
            io_intensive=io_intensive,
            estimated_duration=estimated_duration,
            resource_requirements=task.resource_requirements or {
                'cpu': 0.5 if cpu_intensive else 0.2,
                'memory': 0.4 if memory_intensive else 0.1,
                'network': 0.6 if io_intensive else 0.1
            }
        )


class DynamicResourceAllocator:
    """Dynamic resource allocator that assigns computing resources based on task complexity and system load"""
    
    def __init__(self):
        self.system_monitor = SystemResourceMonitor()
        self.task_complexity_analyzer = TaskComplexityAnalyzer()
    
    async def allocate_resources(self, tasks: List[ParallelTask]) -> Dict[str, Dict[str, float]]:
        """Allocate optimal resources for tasks"""
        try:
            # Get system resource status
            system_resources = await self.system_monitor.get_current_resources()
            
            # Analyze task complexity
            task_complexities = {}
            for task in tasks:
                complexity = await self.task_complexity_analyzer.analyze(task)
                task_complexities[task.task_id] = complexity
            
            # Dynamic allocation strategy
            allocation = {}
            available_cpu = system_resources.get('available_cpu', 0.8)
            available_memory = system_resources.get('available_memory', 0.8)
            available_workers = system_resources.get('available_workers', 4)
            
            # Sort by complexity and priority
            sorted_tasks = sorted(tasks, 
                                key=lambda t: (t.priority.value, task_complexities[t.task_id].complexity_score), 
                                reverse=True)
            
            used_cpu = 0.0
            used_memory = 0.0
            used_workers = 0
            
            for task in sorted_tasks:
                complexity = task_complexities[task.task_id]
                
                # Calculate resource requirements
                cpu_need = complexity.resource_requirements.get('cpu', 0.2)
                memory_need = complexity.resource_requirements.get('memory', 0.1)
                
                # Check if resources are sufficient
                if (used_cpu + cpu_need <= available_cpu and 
                    used_memory + memory_need <= available_memory and 
                    used_workers < available_workers):
                    
                    # Allocate resources
                    allocation[task.task_id] = {
                        'cpu': cpu_need,
                        'memory': memory_need,
                        'workers': 1,
                        'priority_boost': 1.0 if task.priority == TaskPriority.CRITICAL else 0.8
                    }
                    
                    used_cpu += cpu_need
                    used_memory += memory_need
                    used_workers += 1
                else:
                    # Insufficient resources, downgrade allocation
                    allocation[task.task_id] = {
                        'cpu': min(cpu_need, available_cpu - used_cpu),
                        'memory': min(memory_need, available_memory - used_memory),
                        'workers': 0,  # Delayed execution
                        'priority_boost': 0.5
                    }
            
            return allocation
            
        except Exception as e:
            logger.error(f"Resource allocation failed: {e}")
            # Return default allocation
            return {task.task_id: {'cpu': 0.2, 'memory': 0.1, 'workers': 1, 'priority_boost': 1.0} 
                   for task in tasks}


class SystemResourceMonitor:
    """System resource monitor"""
    
    def __init__(self):
        self.last_check = 0
        self.cached_resources = {}
        self.cache_duration = 5.0  # 5-second cache
    
    async def get_current_resources(self) -> Dict[str, float]:
        """Get current system resource status"""
        current_time = time.time()
        
        # Use cache to avoid frequent system calls
        if current_time - self.last_check < self.cache_duration:
            return self.cached_resources
        
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            available_cpu = max(0.1, (100 - cpu_percent) / 100.0)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            available_memory = max(0.1, memory.available / memory.total)
            
            # Estimate available worker threads
            cpu_count = psutil.cpu_count()
            available_workers = max(1, int(cpu_count * available_cpu))
            
            self.cached_resources = {
                'available_cpu': available_cpu,
                'available_memory': available_memory,
                'available_workers': available_workers,
                'cpu_count': cpu_count,
                'total_memory': memory.total
            }
            
            self.last_check = current_time
            return self.cached_resources
            
        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            # Return conservative default values
            return {
                'available_cpu': 0.5,
                'available_memory': 0.5,
                'available_workers': 2,
                'cpu_count': 4,
                'total_memory': 8 * 1024 * 1024 * 1024  # 8GB
            }


# Workflow integration interface
async def optimize_report_generation_workflow(user_query: str, 
                                            workflow_type: str = 'research',
                                            max_workers: int = 4) -> Dict[str, Any]:
    """Optimize parallel processing for report generation workflow"""
    try:
        # Create advanced parallel executor instance
        executor = AdvancedParallelExecutor(max_workers=max_workers, enable_metrics=True)
        await executor.start()
        
        try:
            # Execute optimized research task
            result = await executor.decompose_and_execute_research_task(user_query, workflow_type)
            
            # Add performance metrics
            performance_metrics = await executor.get_metrics()
            result['performance_metrics'] = performance_metrics
            
            return result
            
        finally:
            await executor.stop()
            
    except Exception as e:
        logger.error(f"Workflow optimization failed: {e}")
        return {
            'error': f'Workflow optimization failed: {e}',
            'fallback_used': True
        }


# Global instance (optional)
_global_advanced_executor = None


async def get_global_advanced_executor(max_workers: int = 4) -> 'AdvancedParallelExecutor':
    """Get global advanced parallel executor instance"""
    global _global_advanced_executor
    
    if _global_advanced_executor is None:
        _global_advanced_executor = AdvancedParallelExecutor(max_workers=max_workers, enable_metrics=True)
        await _global_advanced_executor.start()
    
    return _global_advanced_executor


async def shutdown_global_executor():
    """Shutdown global executor"""
    global _global_advanced_executor
    
    if _global_advanced_executor is not None:
        await _global_advanced_executor.stop()
        _global_advanced_executor = None


class AdaptiveLoadBalancer:
    """Adaptive load balancer that dynamically adjusts task allocation to optimize system performance"""
    
    def __init__(self):
        self.worker_states = {}
        self.load_history = deque(maxlen=100)
        self.performance_tracker = {}
    
    async def balance_load(self, tasks: List[ParallelTask], worker_count: int = 4) -> Dict[int, List[ParallelTask]]:
        """Balance load across available worker nodes"""
        try:
            # Initialize worker node states
            if not self.worker_states:
                self.worker_states = {i: {'current_load': 0.0, 'task_count': 0, 'performance_score': 1.0} 
                                    for i in range(worker_count)}
            
            # Predict load for each task
            task_loads = {}
            for task in tasks:
                load = await self._predict_task_load(task)
                task_loads[task.task_id] = load
            
            # Use min-heap for load balancing
            worker_heap = [(state['current_load'], worker_id) 
                          for worker_id, state in self.worker_states.items()]
            heapq.heapify(worker_heap)
            
            # Allocate tasks
            task_assignment = {worker_id: [] for worker_id in self.worker_states.keys()}
            
            # Sort tasks by load from high to low
            sorted_tasks = sorted(tasks, 
                                key=lambda t: (t.priority.value, task_loads[t.task_id]), 
                                reverse=True)
            
            for task in sorted_tasks:
                # Select worker node with lowest load
                current_load, worker_id = heapq.heappop(worker_heap)
                
                # Assign task
                task_assignment[worker_id].append(task)
                
                # Update worker node load
                task_load = task_loads[task.task_id]
                new_load = current_load + task_load
                
                # Update state
                self.worker_states[worker_id]['current_load'] = new_load
                self.worker_states[worker_id]['task_count'] += 1
                
                heapq.heappush(worker_heap, (new_load, worker_id))
            
            return task_assignment
            
        except Exception as e:
            logger.error(f"Load balancing failed: {e}")
            # Simple round-robin allocation as fallback
            return self._simple_round_robin(tasks, worker_count)
    
    async def _predict_task_load(self, task: ParallelTask) -> float:
        """Predict task load"""
        # Predict load based on task type and historical data
        base_load = task.estimated_duration / 60.0  # 转换为分钟
        
        # Adjust based on resource requirements
        cpu_factor = task.resource_requirements.get('cpu', 0.2)
        memory_factor = task.resource_requirements.get('memory', 0.1)
        
        predicted_load = base_load * (1 + cpu_factor + memory_factor)
        
        return min(predicted_load, 2.0)  # Maximum load limit
    
    def _round_robin_assignment(self, tasks: List[ParallelTask], worker_count: int) -> Dict[int, List[ParallelTask]]:
        """Simple round-robin assignment"""
        assignment = {i: [] for i in range(worker_count)}
        
        for i, task in enumerate(tasks):
            worker_id = i % worker_count
            assignment[worker_id].append(task)
        
        return assignment
    
    async def update_worker_performance(self, worker_id: int, task_id: str, 
                                      execution_time: float, success: bool):
        """Update worker node performance"""
        if worker_id in self.worker_states:
            # Update performance score
            current_score = self.worker_states[worker_id]['performance_score']
            
            if success:
                # Successful execution, slightly improve score
                new_score = min(1.0, current_score + 0.01)
            else:
                # Execution failed, reduce score
                new_score = max(0.1, current_score - 0.05)
            
            self.worker_states[worker_id]['performance_score'] = new_score
            
            # Record performance history
            self.performance_tracker[f"{worker_id}_{task_id}"] = {
                'execution_time': execution_time,
                'success': success,
                'timestamp': time.time()
            }


class TaskDependencyResolver:
    """Task dependency resolver"""
    
    def __init__(self):
        self.dependency_graph = {}
        self.completed_tasks = set()
    
    def build_dependency_graph(self, tasks: List[ParallelTask]) -> Dict[str, List[str]]:
        """Build task dependency graph"""
        self.dependency_graph = {}
        
        for task in tasks:
            self.dependency_graph[task.task_id] = task.dependencies.copy()
        
        return self.dependency_graph
    
    def get_ready_tasks(self, tasks: List[ParallelTask]) -> List[ParallelTask]:
        """Get tasks that can be executed immediately (dependencies satisfied)"""
        ready_tasks = []
        
        for task in tasks:
            if self._are_dependencies_satisfied(task.task_id):
                ready_tasks.append(task)
        
        return ready_tasks
    
    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if task dependencies are satisfied"""
        dependencies = self.dependency_graph.get(task_id, [])
        return all(dep in self.completed_tasks for dep in dependencies)
    
    def mark_task_completed(self, task_id: str):
        """Mark task as completed"""
        self.completed_tasks.add(task_id)
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies"""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                dfs(neighbor, path + [neighbor])
            
            rec_stack.remove(node)
        
        for task_id in self.dependency_graph:
            if task_id not in visited:
                dfs(task_id, [task_id])
        
        return cycles


class TaskPriority(Enum):
    """Task priority levels for execution scheduling."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskMetrics:
    """Metrics for task execution."""
    task_id: str
    start_time: float
    end_time: Optional[float] = None
    execution_time: Optional[float] = None
    status: str = "pending"
    error: Optional[str] = None
    retry_count: int = 0


class AdvancedParallelExecutor:
    """Advanced parallel executor with task priority queues, load balancing and intelligent scheduling"""
    
    def __init__(self, max_workers: int = 4, enable_metrics: bool = True):
        self.max_workers = max_workers
        self.enable_metrics = enable_metrics
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)
        self.task_queues = {
            TaskPriority.CRITICAL: asyncio.Queue(),
            TaskPriority.HIGH: asyncio.Queue(),
            TaskPriority.NORMAL: asyncio.Queue(),
            TaskPriority.LOW: asyncio.Queue(),
        }
        self.active_tasks: Dict[str, TaskMetrics] = {}
        self.completed_tasks: List[TaskMetrics] = []
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        
        # New: Intelligent scheduling components
        self.task_decomposer = IntelligentTaskDecomposer()
        self.resource_allocator = DynamicResourceAllocator()
        self.load_balancer = AdaptiveLoadBalancer()
        self.dependency_resolver = TaskDependencyResolver()
    
    async def start(self):
        """Start the parallel executor."""
        if self.is_running:
            return
        
        self.is_running = True
        # Start worker tasks for each priority level
        for priority in TaskPriority:
            worker_task = asyncio.create_task(self._worker(priority))
            self.worker_tasks.append(worker_task)
        
        logger.info(f"Advanced parallel executor started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the parallel executor."""
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Advanced parallel executor stopped")
    
    async def submit_task(
        self,
        task_func: Callable[..., Awaitable[Any]],
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Submit a task for execution."""
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}"
        
        task_data = {
            "task_id": task_id,
            "func": task_func,
            "args": args,
            "kwargs": kwargs,
        }
        
        # Create task metrics
        if self.enable_metrics:
            metrics = TaskMetrics(task_id=task_id, start_time=time.time())
            self.active_tasks[task_id] = metrics
        
        # Add to appropriate priority queue
        await self.task_queues[priority].put(task_data)
        
        logger.debug(f"Task {task_id} submitted with priority {priority.name}")
        return task_id
    
    async def submit_parallel_tasks(self, tasks: List[ParallelTask]) -> List[str]:
        """智能提交并行任务，支持依赖解析和资源分配"""
        try:
            # 1. 检测循环依赖
            self.dependency_resolver.build_dependency_graph(tasks)
            cycles = self.dependency_resolver.detect_circular_dependencies()
            if cycles:
                logger.warning(f"Detected circular dependencies: {cycles}")
                # 移除循环依赖
                tasks = self._resolve_circular_dependencies(tasks, cycles)
            
            # 2. 动态资源分配
            resource_allocation = await self.resource_allocator.allocate_resources(tasks)
            
            # 3. 负载均衡
            load_balanced_assignment = await self.load_balancer.balance_load(tasks, self.max_workers)
            
            # 4. 按批次提交任务
            submitted_task_ids = []
            ready_tasks = self.dependency_resolver.get_ready_tasks(tasks)
            
            for task in ready_tasks:
                # 应用资源分配
                allocation = resource_allocation.get(task.task_id, {})
                priority_boost = allocation.get('priority_boost', 1.0)
                
                # 调整优先级
                adjusted_priority = self._adjust_priority(task.priority, priority_boost)
                
                # Submit task
                task_id = await self.submit_task(
                    self._execute_parallel_task,
                    task,
                    allocation,
                    priority=adjusted_priority
                )
                submitted_task_ids.append(task_id)
            
            return submitted_task_ids
            
        except Exception as e:
            logger.error(f"Failed to submit parallel tasks: {e}")
            # Fallback to simple submission
            return [await self.submit_task(task.func, *task.args, priority=task.priority, **task.kwargs) 
                   for task in tasks]
    
    async def decompose_and_execute_research_task(self, user_query: str, task_type: str = 'research') -> Dict[str, Any]:
        """Decompose and execute research task"""
        try:
            # 1. Intelligent task decomposition
            parallel_tasks = await self.task_decomposer.decompose_research_task(user_query, task_type)
            
            if not parallel_tasks:
                logger.warning("No parallel tasks generated from query decomposition")
                return {'error': 'Task decomposition failed'}
            
            # 2. Submit parallel tasks
            task_ids = await self.submit_parallel_tasks(parallel_tasks)
            
            # 3. Wait for task completion and collect results
            results = {}
            for task_id, parallel_task in zip(task_ids, parallel_tasks):
                try:
                    # Wait for task completion (simplified implementation, should have more complex waiting mechanism)
                    await asyncio.sleep(0.1)  # Give tasks some execution time
                    results[parallel_task.task_id] = f"Result for {parallel_task.task_id}"
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
                    results[parallel_task.task_id] = {'error': str(e)}
            
            return {
                'query': user_query,
                'task_type': task_type,
                'parallel_tasks_count': len(parallel_tasks),
                'results': results,
                'execution_summary': await self._get_execution_summary()
            }
            
        except Exception as e:
            logger.error(f"Research task execution failed: {e}")
            return {'error': f'Research task execution failed: {e}'}
    
    def _resolve_circular_dependencies(self, tasks: List[ParallelTask], cycles: List[List[str]]) -> List[ParallelTask]:
        """Resolve circular dependencies"""
        # Simplified implementation: remove dependencies that cause cycles
        cycle_tasks = set()
        for cycle in cycles:
            cycle_tasks.update(cycle)
        
        for task in tasks:
            if task.task_id in cycle_tasks:
                # Clear dependencies that may cause cycles
                task.dependencies = [dep for dep in task.dependencies if dep not in cycle_tasks]
        
        return tasks
    
    def _adjust_priority(self, base_priority: TaskPriority, boost: float) -> TaskPriority:
        """Adjust task priority based on resource allocation"""
        if boost >= 1.0:
            return base_priority
        elif boost >= 0.8:
            # Slight downgrade
            priority_map = {
                TaskPriority.CRITICAL: TaskPriority.HIGH,
                TaskPriority.HIGH: TaskPriority.NORMAL,
                TaskPriority.NORMAL: TaskPriority.LOW,
                TaskPriority.LOW: TaskPriority.LOW
            }
            return priority_map.get(base_priority, TaskPriority.LOW)
        else:
            # Significant downgrade
            return TaskPriority.LOW
    
    async def _execute_parallel_task(self, task: ParallelTask, allocation: Dict[str, Any]) -> Any:
        """Execute parallel task"""
        start_time = time.time()
        
        try:
            # Apply resource limits (simplified implementation)
            if allocation.get('workers', 1) == 0:
                # Insufficient resources, delay execution
                await asyncio.sleep(1.0)
            
            # Execute task
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                # Execute synchronous function in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, task.func, *task.args
                )
            
            execution_time = time.time() - start_time
            
            # Mark task as completed
            self.dependency_resolver.mark_task_completed(task.task_id)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Parallel task {task.task_id} failed: {e}")
            raise e
    
    async def _get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        total_completed = len(self.completed_tasks)
        successful_tasks = len([t for t in self.completed_tasks if t.status == "completed"])
        failed_tasks = len([t for t in self.completed_tasks if t.status == "failed"])
        
        avg_execution_time = 0
        if self.completed_tasks:
            total_time = sum(t.execution_time or 0 for t in self.completed_tasks)
            avg_execution_time = total_time / total_completed
        
        return {
            'total_tasks': len(self.active_tasks) + total_completed,
            'completed_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': successful_tasks / max(total_completed, 1),
            'average_execution_time': avg_execution_time,
            'active_tasks': len(self.active_tasks),
            'queue_lengths': {priority.name: queue.qsize() for priority, queue in self.task_queues.items()}
        }
    
    async def _worker(self, priority: TaskPriority):
        """Worker coroutine for processing tasks of a specific priority."""
        queue = self.task_queues[priority]
        
        while self.is_running:
            try:
                # Get task from queue with timeout
                task_data = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                # Acquire semaphore for concurrency control
                async with self.semaphore:
                    await self._execute_task(task_data)
                
            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker for priority {priority.name}: {e}")
    
    async def _execute_task(self, task_data: Dict[str, Any]):
        """Execute a single task."""
        task_id = task_data["task_id"]
        func = task_data["func"]
        args = task_data["args"]
        kwargs = task_data["kwargs"]
        
        try:
            # Update metrics
            if self.enable_metrics and task_id in self.active_tasks:
                self.active_tasks[task_id].status = "running"
            
            # Execute the task
            result = await func(*args, **kwargs)
            
            # Update metrics on success
            if self.enable_metrics and task_id in self.active_tasks:
                metrics = self.active_tasks[task_id]
                metrics.end_time = time.time()
                metrics.execution_time = metrics.end_time - metrics.start_time
                metrics.status = "completed"
                
                # Move to completed tasks
                self.completed_tasks.append(metrics)
                del self.active_tasks[task_id]
            
            logger.debug(f"Task {task_id} completed successfully")
            
        except Exception as e:
            # Update metrics on error
            if self.enable_metrics and task_id in self.active_tasks:
                metrics = self.active_tasks[task_id]
                metrics.end_time = time.time()
                metrics.execution_time = metrics.end_time - metrics.start_time
                metrics.status = "failed"
                metrics.error = str(e)
                
                # Move to completed tasks
                self.completed_tasks.append(metrics)
                del self.active_tasks[task_id]
            
            logger.error(f"Task {task_id} failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        if not self.enable_metrics:
            return {"metrics_enabled": False}
        
        total_completed = len(self.completed_tasks)
        successful_tasks = len([t for t in self.completed_tasks if t.status == "completed"])
        failed_tasks = len([t for t in self.completed_tasks if t.status == "failed"])
        
        avg_execution_time = 0
        if self.completed_tasks:
            total_time = sum(t.execution_time or 0 for t in self.completed_tasks)
            avg_execution_time = total_time / total_completed
        
        return {
            "metrics_enabled": True,
            "active_tasks": len(self.active_tasks),
            "total_completed": total_completed,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": successful_tasks / total_completed if total_completed > 0 else 0,
            "average_execution_time": avg_execution_time,
            "max_workers": self.max_workers,
        }


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load."""
    
    def __init__(self, initial_rate: int = 100, time_window: int = 60):
        self.initial_rate = initial_rate
        self.current_rate = initial_rate
        self.time_window = time_window
        self.requests = []
        self.lock = asyncio.Lock()
        self.adaptation_enabled = True
        self.last_adaptation = time.time()
        self.adaptation_interval = 30  # Adapt every 30 seconds
    
    async def acquire(self) -> bool:
        """Acquire a rate limit token."""
        async with self.lock:
            current_time = time.time()
            
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests 
                           if current_time - req_time < self.time_window]
            
            # Check if we can make a request
            if len(self.requests) < self.current_rate:
                self.requests.append(current_time)
                
                # Adaptive rate adjustment
                if self.adaptation_enabled and current_time - self.last_adaptation > self.adaptation_interval:
                    await self._adapt_rate()
                    self.last_adaptation = current_time
                
                return True
            
            return False
    
    async def _adapt_rate(self):
        """Adapt the rate limit based on system performance."""
        try:
            # Simple adaptation logic - increase rate if system is performing well
            current_usage = len(self.requests) / self.current_rate
            
            if current_usage < 0.7:  # Low usage, can increase rate
                self.current_rate = min(self.current_rate * 1.1, self.initial_rate * 2)
                logger.debug(f"Rate limit increased to {self.current_rate}")
            elif current_usage > 0.9:  # High usage, decrease rate
                self.current_rate = max(self.current_rate * 0.9, self.initial_rate * 0.5)
                logger.debug(f"Rate limit decreased to {self.current_rate}")
            
        except Exception as e:
            logger.error(f"Error in rate adaptation: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        current_time = time.time()
        recent_requests = [req_time for req_time in self.requests 
                          if current_time - req_time < self.time_window]
        
        return {
            "current_rate": self.current_rate,
            "initial_rate": self.initial_rate,
            "recent_requests": len(recent_requests),
            "usage_percentage": (len(recent_requests) / self.current_rate) * 100,
            "time_window": self.time_window,
        }


class SmartErrorRecovery:
    """Smart error recovery with exponential backoff and circuit breaker."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.failure_counts: Dict[str, int] = {}
        self.circuit_breaker_states: Dict[str, str] = {}  # "closed", "open", "half-open"
        self.last_failure_times: Dict[str, float] = {}
        self.circuit_breaker_timeout = 60.0  # 1 minute
    
    async def execute_with_recovery(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        operation_id: str = "default",
        **kwargs
    ) -> Any:
        """Execute function with smart error recovery."""
        
        # Check circuit breaker
        if not self._can_execute(operation_id):
            raise RuntimeError(f"Circuit breaker is open for operation: {operation_id}")
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                
                # Reset failure count on success
                self.failure_counts[operation_id] = 0
                self._close_circuit_breaker(operation_id)
                
                return result
                
            except Exception as e:
                last_exception = e
                self.failure_counts[operation_id] = self.failure_counts.get(operation_id, 0) + 1
                self.last_failure_times[operation_id] = time.time()
                
                logger.warning(f"Attempt {attempt + 1} failed for {operation_id}: {e}")
                
                # Check if we should open circuit breaker
                if self.failure_counts[operation_id] >= self.max_retries:
                    self._open_circuit_breaker(operation_id)
                
                # Don't wait after the last attempt
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        raise last_exception
    
    def _can_execute(self, operation_id: str) -> bool:
        """Check if operation can be executed based on circuit breaker state."""
        state = self.circuit_breaker_states.get(operation_id, "closed")
        
        if state == "closed":
            return True
        elif state == "open":
            # Check if we should transition to half-open
            last_failure = self.last_failure_times.get(operation_id, 0)
            if time.time() - last_failure > self.circuit_breaker_timeout:
                self.circuit_breaker_states[operation_id] = "half-open"
                return True
            return False
        elif state == "half-open":
            return True
        
        return False
    
    def _open_circuit_breaker(self, operation_id: str):
        """Open circuit breaker for an operation."""
        self.circuit_breaker_states[operation_id] = "open"
        logger.warning(f"Circuit breaker opened for operation: {operation_id}")
    
    def _close_circuit_breaker(self, operation_id: str):
        """Close circuit breaker for an operation."""
        self.circuit_breaker_states[operation_id] = "closed"
        logger.info(f"Circuit breaker closed for operation: {operation_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error recovery statistics."""
        return {
            "failure_counts": dict(self.failure_counts),
            "circuit_breaker_states": dict(self.circuit_breaker_states),
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
        }


# Global instances for easy access
parallel_executor = AdvancedParallelExecutor()
rate_limiter = AdaptiveRateLimiter()
error_recovery = SmartErrorRecovery()