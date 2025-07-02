#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行执行和速率限制功能测试
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from src.utils.rate_limiter import RateLimiter, RateLimitConfig, TokenBucket, AdaptiveRateLimiter
from src.utils.parallel_executor import (
    ParallelExecutor, ParallelTask, TaskPriority, TaskStatus, TaskResult,
    create_parallel_executor
)
from src.config.parallel_config import (
    ParallelExecutionConfig, RateLimitConfig as ParallelRateLimitConfig, 
    CombinedConfig
)


class TestTokenBucket:
    """令牌桶测试"""
    
    def test_token_bucket_initialization(self):
        """测试令牌桶初始化"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.capacity == 10
        assert bucket.tokens == 10
        assert bucket.refill_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_token_bucket_acquire_immediate(self):
        """测试立即获取令牌"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        # 应该立即获取到令牌
        delay = await bucket.acquire(1)
        assert delay == 0.0
        assert bucket.tokens == 9
    
    @pytest.mark.asyncio
    async def test_token_bucket_acquire_with_delay(self):
        """测试需要等待的令牌获取"""
        bucket = TokenBucket(capacity=2, refill_rate=1.0)
        
        # 消耗所有令牌
        await bucket.acquire(2)
        assert bucket.tokens == 0
        
        # 下一次获取应该需要等待
        start_time = time.time()
        delay = await bucket.acquire(1)
        end_time = time.time()
        
        assert delay > 0
        assert end_time - start_time >= delay * 0.9  # 允许一些时间误差
    
    @pytest.mark.asyncio
    async def test_token_bucket_refill(self):
        """测试令牌补充"""
        bucket = TokenBucket(capacity=5, refill_rate=2.0)
        
        # 消耗所有令牌
        await bucket.acquire(5)
        assert bucket.tokens == 0
        
        # 等待一段时间让令牌补充
        await asyncio.sleep(1.1)  # 稍微超过0.5秒（补充1个令牌的时间）
        
        # 应该能够获取到补充的令牌
        delay = await bucket.acquire(1)
        assert delay == 0.0


class TestRateLimiter:
    """速率限制器测试"""
    
    def test_rate_limiter_initialization(self):
        """测试速率限制器初始化"""
        config = RateLimitConfig(
            requests_per_minute=60,
            burst_capacity=10
        )
        limiter = RateLimiter(config)
        
        assert limiter.config.requests_per_minute == 60
        assert limiter.config.burst_capacity == 10
    
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self):
        """测试速率限制器获取许可"""
        config = RateLimitConfig(
            requests_per_minute=60,
            burst_capacity=5
        )
        limiter = RateLimiter(config)
        
        # 前几次请求应该立即通过
        for _ in range(3):
            delay = await limiter.acquire()
            assert delay == 0.0
    
    def test_rate_limiter_record_success(self):
        """测试记录成功"""
        config = RateLimitConfig()
        limiter = RateLimiter(config)
        
        initial_success = limiter.success_count
        limiter.record_success()
        assert limiter.success_count == initial_success + 1
    
    def test_rate_limiter_record_failure(self):
        """测试记录失败"""
        config = RateLimitConfig()
        limiter = RateLimiter(config)
        
        initial_failure = limiter.failure_count
        limiter.record_failure("Test error")
        assert limiter.failure_count == initial_failure + 1
    
    def test_rate_limiter_get_stats(self):
        """测试获取统计信息"""
        config = RateLimitConfig()
        limiter = RateLimiter(config)
        
        limiter.record_success()
        limiter.record_failure("Test error")
        
        stats = limiter.get_stats()
        assert 'total_requests' in stats
        assert 'success_count' in stats
        assert 'failure_count' in stats
        assert 'success_rate' in stats
        assert stats['success_count'] >= 1
        assert stats['failure_count'] >= 1


class TestAdaptiveRateLimiter:
    """自适应速率限制器测试"""
    
    def test_adaptive_rate_limiter_initialization(self):
        """测试自适应速率限制器初始化"""
        config = RateLimitConfig(
            requests_per_minute=60,
            adaptive_adjustment=True
        )
        limiter = AdaptiveRateLimiter(config)
        
        assert limiter.config.adaptive_adjustment is True
        assert limiter.current_rpm == 60
    
    def test_adaptive_rate_limiter_adjust_on_failure(self):
        """测试失败时的自适应调整"""
        config = RateLimitConfig(
            requests_per_minute=60,
            adaptive_adjustment=True,
            min_requests_per_minute=10
        )
        limiter = AdaptiveRateLimiter(config)
        
        initial_rpm = limiter.current_rpm
        
        # 记录多次失败
        for _ in range(5):
            limiter.record_failure("Rate limit exceeded")
        
        # 触发调整
        limiter._adjust_rate()
        
        # 速率应该降低
        assert limiter.current_rpm < initial_rpm
    
    def test_adaptive_rate_limiter_adjust_on_success(self):
        """测试成功时的自适应调整"""
        config = RateLimitConfig(
            requests_per_minute=30,  # 从较低的速率开始
            adaptive_adjustment=True,
            max_requests_per_minute=120
        )
        limiter = AdaptiveRateLimiter(config)
        
        initial_rpm = limiter.current_rpm
        
        # 记录多次成功
        for _ in range(20):
            limiter.record_success()
        
        # 触发调整
        limiter._adjust_rate()
        
        # 速率应该增加
        assert limiter.current_rpm >= initial_rpm


class TestParallelTask:
    """并行任务测试"""
    
    def test_parallel_task_initialization(self):
        """测试并行任务初始化"""
        def dummy_func():
            return "test"
        
        task = ParallelTask(
            task_id="test-task",
            func=dummy_func,
            args=("arg1",),
            kwargs={"key": "value"},
            priority=TaskPriority.HIGH,
            max_retries=3,
            timeout=60.0,
            dependencies=["dep1", "dep2"]
        )
        
        assert task.task_id == "test-task"
        assert task.func == dummy_func
        assert task.args == ("arg1",)
        assert task.kwargs == {"key": "value"}
        assert task.priority == TaskPriority.HIGH
        assert task.max_retries == 3
        assert task.timeout == 60.0
        assert task.dependencies == ["dep1", "dep2"]
    
    def test_parallel_task_default_values(self):
        """测试并行任务默认值"""
        def dummy_func():
            return "test"
        
        task = ParallelTask(
            task_id="test-task",
            func=dummy_func
        )
        
        assert task.args == ()
        assert task.kwargs == {}
        assert task.priority == TaskPriority.NORMAL
        assert task.max_retries == 3
        assert task.timeout is None
        assert task.dependencies == []


class TestParallelExecutor:
    """并行执行器测试"""
    
    def test_parallel_executor_initialization(self):
        """测试并行执行器初始化"""
        executor = ParallelExecutor(
            max_concurrent_tasks=5,
            enable_adaptive_scheduling=True,
            task_timeout=120.0
        )
        
        assert executor.max_concurrent_tasks == 5
        assert executor.enable_adaptive_scheduling is True
        assert executor.task_timeout == 120.0
        assert len(executor.pending_tasks) == 0
        assert len(executor.running_tasks) == 0
        assert len(executor.completed_tasks) == 0
    
    def test_add_task(self):
        """测试添加任务"""
        executor = ParallelExecutor()
        
        def dummy_func():
            return "test"
        
        task = ParallelTask(
            task_id="test-task",
            func=dummy_func
        )
        
        executor.add_task(task)
        
        assert len(executor.pending_tasks) == 1
        assert executor.pending_tasks[0] == task
        assert executor.total_tasks == 1
    
    def test_add_tasks(self):
        """测试批量添加任务"""
        executor = ParallelExecutor()
        
        def dummy_func():
            return "test"
        
        tasks = [
            ParallelTask(task_id=f"task-{i}", func=dummy_func)
            for i in range(3)
        ]
        
        executor.add_tasks(tasks)
        
        assert len(executor.pending_tasks) == 3
        assert executor.total_tasks == 3
    
    @pytest.mark.asyncio
    async def test_execute_simple_task(self):
        """测试执行简单任务"""
        executor = ParallelExecutor(max_concurrent_tasks=1)
        
        async def simple_task(value):
            await asyncio.sleep(0.1)
            return f"result-{value}"
        
        task = ParallelTask(
            task_id="simple-task",
            func=simple_task,
            args=("test",)
        )
        
        executor.add_task(task)
        results = await executor.execute_all()
        
        assert len(results) == 1
        assert "simple-task" in results
        assert results["simple-task"].status == TaskStatus.COMPLETED
        assert results["simple-task"].result == "result-test"
    
    @pytest.mark.asyncio
    async def test_execute_multiple_tasks(self):
        """测试执行多个任务"""
        executor = ParallelExecutor(max_concurrent_tasks=2)
        
        async def task_func(task_id):
            await asyncio.sleep(0.1)
            return f"result-{task_id}"
        
        tasks = [
            ParallelTask(
                task_id=f"task-{i}",
                func=task_func,
                args=(i,)
            )
            for i in range(3)
        ]
        
        executor.add_tasks(tasks)
        results = await executor.execute_all()
        
        assert len(results) == 3
        for i in range(3):
            task_id = f"task-{i}"
            assert task_id in results
            assert results[task_id].status == TaskStatus.COMPLETED
            assert results[task_id].result == f"result-{i}"
    
    @pytest.mark.asyncio
    async def test_execute_with_dependencies(self):
        """测试执行有依赖关系的任务"""
        executor = ParallelExecutor(max_concurrent_tasks=2)
        
        execution_order = []
        
        async def task_func(task_id):
            execution_order.append(task_id)
            await asyncio.sleep(0.1)
            return f"result-{task_id}"
        
        tasks = [
            ParallelTask(
                task_id="task-1",
                func=task_func,
                args=("task-1",)
            ),
            ParallelTask(
                task_id="task-2",
                func=task_func,
                args=("task-2",),
                dependencies=["task-1"]
            ),
            ParallelTask(
                task_id="task-3",
                func=task_func,
                args=("task-3",),
                dependencies=["task-1", "task-2"]
            )
        ]
        
        executor.add_tasks(tasks)
        results = await executor.execute_all()
        
        assert len(results) == 3
        # 检查执行顺序
        assert execution_order.index("task-1") < execution_order.index("task-2")
        assert execution_order.index("task-2") < execution_order.index("task-3")
    
    @pytest.mark.asyncio
    async def test_execute_with_failure_and_retry(self):
        """测试执行失败和重试"""
        executor = ParallelExecutor(max_concurrent_tasks=1)
        
        call_count = 0
        
        async def failing_task():
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # 前两次失败
                raise Exception("Temporary failure")
            return "success"
        
        task = ParallelTask(
            task_id="failing-task",
            func=failing_task,
            max_retries=3
        )
        
        executor.add_task(task)
        results = await executor.execute_all()
        
        assert len(results) == 1
        assert "failing-task" in results
        assert results["failing-task"].status == TaskStatus.COMPLETED
        assert results["failing-task"].result == "success"
        assert results["failing-task"].retry_count == 2  # 重试了2次
        assert call_count == 3  # 总共调用了3次
    
    def test_get_stats(self):
        """测试获取统计信息"""
        executor = ParallelExecutor()
        
        def dummy_func():
            return "test"
        
        # 添加一些任务
        tasks = [
            ParallelTask(task_id=f"task-{i}", func=dummy_func)
            for i in range(3)
        ]
        executor.add_tasks(tasks)
        
        # 模拟一些执行结果
        executor.successful_tasks = 2
        executor.failed_tasks = 1
        
        stats = executor.get_stats()
        
        assert stats['total_tasks'] == 3
        assert stats['successful_tasks'] == 2
        assert stats['failed_tasks'] == 1
        assert stats['success_rate'] == 2/3
        assert stats['pending_tasks'] == 3
        assert stats['running_tasks'] == 0
        assert stats['completed_tasks'] == 0


class TestCreateParallelExecutor:
    """测试并行执行器创建函数"""
    
    @patch('src.utils.parallel_executor.config_loader')
    def test_create_parallel_executor_with_config(self, mock_config_loader):
        """测试从配置创建并行执行器"""
        mock_config_loader.load_config.return_value = {
            'max_parallel_tasks': 5
        }
        
        executor = create_parallel_executor()
        
        assert executor.max_concurrent_tasks == 5
        assert executor.rate_limiter is not None
        assert executor.enable_adaptive_scheduling is True
    
    def test_create_parallel_executor_with_params(self):
        """测试使用参数创建并行执行器"""
        executor = create_parallel_executor(
            max_concurrent_tasks=3,
            enable_rate_limiting=False,
            enable_adaptive_scheduling=False
        )
        
        assert executor.max_concurrent_tasks == 3
        assert executor.rate_limiter is None
        assert executor.enable_adaptive_scheduling is False


class TestParallelConfig:
    """测试并行配置"""
    
    def test_parallel_execution_config_from_env(self):
        """测试从环境变量加载并行执行配置"""
        with patch.dict('os.environ', {
            'ENABLE_PARALLEL_EXECUTION': 'false',
            'MAX_PARALLEL_TASKS': '5',
            'TASK_TIMEOUT': '600.0'
        }):
            config = ParallelExecutionConfig.from_env()
            
            assert config.enable_parallel_execution is False
            assert config.max_parallel_tasks == 5
            assert config.task_timeout == 600.0
    
    def test_rate_limit_config_from_env(self):
        """测试从环境变量加载速率限制配置"""
        with patch.dict('os.environ', {
            'ENABLE_RATE_LIMITING': 'false',
            'REQUESTS_PER_MINUTE': '30',
            'BURST_CAPACITY': '5'
        }):
            config = ParallelRateLimitConfig.from_env()
            
            assert config.enable_rate_limiting is False
            assert config.requests_per_minute == 30
            assert config.burst_capacity == 5
    
    def test_combined_config_from_env(self):
        """测试从环境变量加载组合配置"""
        with patch.dict('os.environ', {
            'MAX_PARALLEL_TASKS': '4',
            'REQUESTS_PER_MINUTE': '40'
        }):
            config = CombinedConfig.from_env()
            
            assert config.parallel.max_parallel_tasks == 4
            assert config.rate_limit.requests_per_minute == 40
    
    def test_combined_config_to_dict(self):
        """测试组合配置转换为字典"""
        config = CombinedConfig.from_env()
        config_dict = config.to_dict()
        
        assert 'parallel_execution' in config_dict
        assert 'rate_limiting' in config_dict
        assert 'optimization' in config_dict
        
        assert 'max_parallel_tasks' in config_dict['parallel_execution']
        assert 'requests_per_minute' in config_dict['rate_limiting']


@pytest.mark.integration
class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_with_rate_limiting(self):
        """测试完整的并行执行和速率限制流水线"""
        # 创建速率限制配置
        rate_config = RateLimitConfig(
            requests_per_minute=30,  # 较低的速率
            burst_capacity=3
        )
        
        # 创建并行执行器
        executor = ParallelExecutor(
            max_concurrent_tasks=2,
            rate_limiter=RateLimiter(rate_config),
            enable_adaptive_scheduling=False  # 禁用自适应以便测试
        )
        
        request_times = []
        
        async def rate_limited_task(task_id):
            request_times.append(time.time())
            await asyncio.sleep(0.1)  # 模拟处理时间
            return f"result-{task_id}"
        
        # 创建多个任务
        tasks = [
            ParallelTask(
                task_id=f"task-{i}",
                func=rate_limited_task,
                args=(i,)
            )
            for i in range(6)  # 6个任务，应该触发速率限制
        ]
        
        executor.add_tasks(tasks)
        
        start_time = time.time()
        results = await executor.execute_all()
        end_time = time.time()
        
        # 验证结果
        assert len(results) == 6
        assert all(r.status == TaskStatus.COMPLETED for r in results.values())
        
        # 验证速率限制生效（执行时间应该比无限制时更长）
        execution_time = end_time - start_time
        assert execution_time > 1.0  # 由于速率限制，应该需要更长时间
        
        # 验证请求时间间隔
        if len(request_times) > 1:
            intervals = [request_times[i] - request_times[i-1] for i in range(1, len(request_times))]
            # 某些间隔应该大于最小间隔（由于速率限制）
            assert any(interval > 0.5 for interval in intervals)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])