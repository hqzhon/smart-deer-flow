"""性能优化模块测试"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.utils.researcher.performance_monitor import (
    PerformanceMonitor,
    GlobalPerformanceTracker,
)
from src.utils.researcher.config_cache_optimizer import (
    ConfigCacheOptimizer,
    SmartCache,
)
from src.utils.researcher.concurrent_optimizer import (
    ConcurrentTaskManager,
    TaskPriority,
    TaskStatus,
    AdaptiveSemaphore,
)


class TestPerformanceMonitor:
    """性能监控器测试"""

    def test_performance_monitor_initialization(self):
        """测试性能监控器初始化"""
        monitor = PerformanceMonitor("test_monitor")
        assert monitor.name == "test_monitor"
        assert not monitor.monitoring_active
        assert monitor.metrics.start_time is not None  # 初始化时会设置默认时间

    def test_start_stop_monitoring(self):
        """测试开始和停止监控"""
        monitor = PerformanceMonitor("test_monitor")

        # 开始监控
        monitor.start_monitoring()
        assert monitor.monitoring_active
        assert monitor.metrics.start_time is not None

        # 停止监控
        monitor.stop_monitoring()
        assert not monitor.monitoring_active

    def test_task_tracking(self):
        """测试任务跟踪"""
        monitor = PerformanceMonitor("test_monitor")
        monitor.start_monitoring()

        # 记录任务
        initial_count = monitor.metrics.task_count
        monitor.record_task_start()
        assert monitor.metrics.task_count == initial_count + 1

        # 记录成功
        initial_success = monitor.metrics.success_count
        monitor.record_task_success()
        assert monitor.metrics.success_count == initial_success + 1

        # 记录错误
        initial_error = monitor.metrics.error_count
        monitor.record_task_error()
        assert monitor.metrics.error_count == initial_error + 1

    def test_custom_metrics(self):
        """测试自定义指标"""
        monitor = PerformanceMonitor("test_monitor")
        monitor.start_monitoring()

        # 添加自定义指标
        monitor.add_custom_metric("test_metric", "test_value")
        assert "test_metric" in monitor.metrics.custom_metrics
        assert monitor.metrics.custom_metrics["test_metric"] == "test_value"

    @pytest.mark.asyncio
    async def test_async_monitoring_context(self):
        """测试异步监控上下文"""
        monitor = PerformanceMonitor("test_monitor")

        async with monitor.monitor_context(sample_interval=0.1):
            await asyncio.sleep(0.2)
            monitor.add_custom_metric("test", "value")

        # 检查监控数据
        assert len(monitor.cpu_samples) > 0 or len(monitor.memory_samples) > 0
        assert "test" in monitor.metrics.custom_metrics

    def test_performance_summary(self):
        """测试性能摘要"""
        monitor = PerformanceMonitor("test_monitor")
        monitor.start_monitoring()

        # 添加一些数据
        monitor.record_task_start()
        time.sleep(0.1)
        monitor.record_task_success()
        monitor.add_custom_metric("test_metric", 100)

        monitor.stop_monitoring()

        # 获取摘要
        summary = monitor.get_performance_summary()
        assert "monitor_name" in summary
        assert "performance_summary" in summary
        assert "raw_metrics" in summary


class TestGlobalPerformanceTracker:
    """全局性能跟踪器测试"""

    def test_create_monitor(self):
        """测试创建监控器"""
        tracker = GlobalPerformanceTracker()

        monitor = tracker.create_monitor("test_monitor")
        assert isinstance(monitor, PerformanceMonitor)
        assert monitor.name == "test_monitor"
        assert "test_monitor" in tracker.monitors

    def test_get_monitor(self):
        """测试获取监控器"""
        tracker = GlobalPerformanceTracker()

        # 创建监控器
        monitor1 = tracker.create_monitor("test_monitor")

        # 获取监控器
        monitor2 = tracker.get_monitor("test_monitor")
        assert monitor1 is monitor2

        # 获取不存在的监控器
        monitor3 = tracker.get_monitor("nonexistent")
        assert monitor3 is None

    def test_cleanup_inactive_monitors(self):
        """测试清理非活跃监控器"""
        tracker = GlobalPerformanceTracker()

        # 创建监控器
        monitor = tracker.create_monitor("test_monitor")
        monitor.start_monitoring()
        monitor.stop_monitoring()

        # 模拟旧的停止时间
        monitor.metrics.end_time = datetime.now() - timedelta(hours=2)

        # 清理
        tracker.cleanup_inactive_monitors(max_age_hours=1)
        assert "test_monitor" not in tracker.monitors


class TestSmartCache:
    """智能缓存测试"""

    def test_cache_basic_operations(self):
        """测试缓存基本操作"""
        cache = SmartCache(max_size=3)

        # 设置和获取
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # 获取不存在的键
        assert cache.get("nonexistent") is None

        # 删除
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("nonexistent") is False

    def test_cache_ttl(self):
        """测试缓存TTL"""
        cache = SmartCache(default_ttl=1)  # 1秒TTL

        # 设置缓存
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # 等待过期
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_cache_lru_eviction(self):
        """测试LRU驱逐"""
        cache = SmartCache(max_size=2)

        # 填满缓存
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # 访问key1使其成为最近使用
        cache.get("key1")

        # 添加新键，应该驱逐key2
        cache.set("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

    def test_cache_cleanup_expired(self):
        """测试清理过期条目"""
        cache = SmartCache(default_ttl=1)

        # 添加条目
        cache.set("key1", "value1")
        cache.set("key2", "value2", ttl=2)

        # 等待部分过期
        time.sleep(1.1)

        # 清理过期条目
        cleaned = cache.cleanup_expired()
        assert cleaned == 1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_cache_stats(self):
        """测试缓存统计"""
        cache = SmartCache(max_size=10)

        # 添加一些数据
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # 访问数据
        cache.get("key1")  # 命中
        cache.get("key3")  # 未命中

        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 10
        assert stats["hit_rate"] == 0.5
        assert stats["stats"]["hits"] == 1
        assert stats["stats"]["misses"] == 1


class TestConfigCacheOptimizer:
    """配置缓存优化器测试"""

    def test_config_hashing(self):
        """测试配置哈希"""
        optimizer = ConfigCacheOptimizer()

        config1 = {"key1": "value1", "key2": "value2"}
        config2 = {"key2": "value2", "key1": "value1"}  # 相同内容，不同顺序
        config3 = {"key1": "value1", "key2": "different"}

        hash1 = optimizer.get_config_hash(config1)
        hash2 = optimizer.get_config_hash(config2)
        hash3 = optimizer.get_config_hash(config3)

        assert hash1 == hash2  # 相同内容应该有相同哈希
        assert hash1 != hash3  # 不同内容应该有不同哈希

    def test_config_caching(self):
        """测试配置缓存"""
        optimizer = ConfigCacheOptimizer()

        config = {"key1": "value1"}
        processed_config = {"processed": True, **config}

        # 缓存配置
        cache_key = optimizer.cache_config("test_config", config, processed_config)
        assert cache_key is not None

        # 获取缓存的配置
        cached = optimizer.get_cached_config("test_config", config)
        assert cached == processed_config

        # 配置变化后应该缓存未命中
        modified_config = {"key1": "value2"}
        cached_modified = optimizer.get_cached_config("test_config", modified_config)
        assert cached_modified is None

    def test_config_change_detection(self):
        """测试配置变化检测"""
        optimizer = ConfigCacheOptimizer()

        config = {"key1": "value1"}

        # 初始状态，应该检测为变化
        assert optimizer.is_config_changed("test_config", config) is True

        # 缓存配置后
        optimizer.cache_config("test_config", config, config)
        assert optimizer.is_config_changed("test_config", config) is False

        # 配置变化后
        modified_config = {"key1": "value2"}
        assert optimizer.is_config_changed("test_config", modified_config) is True

    def test_tool_caching(self):
        """测试工具缓存"""
        optimizer = ConfigCacheOptimizer()

        tool_config = {"tool_type": "test"}
        tool_instance = Mock()

        # 缓存工具
        cache_key = optimizer.cache_tool("test_tool", tool_config, tool_instance)
        assert cache_key is not None

        # 获取缓存的工具
        cached_tool = optimizer.get_cached_tool("test_tool", tool_config)
        assert cached_tool is tool_instance

        # 配置变化后应该缓存未命中
        modified_config = {"tool_type": "different"}
        cached_modified = optimizer.get_cached_tool("test_tool", modified_config)
        assert cached_modified is None

    def test_cache_invalidation(self):
        """测试缓存失效"""
        optimizer = ConfigCacheOptimizer()

        # 添加一些缓存
        config = {"key1": "value1"}
        tool_config = {"tool_type": "test"}
        tool_instance = Mock()

        optimizer.cache_config("test_config", config, config)
        optimizer.cache_tool("test_tool", tool_config, tool_instance)

        # 失效特定配置缓存
        optimizer.invalidate_config_cache("test_config")
        assert optimizer.get_cached_config("test_config", config) is None

        # 失效特定工具缓存
        optimizer.invalidate_tool_cache("test_tool")
        assert optimizer.get_cached_tool("test_tool", tool_config) is None

    def test_cache_cleanup(self):
        """测试缓存清理"""
        optimizer = ConfigCacheOptimizer(config_ttl=1, tool_ttl=1)

        # 添加缓存
        config = {"key1": "value1"}
        tool_config = {"tool_type": "test"}
        tool_instance = Mock()

        optimizer.cache_config("test_config", config, config)
        optimizer.cache_tool("test_tool", tool_config, tool_instance)

        # 等待过期
        time.sleep(1.1)

        # 清理过期缓存
        cleanup_stats = optimizer.cleanup_expired()
        assert cleanup_stats["config_expired"] >= 0
        assert cleanup_stats["tool_expired"] >= 0


class TestAdaptiveSemaphore:
    """自适应信号量测试"""

    def test_basic_acquire_release(self):
        """测试基本的获取和释放"""
        semaphore = AdaptiveSemaphore(initial_value=2)

        # 获取信号量
        assert semaphore.acquire() is True
        assert semaphore.acquire() is True

        # 第三次获取应该阻塞（使用非阻塞模式测试）
        assert semaphore.acquire(blocking=False) is False

        # 释放信号量
        semaphore.release()
        assert semaphore.acquire(blocking=False) is True

    def test_adaptive_adjustment(self):
        """测试自适应调整"""
        semaphore = AdaptiveSemaphore(initial_value=2, min_value=1, max_value=5)

        # 模拟快速任务，应该增加容量
        for _ in range(15):
            semaphore.acquire()
            semaphore.release(task_duration=0.5)  # 快速任务

        # 强制调整
        semaphore._last_adjustment = datetime.now() - timedelta(seconds=31)
        semaphore._maybe_adjust_capacity()

        # 检查统计信息
        stats = semaphore.get_stats()
        assert "current_capacity" in stats
        assert "avg_task_time" in stats


class TestConcurrentTaskManager:
    """并发任务管理器测试"""

    @pytest.mark.asyncio
    async def test_async_task_execution(self):
        """测试异步任务执行"""
        manager = ConcurrentTaskManager(max_concurrent=2)

        async def test_coro():
            await asyncio.sleep(0.1)
            return "test_result"

        # 执行异步任务
        task_info = await manager.execute_async_task(
            test_coro(), name="test_task", priority=TaskPriority.HIGH
        )

        assert task_info.status == TaskStatus.COMPLETED
        assert task_info.result == "test_result"
        assert task_info.name == "test_task"
        assert task_info.priority == TaskPriority.HIGH

    @pytest.mark.asyncio
    async def test_async_task_timeout(self):
        """测试异步任务超时"""
        manager = ConcurrentTaskManager(max_concurrent=2)

        async def slow_coro():
            await asyncio.sleep(2.0)
            return "should_not_reach"

        # 执行带超时的异步任务
        task_info = await manager.execute_async_task(
            slow_coro(), name="slow_task", timeout=0.5
        )

        assert task_info.status == TaskStatus.FAILED
        assert isinstance(task_info.error, asyncio.TimeoutError)

    def test_sync_task_execution(self):
        """测试同步任务执行"""
        manager = ConcurrentTaskManager(max_concurrent=2)

        def test_func(x, y):
            time.sleep(0.1)
            return x + y

        # 执行同步任务
        task_info = manager.execute_sync_task(test_func, args=(1, 2), name="sync_task")

        assert task_info.status == TaskStatus.COMPLETED
        assert task_info.result == 3
        assert task_info.name == "sync_task"

    @pytest.mark.asyncio
    async def test_batch_async_execution(self):
        """测试批量异步执行"""
        manager = ConcurrentTaskManager(max_concurrent=3)

        async def test_coro(value):
            await asyncio.sleep(0.1)
            return value * 2

        # 批量执行异步任务
        coros = [test_coro(i) for i in range(3)]
        task_infos = await manager.execute_batch_async(
            coros, names=[f"task_{i}" for i in range(3)]
        )

        assert len(task_infos) == 3
        for i, task_info in enumerate(task_infos):
            assert task_info.status == TaskStatus.COMPLETED
            assert task_info.result == i * 2

    def test_batch_sync_execution(self):
        """测试批量同步执行"""
        manager = ConcurrentTaskManager(max_concurrent=3)

        def test_func(value):
            time.sleep(0.1)
            return value * 2

        # 批量执行同步任务
        funcs = [test_func] * 3
        args_list = [(i,) for i in range(3)]
        task_infos = manager.execute_batch_sync(
            funcs, args_list=args_list, names=[f"sync_task_{i}" for i in range(3)]
        )

        assert len(task_infos) == 3
        # 检查所有任务都完成，但不要求特定顺序
        completed_results = [
            task_info.result
            for task_info in task_infos
            if task_info.status == TaskStatus.COMPLETED
        ]
        assert len(completed_results) == 3
        assert all(result in [0, 2, 4] for result in completed_results)

    def test_task_management(self):
        """测试任务管理"""
        manager = ConcurrentTaskManager(max_concurrent=2)

        def test_func():
            return "result"

        # 执行任务
        task_info = manager.execute_sync_task(test_func, name="test_task")

        # 获取任务信息
        retrieved_info = manager.get_task_info(task_info.task_id)
        assert retrieved_info is task_info

        # 获取不存在的任务
        assert manager.get_task_info("nonexistent") is None

    def test_performance_stats(self):
        """测试性能统计"""
        manager = ConcurrentTaskManager(max_concurrent=2)

        def test_func():
            time.sleep(0.1)
            return "result"

        # 执行一些任务
        for i in range(3):
            manager.execute_sync_task(test_func, name=f"task_{i}")

        # 获取性能统计
        stats = manager.get_performance_stats()
        assert stats["total_tasks"] == 3
        assert stats["completed_tasks"] == 3
        assert stats["success_rate"] == 1.0
        assert stats["avg_execution_time"] > 0

    def test_cleanup_completed_tasks(self):
        """测试清理已完成任务"""
        manager = ConcurrentTaskManager(max_concurrent=2)

        def test_func():
            return "result"

        # 执行任务
        task_info = manager.execute_sync_task(test_func, name="test_task")

        # 模拟旧任务
        task_info.created_at = datetime.now() - timedelta(hours=25)

        # 清理旧任务
        cleaned = manager.cleanup_completed_tasks(max_age_hours=24)
        assert cleaned == 1
        assert manager.get_task_info(task_info.task_id) is None

    def test_manager_shutdown(self):
        """测试管理器关闭"""
        manager = ConcurrentTaskManager(max_concurrent=2)

        # 执行一个任务以确保线程池被初始化
        def test_func():
            return "result"

        manager.execute_sync_task(test_func, name="test_task")

        # 关闭管理器
        manager.shutdown(wait=True, timeout=5.0)

        # 验证线程池已关闭
        assert manager.thread_pool._shutdown is True


if __name__ == "__main__":
    pytest.main([__file__])
