import pytest
import asyncio
import hashlib
from unittest.mock import Mock, AsyncMock, patch
from src.utils.performance.parallel_executor import (
    ParallelExecutor,
    SharedTaskContext,
    ParallelTask,
    TaskStatus,
    create_parallel_executor,
)


class TestSharedTaskContext:
    """Test SharedTaskContext functionality"""

    def test_init(self):
        """Test SharedTaskContext initialization"""
        context = SharedTaskContext()
        assert context.search_results_cache == {}
        assert context.content_hashes == set()
        assert context.task_outputs == {}
        assert context.max_cache_size == 100

    def test_add_search_result(self):
        """Test adding search results to cache"""
        context = SharedTaskContext()
        query_hash = "test_hash"
        result = {"data": "test_result"}

        context.add_search_result(query_hash, result)

        assert query_hash in context.search_results_cache
        assert context.search_results_cache[query_hash] == result

    def test_add_search_result_cache_limit(self):
        """Test cache size limit enforcement"""
        context = SharedTaskContext(max_cache_size=2)

        # Add items up to limit
        context.add_search_result("hash1", "result1")
        context.add_search_result("hash2", "result2")
        assert len(context.search_results_cache) == 2

        # Add one more to trigger eviction
        context.add_search_result("hash3", "result3")
        assert len(context.search_results_cache) == 2
        assert "hash1" not in context.search_results_cache  # Oldest removed
        assert "hash3" in context.search_results_cache

    def test_get_search_result(self):
        """Test retrieving cached search results"""
        context = SharedTaskContext()
        query_hash = "test_hash"
        result = {"data": "test_result"}

        # Test cache miss
        assert context.get_search_result(query_hash) is None

        # Test cache hit
        context.add_search_result(query_hash, result)
        assert context.get_search_result(query_hash) == result

    def test_is_content_duplicate_exact_match(self):
        """Test exact content duplication detection"""
        context = SharedTaskContext()
        content = "This is test content"

        # First time should not be duplicate
        assert not context.is_content_duplicate(content)

        # Second time should be duplicate
        assert context.is_content_duplicate(content)

    def test_is_content_duplicate_similarity(self):
        """Test similarity-based duplication detection"""
        context = SharedTaskContext()

        # Add first content
        content1 = "The quick brown fox jumps over the lazy dog"
        context.add_task_output("task1", content1)

        # Similar content should be detected as duplicate
        content2 = "The quick brown fox jumps over the lazy cat"
        assert context.is_content_duplicate(content2, similarity_threshold=0.7)

        # Very different content should not be duplicate
        content3 = "Completely different text about something else entirely"
        assert not context.is_content_duplicate(content3, similarity_threshold=0.7)

    def test_add_task_output(self):
        """Test adding task outputs with deduplication"""
        context = SharedTaskContext()

        # Add unique content
        content1 = "Unique content 1"
        context.add_task_output("task1", content1)
        assert "task1" in context.task_outputs

        # Add duplicate content
        context.add_task_output("task2", content1)
        assert "task2" not in context.task_outputs  # Should be skipped


class TestParallelExecutor:
    """Test ParallelExecutor functionality"""

    @pytest.fixture
    def mock_rate_limiter(self):
        """Mock rate limiter"""
        limiter = Mock()
        limiter.acquire = AsyncMock()
        limiter.get_stats.return_value = {"total_delays": 0, "average_delay": 0.0}
        return limiter

    @pytest.fixture
    def shared_context(self):
        """Shared context fixture"""
        return SharedTaskContext()

    @pytest.fixture
    def executor(self, mock_rate_limiter, shared_context):
        """ParallelExecutor fixture"""
        return ParallelExecutor(
            max_concurrent_tasks=2,
            rate_limiter=mock_rate_limiter,
            enable_adaptive_scheduling=False,
            shared_context=shared_context,
        )

    def test_init(self, mock_rate_limiter, shared_context):
        """Test ParallelExecutor initialization"""
        executor = ParallelExecutor(
            max_concurrent_tasks=3,
            rate_limiter=mock_rate_limiter,
            shared_context=shared_context,
        )

        assert executor.max_concurrent_tasks == 3
        assert executor.rate_limiter == mock_rate_limiter
        assert executor.shared_context == shared_context
        assert executor.pending_tasks == []
        assert executor.running_tasks == {}
        assert executor.completed_tasks == {}

    @pytest.mark.asyncio
    async def test_execute_single_task_with_cache_hit(self, executor):
        """Test task execution with cache hit"""
        # Setup cached result
        query = "test query"
        query_hash = hashlib.md5(str(query).encode()).hexdigest()
        cached_result = "cached result"
        executor.shared_context.add_search_result(query_hash, cached_result)

        # Create task with query
        task = ParallelTask(
            task_id="test_task",
            func=lambda: "should not be called",
            kwargs={"query": query},
        )

        result = await executor._execute_single_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.result == cached_result
        assert result.execution_time < 0.01  # Should be very fast

    @pytest.mark.asyncio
    async def test_execute_single_task_with_caching(self, executor):
        """Test task execution with result caching"""
        query = "test query"
        expected_result = "task result"

        async def mock_task_func():
            return expected_result

        task = ParallelTask(
            task_id="test_task", func=mock_task_func, kwargs={"query": query}
        )

        result = await executor._execute_single_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.result == expected_result

        # Check that result was cached
        query_hash = hashlib.md5(str(query).encode()).hexdigest()
        cached_result = executor.shared_context.get_search_result(query_hash)
        assert cached_result == expected_result

    @pytest.mark.asyncio
    async def test_execute_single_task_content_deduplication(self, executor):
        """Test content deduplication during task execution"""
        content = "This is substantial content that should be checked for duplication"

        async def mock_task_func():
            return content

        task1 = ParallelTask(task_id="task1", func=mock_task_func)
        task2 = ParallelTask(task_id="task2", func=mock_task_func)

        # Execute first task
        result1 = await executor._execute_single_task(task1)
        assert result1.status == TaskStatus.COMPLETED
        assert "task1" in executor.shared_context.task_outputs

        # Execute second task with same content
        result2 = await executor._execute_single_task(task2)
        assert result2.status == TaskStatus.COMPLETED
        # Second task output should not be added due to deduplication
        assert "task2" not in executor.shared_context.task_outputs

    @pytest.mark.asyncio
    async def test_execute_single_task_error_handling(self, executor):
        """Test error handling in task execution"""

        async def failing_task():
            raise ValueError("Test error")

        task = ParallelTask(task_id="failing_task", func=failing_task, max_retries=1)

        result = await executor._execute_single_task(task)

        assert result.status == TaskStatus.FAILED
        assert "Test error" in result.error
        assert result.attempts == 2  # Initial attempt + 1 retry


class TestCreateParallelExecutor:
    """Test create_parallel_executor function"""

    @patch("src.utils.parallel_executor.get_global_rate_limiter")
    @patch("src.config.config_loader.config_loader.load_config")
    def test_create_parallel_executor_default(
        self, mock_load_config, mock_rate_limiter
    ):
        """Test creating executor with default settings"""
        mock_load_config.return_value = {"max_parallel_tasks": 5}
        mock_limiter = Mock()
        mock_rate_limiter.return_value = mock_limiter

        executor = create_parallel_executor()

        assert executor.max_concurrent_tasks == 5
        assert executor.rate_limiter == mock_limiter
        assert isinstance(executor.shared_context, SharedTaskContext)

    @patch("src.utils.parallel_executor.get_global_rate_limiter")
    def test_create_parallel_executor_with_shared_context(self, mock_rate_limiter):
        """Test creating executor with custom shared context"""
        mock_limiter = Mock()
        mock_rate_limiter.return_value = mock_limiter
        custom_context = SharedTaskContext(max_cache_size=50)

        executor = create_parallel_executor(
            max_concurrent_tasks=3, shared_context=custom_context
        )

        assert executor.max_concurrent_tasks == 3
        assert executor.shared_context == custom_context
        assert executor.shared_context.max_cache_size == 50

    @patch("src.utils.parallel_executor.get_global_rate_limiter")
    @patch("src.config.config_loader.config_loader.load_config")
    def test_create_parallel_executor_config_error(
        self, mock_load_config, mock_rate_limiter
    ):
        """Test creating executor when config loading fails"""
        mock_load_config.side_effect = Exception("Config error")
        mock_limiter = Mock()
        mock_rate_limiter.return_value = mock_limiter

        executor = create_parallel_executor()

        # Should use default value when config fails
        assert executor.max_concurrent_tasks == 3
        assert executor.rate_limiter == mock_limiter


class TestParallelTaskIntegration:
    """Integration tests for parallel task execution"""

    @pytest.mark.asyncio
    async def test_parallel_execution_with_deduplication(self):
        """Test full parallel execution with deduplication"""
        shared_context = SharedTaskContext()

        with patch(
            "src.utils.parallel_executor.get_global_rate_limiter"
        ) as mock_rate_limiter:
            mock_limiter = Mock()
            mock_limiter.acquire = AsyncMock()
            mock_limiter.get_stats.return_value = {
                "total_delays": 0,
                "average_delay": 0.0,
            }
            mock_rate_limiter.return_value = mock_limiter

            executor = ParallelExecutor(
                max_concurrent_tasks=2,
                rate_limiter=mock_limiter,
                shared_context=shared_context,
            )

            # Create tasks with same query (should use cache)
            async def search_task(query):
                await asyncio.sleep(0.01)  # Simulate work
                return f"Result for {query}"

            tasks = [
                ParallelTask(
                    task_id="task1",
                    func=search_task,
                    args=("same query",),
                    kwargs={"query": "same query"},
                ),
                ParallelTask(
                    task_id="task2",
                    func=search_task,
                    args=("same query",),
                    kwargs={"query": "same query"},
                ),
                ParallelTask(
                    task_id="task3",
                    func=search_task,
                    args=("different query",),
                    kwargs={"query": "different query"},
                ),
            ]

            executor.add_tasks(tasks)
            results = await executor.execute_all()

            assert len(results) == 3
            assert all(
                result.status == TaskStatus.COMPLETED for result in results.values()
            )

            # Check that cache was used
            query_hash = hashlib.md5("same query".encode()).hexdigest()
            assert shared_context.get_search_result(query_hash) is not None

    @pytest.mark.asyncio
    async def test_content_similarity_detection(self):
        """Test content similarity detection in real scenario"""
        shared_context = SharedTaskContext()

        # Add similar content
        content1 = (
            "The research shows that artificial intelligence is transforming healthcare"
        )
        content2 = "Research indicates that AI is revolutionizing healthcare systems"
        content3 = "Weather forecast shows sunny skies tomorrow"

        # First content should not be duplicate
        assert not shared_context.is_content_duplicate(content1)
        shared_context.add_task_output("task1", content1)

        # Similar content should be detected as duplicate
        assert shared_context.is_content_duplicate(content2, similarity_threshold=0.3)

        # Different content should not be duplicate
        assert not shared_context.is_content_duplicate(
            content3, similarity_threshold=0.3
        )
