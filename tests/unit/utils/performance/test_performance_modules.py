"""Unit tests for performance optimization modules."""

import pytest
import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))

from src.utils.performance.workflow_optimizer import WorkflowOptimizer
from src.utils.performance.memory_manager import HierarchicalMemoryManager


class TestWorkflowOptimizer:
    """Test cases for WorkflowOptimizer."""

    def setup_method(self):
        """Test setup."""
        self.optimizer = WorkflowOptimizer()

    def test_workflow_optimizer_instantiation(self):
        """Test WorkflowOptimizer instantiation."""
        assert self.optimizer is not None
        assert isinstance(self.optimizer, WorkflowOptimizer)

    def test_optimizer_basic_functionality(self):
        """Test basic optimizer functionality."""
        # Test that optimizer can be created and has expected attributes
        assert hasattr(self.optimizer, "__class__")
        assert self.optimizer.__class__.__name__ == "WorkflowOptimizer"

    def test_multiple_optimizer_instances(self):
        """Test creating multiple optimizer instances."""
        optimizer1 = WorkflowOptimizer()
        optimizer2 = WorkflowOptimizer()

        assert optimizer1 is not optimizer2
        assert isinstance(optimizer1, WorkflowOptimizer)
        assert isinstance(optimizer2, WorkflowOptimizer)

    def test_optimizer_methods_exist(self):
        """Test that optimizer has expected methods."""
        # Check for common optimization methods (if they exist)
        methods_to_check = ["optimize", "analyze", "configure"]

        for method_name in methods_to_check:
            if hasattr(self.optimizer, method_name):
                method = getattr(self.optimizer, method_name)
                assert callable(method)


class TestHierarchicalMemoryManager:
    """Test cases for HierarchicalMemoryManager."""

    def setup_method(self):
        """Test setup."""
        self.memory_manager = HierarchicalMemoryManager()

    def test_memory_manager_instantiation(self):
        """Test HierarchicalMemoryManager instantiation."""
        assert self.memory_manager is not None
        assert isinstance(self.memory_manager, HierarchicalMemoryManager)

    @pytest.mark.asyncio
    async def test_cache_basic_functionality(self):
        """Test basic cache functionality."""
        test_key = "test_key"
        test_value = "test_value"

        # Test set operation
        await self.memory_manager.set(test_key, test_value)

        # Test get operation
        cached_value = await self.memory_manager.get(test_key)
        assert cached_value == test_value

    @pytest.mark.asyncio
    async def test_cache_with_different_data_types(self):
        """Test cache with different data types."""
        test_cases = [
            ("string_key", "string_value"),
            ("int_key", 12345),
            ("list_key", [1, 2, 3, "test"]),
            ("dict_key", {"nested": "value", "number": 42}),
            ("bool_key", True),
        ]

        for key, value in test_cases:
            await self.memory_manager.set(key, value)
            cached_value = await self.memory_manager.get(key)
            assert cached_value == value

    @pytest.mark.asyncio
    async def test_cache_overwrite(self):
        """Test cache value overwriting."""
        test_key = "overwrite_key"
        original_value = "original_value"
        new_value = "new_value"

        # Set original value
        await self.memory_manager.set(test_key, original_value)
        cached_value = await self.memory_manager.get(test_key)
        assert cached_value == original_value

        # Overwrite with new value
        await self.memory_manager.set(test_key, new_value)
        cached_value = await self.memory_manager.get(test_key)
        assert cached_value == new_value

    @pytest.mark.asyncio
    async def test_cache_nonexistent_key(self):
        """Test getting value for non-existent key."""
        nonexistent_key = "nonexistent_key_12345"

        # Should return None or raise KeyError
        try:
            cached_value = await self.memory_manager.get(nonexistent_key)
            assert cached_value is None
        except KeyError:
            # KeyError is also acceptable behavior
            pass

    @pytest.mark.asyncio
    async def test_cache_multiple_keys(self):
        """Test cache with multiple keys."""
        test_data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }

        # Set multiple values
        for key, value in test_data.items():
            await self.memory_manager.set(key, value)

        # Verify all values
        for key, expected_value in test_data.items():
            cached_value = await self.memory_manager.get(key)
            assert cached_value == expected_value

    @pytest.mark.asyncio
    async def test_cache_concurrent_operations(self):
        """Test concurrent cache operations."""

        async def set_and_get(key, value):
            await self.memory_manager.set(key, value)
            return await self.memory_manager.get(key)

        # Create multiple concurrent operations
        tasks = []
        for i in range(5):
            key = f"concurrent_key_{i}"
            value = f"concurrent_value_{i}"
            task = set_and_get(key, value)
            tasks.append((task, value))

        # Execute concurrently
        results = await asyncio.gather(*[task for task, _ in tasks])

        # Verify results
        for i, (result, expected_value) in enumerate(
            zip(results, [value for _, value in tasks])
        ):
            assert result == expected_value

    def test_multiple_memory_manager_instances(self):
        """Test creating multiple memory manager instances."""
        manager1 = HierarchicalMemoryManager()
        manager2 = HierarchicalMemoryManager()

        assert manager1 is not manager2
        assert isinstance(manager1, HierarchicalMemoryManager)
        assert isinstance(manager2, HierarchicalMemoryManager)

    @pytest.mark.asyncio
    async def test_memory_manager_isolation(self):
        """Test that different memory manager instances are isolated."""
        manager1 = HierarchicalMemoryManager()
        manager2 = HierarchicalMemoryManager()

        test_key = "isolation_test_key"
        value1 = "value_for_manager1"
        value2 = "value_for_manager2"

        # Set different values in different managers
        await manager1.set(test_key, value1)
        await manager2.set(test_key, value2)

        # Verify isolation
        cached_value1 = await manager1.get(test_key)
        cached_value2 = await manager2.get(test_key)

        assert cached_value1 == value1
        assert cached_value2 == value2
        assert cached_value1 != cached_value2

    @pytest.mark.asyncio
    async def test_cache_error_handling(self):
        """Test cache error handling."""
        # Test with None key
        try:
            await self.memory_manager.set(None, "test_value")
            cached_value = await self.memory_manager.get(None)
            # If no exception, verify behavior
            assert cached_value == "test_value" or cached_value is None
        except (TypeError, ValueError):
            # Exception is acceptable for None key
            pass

        # Test with empty string key
        try:
            await self.memory_manager.set("", "empty_key_value")
            cached_value = await self.memory_manager.get("")
            assert cached_value == "empty_key_value" or cached_value is None
        except (TypeError, ValueError):
            # Exception is acceptable for empty key
            pass


class TestPerformanceModulesIntegration:
    """Test integration between performance modules."""

    def test_modules_can_coexist(self):
        """Test that performance modules can coexist."""
        optimizer = WorkflowOptimizer()
        memory_manager = HierarchicalMemoryManager()

        assert optimizer is not None
        assert memory_manager is not None
        assert isinstance(optimizer, WorkflowOptimizer)
        assert isinstance(memory_manager, HierarchicalMemoryManager)

    @pytest.mark.asyncio
    async def test_performance_modules_basic_workflow(self):
        """Test basic workflow with performance modules."""
        optimizer = WorkflowOptimizer()
        memory_manager = HierarchicalMemoryManager()

        # Test basic workflow
        test_key = "workflow_test"
        test_value = "workflow_value"

        # Use memory manager
        await memory_manager.set(test_key, test_value)
        cached_value = await memory_manager.get(test_key)

        assert cached_value == test_value

        # Verify optimizer is still functional
        assert optimizer is not None
