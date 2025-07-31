"""Test cases for IterativeResearchEngine fixes"""

import unittest
from unittest.mock import MagicMock, patch
import asyncio


class TestIterativeResearchEngineFixes(unittest.TestCase):
    """Test cases for IterativeResearchEngine fixes"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_unified_config = MagicMock()
        self.mock_unified_config.max_follow_up_iterations = 3
        self.mock_unified_config.max_research_time_minutes = 30
        self.mock_unified_config.max_follow_up_queries = 3

    def test_execute_iterative_research_method_exists(self):
        """Test that execute_iterative_research method exists"""
        from src.utils.researcher.iterative_research_engine import (
            IterativeResearchEngine,
        )

        engine = IterativeResearchEngine(self.mock_unified_config)

        # Check that the method exists
        self.assertTrue(hasattr(engine, "execute_iterative_research"))
        self.assertTrue(callable(getattr(engine, "execute_iterative_research")))

    async def test_execute_iterative_research_basic_flow(self):
        """Test basic flow of execute_iterative_research method"""
        from src.utils.researcher.iterative_research_engine import (
            IterativeResearchEngine,
        )

        engine = IterativeResearchEngine(self.mock_unified_config)

        # Test basic execution
        query = "Test research query"
        state = {"follow_up_queries": ["query1", "query2"]}

        result = await engine.execute_iterative_research(query, state)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("terminated", result)
        self.assertIn("iteration", result)
        self.assertIn("query", result)
        self.assertIn("research_summary", result)

        # Should not be terminated on first iteration
        self.assertFalse(result["terminated"])
        self.assertEqual(result["iteration"], 1)
        self.assertEqual(result["query"], query)

    async def test_execute_iterative_research_termination_conditions(self):
        """Test termination conditions in execute_iterative_research"""
        from src.utils.researcher.iterative_research_engine import (
            IterativeResearchEngine,
        )

        # Set max iterations to 1 for quick termination
        self.mock_unified_config.max_follow_up_iterations = 1
        engine = IterativeResearchEngine(self.mock_unified_config)

        query = "Test research query"
        state = {"follow_up_queries": ["query1"]}

        # First execution should not terminate
        result1 = await engine.execute_iterative_research(query, state)
        self.assertFalse(result1["terminated"])

        # Second execution should terminate due to max iterations
        result2 = await engine.execute_iterative_research(query, state)
        self.assertTrue(result2["terminated"])
        self.assertIn("termination_reason", result2)
        self.assertIn("maximum iterations", result2["termination_reason"])

    async def test_execute_iterative_research_with_reflection(self):
        """Test execute_iterative_research with reflection result"""
        from src.utils.researcher.iterative_research_engine import (
            IterativeResearchEngine,
        )

        engine = IterativeResearchEngine(self.mock_unified_config)

        # Mock reflection result
        mock_reflection = MagicMock()
        mock_reflection.is_sufficient = False
        mock_reflection.knowledge_gaps = [
            MagicMock(suggested_query="Follow-up query 1"),
            MagicMock(description="Knowledge gap 2"),
        ]

        query = "Test research query"
        state = {"follow_up_queries": ["existing_query"]}

        result = await engine.execute_iterative_research(query, state, mock_reflection)

        # Verify reflection was processed
        self.assertIn("follow_up_queries", result)
        self.assertIsInstance(result["follow_up_queries"], list)

    def test_unified_config_attributes_fixed(self):
        """Test that unified config attributes are correctly referenced"""
        from src.utils.researcher.iterative_research_engine import (
            IterativeResearchEngine,
        )

        engine = IterativeResearchEngine(self.mock_unified_config)

        # Test check_termination_conditions uses correct attribute
        state = {"follow_up_queries": ["query1"]}
        should_terminate, reason, factors = engine.check_termination_conditions(state)

        # Should reference max_follow_up_iterations, not max_research_iterations
        self.assertEqual(
            factors["max_iterations"], self.mock_unified_config.max_follow_up_iterations
        )

        # Test get_research_summary uses correct attribute
        summary = engine.get_research_summary()
        self.assertEqual(
            summary["max_iterations"], self.mock_unified_config.max_follow_up_iterations
        )


class TestUnifiedResearchConfigFixes(unittest.TestCase):
    """Test cases for UnifiedResearchConfig fixes"""

    def test_unified_config_has_parallel_attributes(self):
        """Test that UnifiedResearchConfig has required parallel execution attributes"""
        from src.utils.researcher.isolation_config_manager import UnifiedResearchConfig

        config = UnifiedResearchConfig()

        # Check that new attributes exist
        self.assertTrue(hasattr(config, "max_context_steps_parallel"))
        self.assertTrue(hasattr(config, "disable_context_parallel"))

        # Check default values
        self.assertEqual(config.max_context_steps_parallel, 1)
        self.assertEqual(config.disable_context_parallel, False)

    @patch(
        "src.utils.researcher.isolation_config_manager.get_configuration_from_config"
    )
    def test_isolation_config_manager_builds_parallel_config(self, mock_get_config):
        """Test that IsolationConfigManager properly builds parallel configuration"""
        from src.utils.researcher.isolation_config_manager import IsolationConfigManager

        # Mock configurable object
        mock_configurable = MagicMock()
        mock_configurable.max_context_steps_parallel = 2
        mock_configurable.disable_context_parallel = True
        mock_configurable.enable_parallel_execution = True
        mock_configurable.max_parallel_tasks = 5

        # Set up other required attributes with defaults
        for attr in [
            "reflection_enabled",
            "reflection_max_loops",
            "reflection_model",
            "reflection_knowledge_gap_threshold",
            "reflection_sufficiency_threshold",
            "reflection_max_followup_queries",
            "reflection_skip_initial_stage",
            "reflection_temperature",
            "max_search_results",
            "enable_smart_filtering",
            "mcp_enabled",
            "mcp_servers",
            "max_step_num",
        ]:
            if not hasattr(mock_configurable, attr):
                setattr(mock_configurable, attr, None)

        mock_get_config.return_value = mock_configurable

        manager = IsolationConfigManager(MagicMock(), MagicMock())
        unified_config = manager.get_unified_config()

        # Verify parallel configuration attributes
        self.assertEqual(unified_config.max_context_steps_parallel, 2)
        self.assertEqual(unified_config.disable_context_parallel, True)
        self.assertEqual(unified_config.enable_parallel_execution, True)
        self.assertEqual(unified_config.max_parallel_tasks, 5)

    @patch(
        "src.utils.researcher.isolation_config_manager.get_configuration_from_config"
    )
    async def test_isolation_context_includes_parallel_config(self, mock_get_config):
        """Test that isolation context includes parallel configuration"""
        from src.utils.researcher.isolation_config_manager import IsolationConfigManager

        # Mock configurable object with minimal required attributes
        mock_configurable = MagicMock()
        for attr in [
            "reflection_enabled",
            "reflection_max_loops",
            "reflection_model",
            "reflection_knowledge_gap_threshold",
            "reflection_sufficiency_threshold",
            "reflection_max_followup_queries",
            "max_follow_up_iterations",
            "enable_iterative_research",
            "max_queries_per_iteration",
            "follow_up_delay_seconds",
            "max_search_results",
            "enable_smart_filtering",
            "mcp_enabled",
            "mcp_servers",
            "max_step_num",
            "max_context_steps_parallel",
            "disable_context_parallel",
            "enable_parallel_execution",
            "max_parallel_tasks",
        ]:
            setattr(mock_configurable, attr, getattr(mock_configurable, attr, None))

        mock_get_config.return_value = mock_configurable

        manager = IsolationConfigManager(MagicMock(), MagicMock())
        isolation_context = await manager.setup_isolation_context()

        # Verify parallel_config is included
        self.assertIn("parallel_config", isolation_context)
        parallel_config = isolation_context["parallel_config"]

        # Verify all parallel configuration attributes are included
        self.assertIn("enable_parallel_execution", parallel_config)
        self.assertIn("max_parallel_tasks", parallel_config)
        self.assertIn("max_context_steps_parallel", parallel_config)
        self.assertIn("disable_context_parallel", parallel_config)


if __name__ == "__main__":
    # Run async tests
    async def run_async_tests():
        test_instance = TestIterativeResearchEngineFixes()
        test_instance.setUp()

        await test_instance.test_execute_iterative_research_basic_flow()
        await test_instance.test_execute_iterative_research_termination_conditions()
        await test_instance.test_execute_iterative_research_with_reflection()

        config_test_instance = TestUnifiedResearchConfigFixes()
        await config_test_instance.test_isolation_context_includes_parallel_config()

        print("All async tests passed!")

    # Run sync tests
    unittest.main(verbosity=2, exit=False)

    # Run async tests
    asyncio.run(run_async_tests())
