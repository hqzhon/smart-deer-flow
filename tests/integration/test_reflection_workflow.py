"""Integration tests for reflection workflow."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.utils.reflection.reflection_manager import (
    get_reflection_manager,
    reset_reflection_manager,
)
from src.graph.nodes import reflection_node, update_plan_node


class TestReflectionWorkflowIntegration:
    """Integration tests for the complete reflection workflow."""

    def setup_method(self):
        """Reset reflection manager before each test."""
        reset_reflection_manager()

    def teardown_method(self):
        """Clean up after each test."""
        reset_reflection_manager()

    @pytest.fixture
    def base_state(self):
        """Create base state for testing."""
        return {
            "research_result": "Initial research findings",
            "session_id": "integration_test_session",
            "current_step": Mock(title="Research Step 1"),
            "research_topic": "AI Integration Testing",
            "observations": ["Initial observation"],
            "current_plan": Mock(model_copy=Mock(return_value=Mock(steps=[]))),
            "reflection": {},
        }

    @pytest.fixture
    def base_config(self):
        """Create base config for testing."""
        return {"configurable": {"session_id": "integration_test_session"}}

    @pytest.fixture
    def mock_configurable(self):
        """Create mock configurable settings."""
        mock_reflection = Mock()
        mock_reflection.enabled = True
        mock_reflection.max_loops = 2

        mock_configurable = Mock()
        mock_configurable.reflection = mock_reflection

        return mock_configurable

    @patch("src.graph.nodes.get_configuration_from_config")
    @patch("src.utils.reflection.enhanced_reflection.EnhancedReflectionAgent")
    async def test_single_reflection_execution(
        self,
        mock_enhanced_reflection_agent,
        mock_get_config,
        base_state,
        base_config,
        mock_configurable,
    ):
        """Test single reflection execution within limit."""
        # Setup mocks
        mock_get_config.return_value = mock_configurable

        mock_agent_instance = Mock()
        mock_agent_instance.analyze_knowledge_gaps = AsyncMock(
            return_value={
                "is_sufficient": False,
                "primary_follow_up_query": "What about edge cases?",
                "reason": "Need more analysis",
            }
        )
        mock_enhanced_reflection_agent.return_value = mock_agent_instance

        # Get reflection manager and verify initial state
        manager = get_reflection_manager(max_reflections=2)
        can_execute, reason = manager.can_execute_reflection("integration_test_session")
        assert can_execute is True
        assert "remaining 2 times" in reason

        # Execute reflection
        result = await reflection_node(base_state, base_config)

        # Verify reflection was executed and recorded
        assert "reflection_insights" in result
        assert result["reflection_insights"]["is_sufficient"] is False

        # Verify manager state updated
        summary = manager.get_summary("integration_test_session")
        assert summary["total_reflections"] == 1
        assert summary["remaining"] == 1
        assert summary["is_limit_reached"] is False

    @patch("src.graph.nodes.get_configuration_from_config")
    @patch("src.utils.reflection.enhanced_reflection.EnhancedReflectionAgent")
    async def test_reflection_limit_enforcement(
        self,
        mock_enhanced_reflection_agent,
        mock_get_config,
        base_state,
        base_config,
        mock_configurable,
    ):
        """Test reflection limit enforcement across multiple executions."""
        # Setup mocks
        mock_get_config.return_value = mock_configurable

        mock_agent_instance = Mock()
        mock_agent_instance.analyze_knowledge_gaps = AsyncMock(
            return_value={
                "is_sufficient": False,
                "primary_follow_up_query": "More questions",
                "reason": "Need deeper analysis",
            }
        )
        mock_enhanced_reflection_agent.return_value = mock_agent_instance

        # Set max_loops to 1 for this test
        mock_configurable.reflection.max_loops = 1

        # First reflection should succeed
        result1 = await reflection_node(base_state, base_config)
        assert "reflection_insights" in result1
        assert result1["reflection_insights"]["is_sufficient"] is False

        # Second reflection should be blocked
        result2 = await reflection_node(base_state, base_config)
        assert "reflection_insights" in result2
        assert result2["reflection_insights"]["is_sufficient"] is True
        assert (
            "Maximum reflection count limit reached"
            in result2["reflection_insights"]["reason"]
        )

        # Verify manager state
        manager = get_reflection_manager()
        summary = manager.get_summary("integration_test_session")
        assert summary["total_reflections"] == 1
        assert summary["is_limit_reached"] is True

    @patch("src.graph.nodes.get_configuration_from_config")
    @patch("src.utils.reflection.enhanced_reflection.EnhancedReflectionAgent")
    @patch("src.utils.researcher.isolation_config_manager.IsolationConfigManager")
    @patch("src.utils.context.execution_context_manager.ExecutionContextManager")
    async def test_update_plan_node_no_longer_increments_count(
        self,
        mock_context_manager,
        mock_config_manager,
        mock_enhanced_reflection_agent,
        mock_get_config,
        base_state,
        base_config,
        mock_configurable,
    ):
        """Test that update_plan_node no longer increments reflection count."""
        # Setup mocks
        mock_get_config.return_value = mock_configurable

        # Mock context managers
        mock_config_instance = Mock()
        mock_config_instance.get_unified_config.return_value = Mock(
            max_context_steps_parallel=10
        )
        mock_config_manager.return_value = mock_config_instance

        mock_context_instance = Mock()
        mock_context_instance.manage_observations_advanced.return_value = [
            "Managed observation"
        ]
        mock_context_manager.return_value = mock_context_instance

        # Create reflection insights with primary_follow_up_query
        reflection_insights = Mock()
        reflection_insights.primary_follow_up_query = "Follow up question"
        reflection_insights.is_sufficient = False

        # Prepare state for update_plan_node
        update_state = {**base_state, "reflection": {"count": 0}}  # Initial count

        # Execute update_plan_node

        # Prepare complete state for update_plan_node
        update_state["research_result"] = "Test research result"
        update_state["reflection_insights"] = reflection_insights

        result = await update_plan_node(update_state, base_config)

        # Verify that reflection count was NOT incremented
        # (ReflectionManager now handles counting)
        assert "reflection" in result
        reflection_state = result["reflection"]
        # The count should remain as it was in the original state
        # since update_plan_node no longer modifies it
        assert reflection_state.get("count", 0) == 0

    async def test_session_isolation(self):
        """Test that different sessions have independent reflection counts."""
        # Get manager with limit of 1
        manager = get_reflection_manager(max_reflections=1)

        # Session 1: Execute reflection
        can_execute1, _ = manager.can_execute_reflection("session_1")
        assert can_execute1 is True

        success1 = manager.record_reflection("session_1")
        assert success1 is True

        # Session 1 should now be at limit
        can_execute1_after, reason1 = manager.can_execute_reflection("session_1")
        assert can_execute1_after is False
        assert "Maximum reflection count limit reached" in reason1

        # Session 2 should still be able to execute
        can_execute2, reason2 = manager.can_execute_reflection("session_2")
        assert can_execute2 is True
        assert "remaining 1 times" in reason2

        # Execute reflection for session 2
        success2 = manager.record_reflection("session_2")
        assert success2 is True

        # Both sessions should now be at limit
        can_execute1_final, _ = manager.can_execute_reflection("session_1")
        can_execute2_final, _ = manager.can_execute_reflection("session_2")
        assert can_execute1_final is False
        assert can_execute2_final is False

        # Verify independent summaries
        summary1 = manager.get_summary("session_1")
        summary2 = manager.get_summary("session_2")

        assert summary1["session_id"] == "session_1"
        assert summary2["session_id"] == "session_2"
        assert summary1["total_reflections"] == 1
        assert summary2["total_reflections"] == 1

    @patch("src.graph.nodes.get_configuration_from_config")
    @patch("src.utils.reflection.enhanced_reflection.EnhancedReflectionAgent")
    @patch("src.utils.researcher.reflection_system_manager.ReflectionSystemManager")
    async def test_reflection_workflow_with_different_limits(
        self,
        mock_system_manager,
        mock_enhanced_reflection_agent,
        mock_get_config,
        base_state,
        base_config,
    ):
        """Test reflection workflow with different max_loops configurations."""
        # Test with max_loops = 3
        mock_reflection = Mock()
        mock_reflection.enabled = True
        mock_reflection.max_loops = 3

        mock_configurable = Mock()
        mock_configurable.reflection = mock_reflection
        mock_get_config.return_value = mock_configurable

        mock_system_instance = Mock()
        mock_system_instance.initialize_reflection_system = AsyncMock()
        mock_system_instance.execute_reflection_analysis = AsyncMock(
            return_value={
                "is_sufficient": False,
                "primary_follow_up_query": "Question",
                "reason": "More analysis needed",
            }
        )
        mock_system_manager.return_value = mock_system_instance

        # Execute reflections up to the limit
        for i in range(3):
            result = await reflection_node(base_state, base_config)
            assert "reflection_insights" in result
            assert result["reflection_insights"]["is_sufficient"] is False

        # Fourth reflection should be blocked
        result = await reflection_node(base_state, base_config)
        assert "reflection_insights" in result
        assert result["reflection_insights"]["is_sufficient"] is True
        assert (
            "Maximum reflection count limit reached"
            in result["reflection_insights"]["reason"]
        )

        # Verify final state
        manager = get_reflection_manager()
        summary = manager.get_summary("integration_test_session")
        assert summary["total_reflections"] == 3
        assert summary["max_allowed"] == 3
        assert summary["is_limit_reached"] is True

    async def test_reflection_manager_configuration_consistency(self):
        """Test that ReflectionManager uses consistent configuration."""
        # Test with different max_reflections values
        manager1 = get_reflection_manager(max_reflections=2)
        assert manager1.max_reflections == 2

        # Getting manager again should return the same instance
        manager2 = get_reflection_manager(max_reflections=3)  # Different value
        assert manager2 is manager1  # Same instance
        assert manager2.max_reflections == 2  # Original value preserved

        # Reset and create new manager
        reset_reflection_manager()
        manager3 = get_reflection_manager(max_reflections=3)
        assert manager3.max_reflections == 3
        assert manager3 is not manager1  # New instance

    @patch("src.graph.nodes.get_configuration_from_config")
    @patch("src.utils.reflection.enhanced_reflection.EnhancedReflectionAgent")
    @patch("src.utils.researcher.reflection_system_manager.ReflectionSystemManager")
    async def test_reflection_error_handling(
        self,
        mock_system_manager,
        mock_enhanced_reflection_agent,
        mock_get_config,
        base_state,
        base_config,
        mock_configurable,
    ):
        """Test reflection error handling doesn't affect count."""
        # Setup mocks
        mock_get_config.return_value = mock_configurable

        mock_system_instance = Mock()
        mock_system_instance.initialize_reflection_system = AsyncMock()
        mock_system_instance.execute_reflection_analysis = AsyncMock(
            side_effect=Exception("Reflection analysis failed")
        )
        mock_system_manager.return_value = mock_system_instance

        # Get initial manager state
        manager = get_reflection_manager(max_reflections=2)
        initial_summary = manager.get_summary("integration_test_session")
        assert initial_summary["total_reflections"] == 0

        # Execute reflection (should fail gracefully)
        result = await reflection_node(base_state, base_config)

        # Verify graceful error handling
        assert "reflection_insights" in result
        assert result["reflection_insights"]["is_sufficient"] is True
        assert "Reflection failed" in result["reflection_insights"]["reason"]

        # Verify count was not incremented due to error
        # (record_reflection is only called after successful execution)
        final_summary = manager.get_summary("integration_test_session")
        assert final_summary["total_reflections"] == 0

        # Verify can still execute reflection
        can_execute, reason = manager.can_execute_reflection("integration_test_session")
        assert can_execute is True
        assert "remaining 2 times" in reason
